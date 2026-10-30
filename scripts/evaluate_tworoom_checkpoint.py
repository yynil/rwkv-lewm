import argparse
import json
import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import build_model, load_stable_worldmodel_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="tworoom_stage2_ds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--split", choices=["tail", "head"], default="tail")
    parser.add_argument("--output", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_cfg(config_name: str, overrides: list[str]):
    loader_overrides = [
        "loader.num_workers=0",
        "loader.persistent_workers=false",
        "loader.prefetch_factor=null",
    ]
    with hydra.initialize(version_base=None, config_path="../config/train"):
        return hydra.compose(config_name=config_name, overrides=loader_overrides + overrides)


def resolve_model_state(path: str) -> Path:
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "mp_rank_00_model_states.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt}")
    return ckpt


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "module" in state:
        return state["module"]
    return state


def make_eval_subset(dataset, eval_samples: int, split: str):
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")
    k = min(eval_samples, n)
    if split == "tail":
        indices = range(n - k, n)
    else:
        indices = range(0, k)
    return Subset(dataset, list(indices))


def update_sums(sums: dict[str, float], count: int, **values: torch.Tensor):
    for key, value in values.items():
        sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    cfg = load_cfg(args.config_name, args.overrides)
    model = build_model(cfg).to(device)
    ckpt = resolve_model_state(args.checkpoint)
    model.load_state_dict(load_state_dict(ckpt), strict=True)
    model.eval()

    dataset = load_stable_worldmodel_dataset(cfg)
    subset = make_eval_subset(dataset, args.eval_samples, args.split)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )

    sums: dict[str, float] = {}
    total = 0
    total_transitions = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            out = model.encode(batch)
            emb = out["emb"].float()
            act_emb = out["act_emb"]
            pred, _ = model.predict(out["emb"][:, :-1], act_emb[:, :-1])
            pred = pred.float()
            target = emb[:, 1:]
            prev = emb[:, :-1]

            pred_delta = pred - prev
            target_delta = target - prev
            pred_mse = (pred - target).pow(2).mean()
            copy_mse = (prev - target).pow(2).mean()
            pred_cos = F.cosine_similarity(pred.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()
            delta_cos = F.cosine_similarity(
                pred_delta.flatten(0, 1), target_delta.flatten(0, 1), dim=-1
            ).mean()
            delta_mse = (pred_delta - target_delta).pow(2).mean()
            emb_std = emb.std()
            temporal_mse = target_delta.pow(2).mean()
            pred_delta_norm = pred_delta.norm(dim=-1).mean()
            target_delta_norm = target_delta.norm(dim=-1).mean()

            b = emb.size(0)
            transitions = pred.numel() // pred.size(-1)
            total += b
            total_transitions += transitions
            update_sums(
                sums,
                b,
                pred_mse=pred_mse,
                copy_mse=copy_mse,
                delta_mse=delta_mse,
                pred_cos=pred_cos,
                delta_cos=delta_cos,
                emb_std=emb_std,
                temporal_mse=temporal_mse,
                pred_delta_norm=pred_delta_norm,
                target_delta_norm=target_delta_norm,
            )

    metrics = {key: value / total for key, value in sums.items()}
    metrics["relative_mse_vs_copy"] = metrics["pred_mse"] / max(metrics["copy_mse"], 1e-12)
    metrics["copy_improvement"] = 1.0 - metrics["relative_mse_vs_copy"]
    metrics["num_sequences"] = total
    metrics["num_transitions"] = total_transitions
    metrics["dataset_len"] = len(dataset)
    metrics["checkpoint"] = str(ckpt)
    metrics["config_name"] = args.config_name
    metrics["split"] = args.split

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
