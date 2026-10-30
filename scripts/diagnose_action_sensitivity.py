import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import ninja

    os.environ["PATH"] = f"{ninja.BIN_DIR}:{os.environ.get('PATH', '')}"
except Exception:
    pass

from train import build_model, load_stable_worldmodel_dataset


def load_cfg(config_name: str, overrides: list[str]):
    config_path = ROOT / "config" / "train"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_path)):
        return hydra.compose(config_name=config_name, overrides=overrides)


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


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).square().mean()


def norm_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).norm(dim=-1).mean()


def batch_metrics(model, batch, history_size: int, num_preds: int, noise_scale: float, generator: torch.Generator):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    with torch.no_grad():
        output = model.encode(batch)
        emb = output["emb"].detach()
        act_emb = output["act_emb"].detach()

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    target = emb[:, num_preds:]
    if target.shape != ctx_emb.shape:
        raise ValueError(f"expected target shape {tuple(ctx_emb.shape)}, got {tuple(target.shape)}")

    with torch.no_grad():
        base_pred, _ = model.predict(ctx_emb, ctx_act)
        base_loss = mse(base_pred, target)
        base_target_norm = norm_mean(base_pred, target)
        temporal_norm = norm_mean(target, ctx_emb)

        perm_b = torch.randperm(ctx_act.size(0), generator=generator, device=ctx_act.device)
        perm_t = torch.randperm(ctx_act.size(1), generator=generator, device=ctx_act.device)
        action_noise = torch.randn(
            ctx_act.shape,
            generator=generator,
            device=ctx_act.device,
            dtype=ctx_act.dtype,
        ) * ctx_act.float().std().clamp_min(1e-6).to(dtype=ctx_act.dtype) * noise_scale
        image_noise = torch.randn(
            ctx_emb.shape,
            generator=generator,
            device=ctx_emb.device,
            dtype=ctx_emb.dtype,
        ) * ctx_emb.float().std().clamp_min(1e-6).to(dtype=ctx_emb.dtype) * noise_scale

        variants = {
            "action_zero": (ctx_emb, torch.zeros_like(ctx_act)),
            "action_shuffle_batch": (ctx_emb, ctx_act[perm_b]),
            "action_shuffle_time": (ctx_emb, ctx_act[:, perm_t]),
            "action_noise": (ctx_emb, ctx_act + action_noise),
            "image_zero": (torch.zeros_like(ctx_emb), ctx_act),
            "image_shuffle_batch": (ctx_emb[perm_b], ctx_act),
            "image_noise": (ctx_emb + image_noise, ctx_act),
        }

        out = {
            "base_loss": base_loss,
            "base_target_norm": base_target_norm,
            "temporal_norm": temporal_norm,
            "ctx_emb_norm": ctx_emb.float().norm(dim=-1).mean(),
            "ctx_act_norm": ctx_act.float().norm(dim=-1).mean(),
        }
        for name, (alt_emb, alt_act) in variants.items():
            alt_pred, _ = model.predict(alt_emb, alt_act)
            out[f"{name}_pred_change_mse"] = mse(alt_pred, base_pred)
            out[f"{name}_pred_change_norm"] = norm_mean(alt_pred, base_pred)
            out[f"{name}_loss"] = mse(alt_pred, target)
            out[f"{name}_loss_delta"] = out[f"{name}_loss"] - base_loss

        if getattr(model, "token_fusion", None) == "interleave_action":
            tokens = model.make_interleaved_tokens(ctx_emb, ctx_act)
            out["image_token_count"] = torch.tensor(float(tokens[:, 0::2].size(1)), device=ctx_emb.device)
            out["action_token_count"] = torch.tensor(float(tokens[:, 1::2].size(1)), device=ctx_emb.device)
            out["image_token_norm"] = tokens[:, 0::2].float().norm(dim=-1).mean()
            out["action_token_norm"] = tokens[:, 1::2].float().norm(dim=-1).mean()
            if model.token_type_embeddings is not None:
                type_emb = model.token_type_embeddings.detach().float()
                out["type_image_norm"] = type_emb[0].norm().to(ctx_emb.device)
                out["type_action_norm"] = type_emb[1].norm().to(ctx_emb.device)
                out["type_diff_norm"] = (type_emb[0] - type_emb[1]).norm().to(ctx_emb.device)

    ctx_emb_g = ctx_emb.detach().clone().requires_grad_(True)
    ctx_act_g = ctx_act.detach().clone().requires_grad_(True)
    pred_g, _ = model.predict(ctx_emb_g, ctx_act_g)
    pred_loss_g = mse(pred_g, target.detach())
    model.zero_grad(set_to_none=True)
    pred_loss_g.backward()
    out["grad_emb_norm"] = ctx_emb_g.grad.float().norm(dim=-1).mean().detach()
    out["grad_act_norm"] = ctx_act_g.grad.float().norm(dim=-1).mean().detach()
    for t in range(history_size):
        out[f"grad_emb_t{t}"] = ctx_emb_g.grad[:, t].float().norm(dim=-1).mean().detach()
        out[f"grad_act_t{t}"] = ctx_act_g.grad[:, t].float().norm(dim=-1).mean().detach()
    return {k: v.detach().float().cpu() for k, v in out.items()}


def summarize(records: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    keys = records[0].keys()
    return {
        k: float(torch.stack([record[k] for record in records]).mean().item())
        for k in keys
    }


def add_ratios(summary: dict[str, float]) -> dict[str, float]:
    base_loss = max(summary["base_loss"], 1e-12)
    base_target_norm = max(summary["base_target_norm"], 1e-12)
    grad_emb = max(summary["grad_emb_norm"], 1e-12)
    out = dict(summary)
    out["grad_act_over_emb"] = summary["grad_act_norm"] / grad_emb
    for prefix in [
        "action_zero",
        "action_shuffle_batch",
        "action_shuffle_time",
        "action_noise",
        "image_zero",
        "image_shuffle_batch",
        "image_noise",
    ]:
        out[f"{prefix}_change_over_base_error"] = summary[f"{prefix}_pred_change_norm"] / base_target_norm
        out[f"{prefix}_loss_over_base"] = summary[f"{prefix}_loss"] / base_loss
    return out


def run_one(args, checkpoint: str, config_name: str, overrides: list[str]) -> dict:
    cfg = load_cfg(config_name, overrides)
    if args.dataset_name:
        cfg.data.name = args.dataset_name
    if args.batch_size:
        cfg.loader.batch_size = args.batch_size
    cfg.loader.num_workers = args.num_workers
    cfg.loader.persistent_workers = args.num_workers > 0

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    ckpt = resolve_model_state(checkpoint)
    model.load_state_dict(load_state_dict(ckpt), strict=True)
    model.eval()
    model.requires_grad_(False)

    dataset = load_stable_worldmodel_dataset(cfg)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "shuffle": True,
        "drop_last": True,
        "generator": generator,
        "batch_size": cfg.loader.batch_size,
        "num_workers": cfg.loader.num_workers,
        "persistent_workers": cfg.loader.persistent_workers,
        "pin_memory": cfg.loader.get("pin_memory", True),
    }
    if cfg.loader.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.loader.get("prefetch_factor", 2)
    loader = DataLoader(dataset, **loader_kwargs)

    cuda_generator = torch.Generator(device=device)
    cuda_generator.manual_seed(args.seed + 17)
    records = []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        records.append(
            batch_metrics(
                model=model,
                batch=batch,
                history_size=cfg.loss.get("history_size", 3),
                num_preds=cfg.loss.get("num_preds", 1),
                noise_scale=args.noise_scale,
                generator=cuda_generator,
            )
        )

    summary = add_ratios(summarize(records))
    return {
        "checkpoint": str(ckpt),
        "config_name": config_name,
        "dataset_name": cfg.data.name,
        "batches": len(records),
        "batch_size": int(cfg.loader.batch_size),
        "history_size": int(cfg.loss.get("history_size", 3)),
        "num_preds": int(cfg.loss.get("num_preds", 1)),
        "token_fusion": cfg.model.get("token_fusion", "concat_action"),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint path or DeepSpeed epoch directory. Can be passed multiple times.",
    )
    parser.add_argument("--config-name", default="pusht_rwkv_v7_stage2_ds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if not args.checkpoint:
        args.checkpoint = [
            "outputs/pusht_rwkv_v7_stage2_ds/epoch_8",
            "outputs/pusht_rwkv_v7_stage2_cont_ds/epoch_8",
        ]

    torch.set_float32_matmul_precision("high")
    results = [run_one(args, checkpoint, args.config_name, args.overrides) for checkpoint in args.checkpoint]
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
