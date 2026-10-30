import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
LEWM_ROOT = Path(os.environ.get("LEWM_ROOT", ROOT.parent / "le-wm"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LEWM_ROOT) not in sys.path:
    sys.path.append(str(LEWM_ROOT))

try:
    import ninja

    os.environ["PATH"] = f"{ninja.BIN_DIR}:{os.environ.get('PATH', '')}"
except Exception:
    pass

from train import build_model


def load_train_cfg(config_name: str, overrides: list[str]):
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


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def transform_pixels(pixels: np.ndarray, transform) -> torch.Tensor:
    b, t = pixels.shape[:2]
    flat = pixels.reshape(b * t, *pixels.shape[2:])
    out = torch.stack([transform(x) for x in flat])
    return out.reshape(b, t, *out.shape[1:])


def make_dataset(name: str):
    return swm.data.HDF5Dataset(
        name,
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=swm.data.utils.get_cache_dir(),
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def sample_eval_points(dataset, num_eval: int, goal_offset_steps: int, seed: int):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < num_eval:
        raise ValueError(f"Only {len(valid_indices)} valid starts found for {num_eval} evals")
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    random_indices = np.sort(valid_indices[random_indices])
    rows = dataset.get_row_data(random_indices)
    return random_indices, rows[col_name], rows["step_idx"], valid_indices


def build_action_processor(dataset):
    processor = preprocessing.StandardScaler()
    action = dataset.get_col_data("action")
    action = action[~np.isnan(action).any(axis=1)]
    processor.fit(action)
    return processor


def make_expert_batch(dataset, row_indices, horizon: int, action_block: int, img_size: int, device):
    transform = img_transform(img_size)
    action_processor = build_action_processor(dataset)
    frame_offsets = np.arange(horizon + 1) * action_block
    all_pixels = []
    all_actions = []
    for row in row_indices:
        pixel_rows = dataset.get_row_data(row + frame_offsets)["pixels"]
        raw_action = dataset.get_row_data(np.arange(row, row + horizon * action_block))["action"]
        norm_action = action_processor.transform(raw_action).astype(np.float32)
        all_pixels.append(pixel_rows)
        all_actions.append(norm_action.reshape(horizon, -1))
    pixels = transform_pixels(np.stack(all_pixels), transform).to(device)
    action = torch.tensor(np.stack(all_actions), dtype=torch.float32, device=device)
    return pixels, action


def load_rwkv_model(checkpoint: str, config_name: str, overrides: list[str], device: torch.device):
    cfg = load_train_cfg(config_name, overrides)
    model = build_model(cfg).to(device)
    ckpt = resolve_model_state(checkpoint)
    model.load_state_dict(load_state_dict(ckpt), strict=True)
    return model.eval().requires_grad_(False), str(ckpt)


def load_models(args, device):
    models = {}
    metadata = {}
    lewm = swm.policy.AutoCostModel(args.lewm_policy).to(device).eval().requires_grad_(False)
    models["lewm"] = lewm
    metadata["lewm"] = args.lewm_policy
    base, base_ckpt = load_rwkv_model(args.base_checkpoint, args.rwkv_config, args.overrides, device)
    cont, cont_ckpt = load_rwkv_model(args.cont_checkpoint, args.rwkv_config, args.overrides, device)
    models["rwkv_base"] = base
    models["rwkv_cont"] = cont
    metadata["rwkv_base"] = base_ckpt
    metadata["rwkv_cont"] = cont_ckpt
    return models, metadata


def call_predict(model, emb, act_emb):
    out = model.predict(emb, act_emb)
    if isinstance(out, tuple):
        return out[0]
    return out


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def step_errors(pred: torch.Tensor, target: torch.Tensor):
    diff = pred.float() - target.float()
    mse = diff.square().mean(dim=-1)
    sum_mse = diff.square().sum(dim=-1)
    norm = diff.norm(dim=-1)
    return mse, sum_mse, norm


@torch.inference_mode()
def diagnose_model(model, pixels: torch.Tensor, action: torch.Tensor, history_sizes: list[int]):
    horizon = action.size(1)
    encoded = model.encode({"pixels": pixels.clone(), "action": action.clone()})
    emb = encoded["emb"].detach()
    act_emb = encoded["act_emb"].detach()

    teacher = {"mse": [], "sum_mse": [], "norm": []}
    for t in range(horizon):
        start = max(0, t - 2)
        pred = call_predict(model, emb[:, start : t + 1], act_emb[:, start : t + 1])[:, -1]
        mse, sum_mse, norm = step_errors(pred, emb[:, t + 1])
        teacher["mse"].append(mse)
        teacher["sum_mse"].append(sum_mse)
        teacher["norm"].append(norm)
    teacher = {k: torch.stack(v, dim=1) for k, v in teacher.items()}

    rollout = {}
    for history in history_sizes:
        info = {
            "pixels": pixels[:, None, :history].clone(),
            "action": torch.zeros(pixels.size(0), 1, history, 2, device=pixels.device),
        }
        out = model.rollout(info, action[:, None].clone())
        pred = out["predicted_emb"][:, 0, : horizon + 1]
        mse, sum_mse, norm = step_errors(pred, emb[:, : horizon + 1])
        rollout[f"history{history}"] = {
            "mse": mse,
            "sum_mse": sum_mse,
            "norm": norm,
            "final_sum_mse": sum_mse[:, -1],
            "final_norm": norm[:, -1],
        }

    result = {
        "teacher_forced": {
            "per_step_mse": [summarize(teacher["mse"][:, i]) for i in range(horizon)],
            "per_step_sum_mse": [summarize(teacher["sum_mse"][:, i]) for i in range(horizon)],
            "per_step_norm": [summarize(teacher["norm"][:, i]) for i in range(horizon)],
            "final_sum_mse": summarize(teacher["sum_mse"][:, -1]),
            "final_norm": summarize(teacher["norm"][:, -1]),
        },
        "autoregressive": {},
    }
    for name, data in rollout.items():
        result["autoregressive"][name] = {
            "per_step_sum_mse": [summarize(data["sum_mse"][:, i]) for i in range(horizon + 1)],
            "per_step_norm": [summarize(data["norm"][:, i]) for i in range(horizon + 1)],
            "final_sum_mse": summarize(data["final_sum_mse"]),
            "final_norm": summarize(data["final_norm"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default="outputs/pusht_rwkv_v7_stage2_ds/epoch_8")
    parser.add_argument("--cont-checkpoint", default="outputs/pusht_rwkv_v7_stage2_cont_ds/epoch_8")
    parser.add_argument("--rwkv-config", default="pusht_rwkv_v7_stage2_ds")
    parser.add_argument("--lewm-policy", default="pusht_lewm_repro_epoch10/lewm_epoch_10")
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--history-sizes", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--output", default="outputs/pusht_expert_rollout_error/rwkv_v7_base_cont_vs_lewm_seed42_50.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dataset = make_dataset(args.dataset_name)
    row_indices, eval_episodes, eval_starts, valid_indices = sample_eval_points(
        dataset, args.num_eval, args.goal_offset_steps, args.seed
    )
    pixels, action = make_expert_batch(dataset, row_indices, args.horizon, args.action_block, args.img_size, device)
    models, metadata = load_models(args, device)
    diagnostics = {}
    for name, model in models.items():
        print(f"diagnosing {name}")
        diagnostics[name] = diagnose_model(model, pixels, action, args.history_sizes)

    result = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "goal_offset_steps": args.goal_offset_steps,
        "horizon": args.horizon,
        "action_block": args.action_block,
        "num_eval": args.num_eval,
        "metadata": metadata,
        "eval_rows": {
            "row_indices": row_indices.tolist(),
            "episodes": eval_episodes.tolist(),
            "starts": eval_starts.tolist(),
        },
        "diagnostics": diagnostics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
