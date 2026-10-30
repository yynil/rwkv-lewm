import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torch import nn
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwkv-checkpoint", default="outputs/pusht_rwkv_v4_stage2_ds/epoch_3")
    parser.add_argument("--rwkv-config", default="pusht_rwkv_v4_stage2_ds")
    parser.add_argument("--lewm-policy", default="pusht_lewm_repro_epoch10/lewm_epoch_10")
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--output", default="outputs/pusht_fixed_candidate_diagnostics/rwkv_v4_epoch3_vs_lewm_epoch10.json")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_train_cfg(config_name: str, overrides: list[str]):
    with hydra.initialize(version_base=None, config_path="../config/train"):
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


def transform_pixels(pixels: np.ndarray, transform) -> torch.Tensor:
    # pixels: (B, T, H, W, C), uint8
    b, t = pixels.shape[:2]
    flat = pixels.reshape(b * t, *pixels.shape[2:])
    out = torch.stack([transform(x) for x in flat])
    return out.reshape(b, t, *out.shape[1:])


def make_info(dataset, row_indices: np.ndarray, goal_offset_steps: int, img_size: int, num_candidates: int):
    start_rows = dataset.get_row_data(row_indices)
    goal_rows = dataset.get_row_data(row_indices + goal_offset_steps)
    transform = img_transform(img_size)
    pixels = transform_pixels(start_rows["pixels"][:, None], transform)
    goal = transform_pixels(goal_rows["pixels"][:, None], transform)

    info = {
        "pixels": pixels.unsqueeze(1).expand(-1, num_candidates, -1, -1, -1, -1).contiguous(),
        "goal": goal.unsqueeze(1).expand(-1, num_candidates, -1, -1, -1, -1).contiguous(),
        # LeWM's get_cost pops action from the goal dict before encoding; the
        # rollout itself overwrites info["action"] with candidate actions.
        "action": torch.zeros(pixels.size(0), num_candidates, 1, 2, dtype=torch.float32),
    }
    return info


def make_candidates(
    dataset,
    row_indices: np.ndarray,
    num_candidates: int,
    horizon: int,
    action_block: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_processor = build_action_processor(dataset)
    action_dim = int(dataset.get_col_data("action").shape[-1])
    flat_dim = action_dim * action_block
    gen = torch.Generator(device=device).manual_seed(seed)
    candidates = torch.randn(
        len(row_indices),
        num_candidates,
        horizon,
        flat_dim,
        generator=gen,
        device=device,
    )
    candidates[:, 0] = 0.0

    expert = []
    for row in row_indices:
        raw = dataset.get_row_data(np.arange(row, row + horizon * action_block))["action"]
        norm = action_processor.transform(raw).astype(np.float32)
        expert.append(norm.reshape(horizon, flat_dim))
    expert = torch.tensor(np.stack(expert), dtype=torch.float32, device=device)
    candidates = torch.cat([candidates, expert.unsqueeze(1)], dim=1)
    return candidates, expert


def load_rwkv_model(args, device: torch.device):
    cfg = load_train_cfg(args.rwkv_config, args.overrides)
    model = build_model(cfg).to(device)
    ckpt = resolve_model_state(args.rwkv_checkpoint)
    state = load_state_dict(ckpt)
    if (
        "dynamics.input_proj.weight" in state
        and hasattr(model.dynamics, "input_proj")
        and isinstance(model.dynamics.input_proj, nn.Identity)
    ):
        weight = state["dynamics.input_proj.weight"]
        bias = state.get("dynamics.input_proj.bias")
        model.dynamics.input_proj = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False), ckpt


def load_lewm_model(policy: str, device: torch.device):
    model = swm.policy.AutoCostModel(policy)
    model = model.to(device)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def batched_cost(model, info, candidates, batch_size: int):
    chunks = []
    for start in range(0, candidates.size(0), batch_size):
        end = min(start + batch_size, candidates.size(0))
        info_batch = {k: v[start:end].clone() if torch.is_tensor(v) else v for k, v in info.items()}
        chunks.append(model.get_cost(info_batch, candidates[start:end]).detach().float().cpu())
    return torch.cat(chunks, dim=0)


def ranks_from_cost(cost: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(cost, dim=1)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(cost.size(1)).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks


def summarize_costs(lewm_cost: torch.Tensor, rwkv_cost: torch.Tensor, topk: int, random_count: int):
    n, total = lewm_cost.shape
    expert_idx = total - 1
    lewm_random = lewm_cost[:, :random_count]
    rwkv_random = rwkv_cost[:, :random_count]
    s = random_count
    k = min(topk, s)
    lewm_ranks = ranks_from_cost(lewm_random)
    rwkv_ranks = ranks_from_cost(rwkv_random)
    lewm_all_ranks = ranks_from_cost(lewm_cost)
    rwkv_all_ranks = ranks_from_cost(rwkv_cost)
    lewm_top1 = torch.argmin(lewm_random, dim=1)
    rwkv_top1 = torch.argmin(rwkv_random, dim=1)
    batch = torch.arange(n)

    lewm_topk = torch.topk(lewm_random, k=k, dim=1, largest=False).indices
    rwkv_topk = torch.topk(rwkv_random, k=k, dim=1, largest=False).indices
    overlap = []
    for i in range(n):
        overlap.append(len(set(lewm_topk[i].tolist()) & set(rwkv_topk[i].tolist())) / k)
    overlap = torch.tensor(overlap)

    lewm_centered = lewm_random - lewm_random.mean(dim=1, keepdim=True)
    rwkv_centered = rwkv_random - rwkv_random.mean(dim=1, keepdim=True)
    pearson = (lewm_centered * rwkv_centered).sum(dim=1) / (
        lewm_centered.square().sum(dim=1).sqrt() * rwkv_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)

    lewm_rank_f = lewm_ranks.float()
    rwkv_rank_f = rwkv_ranks.float()
    lewm_rank_c = lewm_rank_f - lewm_rank_f.mean(dim=1, keepdim=True)
    rwkv_rank_c = rwkv_rank_f - rwkv_rank_f.mean(dim=1, keepdim=True)
    spearman = (lewm_rank_c * rwkv_rank_c).sum(dim=1) / (
        lewm_rank_c.square().sum(dim=1).sqrt() * rwkv_rank_c.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)

    result = {
        "num_eval": n,
        "num_random_candidates": s,
        "num_total_candidates": total,
        "expert_candidate_index": expert_idx,
        "topk": k,
        "top1_agreement_rate": (lewm_top1 == rwkv_top1).float().mean().item(),
        "topk_overlap_mean": overlap.mean().item(),
        "topk_overlap_median": overlap.median().item(),
        "cost_pearson_mean": pearson.mean().item(),
        "cost_pearson_median": pearson.median().item(),
        "rank_spearman_mean": spearman.mean().item(),
        "rank_spearman_median": spearman.median().item(),
        "expert_rank_lewm_mean": lewm_all_ranks[:, expert_idx].float().mean().item(),
        "expert_rank_rwkv_mean": rwkv_all_ranks[:, expert_idx].float().mean().item(),
        "expert_rank_lewm_median": lewm_all_ranks[:, expert_idx].float().median().item(),
        "expert_rank_rwkv_median": rwkv_all_ranks[:, expert_idx].float().median().item(),
        "lewm_top1_rank_under_rwkv_mean": rwkv_ranks[batch, lewm_top1].float().mean().item(),
        "rwkv_top1_rank_under_lewm_mean": lewm_ranks[batch, rwkv_top1].float().mean().item(),
        "lewm_top1_rank_under_rwkv_median": rwkv_ranks[batch, lewm_top1].float().median().item(),
        "rwkv_top1_rank_under_lewm_median": lewm_ranks[batch, rwkv_top1].float().median().item(),
        "lewm_top1_cost_lewm_mean": lewm_cost[batch, lewm_top1].mean().item(),
        "rwkv_top1_cost_lewm_mean": lewm_cost[batch, rwkv_top1].mean().item(),
        "lewm_top1_cost_rwkv_mean": rwkv_cost[batch, lewm_top1].mean().item(),
        "rwkv_top1_cost_rwkv_mean": rwkv_cost[batch, rwkv_top1].mean().item(),
        "per_eval": [],
    }
    for i in range(n):
        result["per_eval"].append(
            {
                "idx": i,
                "lewm_top1": int(lewm_top1[i]),
                "rwkv_top1": int(rwkv_top1[i]),
                "same_top1": bool(lewm_top1[i] == rwkv_top1[i]),
                "topk_overlap": float(overlap[i]),
                "cost_pearson": float(pearson[i]),
                "rank_spearman": float(spearman[i]),
                "expert_rank_lewm": int(lewm_all_ranks[i, expert_idx]),
                "expert_rank_rwkv": int(rwkv_all_ranks[i, expert_idx]),
                "lewm_top1_rank_under_rwkv": int(rwkv_ranks[i, lewm_top1[i]]),
                "rwkv_top1_rank_under_lewm": int(lewm_ranks[i, rwkv_top1[i]]),
            }
        )
    return result


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dataset = make_dataset(args.dataset_name)
    row_indices, eval_episodes, eval_starts, valid_indices = sample_eval_points(
        dataset, args.num_eval, args.goal_offset_steps, args.seed
    )
    print(f"{len(valid_indices)} valid starting points found for evaluation.")
    print("eval_rows indices", row_indices)
    print("eval_rows episodes", eval_episodes)
    print("eval_rows starts", eval_starts)

    info = make_info(dataset, row_indices, args.goal_offset_steps, args.img_size, args.num_candidates)
    info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info.items()}
    candidates, _expert = make_candidates(
        dataset,
        row_indices,
        args.num_candidates,
        args.horizon,
        args.action_block,
        args.seed,
        device,
    )

    start = time.time()
    rwkv_model, rwkv_ckpt = load_rwkv_model(args, device)
    lewm_model = load_lewm_model(args.lewm_policy, device)

    print("computing LeWM costs")
    lewm_cost = batched_cost(lewm_model, info, candidates, args.batch_size)
    print("computing RWKV costs")
    rwkv_cost = batched_cost(rwkv_model, info, candidates, args.batch_size)

    summary = summarize_costs(lewm_cost, rwkv_cost, args.topk, args.num_candidates)
    result = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "goal_offset_steps": args.goal_offset_steps,
        "horizon": args.horizon,
        "action_block": args.action_block,
        "num_eval": args.num_eval,
        "num_candidates": args.num_candidates,
        "num_total_candidates_with_expert": args.num_candidates + 1,
        "rwkv_checkpoint": str(rwkv_ckpt),
        "rwkv_config": args.rwkv_config,
        "lewm_policy": args.lewm_policy,
        "eval_rows": {
            "row_indices": row_indices.tolist(),
            "episodes": eval_episodes.tolist(),
            "starts": eval_starts.tolist(),
        },
        "summary": summary,
        "elapsed_seconds": time.time() - start,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "lewm_cost": lewm_cost,
            "rwkv_cost": rwkv_cost,
            "row_indices": torch.tensor(row_indices),
            "episodes": torch.tensor(eval_episodes),
            "starts": torch.tensor(eval_starts),
        },
        out.with_suffix(".pt"),
    )


if __name__ == "__main__":
    main()
