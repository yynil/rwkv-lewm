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
from gymnasium.spaces import Box
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


def make_solver_info(dataset, row_indices: np.ndarray, goal_offset_steps: int, img_size: int):
    start_rows = dataset.get_row_data(row_indices)
    goal_rows = dataset.get_row_data(row_indices + goal_offset_steps)
    transform = img_transform(img_size)
    pixels = transform_pixels(start_rows["pixels"][:, None], transform)
    goal = transform_pixels(goal_rows["pixels"][:, None], transform)
    return {
        "pixels": pixels.contiguous(),
        "goal": goal.contiguous(),
        "action": torch.zeros(pixels.size(0), 1, 2, dtype=torch.float32),
    }


def make_baseline_actions(
    dataset,
    row_indices: np.ndarray,
    horizon: int,
    action_block: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    action_processor = build_action_processor(dataset)
    action_dim = int(dataset.get_col_data("action").shape[-1])
    flat_dim = action_dim * action_block
    expert = []
    for row in row_indices:
        raw = dataset.get_row_data(np.arange(row, row + horizon * action_block))["action"]
        norm = action_processor.transform(raw).astype(np.float32)
        expert.append(norm.reshape(horizon, flat_dim))
    expert = torch.tensor(np.stack(expert), dtype=torch.float32, device=device)
    return {
        "expert": expert,
        "zero": torch.zeros_like(expert),
    }


def load_rwkv_model(checkpoint: str, config_name: str, overrides: list[str], device: torch.device):
    cfg = load_train_cfg(config_name, overrides)
    model = build_model(cfg).to(device)
    ckpt = resolve_model_state(checkpoint)
    model.load_state_dict(load_state_dict(ckpt), strict=True)
    return model.eval().requires_grad_(False), str(ckpt)


def load_models(args, device: torch.device):
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


def clone_info(info: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device).clone() if torch.is_tensor(v) else v for k, v in info.items()}


@torch.inference_mode()
def cost_actions(model, info: dict[str, torch.Tensor], actions: torch.Tensor, batch_size: int):
    chunks = []
    for start in range(0, actions.size(0), batch_size):
        end = min(start + batch_size, actions.size(0))
        info_batch = {k: v[start:end].unsqueeze(1).clone() if torch.is_tensor(v) else v for k, v in info.items()}
        chunks.append(model.get_cost(info_batch, actions[start:end].unsqueeze(1)).detach().float().cpu().squeeze(1))
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def solve_cem(model, info, args, device: torch.device):
    plan_config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=args.action_block,
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=args.cem_batch_size,
        num_samples=args.cem_samples,
        var_scale=args.var_scale,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=device,
        seed=args.seed,
    )
    action_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(args.num_eval, args.raw_action_dim),
        dtype=np.float32,
    )
    solver.configure(action_space=action_space, n_envs=args.num_eval, config=plan_config)
    out = solver(clone_info(info, device), init_action=None)
    return out["actions"].to(device), torch.tensor(out["costs"], dtype=torch.float32)


def summarize_cost_vector(cost: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(cost.mean().item()),
        "median": float(cost.median().item()),
        "min": float(cost.min().item()),
        "max": float(cost.max().item()),
        "std": float(cost.std(unbiased=False).item()),
    }


def summarize_action(actions: torch.Tensor) -> dict[str, float]:
    flat = actions.float().flatten(1)
    return {
        "norm_mean": float(flat.norm(dim=1).mean().item()),
        "norm_median": float(flat.norm(dim=1).median().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "std": float(flat.std(unbiased=False).item()),
    }


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
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--raw-action-dim", type=int, default=2)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--cem-batch-size", type=int, default=1)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--cost-batch-size", type=int, default=4)
    parser.add_argument("--output", default="outputs/pusht_cem_cross_cost/rwkv_v7_base_cont_vs_lewm_seed42_50.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

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

    info = make_solver_info(dataset, row_indices, args.goal_offset_steps, args.img_size)
    info = clone_info(info, device)
    baseline_actions = make_baseline_actions(dataset, row_indices, args.horizon, args.action_block, device)
    models, metadata = load_models(args, device)

    start = time.time()
    planned_actions = {}
    own_final_costs = {}
    for name, model in models.items():
        print(f"running CEM for {name}")
        actions, own_costs = solve_cem(model, info, args, device)
        planned_actions[name] = actions
        own_final_costs[name] = own_costs

    action_sets = {**planned_actions, **baseline_actions}
    cross_costs: dict[str, dict[str, torch.Tensor]] = {}
    for evaluator_name, evaluator in models.items():
        cross_costs[evaluator_name] = {}
        for action_name, actions in action_sets.items():
            print(f"scoring {action_name} under {evaluator_name}")
            cross_costs[evaluator_name][action_name] = cost_actions(
                evaluator, info, actions, args.cost_batch_size
            )

    summary = {
        "own_final_elite_cost": {name: summarize_cost_vector(cost) for name, cost in own_final_costs.items()},
        "action_stats": {name: summarize_action(actions) for name, actions in action_sets.items()},
        "cross_cost": {
            evaluator_name: {
                action_name: summarize_cost_vector(cost)
                for action_name, cost in costs.items()
            }
            for evaluator_name, costs in cross_costs.items()
        },
        "pairwise_action_l2": {},
        "per_eval": [],
    }
    names = list(action_sets.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            dist = (action_sets[a].float().flatten(1) - action_sets[b].float().flatten(1)).norm(dim=1)
            summary["pairwise_action_l2"][f"{a}_vs_{b}"] = summarize_cost_vector(dist.cpu())

    for i in range(args.num_eval):
        row = {"idx": i, "episode": int(eval_episodes[i]), "start": int(eval_starts[i])}
        for evaluator_name, costs in cross_costs.items():
            for action_name, cost in costs.items():
                row[f"{evaluator_name}_cost_{action_name}"] = float(cost[i].item())
        for a in names:
            row[f"action_norm_{a}"] = float(action_sets[a][i].float().norm().item())
        summary["per_eval"].append(row)

    result = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "goal_offset_steps": args.goal_offset_steps,
        "horizon": args.horizon,
        "receding_horizon": args.receding_horizon,
        "action_block": args.action_block,
        "num_eval": args.num_eval,
        "cem_samples": args.cem_samples,
        "cem_steps": args.cem_steps,
        "cem_topk": args.cem_topk,
        "var_scale": args.var_scale,
        "metadata": metadata,
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
            "planned_actions": {k: v.detach().cpu() for k, v in planned_actions.items()},
            "baseline_actions": {k: v.detach().cpu() for k, v in baseline_actions.items()},
            "cross_costs": cross_costs,
            "own_final_costs": own_final_costs,
            "row_indices": torch.tensor(row_indices),
            "episodes": torch.tensor(eval_episodes),
            "starts": torch.tensor(eval_starts),
        },
        out.with_suffix(".pt"),
    )


if __name__ == "__main__":
    main()
