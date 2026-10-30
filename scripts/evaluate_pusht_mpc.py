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
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import ninja

    os.environ["PATH"] = f"{ninja.BIN_DIR}:{os.environ.get('PATH', '')}"
except Exception:
    pass

from train import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", default="pusht_stage2_ds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--world-history-size", type=int, default=1)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--cem-batch-size", type=int, default=1)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output", default=None)
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


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def make_dataset(name: str):
    return swm.data.HDF5Dataset(
        name,
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=swm.data.utils.get_cache_dir(),
    )


def build_processors(dataset):
    process = {}
    for col in ["action", "proprio", "state"]:
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


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
    return rows[col_name], rows["step_idx"], valid_indices


def main():
    args = parse_args()
    assert args.horizon * args.action_block <= args.eval_budget, "Planning horizon must be <= eval budget"
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    train_cfg = load_train_cfg(args.train_config, args.overrides)
    model = build_model(train_cfg).to(device)
    ckpt = resolve_model_state(args.checkpoint)
    model.load_state_dict(load_state_dict(ckpt), strict=True)
    model.eval().requires_grad_(False)

    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=args.num_eval,
        max_episode_steps=2 * args.eval_budget,
        history_size=args.world_history_size,
        frame_skip=1,
        image_shape=(args.img_size, args.img_size),
    )

    dataset = make_dataset(args.dataset_name)
    process = build_processors(dataset)
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }

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
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)

    eval_episodes, eval_start_idx, valid_indices = sample_eval_points(
        dataset, args.num_eval, args.goal_offset_steps, args.seed
    )
    print(f"{len(valid_indices)} valid starting points found for evaluation.")
    print("eval_rows episodes", eval_episodes)
    print("eval_rows starts", eval_start_idx)

    start = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=args.goal_offset_steps,
        eval_budget=args.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state"}},
            },
        ],
        save_video=args.save_video,
        video_path=Path(args.output).parent if args.output else Path("outputs/mpc_eval"),
    )
    elapsed = time.time() - start

    result = {
        "checkpoint": str(ckpt),
        "dataset_name": args.dataset_name,
        "num_eval": args.num_eval,
        "seed": args.seed,
        "goal_offset_steps": args.goal_offset_steps,
        "eval_budget": args.eval_budget,
        "horizon": args.horizon,
        "receding_horizon": args.receding_horizon,
        "action_block": args.action_block,
        "world_history_size": args.world_history_size,
        "cem_samples": args.cem_samples,
        "cem_steps": args.cem_steps,
        "cem_topk": args.cem_topk,
        "cem_batch_size": args.cem_batch_size,
        "metrics": metrics,
        "evaluation_time_seconds": elapsed,
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=lambda x: x.tolist()))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=lambda x: x.tolist()) + "\n")


if __name__ == "__main__":
    main()
