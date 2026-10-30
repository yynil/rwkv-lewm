import argparse
import json
import os
from pathlib import Path

import deepspeed
import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from rwkv_lewm.losses import rwkv_lewm_loss
from rwkv_lewm.modules import SIGReg
from train import build_model, build_optimizer, load_stable_worldmodel_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="tworoom_stage1")
    parser.add_argument("--deepspeed_config", default="config/deepspeed/zero2.json")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def move_batch(batch: dict, device: torch.device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_cfg(config_name: str, overrides: list[str]):
    with hydra.initialize(version_base=None, config_path="config/train"):
        return hydra.compose(config_name=config_name, overrides=overrides)


def load_deepspeed_checkpoint(engine, resume_from: str):
    path = Path(resume_from)
    if path.name.startswith(("epoch_", "step_")):
        engine.load_checkpoint(str(path.parent), tag=path.name)
    else:
        engine.load_checkpoint(str(path))


def distributed_barrier():
    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()


def save_deepspeed_checkpoint(engine, out_dir: Path, tag: str):
    engine.save_checkpoint(str(out_dir), tag=tag)
    distributed_barrier()


def shutdown_distributed():
    if dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()


def main():
    args = parse_args()
    deepspeed.init_distributed()
    torch.cuda.set_device(args.local_rank)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", args.local_rank)

    cfg = load_cfg(args.config_name, args.overrides)
    ds_config = json.loads(Path(args.deepspeed_config).read_text())
    ds_config["train_micro_batch_size_per_gpu"] = int(cfg.loader.batch_size)
    ds_config["gradient_clipping"] = float(cfg.trainer.gradient_clip_val)

    model = build_model(cfg)
    if cfg.get("resume_from") and str(cfg.resume_from).endswith(".pt"):
        if is_rank0():
            print(f"Loading PyTorch checkpoint before DeepSpeed init: {cfg.resume_from}")
        state = torch.load(cfg.resume_from, map_location="cpu")
        model.load_state_dict(state, strict=True)

    optimizer = build_optimizer(model, cfg)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    del optimizer

    if cfg.get("resume_from") and not str(cfg.resume_from).endswith(".pt"):
        if is_rank0():
            print(f"Loading DeepSpeed checkpoint: {cfg.resume_from}")
        load_deepspeed_checkpoint(engine, str(cfg.resume_from))

    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)
    dataset = load_stable_worldmodel_dataset(cfg)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )
    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    loader_kwargs.pop("shuffle", None)
    loader = DataLoader(dataset, sampler=sampler, drop_last=True, **loader_kwargs)

    out_dir = Path(cfg.output_dir)
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
        Path(args.deepspeed_config).resolve()
        (out_dir / "deepspeed_config.json").write_text(json.dumps(ds_config, indent=2))
    distributed_barrier()

    global_step = 0
    engine.train()
    for epoch in range(cfg.trainer.max_epochs):
        sampler.set_epoch(epoch)
        iterator = loader
        if is_rank0():
            iterator = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)

        for batch in iterator:
            batch = move_batch(batch, device)
            metrics = rwkv_lewm_loss(
                engine.module,
                batch,
                sigreg,
                cfg.loss.sigreg.weight,
                mode=cfg.loss.get("mode", "next_step"),
                history_size=cfg.loss.get("history_size", 3),
                num_preds=cfg.loss.get("num_preds", 1),
                rollout_start_size=cfg.loss.get("rollout_start_size", 1),
                rollout_horizon=cfg.loss.get("rollout_horizon", None),
                rollout_weight=cfg.loss.get("rollout_weight", 1.0),
                teacher_weight=cfg.loss.get("teacher_weight", 1.0),
            )
            engine.backward(metrics["loss"])
            engine.step()

            if global_step % cfg.log_every == 0:
                loss = reduce_mean(metrics["loss"])
                pred = reduce_mean(metrics["pred_loss"])
                teacher = reduce_mean(metrics["teacher_loss"])
                sig = reduce_mean(metrics["sigreg_loss"])
                emb_std = reduce_mean(metrics["emb_std"])
                temporal = reduce_mean(metrics["temporal_mse"])
                delta = reduce_mean(metrics["pred_delta_norm"])
                rollout = reduce_mean(metrics["rollout_loss"]) if "rollout_loss" in metrics else None
                rollout_final = (
                    reduce_mean(metrics["rollout_final_mse"]) if "rollout_final_mse" in metrics else None
                )
                if is_rank0():
                    extra = ""
                    if rollout is not None and rollout_final is not None:
                        extra = (
                            f" teacher={teacher.float().item():.8f} "
                            f"rollout={rollout.float().item():.8f} "
                            f"rollout_final={rollout_final.float().item():.8f}"
                        )
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={loss.float().item():.8f} "
                        f"pred={pred.float().item():.8f} "
                        f"sigreg={sig.float().item():.8f} "
                        f"emb_std={emb_std.float().item():.6f} "
                        f"temporal={temporal.float().item():.3e} "
                        f"delta={delta.float().item():.6f}"
                        f"{extra}",
                        flush=True,
                    )

            global_step += 1
            if cfg.trainer.get("max_steps") and global_step >= cfg.trainer.max_steps:
                tag = f"step_{global_step}"
                save_deepspeed_checkpoint(engine, out_dir, tag)
                shutdown_distributed()
                return

        tag = f"epoch_{epoch + 1}"
        save_deepspeed_checkpoint(engine, out_dir, tag)

    shutdown_distributed()


if __name__ == "__main__":
    main()
