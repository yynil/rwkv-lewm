from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from rwkv_lewm.encoders import build_vit_encoder, infer_encoder_dim
from rwkv_lewm.losses import rwkv_lewm_loss
from rwkv_lewm.model import RWKVLeWorldModel
from rwkv_lewm.modules import SIGReg
from rwkv_lewm.rwkv_dynamics import rwkv_optimizer_groups
from rwkv_lewm.rwkv_v7_cuda_dynamics import rwkv7_cuda_optimizer_groups


def build_model(cfg: DictConfig) -> RWKVLeWorldModel:
    encoder = build_vit_encoder(**cfg.encoder)
    encoder_dim = infer_encoder_dim(encoder)
    action_dim = cfg.data.frameskip * cfg.data.action_dim
    return RWKVLeWorldModel.from_components(
        encoder=encoder,
        encoder_dim=encoder_dim,
        action_dim=action_dim,
        embed_dim=cfg.model.embed_dim,
        dynamics_dim=cfg.model.dynamics_dim,
        dynamics_layers=cfg.model.dynamics_layers,
        rwkv_ffn_dim=cfg.model.get("rwkv_ffn_dim", None),
        dropout=cfg.model.dropout,
        backend=cfg.model.backend,
        residual_prediction=cfg.model.get("residual_prediction", True),
        action_encoder_type=cfg.model.get("action_encoder_type", "mlp"),
        prediction_head_type=cfg.model.get("prediction_head_type", "direct"),
        projector_norm=cfg.model.get("projector_norm", "batchnorm"),
        pred_proj_norm=cfg.model.get("pred_proj_norm", "batchnorm"),
        action_modulation=cfg.model.get("action_modulation", "none"),
        action_modulation_scope=cfg.model.get("action_modulation_scope", "full"),
        action_modulate_ffn=cfg.model.get("action_modulate_ffn", True),
        action_residual_gate=cfg.model.get("action_residual_gate", "adaln"),
        action_residual_gate_scale=cfg.model.get("action_residual_gate_scale", 0.1),
        kv_max_scale=cfg.model.get("kv_max_scale", 0.05),
        kv_max_shift=cfg.model.get("kv_max_shift", 0.05),
        action_adapter_residual_scale=cfg.model.get("action_adapter_residual_scale", 0.1),
        action_adapter_max_scale=cfg.model.get("action_adapter_max_scale", 0.05),
        action_adapter_max_shift=cfg.model.get("action_adapter_max_shift", 0.05),
        action_adapter_position=cfg.model.get("action_adapter_position", "post"),
        token_fusion=cfg.model.get("token_fusion", "concat_action"),
        use_token_type_embeddings=cfg.model.get("use_token_type_embeddings", True),
        use_sequence_pos_embedding=cfg.model.get("use_sequence_pos_embedding", False),
        max_position_embeddings=cfg.model.get("max_position_embeddings", 512),
        normalize_embeddings=cfg.model.get("normalize_embeddings", True),
        rollout_history_size=cfg.model.get("rollout_history_size", cfg.loss.get("history_size", 3)),
        modulate_v_first=cfg.model.get("modulate_v_first", False),
        rwkv_output_init_scale=cfg.model.get("rwkv_output_init_scale", 0.0),
        rwkv_output_head_type=cfg.model.get("rwkv_output_head_type", "linear"),
        action_transition_type=cfg.model.get("action_transition_type", "none"),
        action_transition_hidden_scale=cfg.model.get("action_transition_hidden_scale", 4),
        action_transition_residual_scale=cfg.model.get("action_transition_residual_scale", 1.0),
        action_transition_bounded=cfg.model.get("action_transition_bounded", False),
        action_transition_max_scale=cfg.model.get("action_transition_max_scale", 0.05),
        action_transition_max_shift=cfg.model.get("action_transition_max_shift", 0.05),
        action_transition_gate_tanh=cfg.model.get("action_transition_gate_tanh", False),
        adaptive_action_hidden_scale=cfg.model.get("adaptive_action_hidden_scale", 4),
        adaptive_action_alpha_init=cfg.model.get("adaptive_action_alpha_init", 1.0),
        adaptive_action_alpha_delta=cfg.model.get("adaptive_action_alpha_delta", 0.5),
        adaptive_action_modulate=cfg.model.get("adaptive_action_modulate", False),
        adaptive_action_max_scale=cfg.model.get("adaptive_action_max_scale", 0.05),
        adaptive_action_max_shift=cfg.model.get("adaptive_action_max_shift", 0.05),
        learned_action_scale_init=cfg.model.get("learned_action_scale_init", 1.0),
        learned_action_per_channel_scale=cfg.model.get("learned_action_per_channel_scale", False),
        learned_action_normalize_streams=cfg.model.get("learned_action_normalize_streams", True),
        scaled_action_scale_init=cfg.model.get("scaled_action_scale_init", 1.0),
        scaled_action_per_channel_scale=cfg.model.get("scaled_action_per_channel_scale", False),
        action_history_size=cfg.model.get("action_history_size", 3),
        action_history_include_current=cfg.model.get("action_history_include_current", True),
        action_history_scale_init=cfg.model.get("action_history_scale_init", 0.1),
        action_history_per_channel_scale=cfg.model.get("action_history_per_channel_scale", False),
        action_history_hidden_scale=cfg.model.get("action_history_hidden_scale", 4),
        action_history_zero_init_output=cfg.model.get("action_history_zero_init_output", True),
        block_action_transition_type=cfg.model.get("block_action_transition_type", "none"),
        block_action_transition_hidden_scale=cfg.model.get("block_action_transition_hidden_scale", 4),
        block_action_transition_residual_scale=cfg.model.get("block_action_transition_residual_scale", 0.1),
        block_action_transition_bounded=cfg.model.get("block_action_transition_bounded", True),
        block_action_transition_max_scale=cfg.model.get("block_action_transition_max_scale", 0.05),
        block_action_transition_max_shift=cfg.model.get("block_action_transition_max_shift", 0.05),
        block_action_transition_gate_tanh=cfg.model.get("block_action_transition_gate_tanh", True),
        block_action_transition_target=cfg.model.get("block_action_transition_target", "last"),
    )


def build_optimizer(model: RWKVLeWorldModel, cfg: DictConfig):
    if cfg.model.backend == "rwkv7_cuda":
        groups = rwkv7_cuda_optimizer_groups(model, weight_decay=cfg.optimizer.weight_decay)
    else:
        groups = rwkv_optimizer_groups(model, weight_decay=cfg.optimizer.weight_decay)
    return torch.optim.AdamW(groups, lr=cfg.optimizer.lr, betas=tuple(cfg.optimizer.betas))


def load_stable_worldmodel_dataset(cfg: DictConfig):
    import stable_pretraining as spt
    import stable_worldmodel as swm

    from lewm_compat import get_column_normalizer, get_img_preprocessor

    dataset = swm.data.HDF5Dataset(
        name=cfg.data.name,
        num_steps=cfg.data.num_steps,
        frameskip=cfg.data.frameskip,
        keys_to_load=list(cfg.data.keys_to_load),
        keys_to_cache=list(cfg.data.keys_to_cache),
        transform=None,
    )
    transforms = [get_img_preprocessor("pixels", "pixels", cfg.data.img_size)]
    for col in cfg.data.keys_to_load:
        if not str(col).startswith("pixels"):
            transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    if len(dataset) == 0:
        raise ValueError(
            "Stable-worldmodel dataset is empty for "
            f"name={cfg.data.name!r}, num_steps={cfg.data.num_steps}, "
            f"frameskip={cfg.data.frameskip}, span={cfg.data.num_steps * cfg.data.frameskip}. "
            "For TwoRoom, episode lengths are at most about 101 raw steps; reduce num_steps "
            "or frameskip while keeping frameskip compatible with any resumed checkpoint."
        )
    return dataset


@hydra.main(version_base=None, config_path="config/train", config_name="rwkv_lewm")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    if cfg.get("resume_from"):
        print(f"Loading checkpoint: {cfg.resume_from}")
        state = torch.load(cfg.resume_from, map_location="cpu")
        model.load_state_dict(state, strict=True)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)
    optimizer = build_optimizer(model, cfg)

    dataset = load_stable_worldmodel_dataset(cfg)
    loader = DataLoader(dataset, shuffle=True, drop_last=True, **cfg.loader)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    model.train()
    global_step = 0
    for epoch in range(cfg.trainer.max_epochs):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            metrics = rwkv_lewm_loss(
                model,
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
            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip_val)
            optimizer.step()

            if global_step % cfg.log_every == 0:
                extra = ""
                if "rollout_loss" in metrics:
                    extra = (
                        f" teacher={metrics['teacher_loss'].float().item():.8f}"
                        f" rollout={metrics['rollout_loss'].float().item():.8f}"
                        f" rollout_final={metrics['rollout_final_mse'].float().item():.8f}"
                    )
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={metrics['loss'].float().item():.8f} "
                    f"pred={metrics['pred_loss'].float().item():.8f} "
                    f"sigreg={metrics['sigreg_loss'].float().item():.8f} "
                    f"emb_std={metrics['emb_std'].float().item():.6f} "
                    f"temporal={metrics['temporal_mse'].float().item():.3e} "
                    f"delta={metrics['pred_delta_norm'].float().item():.6f}"
                    f"{extra}"
                )
            global_step += 1
            if cfg.trainer.get("max_steps") and global_step >= cfg.trainer.max_steps:
                torch.save(model.state_dict(), out_dir / f"rwkv_lewm_step_{global_step}.pt")
                return

        torch.save(model.state_dict(), out_dir / f"rwkv_lewm_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
