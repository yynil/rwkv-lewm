import argparse
import sys
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import build_model


def load_cfg(config_name: str, overrides: list[str]):
    config_path = Path(__file__).resolve().parents[1] / "config" / "train"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_path)):
        return hydra.compose(config_name=config_name, overrides=overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pusht_rwkv_v7_stage2_ds")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_cfg(args.config_name, args.overrides)
    model = build_model(cfg).cuda().train()
    dynamics = model.dynamics

    print("config", args.config_name)
    print("embed_dim", cfg.model.embed_dim)
    print("dynamics_dim", cfg.model.dynamics_dim)
    print("token_fusion", cfg.model.token_fusion)
    print("use_token_type_embeddings", cfg.model.use_token_type_embeddings)
    print("action_modulation", dynamics.cfg.action_modulation)
    print("modulate_v_first", dynamics.cfg.modulate_v_first)
    print("rwkv_output_init_scale", dynamics.cfg.rwkv_output_init_scale)
    print("fuse", type(model.fuse).__name__)

    if cfg.model.token_fusion != "interleave_action":
        raise AssertionError("v7 expects token_fusion=interleave_action")
    if dynamics.cfg.action_modulation != "none":
        raise AssertionError("v7 should not use action modulation")
    if dynamics.cfg.modulate_v_first:
        raise AssertionError("v7 should not modulate v_first")
    if model.token_type_embeddings is None:
        raise AssertionError("v7 expects explicit image/action type embeddings")
    if model.token_type_embeddings.shape != (2, cfg.model.dynamics_dim):
        raise AssertionError(f"unexpected type embedding shape: {tuple(model.token_type_embeddings.shape)}")

    type_diff = (model.token_type_embeddings[0] - model.token_type_embeddings[1]).detach().float().norm().item()
    print(f"type_embedding_diff_norm={type_diff:.6f}")
    if type_diff <= 0.0:
        raise AssertionError("image/action type embeddings are identical")

    batch = 2
    steps = 3
    embed_dim = cfg.model.embed_dim
    action_dim = cfg.data.frameskip * cfg.data.action_dim
    emb = torch.randn(batch, steps, embed_dim, device="cuda")
    action = torch.randn(batch, steps, action_dim, device="cuda")
    act_emb = model.action_encoder(action)

    tokens = model.make_interleaved_tokens(emb, act_emb)
    print("interleaved_tokens", tuple(tokens.shape))
    expected_tokens = (batch, 2 * steps, cfg.model.dynamics_dim)
    if tuple(tokens.shape) != expected_tokens:
        raise AssertionError(f"expected tokens={expected_tokens}, got {tuple(tokens.shape)}")

    pred, _ = model.predict(emb, act_emb)
    print("pred", tuple(pred.shape))
    if tuple(pred.shape) != (batch, steps, embed_dim):
        raise AssertionError(f"unexpected pred shape: {tuple(pred.shape)}")

    model.zero_grad(set_to_none=True)
    loss = pred.float().square().mean()
    loss.backward()
    grad = model.token_type_embeddings.grad
    grad_max = 0.0 if grad is None else grad.detach().float().abs().max().item()
    print(f"type_embedding_grad_max_abs={grad_max:.6e}")
    if grad_max <= 0.0:
        raise AssertionError("type embeddings did not receive gradient")

    print("v7 interleaved smoke passed")


if __name__ == "__main__":
    main()
