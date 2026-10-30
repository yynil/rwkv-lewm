import argparse
import sys
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_lewm.modules import AdaLNZeroModulation
from train import build_model


def load_cfg(config_name: str, overrides: list[str]):
    config_path = Path(__file__).resolve().parents[1] / "config" / "train"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_path)):
        return hydra.compose(config_name=config_name, overrides=overrides)


def check_adaln_zero(model: torch.nn.Module) -> None:
    max_param_abs = 0.0
    max_output_abs = 0.0
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, AdaLNZeroModulation):
            continue
        count += 1
        param_abs = module.zero_init_max_abs()
        last = module.net[-1]
        cond_dim = last.in_features
        cond = torch.randn(2, 3, cond_dim, device=last.weight.device, dtype=last.weight.dtype)
        output_abs = max(chunk.detach().abs().max().item() for chunk in module(cond))
        print(f"{name}: chunks={module.num_chunks} param_max_abs={param_abs:.3e} output_max_abs={output_abs:.3e}")
        max_param_abs = max(max_param_abs, param_abs)
        max_output_abs = max(max_output_abs, output_abs)

    if count == 0:
        raise AssertionError("No AdaLNZeroModulation modules found")
    if max_param_abs != 0.0 or max_output_abs != 0.0:
        raise AssertionError(
            f"AdaLN-zero init failed: param_max_abs={max_param_abs}, output_max_abs={max_output_abs}"
        )
    print(f"checked {count} AdaLNZeroModulation modules; all zero initialized")


def check_rwkv_output_init(dynamics: torch.nn.Module) -> None:
    scale = float(getattr(dynamics.cfg, "rwkv_output_init_scale", 0.0))
    print("rwkv_output_init_scale", scale)
    if scale <= 0:
        raise AssertionError("v6 expects rwkv_output_init_scale > 0")

    for i, block in enumerate(dynamics.blocks):
        att_abs = block.att.output.weight.detach().float().abs().max().item()
        ffn_abs = block.ffn.value.weight.detach().float().abs().max().item()
        print(f"block={i} att.output.max_abs={att_abs:.6e} ffn.value.max_abs={ffn_abs:.6e}")
        if att_abs <= 0.0 or ffn_abs <= 0.0:
            raise AssertionError(f"block {i} residual output init is still zero")


def check_adaln_grad(model: torch.nn.Module) -> None:
    device = next(model.parameters()).device
    batch = 2
    steps = 3
    embed_dim = model.dynamics.cfg.input_dim
    action_dim = model.action_encoder.patch_embed.in_channels
    emb = torch.randn(batch, steps, embed_dim, device=device)
    action = torch.randn(batch, steps, action_dim, device=device)

    model.zero_grad(set_to_none=True)
    act_emb = model.action_encoder(action).to(torch.bfloat16)
    pred, _ = model.predict(emb, act_emb)
    loss = pred.float().square().mean()
    loss.backward()

    grad_max = 0.0
    for name, module in model.named_modules():
        if not isinstance(module, AdaLNZeroModulation):
            continue
        grad = module.net[-1].weight.grad
        value = 0.0 if grad is None else grad.detach().float().abs().max().item()
        print(f"{name}: grad_max_abs={value:.6e}")
        grad_max = max(grad_max, value)

    if grad_max <= 0.0:
        raise AssertionError("AdaLN-zero final Linear still has zero gradient")
    print(f"AdaLN-zero gradient smoke passed; max_grad_abs={grad_max:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pusht_rwkv_v6_a_stage2_ds")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_cfg(args.config_name, args.overrides)
    model = build_model(cfg).cuda().train()
    dynamics = model.dynamics

    print("config", args.config_name)
    print("embed_dim", cfg.model.embed_dim)
    print("dynamics_dim", cfg.model.dynamics_dim)
    print("token_fusion", cfg.model.token_fusion, type(model.fuse).__name__)
    print("input_proj", type(dynamics.input_proj).__name__)
    print("action_modulation", dynamics.cfg.action_modulation)
    print("action_modulation_scope", dynamics.cfg.action_modulation_scope)
    print("action_modulate_ffn", dynamics.cfg.action_modulate_ffn)
    print("action_residual_gate", dynamics.cfg.action_residual_gate)
    print("action_residual_gate_scale", dynamics.cfg.action_residual_gate_scale)
    print("modulate_v_first", getattr(dynamics.cfg, "modulate_v_first", False))

    if cfg.model.dynamics_dim != cfg.model.embed_dim:
        raise AssertionError("v6 expects dynamics_dim == embed_dim")
    if type(model.fuse).__name__ != "Identity":
        raise AssertionError("v6 expects latent_identity token fusion")
    if type(dynamics.input_proj).__name__ != "Identity":
        raise AssertionError("v6 expects identity dynamics input projection")
    if dynamics.cfg.action_modulation_scope == "full":
        raise AssertionError("v6 configs should use scoped action modulation, not full")
    if dynamics.cfg.modulate_v_first:
        raise AssertionError("v6 scoped experiments should not modulate v_first")

    check_adaln_zero(model)
    check_rwkv_output_init(dynamics)
    check_adaln_grad(model)


if __name__ == "__main__":
    main()
