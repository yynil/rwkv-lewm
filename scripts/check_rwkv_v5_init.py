import argparse
import sys
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_lewm.modules import AdaLNZeroModulation, modulate
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
        raise AssertionError("v5 expects rwkv_output_init_scale > 0")

    for i, block in enumerate(dynamics.blocks):
        att_abs = block.att.output.weight.detach().float().abs().max().item()
        ffn_abs = block.ffn.value.weight.detach().float().abs().max().item()
        print(f"block={i} att.output.max_abs={att_abs:.6e} ffn.value.max_abs={ffn_abs:.6e}")
        if att_abs <= 0.0 or ffn_abs <= 0.0:
            raise AssertionError(f"block {i} residual output init is still zero")


def check_branch_outputs(dynamics: torch.nn.Module) -> None:
    device = next(dynamics.parameters()).device
    hidden = dynamics.cfg.hidden_dim
    x = torch.randn(2, dynamics.cfg.chunk_len, hidden, device=device, dtype=torch.bfloat16)
    cond = torch.randn_like(x)
    block = dynamics.blocks[0]
    modulation = dynamics.modulations[0]
    v_first = torch.empty_like(x)

    with torch.no_grad():
        x0 = block.ln0(x)
        shift_attn, scale_attn, _, shift_ffn, scale_ffn, _ = modulation(cond)
        attn_in = modulate(block.ln1(x0), shift_attn, scale_attn)
        x_attn, _ = block.att(attn_in, v_first)
        ffn_out = block.ffn(modulate(block.ln2(x0), shift_ffn, scale_ffn))
        attn_max = x_attn.detach().float().abs().max().item()
        ffn_max = ffn_out.detach().float().abs().max().item()
        print(f"ungated branch output max_abs: attn={attn_max:.6e} ffn={ffn_max:.6e}")
        if attn_max <= 0.0 or ffn_max <= 0.0:
            raise AssertionError("Ungated RWKV branch output is zero; AdaLN gates cannot receive gradients")


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
        if grad is None:
            value = 0.0
        else:
            value = grad.detach().float().abs().max().item()
        print(f"{name}: grad_max_abs={value:.6e}")
        grad_max = max(grad_max, value)

    if grad_max <= 0.0:
        raise AssertionError("AdaLN-zero final Linear still has zero gradient")
    print(f"AdaLN-zero gradient smoke passed; max_grad_abs={grad_max:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pusht_rwkv_v5_stage2_ds")
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
    print("modulate_v_first", getattr(dynamics.cfg, "modulate_v_first", False))

    if cfg.model.dynamics_dim != cfg.model.embed_dim:
        raise AssertionError("v5 expects dynamics_dim == embed_dim")
    if type(model.fuse).__name__ != "Identity":
        raise AssertionError("v5 expects latent_identity token fusion")
    if type(dynamics.input_proj).__name__ != "Identity":
        raise AssertionError("v5 expects identity dynamics input projection")

    check_adaln_zero(model)
    check_rwkv_output_init(dynamics)
    check_branch_outputs(dynamics)
    check_adaln_grad(model)


if __name__ == "__main__":
    main()
