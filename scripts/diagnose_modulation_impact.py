import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LEWM_ROOT = Path(os.environ.get("LEWM_ROOT", ROOT.parent / "le-wm"))
if str(LEWM_ROOT) not in sys.path:
    sys.path.append(str(LEWM_ROOT))

try:
    import ninja

    os.environ["PATH"] = f"{ninja.BIN_DIR}:{os.environ.get('PATH', '')}"
except Exception:
    pass

from rwkv_lewm.modules import modulate
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


def load_rwkv_model(config_name: str, checkpoint: str, device: torch.device, overrides: list[str]):
    cfg = load_cfg(config_name, overrides)
    model = build_model(cfg).to(device)
    state = torch.load(resolve_model_state(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(state["module"] if "module" in state else state, strict=True)
    model.eval().requires_grad_(False)
    return cfg, model


def load_lewm_model(checkpoint: str, device: torch.device):
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.to(device).eval().requires_grad_(False)
    return model


def mean_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(dim=-1).mean()


def mean_abs(x: torch.Tensor) -> torch.Tensor:
    return x.float().abs().mean()


def ratio_pct(num: torch.Tensor, den: torch.Tensor) -> float:
    return float((num / den.clamp_min(1e-12) * 100).detach().cpu().item())


def add_record(records: dict[str, list[float]], prefix: str, metrics: dict[str, torch.Tensor | float]):
    for name, value in metrics.items():
        if torch.is_tensor(value):
            value = float(value.detach().float().cpu().item())
        records[f"{prefix}.{name}"].append(float(value))


def summarize(records: dict[str, list[float]]) -> dict[str, float]:
    return {k: sum(v) / len(v) for k, v in sorted(records.items()) if v}


def transformer_impact(model, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> dict[str, float]:
    records = defaultdict(list)
    predictor = model.predictor
    x = ctx_emb + predictor.pos_embedding[:, : ctx_emb.size(1)]
    x = predictor.dropout(x)
    transformer = predictor.transformer
    x = transformer.input_proj(x)
    c = transformer.cond_proj(ctx_act)

    with torch.no_grad():
        for i, block in enumerate(transformer.layers):
            x_in = x
            den = mean_norm(x_in)
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
                block.adaLN_modulation(c).chunk(6, dim=-1)
            )
            attn_in = modulate(block.norm1(x), shift_attn, scale_attn)
            attn_raw = block.attn(attn_in)
            attn_res = gate_attn * attn_raw
            x = x + attn_res

            ffn_in = modulate(block.norm2(x), shift_ffn, scale_ffn)
            ffn_raw = block.mlp(ffn_in)
            ffn_res = gate_ffn * ffn_raw
            x = x + ffn_res

            add_record(
                records,
                f"layer_{i}",
                {
                    "attn_residual_pct": ratio_pct(mean_norm(attn_res), den),
                    "ffn_residual_pct": ratio_pct(mean_norm(ffn_res), den),
                    "shift_attn_abs": mean_abs(shift_attn),
                    "scale_attn_abs": mean_abs(scale_attn),
                    "gate_attn_abs": mean_abs(gate_attn),
                    "shift_ffn_abs": mean_abs(shift_ffn),
                    "scale_ffn_abs": mean_abs(scale_ffn),
                    "gate_ffn_abs": mean_abs(gate_ffn),
                },
            )
    return summarize(records)


def rwkv_full_impact(model, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> dict[str, float]:
    records = defaultdict(list)
    tokens = model.make_tokens(ctx_emb, ctx_act)
    dyn_dtype = next(model.dynamics.parameters()).dtype
    cond = model.dynamics.cond_proj(ctx_act.to(dtype=dyn_dtype))
    real_t = tokens.size(1)
    chunk_len = int(model.dynamics.cfg.chunk_len)
    pad_t = (-real_t) % chunk_len
    if pad_t:
        tokens = torch.cat(
            [tokens, tokens.new_zeros(tokens.size(0), pad_t, tokens.size(2))],
            dim=1,
        )
        cond = torch.cat(
            [cond, cond.new_zeros(cond.size(0), pad_t, cond.size(2))],
            dim=1,
        )
    x = model.dynamics.input_proj(tokens.to(dtype=torch.bfloat16))
    v_first = torch.empty_like(x)

    with torch.no_grad():
        for i, (block, modulation) in enumerate(zip(model.dynamics.blocks, model.dynamics.modulations)):
            if block.layer_id == 0:
                x = block.ln0(x)
            x_in = x
            den = mean_norm(x_in[:, :real_t])

            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation(cond)
            attn_in = modulate(block.ln1(x), shift_attn, scale_attn)
            x_attn, v_first = block.att(attn_in, v_first)
            attn_res = gate_attn * x_attn
            x = x + attn_res

            ffn_in = modulate(block.ln2(x), shift_ffn, scale_ffn)
            ffn_raw = block.ffn(ffn_in)
            ffn_res = gate_ffn * ffn_raw
            x = x + ffn_res

            add_record(
                records,
                f"layer_{i}",
                {
                    "attn_residual_pct": ratio_pct(mean_norm(attn_res[:, :real_t]), den),
                    "ffn_residual_pct": ratio_pct(mean_norm(ffn_res[:, :real_t]), den),
                    "attn_raw_pct": ratio_pct(mean_norm(x_attn[:, :real_t]), den),
                    "ffn_raw_pct": ratio_pct(mean_norm(ffn_raw[:, :real_t]), den),
                    "shift_attn_abs": mean_abs(shift_attn[:, :real_t]),
                    "scale_attn_abs": mean_abs(scale_attn[:, :real_t]),
                    "gate_attn_abs": mean_abs(gate_attn[:, :real_t]),
                    "shift_ffn_abs": mean_abs(shift_ffn[:, :real_t]),
                    "scale_ffn_abs": mean_abs(scale_ffn[:, :real_t]),
                    "gate_ffn_abs": mean_abs(gate_ffn[:, :real_t]),
                },
            )
    return summarize(records)


def rwkv_scoped_impact(model, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> dict[str, float]:
    records = defaultdict(list)
    tokens = model.make_tokens(ctx_emb, ctx_act)
    dyn_dtype = next(model.dynamics.parameters()).dtype
    cond = model.dynamics.cond_proj(ctx_act.to(dtype=dyn_dtype))
    real_t = tokens.size(1)
    chunk_len = int(model.dynamics.cfg.chunk_len)
    pad_t = (-real_t) % chunk_len
    if pad_t:
        tokens = torch.cat(
            [tokens, tokens.new_zeros(tokens.size(0), pad_t, tokens.size(2))],
            dim=1,
        )
        cond = torch.cat(
            [cond, cond.new_zeros(cond.size(0), pad_t, cond.size(2))],
            dim=1,
        )
    x = model.dynamics.input_proj(tokens.to(dtype=torch.bfloat16))
    v_first = torch.empty_like(x)

    with torch.no_grad():
        for i, (block, modulation, ffn_modulation) in enumerate(
            zip(
                model.dynamics.blocks,
                model.dynamics.modulations,
                model.dynamics.ffn_modulations or [None] * len(model.dynamics.blocks),
            )
        ):
            if block.layer_id == 0:
                x = block.ln0(x)
            x_in = x
            den = mean_norm(x_in[:, :real_t])
            attn_in = block.ln1(x)
            x_attn, v_first = model.dynamics._time_mixer_forward_scoped(
                block.att,
                attn_in,
                v_first,
                modulation(cond),
            )
            x = x + x_attn
            ffn_in = block.ln2(x)
            if ffn_modulation is not None:
                shift_ffn, scale_ffn = ffn_modulation(cond)
                ffn_in = modulate(ffn_in, shift_ffn, scale_ffn)
            ffn_raw = block.ffn(ffn_in)
            x = x + ffn_raw

            metrics = {
                "attn_residual_pct": ratio_pct(mean_norm(x_attn[:, :real_t]), den),
                "ffn_residual_pct": ratio_pct(mean_norm(ffn_raw[:, :real_t]), den),
            }
            chunks = modulation(cond)
            labels = ("shift_k", "scale_k", "shift_v", "scale_v")
            if model.dynamics.cfg.action_modulation_scope == "rkv":
                labels = ("shift_r", "scale_r", "shift_k", "scale_k", "shift_v", "scale_v")
            for label, chunk in zip(labels, chunks):
                metrics[f"{label}_abs"] = mean_abs(chunk[:, :real_t])
            if model.dynamics.cfg.action_modulation_scope == "kv_limited":
                shift_k, scale_k, shift_v, scale_v = chunks
                metrics["effective_shift_k_abs"] = mean_abs(
                    model.dynamics.cfg.kv_max_shift * torch.tanh(shift_k[:, :real_t])
                )
                metrics["effective_scale_k_abs"] = mean_abs(
                    model.dynamics.cfg.kv_max_scale * torch.tanh(scale_k[:, :real_t])
                )
                metrics["effective_shift_v_abs"] = mean_abs(
                    model.dynamics.cfg.kv_max_shift * torch.tanh(shift_v[:, :real_t])
                )
                metrics["effective_scale_v_abs"] = mean_abs(
                    model.dynamics.cfg.kv_max_scale * torch.tanh(scale_v[:, :real_t])
                )
            if ffn_modulation is not None:
                metrics["shift_ffn_abs"] = mean_abs(shift_ffn[:, :real_t])
                metrics["scale_ffn_abs"] = mean_abs(scale_ffn[:, :real_t])
            add_record(records, f"layer_{i}", metrics)
    return summarize(records)


def rwkv_pre_x_impact(model, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> dict[str, float]:
    records = defaultdict(list)
    tokens = model.make_tokens(ctx_emb, ctx_act)
    dyn_dtype = next(model.dynamics.parameters()).dtype
    cond = model.dynamics.cond_proj(ctx_act.to(dtype=dyn_dtype))
    real_t = tokens.size(1)
    chunk_len = int(model.dynamics.cfg.chunk_len)
    pad_t = (-real_t) % chunk_len
    if pad_t:
        tokens = torch.cat(
            [tokens, tokens.new_zeros(tokens.size(0), pad_t, tokens.size(2))],
            dim=1,
        )
        cond = torch.cat(
            [cond, cond.new_zeros(cond.size(0), pad_t, cond.size(2))],
            dim=1,
        )
    x = model.dynamics.input_proj(tokens.to(dtype=torch.bfloat16))
    v_first = torch.empty_like(x)

    with torch.no_grad():
        for i, (block, modulation) in enumerate(zip(model.dynamics.blocks, model.dynamics.modulations)):
            if block.layer_id == 0:
                x = block.ln0(x)
            x_in = x
            den = mean_norm(x_in[:, :real_t])
            shift_attn, scale_attn, shift_ffn, scale_ffn = modulation(cond)

            raw_attn_in = block.ln1(x)
            attn_in = modulate(raw_attn_in, shift_attn, scale_attn)
            x_attn, v_first = block.att(attn_in, v_first)
            x = x + x_attn

            raw_ffn_in = block.ln2(x)
            ffn_in = modulate(raw_ffn_in, shift_ffn, scale_ffn)
            ffn_raw = block.ffn(ffn_in)
            x = x + ffn_raw

            add_record(
                records,
                f"layer_{i}",
                {
                    "attn_residual_pct": ratio_pct(mean_norm(x_attn[:, :real_t]), den),
                    "ffn_residual_pct": ratio_pct(mean_norm(ffn_raw[:, :real_t]), den),
                    "attn_input_delta_pct": ratio_pct(
                        mean_norm((attn_in - raw_attn_in)[:, :real_t]),
                        mean_norm(raw_attn_in[:, :real_t]),
                    ),
                    "ffn_input_delta_pct": ratio_pct(
                        mean_norm((ffn_in - raw_ffn_in)[:, :real_t]),
                        mean_norm(raw_ffn_in[:, :real_t]),
                    ),
                    "shift_attn_abs": mean_abs(shift_attn[:, :real_t]),
                    "scale_attn_abs": mean_abs(scale_attn[:, :real_t]),
                    "shift_ffn_abs": mean_abs(shift_ffn[:, :real_t]),
                    "scale_ffn_abs": mean_abs(scale_ffn[:, :real_t]),
                },
            )
    return summarize(records)


def rwkv_adapter_impact(model, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> dict[str, float]:
    records = defaultdict(list)
    tokens = model.make_tokens(ctx_emb, ctx_act)
    dyn_dtype = next(model.dynamics.parameters()).dtype
    cond = model.dynamics.cond_proj(ctx_act.to(dtype=dyn_dtype))
    real_t = tokens.size(1)
    chunk_len = int(model.dynamics.cfg.chunk_len)
    pad_t = (-real_t) % chunk_len
    if pad_t:
        tokens = torch.cat(
            [tokens, tokens.new_zeros(tokens.size(0), pad_t, tokens.size(2))],
            dim=1,
        )
        cond = torch.cat(
            [cond, cond.new_zeros(cond.size(0), pad_t, cond.size(2))],
            dim=1,
        )
    x = model.dynamics.input_proj(tokens.to(dtype=torch.bfloat16))
    v_first = torch.empty_like(x)

    with torch.no_grad():
        for i, (block, modulation) in enumerate(zip(model.dynamics.blocks, model.dynamics.modulations)):
            x_in = x
            den = mean_norm(x_in[:, :real_t])

            pre_adapter_res = torch.zeros_like(x)
            post_adapter_res = torch.zeros_like(x)
            if model.dynamics.cfg.action_adapter_position in {"pre", "pre_post"}:
                x_before_adapter = x
                x = model.dynamics._apply_action_adapter(i, x, cond, modulation)
                pre_adapter_res = x - x_before_adapter

            x_before_block = x
            x, v_first = block(x, v_first)
            block_res = x - x_before_block

            if model.dynamics.cfg.action_adapter_position in {"post", "pre_post"}:
                x_before_adapter = x
                x = model.dynamics._apply_action_adapter(i, x, cond, modulation)
                post_adapter_res = x - x_before_adapter

            shift, scale, gate = modulation(cond)
            add_record(
                records,
                f"layer_{i}",
                {
                    "block_residual_pct": ratio_pct(mean_norm(block_res[:, :real_t]), den),
                    "pre_adapter_residual_pct": ratio_pct(mean_norm(pre_adapter_res[:, :real_t]), den),
                    "post_adapter_residual_pct": ratio_pct(mean_norm(post_adapter_res[:, :real_t]), den),
                    "shift_abs": mean_abs(shift[:, :real_t]),
                    "scale_abs": mean_abs(scale[:, :real_t]),
                    "gate_abs": mean_abs(gate[:, :real_t]),
                    "effective_shift_abs": mean_abs(
                        model.dynamics.cfg.action_adapter_max_shift * torch.tanh(shift[:, :real_t])
                    ),
                    "effective_scale_abs": mean_abs(
                        model.dynamics.cfg.action_adapter_max_scale * torch.tanh(scale[:, :real_t])
                    ),
                    "effective_gate_abs": mean_abs(
                        model.dynamics.cfg.action_adapter_residual_scale * torch.tanh(gate[:, :real_t])
                    ),
                },
            )
    return summarize(records)


def rwkv_transition_impact(model, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> dict[str, float]:
    records = defaultdict(list)
    tokens = model.make_tokens(ctx_emb, ctx_act)

    with torch.no_grad():
        dynamics_out, _ = model.dynamics(tokens, None, cond=None)
        if model.transition_head is None:
            add_record(records, "transition", {"state_norm": mean_norm(dynamics_out)})
            return summarize(records)

        transition_out = model.transition_head(dynamics_out, ctx_act)
        transition_res = transition_out - dynamics_out
        metrics = {
            "state_norm": mean_norm(dynamics_out),
            "transition_residual_pct": ratio_pct(mean_norm(transition_res), mean_norm(dynamics_out)),
        }
        if hasattr(model.transition_head, "modulation"):
            dtype = next(model.transition_head.parameters()).dtype
            cond = model.transition_head.cond_proj(ctx_act.to(dtype=dtype))
            shift, scale, gate = model.transition_head.modulation(cond)
            metrics.update(
                {
                    "shift_abs": mean_abs(shift),
                    "scale_abs": mean_abs(scale),
                    "gate_abs": mean_abs(gate),
                }
            )
            if getattr(model.transition_head, "bounded", False):
                metrics.update(
                    {
                        "effective_shift_abs": mean_abs(
                            model.transition_head.max_shift * torch.tanh(shift)
                        ),
                        "effective_scale_abs": mean_abs(
                            model.transition_head.max_scale * torch.tanh(scale)
                        ),
                    }
                )
            if getattr(model.transition_head, "gate_tanh", False):
                metrics["effective_gate_abs"] = mean_abs(
                    model.transition_head.residual_scale * torch.tanh(gate)
                )
            else:
                metrics["effective_gate_abs"] = mean_abs(model.transition_head.residual_scale * gate)
        add_record(records, "transition", metrics)
    return summarize(records)


def rwkv_block_transition_impact(model, ctx_emb: torch.Tensor, ctx_act: torch.Tensor) -> dict[str, float]:
    records = defaultdict(list)
    tokens = model.make_tokens(ctx_emb, ctx_act)
    real_t = tokens.size(1)
    current_act = ctx_act[:, -1:]

    with torch.no_grad():
        x = model.dynamics.input_proj(tokens.to(dtype=torch.bfloat16))
        x, _ = model.dynamics._pad_to_chunk(x)
        v_first = torch.empty_like(x)

        for i, block in enumerate(model.dynamics.blocks):
            x_before_block = x
            if model.dynamics.cfg.block_action_transition_target == "all":
                target = slice(0, real_t)
            else:
                target = slice(real_t - 1, real_t)
            den = mean_norm(x_before_block[:, target])

            x, v_first = block(x, v_first)
            block_res = x - x_before_block
            x_before_transition = x
            x = model.dynamics._apply_block_action_transition(i, x, current_act, real_t)
            transition_res = x - x_before_transition

            transition = model.dynamics.block_action_transitions[i]
            dtype = next(transition.parameters()).dtype
            cond = transition.cond_proj(current_act.to(dtype=dtype))
            shift, scale, gate = transition.modulation(cond)
            metrics = {
                "block_residual_pct": ratio_pct(mean_norm(block_res[:, target]), den),
                "transition_residual_pct": ratio_pct(mean_norm(transition_res[:, target]), den),
                "state_norm": mean_norm(x_before_transition[:, target]),
                "shift_abs": mean_abs(shift),
                "scale_abs": mean_abs(scale),
                "gate_abs": mean_abs(gate),
            }
            if getattr(transition, "bounded", False):
                metrics.update(
                    {
                        "effective_shift_abs": mean_abs(transition.max_shift * torch.tanh(shift)),
                        "effective_scale_abs": mean_abs(transition.max_scale * torch.tanh(scale)),
                    }
                )
            if getattr(transition, "gate_tanh", False):
                metrics["effective_gate_abs"] = mean_abs(transition.residual_scale * torch.tanh(gate))
            else:
                metrics["effective_gate_abs"] = mean_abs(transition.residual_scale * gate)
            add_record(records, f"layer_{i}", metrics)
    return summarize(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["rwkv", "lewm"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="pusht_rwkv_v9_statecond_lewmctx_stage2_ds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    cfg = load_cfg(args.config_name, args.overrides)
    cfg.loader.batch_size = args.batch_size
    cfg.loader.num_workers = args.num_workers
    cfg.loader.persistent_workers = args.num_workers > 0
    cfg.data.num_steps = 4

    dataset = load_stable_worldmodel_dataset(cfg)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    if args.kind == "rwkv":
        _, model = load_rwkv_model(args.config_name, args.checkpoint, device, args.overrides)
    else:
        model = load_lewm_model(args.checkpoint, device)

    records = defaultdict(list)
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        with torch.no_grad():
            out = model.encode(batch)
            emb = out["emb"].detach()
            act_emb = out["act_emb"].detach()
        ctx_emb = emb[:, :3]
        ctx_act = act_emb[:, :3]

        if args.kind == "lewm":
            summary = transformer_impact(model, ctx_emb, ctx_act)
        elif getattr(model.dynamics, "block_action_transitions", None) is not None:
            summary = rwkv_block_transition_impact(model, ctx_emb, ctx_act)
        elif getattr(model, "transition_head", None) is not None and model.dynamics.modulations is None:
            summary = rwkv_transition_impact(model, ctx_emb, ctx_act)
        elif model.dynamics.cfg.action_modulation_scope == "full":
            summary = rwkv_full_impact(model, ctx_emb, ctx_act)
        elif model.dynamics.cfg.action_modulation_scope == "pre_x":
            summary = rwkv_pre_x_impact(model, ctx_emb, ctx_act)
        elif model.dynamics.cfg.action_modulation_scope == "adapter":
            summary = rwkv_adapter_impact(model, ctx_emb, ctx_act)
        else:
            summary = rwkv_scoped_impact(model, ctx_emb, ctx_act)
        for key, value in summary.items():
            records[key].append(value)

    result = {
        "kind": args.kind,
        "checkpoint": args.checkpoint,
        "config_name": args.config_name,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "summary": summarize(records),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
