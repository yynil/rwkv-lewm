import importlib.util
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .modules import ActionTransitionHead, AdaLNZeroModulation, modulate


@dataclass
class RWKV7CudaConfig:
    input_dim: int = 384
    hidden_dim: int = 384
    output_dim: int = 192
    num_layers: int = 6
    head_size: int = 64
    ffn_dim: int | None = None
    chunk_len: int = 16
    action_cond_dim: int | None = None
    action_modulation: str = "none"
    action_modulation_scope: str = "full"
    action_modulate_ffn: bool = True
    action_residual_gate: str = "adaln"
    action_residual_gate_scale: float = 0.1
    kv_max_scale: float = 0.05
    kv_max_shift: float = 0.05
    action_adapter_residual_scale: float = 0.1
    action_adapter_max_scale: float = 0.05
    action_adapter_max_shift: float = 0.05
    action_adapter_position: str = "post"
    project_action_condition: bool = True
    modulate_v_first: bool = False
    rwkv_output_init_scale: float = 0.0
    zero_head: bool = True
    output_head_type: str = "linear"
    block_action_transition_type: str = "none"
    block_action_transition_hidden_scale: int = 4
    block_action_transition_residual_scale: float = 0.1
    block_action_transition_bounded: bool = True
    block_action_transition_max_scale: float = 0.05
    block_action_transition_max_shift: float = 0.05
    block_action_transition_gate_tanh: bool = True
    block_action_transition_target: str = "last"


@contextmanager
def _temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


_OFFICIAL_MODULE = None


def load_official_rwkv_v7():
    """Load vendored RWKV-v7 train_temp/src/model.py with its CUDA sources."""

    global _OFFICIAL_MODULE
    if _OFFICIAL_MODULE is not None:
        return _OFFICIAL_MODULE

    try:
        import ninja

        os.environ["PATH"] = f"{ninja.BIN_DIR}:{os.environ.get('PATH', '')}"
    except Exception:
        pass

    vendor_dir = Path(__file__).parent / "backends" / "rwkv_v7_official"
    model_path = vendor_dir / "model.py"
    os.environ.setdefault("RWKV_JIT_ON", "0")
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")

    spec = importlib.util.spec_from_file_location("rwkv_lewm_official_rwkv_v7", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RWKV-v7 model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    with _temporary_cwd(vendor_dir):
        spec.loader.exec_module(module)
    _OFFICIAL_MODULE = module
    return module


class RWKV7CudaDynamics(nn.Module):
    """RWKV-v7 CUDA TimeMixer dynamics adapter.

    This uses BlinkDL's current RWKV-v7 train_temp implementation and CUDA
    forward/backward kernels. The adapter removes token embeddings and LM head,
    taking continuous LeWM latent-action tokens instead.
    """

    def __init__(self, cfg: RWKV7CudaConfig):
        super().__init__()
        if cfg.head_size != 64:
            raise ValueError("The vendored RWKV-v7 CUDA model.py currently hard-codes head_size=64")
        if cfg.hidden_dim % cfg.head_size != 0:
            raise ValueError("hidden_dim must be divisible by head_size")
        self.cfg = cfg
        official = load_official_rwkv_v7()
        args = SimpleNamespace(
            n_embd=cfg.hidden_dim,
            dim_att=cfg.hidden_dim,
            dim_ffn=cfg.ffn_dim if cfg.ffn_dim is not None else cfg.hidden_dim * 4,
            n_layer=cfg.num_layers,
            head_size=cfg.head_size,
            my_testing="x070",
            grad_cp=0,
        )
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim) if cfg.input_dim != cfg.hidden_dim else nn.Identity()
        self.blocks = nn.ModuleList([official.Block(args, i) for i in range(cfg.num_layers)])
        self._init_rwkv_residual_outputs(cfg.rwkv_output_init_scale)
        if cfg.action_modulation not in {"none", "adaln_zero"}:
            raise ValueError(f"Unknown action_modulation: {cfg.action_modulation}")
        if cfg.action_modulation_scope not in {"full", "kv", "kv_limited", "rkv", "adapter", "pre_x"}:
            raise ValueError(f"Unknown action_modulation_scope: {cfg.action_modulation_scope}")
        if cfg.action_residual_gate not in {"adaln", "none", "small"}:
            raise ValueError(f"Unknown action_residual_gate: {cfg.action_residual_gate}")
        if cfg.action_adapter_position not in {"pre", "post", "pre_post"}:
            raise ValueError(f"Unknown action_adapter_position: {cfg.action_adapter_position}")
        if cfg.block_action_transition_type not in {"none", "adaln"}:
            raise ValueError(f"Unknown block_action_transition_type: {cfg.block_action_transition_type}")
        if cfg.block_action_transition_target not in {"last", "all"}:
            raise ValueError(f"Unknown block_action_transition_target: {cfg.block_action_transition_target}")
        if cfg.output_head_type not in {"linear", "identity"}:
            raise ValueError(f"Unknown output_head_type: {cfg.output_head_type}")
        if cfg.output_head_type == "identity" and cfg.hidden_dim != cfg.output_dim:
            raise ValueError(
                "RWKV identity output head requires hidden_dim == output_dim, "
                f"got hidden_dim={cfg.hidden_dim}, output_dim={cfg.output_dim}"
            )
        if cfg.action_modulation == "adaln_zero":
            if cfg.action_cond_dim is None:
                raise ValueError("action_cond_dim is required for adaln_zero action modulation")
            if cfg.action_modulation_scope != "full" and cfg.modulate_v_first:
                raise ValueError("scoped action modulation intentionally does not support v_first modulation")
            self.cond_proj = (
                nn.Linear(cfg.action_cond_dim, cfg.hidden_dim)
                if cfg.project_action_condition and cfg.action_cond_dim != cfg.hidden_dim
                else nn.Identity()
            )
            if cfg.action_modulation_scope == "adapter":
                tmix_chunks = 3
            elif cfg.action_modulation_scope in {"full", "rkv"}:
                tmix_chunks = 6
            else:
                tmix_chunks = 4
            self.modulations = nn.ModuleList(
                [
                    AdaLNZeroModulation(cfg.hidden_dim, cfg.hidden_dim, num_chunks=tmix_chunks)
                    for _ in range(cfg.num_layers)
                ]
            )
            self.ffn_modulations = (
                nn.ModuleList(
                    [AdaLNZeroModulation(cfg.hidden_dim, cfg.hidden_dim, num_chunks=2) for _ in range(cfg.num_layers)]
                )
                if cfg.action_modulation_scope in {"kv", "kv_limited", "rkv"} and cfg.action_modulate_ffn
                else None
            )
            self.residual_gate_modulations = (
                nn.ModuleList(
                    [AdaLNZeroModulation(cfg.hidden_dim, cfg.hidden_dim, num_chunks=1) for _ in range(cfg.num_layers)]
                )
                if cfg.action_modulation_scope in {"kv", "kv_limited", "rkv"} and cfg.action_residual_gate == "small"
                else None
            )
            self.v_first_modulations = (
                nn.ModuleList(
                    [AdaLNZeroModulation(cfg.hidden_dim, cfg.hidden_dim, num_chunks=2) for _ in range(cfg.num_layers)]
                )
                if cfg.modulate_v_first
                else None
            )
            self.action_adapter_norms = (
                nn.ModuleList([nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_layers)])
                if cfg.action_modulation_scope == "adapter"
                else None
            )
            self.action_adapters = (
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim),
                            nn.SiLU(),
                            nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim),
                        )
                        for _ in range(cfg.num_layers)
                    ]
                )
                if cfg.action_modulation_scope == "adapter"
                else None
            )
        else:
            self.cond_proj = None
            self.modulations = None
            self.ffn_modulations = None
            self.residual_gate_modulations = None
            self.v_first_modulations = None
            self.action_adapter_norms = None
            self.action_adapters = None
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Identity() if cfg.output_head_type == "identity" else nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.block_action_transitions = (
            nn.ModuleList(
                [
                    ActionTransitionHead(
                        hidden_dim=cfg.hidden_dim,
                        action_dim=cfg.action_cond_dim or cfg.hidden_dim,
                        hidden_scale=cfg.block_action_transition_hidden_scale,
                        residual_scale=cfg.block_action_transition_residual_scale,
                        bounded=cfg.block_action_transition_bounded,
                        max_scale=cfg.block_action_transition_max_scale,
                        max_shift=cfg.block_action_transition_max_shift,
                        gate_tanh=cfg.block_action_transition_gate_tanh,
                    )
                    for _ in range(cfg.num_layers)
                ]
            )
            if cfg.block_action_transition_type == "adaln"
            else None
        )
        if cfg.zero_head and isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.blocks.to(dtype=torch.bfloat16)
        if self.modulations is not None:
            self.cond_proj.to(dtype=torch.bfloat16)
            self.modulations.to(dtype=torch.bfloat16)
            if self.ffn_modulations is not None:
                self.ffn_modulations.to(dtype=torch.bfloat16)
            if self.residual_gate_modulations is not None:
                self.residual_gate_modulations.to(dtype=torch.bfloat16)
            if self.v_first_modulations is not None:
                self.v_first_modulations.to(dtype=torch.bfloat16)
            if self.action_adapter_norms is not None:
                self.action_adapter_norms.to(dtype=torch.bfloat16)
            if self.action_adapters is not None:
                self.action_adapters.to(dtype=torch.bfloat16)
        if self.block_action_transitions is not None:
            self.block_action_transitions.to(dtype=torch.bfloat16)
        self.norm.to(dtype=torch.bfloat16)
        self.head.to(dtype=torch.bfloat16)

    def _init_rwkv_residual_outputs(self, scale: float) -> None:
        """Optionally undo RWKV-v7's zero output init for AdaLN-zero gating.

        Official RWKV-v7 initializes att.output and ffn.value to zero so the
        plain residual block starts as an identity map while those weights learn.
        With an additional zero-initialized AdaLN gate, that creates a double
        zero: the gated branch output is zero and the gate gradient is also zero.
        A small nonzero branch init preserves AdaLN-zero's initial no-op output
        because the gate is still zero, while allowing the gate to receive
        gradients on the first step.
        """

        if scale <= 0:
            return
        for block in self.blocks:
            for linear in (block.att.output, block.ffn.value):
                bound = scale / math.sqrt(linear.in_features)
                nn.init.uniform_(linear.weight, -bound, bound)

    @property
    def chunk_len(self) -> int:
        return self.cfg.chunk_len

    def _pad_to_chunk(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        remainder = x.size(1) % self.chunk_len
        if remainder == 0:
            return x, 0
        pad = self.chunk_len - remainder
        return torch.cat([x, x[:, -1:].expand(-1, pad, -1)], dim=1), pad

    def _block_forward(
        self,
        block: nn.Module,
        x: torch.Tensor,
        v_first: torch.Tensor,
        cond: torch.Tensor | None,
        modulation: AdaLNZeroModulation | None,
        ffn_modulation: AdaLNZeroModulation | None,
        residual_gate_modulation: AdaLNZeroModulation | None,
        v_first_modulation: AdaLNZeroModulation | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modulation is None or cond is None:
            return block(x, v_first)

        if self.cfg.action_modulation_scope == "adapter":
            if self.cfg.action_adapter_position in {"pre", "pre_post"}:
                x = self._apply_action_adapter(block.layer_id, x, cond, modulation)
            x, v_first = block(x, v_first)
            if self.cfg.action_adapter_position in {"post", "pre_post"}:
                x = self._apply_action_adapter(block.layer_id, x, cond, modulation)
            return x, v_first

        if block.layer_id == 0:
            x = block.ln0(x)

        if self.cfg.action_modulation_scope == "pre_x":
            shift_attn, scale_attn, shift_ffn, scale_ffn = modulation(cond)
            attn_in = modulate(block.ln1(x), shift_attn, scale_attn)
            x_attn, v_first = block.att(attn_in, v_first)
            x = x + x_attn
            ffn_in = modulate(block.ln2(x), shift_ffn, scale_ffn)
            x = x + block.ffn(ffn_in)
            return x, v_first

        if self.cfg.action_modulation_scope != "full":
            return self._block_forward_scoped(
                block,
                x,
                v_first,
                cond,
                modulation,
                ffn_modulation,
                residual_gate_modulation,
            )

        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation(cond)
        attn_in = modulate(block.ln1(x), shift_attn, scale_attn)
        if v_first_modulation is None:
            x_attn, v_first = block.att(attn_in, v_first)
        else:
            shift_v_first, scale_v_first = v_first_modulation(cond)
            if block.layer_id == 0:
                x_attn, raw_v_first = block.att(attn_in, v_first)
                v_first = modulate(raw_v_first, shift_v_first, scale_v_first)
            else:
                x_attn, _ = block.att(attn_in, modulate(v_first, shift_v_first, scale_v_first))
        x = x + gate_attn * x_attn
        x = x + gate_ffn * block.ffn(modulate(block.ln2(x), shift_ffn, scale_ffn))
        return x, v_first

    def _bounded_modulate(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        max_shift: float,
        max_scale: float,
    ) -> torch.Tensor:
        return x * (1 + max_scale * torch.tanh(scale)) + max_shift * torch.tanh(shift)

    def _apply_action_adapter(
        self,
        layer_id: int,
        x: torch.Tensor,
        cond: torch.Tensor,
        modulation: AdaLNZeroModulation,
    ) -> torch.Tensor:
        if self.action_adapter_norms is None or self.action_adapters is None:
            raise ValueError("action adapter modules are not initialized")
        shift, scale, gate = modulation(cond)
        adapter_in = self._bounded_modulate(
            self.action_adapter_norms[layer_id](x),
            shift,
            scale,
            self.cfg.action_adapter_max_shift,
            self.cfg.action_adapter_max_scale,
        )
        adapter_raw = torch.tanh(self.action_adapters[layer_id](adapter_in))
        adapter_gate = self.cfg.action_adapter_residual_scale * torch.tanh(gate)
        return x + adapter_gate * adapter_raw

    def _apply_block_action_transition(
        self,
        layer_id: int,
        x: torch.Tensor,
        block_action: torch.Tensor,
        real_t: int,
    ) -> torch.Tensor:
        if self.block_action_transitions is None:
            return x
        if block_action.ndim == 2:
            block_action = block_action.unsqueeze(1)
        if block_action.size(1) != 1:
            raise ValueError(
                "block action transition expects one current action embedding, "
                f"got shape {tuple(block_action.shape)}"
            )
        transition = self.block_action_transitions[layer_id]
        if self.cfg.block_action_transition_target == "all":
            action = block_action.expand(-1, real_t, -1)
            updated = transition(x[:, :real_t], action)
            if real_t == x.size(1):
                return updated
            return torch.cat([updated, x[:, real_t:]], dim=1)

        idx = real_t - 1
        updated = transition(x[:, idx : idx + 1], block_action)
        return torch.cat([x[:, :idx], updated, x[:, idx + 1 :]], dim=1)

    def _time_mixer_forward_scoped(
        self,
        att: nn.Module,
        x: torch.Tensor,
        v_first: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        official = load_official_rwkv_v7()

        xr, xw, xk, xv, xa, xg = official.tmix_mix6_bf16_v5(
            x,
            att.x_r.view(-1),
            att.x_w.view(-1),
            att.x_k.view(-1),
            att.x_v.view(-1),
            att.x_a.view(-1),
            att.x_g.view(-1),
        )

        if self.cfg.action_modulation_scope == "kv":
            shift_k, scale_k, shift_v, scale_v = modulation
            xk = modulate(xk, shift_k, scale_k)
            xv = modulate(xv, shift_v, scale_v)
        elif self.cfg.action_modulation_scope == "kv_limited":
            shift_k, scale_k, shift_v, scale_v = modulation
            xk = self._bounded_modulate(xk, shift_k, scale_k, self.cfg.kv_max_shift, self.cfg.kv_max_scale)
            xv = self._bounded_modulate(xv, shift_v, scale_v, self.cfg.kv_max_shift, self.cfg.kv_max_scale)
        elif self.cfg.action_modulation_scope == "rkv":
            shift_r, scale_r, shift_k, scale_k, shift_v, scale_v = modulation
            xr = modulate(xr, shift_r, scale_r)
            xk = modulate(xk, shift_k, scale_k)
            xv = modulate(xv, shift_v, scale_v)
        else:
            raise ValueError(f"Unsupported scoped modulation: {self.cfg.action_modulation_scope}")

        r = att.receptance(xr)
        w = att.w0 + torch.tanh(xw @ att.w1) @ att.w2
        k = att.key(xk)
        v = att.value(xv)
        if att.layer_id == 0:
            v_first = v
        else:
            v12 = (xv @ att.v1) @ att.v2
            v = official.tmix_vres_gate_bf16_v1(v, v_first, att.v0, v12)

        a = official.tmix_a_gate_bf16(att.a0, (xa @ att.a1) @ att.a2)
        g = torch.sigmoid(xg @ att.g1) @ att.g2
        k, neg_kk, kka = official.tmix_kk_pre_bf16_v5(
            k,
            att.k_k.view(-1),
            a,
            att.k_a.view(-1),
        )
        x = official.RWKV7_CLAMPW_CUDA(r, w, k, v, neg_kk, kka)
        x = official.tmix_lnx_rkvres_xg_bf16_v1(
            x,
            r,
            k,
            v,
            att.r_k,
            att.ln_x.weight,
            att.ln_x.bias,
            g,
        )
        return att.output(x), v_first

    def _block_forward_scoped(
        self,
        block: nn.Module,
        x: torch.Tensor,
        v_first: torch.Tensor,
        cond: torch.Tensor,
        modulation: AdaLNZeroModulation,
        ffn_modulation: AdaLNZeroModulation | None,
        residual_gate_modulation: AdaLNZeroModulation | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_in = block.ln1(x)
        x_attn, v_first = self._time_mixer_forward_scoped(block.att, attn_in, v_first, modulation(cond))
        if self.cfg.action_residual_gate == "small":
            if residual_gate_modulation is None:
                raise ValueError("small residual gate requires residual_gate_modulations")
            (gate_attn,) = residual_gate_modulation(cond)
            x = x + (1 + self.cfg.action_residual_gate_scale * torch.tanh(gate_attn)) * x_attn
        elif self.cfg.action_residual_gate == "none":
            x = x + x_attn
        else:
            raise ValueError("adaln residual gate is only used by full action modulation")

        ffn_in = block.ln2(x)
        if ffn_modulation is not None:
            shift_ffn, scale_ffn = ffn_modulation(cond)
            ffn_in = modulate(ffn_in, shift_ffn, scale_ffn)
        x = x + block.ffn(ffn_in)
        return x, v_first

    def forward(
        self,
        x: torch.Tensor,
        state: object | None = None,
        cond: torch.Tensor | None = None,
        block_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object | None]:
        del state
        out_dtype = x.dtype
        real_t = x.size(1)
        x = self.input_proj(x).to(torch.bfloat16)
        if cond is not None:
            if self.cond_proj is None:
                raise ValueError("cond was provided but action modulation is disabled")
            cond = self.cond_proj(cond.to(torch.bfloat16))
        if block_action is not None:
            if self.block_action_transitions is None:
                raise ValueError("block_action was provided but block action transition is disabled")
            block_action = block_action.to(torch.bfloat16)
        x, pad = self._pad_to_chunk(x)
        if cond is not None:
            cond, _ = self._pad_to_chunk(cond)
        v_first = torch.empty_like(x)
        if self.modulations is None:
            modulations = [None] * len(self.blocks)
            ffn_modulations = [None] * len(self.blocks)
            residual_gate_modulations = [None] * len(self.blocks)
            v_first_modulations = [None] * len(self.blocks)
        else:
            modulations = self.modulations
            ffn_modulations = self.ffn_modulations or [None] * len(self.blocks)
            residual_gate_modulations = self.residual_gate_modulations or [None] * len(self.blocks)
            v_first_modulations = self.v_first_modulations or [None] * len(self.blocks)
        for block, modulation, ffn_modulation, residual_gate_modulation, v_first_modulation in zip(
            self.blocks,
            modulations,
            ffn_modulations,
            residual_gate_modulations,
            v_first_modulations,
        ):
            x, v_first = self._block_forward(
                block,
                x,
                v_first,
                cond,
                modulation,
                ffn_modulation,
                residual_gate_modulation,
                v_first_modulation,
            )
            if block_action is not None:
                x = self._apply_block_action_transition(block.layer_id, x, block_action, real_t)
        x = self.head(self.norm(x)).to(out_dtype)
        if pad:
            x = x[:, :-pad]
        return x, None

    def step(
        self,
        x_t: torch.Tensor,
        state: object | None = None,
        cond_t: torch.Tensor | None = None,
        block_action_t: torch.Tensor | None = None,
    ):
        """Use a one-token chunk by padding internally.

        Official train_temp kernels are sequence kernels. This method preserves
        the LeWM planning API, but it is not a true O(1) recurrent inference
        kernel yet. For planning efficiency, prefer batched sequence rollout.
        """

        cond = cond_t.unsqueeze(1) if cond_t is not None else None
        block_action = block_action_t.unsqueeze(1) if block_action_t is not None else None
        y, _ = self.forward(x_t.unsqueeze(1), state, cond=cond, block_action=block_action)
        return y[:, 0], None


def rwkv7_cuda_optimizer_groups(
    module: nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    lr_decay, lr_1x, lr_2x = [], [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if "att.w0" in name:
            lr_2x.append(param)
        elif param.ndim >= 2 and ".weight" in name and weight_decay > 0:
            lr_decay.append(param)
        else:
            lr_1x.append(param)
    groups: list[dict[str, object]] = [
        {"params": lr_1x, "weight_decay": 0.0, "my_lr_scale": 1.0},
        {"params": lr_2x, "weight_decay": 0.0, "my_lr_scale": 2.0},
    ]
    if lr_decay:
        groups.append({"params": lr_decay, "weight_decay": weight_decay, "my_lr_scale": 1.0})
    return groups
