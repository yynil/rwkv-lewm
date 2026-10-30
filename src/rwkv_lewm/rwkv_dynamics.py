from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RWKVDynamicsConfig:
    input_dim: int = 384
    hidden_dim: int = 384
    output_dim: int = 192
    num_layers: int = 6
    dropout: float = 0.1
    ffn_mult: int = 4
    residual_prediction: bool = True


class RWKVStateMix(nn.Module):
    """A compact PyTorch recurrent state-mixing block.

    This is intentionally shaped like an RWKV trajectory backend: it processes a
    stream with fixed-size recurrent state and no KV cache. It is not a verbatim
    RWKV-v7 CUDA kernel. The public API is kept small so the official RWKV-v7
    backend can replace this block without touching the LeWM wrapper.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Parameter(torch.linspace(-4.0, -1.0, dim).view(1, dim))
        self.mix = nn.Parameter(torch.zeros(1, dim))
        self.output = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.output.weight)

    def step(self, x_t: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros_like(x_t)
        x = self.ln(x_t)
        mixed = x + self.mix * (state - x)
        r = torch.sigmoid(self.receptance(mixed))
        k = self.key(mixed)
        v = self.value(mixed)
        decay = torch.exp(-F.softplus(self.decay)).to(dtype=x.dtype)
        new_state = decay * state + (1.0 - decay) * torch.tanh(k)
        y = r * (v + new_state)
        return self.dropout(self.output(y)), new_state

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for t in range(x.size(1)):
            y_t, state = self.step(x[:, t], state)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1), state


class ChannelMix(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, mult * dim, bias=False)
        self.value = nn.Linear(mult * dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.key(self.ln(x))).square()
        return self.dropout(self.value(y))


class RWKVBlock(nn.Module):
    def __init__(self, dim: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.time_mix = RWKVStateMix(dim, dropout=dropout)
        self.channel_mix = ChannelMix(dim, mult=ffn_mult, dropout=dropout)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, state = self.time_mix(x, state)
        x = x + y
        x = x + self.channel_mix(x)
        return x, state

    def step(self, x_t: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        y_t, state = self.time_mix.step(x_t, state)
        x_t = x_t + y_t
        x_t = x_t + self.channel_mix(x_t.unsqueeze(1)).squeeze(1)
        return x_t, state


class RWKVDynamics(nn.Module):
    """Trajectory dynamics model with fixed-size recurrent state."""

    def __init__(self, cfg: RWKVDynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [RWKVBlock(cfg.hidden_dim, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            torch.zeros(batch_size, self.cfg.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.cfg.num_layers)
        ]

    def forward(
        self, x: torch.Tensor, state: list[torch.Tensor] | None = None, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del cond
        if state is None:
            state = self.initial_state(x.size(0), x.device, x.dtype)
        x = self.input_proj(x)
        next_state = []
        for block, layer_state in zip(self.blocks, state):
            x, layer_state = block(x, layer_state)
            next_state.append(layer_state)
        return self.head(self.norm(x)), next_state

    def step(
        self, x_t: torch.Tensor, state: list[torch.Tensor] | None = None, cond_t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del cond_t
        if state is None:
            state = self.initial_state(x_t.size(0), x_t.device, x_t.dtype)
        x_t = self.input_proj(x_t)
        next_state = []
        for block, layer_state in zip(self.blocks, state):
            x_t, layer_state = block.step(x_t, layer_state)
            next_state.append(layer_state)
        return self.head(self.norm(x_t)), next_state


def rwkv_optimizer_groups(
    module: nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    """RWKV-style optimizer grouping: decay only large matrix weights."""

    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "norm" not in name and "ln" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
