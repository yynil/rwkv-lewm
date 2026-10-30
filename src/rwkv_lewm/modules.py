import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer used by LeWM.

    Input shape is (T, B, D). The regularizer samples random unit directions and
    matches the one-dimensional projected empirical characteristic functions to
    a standard Gaussian target.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"SIGReg expects (T, B, D), got {tuple(z.shape)}")
        directions = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        x_t = (z @ directions).unsqueeze(-1) * self.t.to(dtype=z.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(dtype=z.dtype)).square()
        err = err + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(dtype=z.dtype)) * z.size(-2)
        return statistic.mean()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn: type[nn.Module] | None = nn.LayerNorm,
        act_fn: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceMLP(MLP):
    """MLP that also accepts (..., D) by flattening non-feature dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return super().forward(x)
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.size(-1))
        y = super().forward(x_flat)
        return y.reshape(*shape, y.size(-1))


class ActionEncoder(nn.Module):
    """Encode action blocks into the latent token dimension.

    The input is expected to be (B, T, action_dim). For frame-skipped world
    models action_dim is usually raw_action_dim * frameskip.
    """

    def __init__(self, input_dim: int, emb_dim: int, hidden_scale: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(hidden_scale * emb_dim, emb_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        dtype = next(self.net.parameters()).dtype
        return self.net(action.to(dtype=dtype))


class LeWMActionEncoder(nn.Module):
    """Original LeWM-style action embedder.

    This matches the Embedder used in le-wm/module.py: a 1x1 Conv1d over the
    action feature axis, followed by a SiLU MLP. The convolution is not temporal,
    but it preserves the same parameterization as the reference LeWM code.
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        smoothed_dim: int | None = None,
        mlp_scale: int = 4,
    ):
        super().__init__()
        smoothed_dim = input_dim if smoothed_dim is None else smoothed_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        squeeze_time = action.ndim == 2
        if squeeze_time:
            action = action.unsqueeze(1)
        if action.ndim != 3:
            raise ValueError(f"LeWMActionEncoder expects (B, D) or (B, T, D), got {tuple(action.shape)}")
        x = action.to(dtype=dtype).permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        x = self.embed(x)
        return x.squeeze(1) if squeeze_time else x


class AdaptiveAddActionFusion(nn.Module):
    """State-conditioned action injection for RWKV input tokens.

    The zero-initialized controller makes the module exactly match `z + action`
    at initialization, then learns a nonlinear, data-dependent action scale.
    """

    def __init__(
        self,
        embed_dim: int,
        dynamics_dim: int,
        gate_type: str = "scalar",
        hidden_scale: int = 4,
        alpha_init: float = 1.0,
        alpha_delta: float = 0.5,
        modulate_action: bool = False,
        max_scale: float = 0.05,
        max_shift: float = 0.05,
    ):
        super().__init__()
        if gate_type not in {"scalar", "channel"}:
            raise ValueError(f"Unknown adaptive action gate_type: {gate_type}")
        self.gate_type = gate_type
        self.alpha_init = alpha_init
        self.alpha_delta = alpha_delta
        self.modulate_action = modulate_action
        self.max_scale = max_scale
        self.max_shift = max_shift
        self.image = nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim)
        self.action = nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim)
        self.image_norm = nn.LayerNorm(dynamics_dim, elementwise_affine=False)
        self.action_norm = nn.LayerNorm(dynamics_dim, elementwise_affine=False)
        gate_dim = 1 if gate_type == "scalar" else dynamics_dim
        chunks = 3 if modulate_action else 1
        self.controller = nn.Sequential(
            nn.Linear(3 * dynamics_dim, hidden_scale * dynamics_dim),
            nn.SiLU(),
            nn.Linear(hidden_scale * dynamics_dim, chunks * gate_dim),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        z = self.image(emb)
        u = self.action(act_emb)
        z_norm = self.image_norm(z)
        u_norm = self.action_norm(u)
        cond = torch.cat([z_norm, u_norm, z_norm * u_norm], dim=-1)
        params = self.controller(cond)
        if self.modulate_action:
            gate, scale, shift = params.chunk(3, dim=-1)
            u = u * (1 + self.max_scale * torch.tanh(scale)) + self.max_shift * torch.tanh(shift)
        else:
            gate = params
        alpha = self.alpha_init + self.alpha_delta * torch.tanh(gate)
        return z + alpha * u


class LearnedAddActionFusion(nn.Module):
    """Learned input-side state/action fusion for RWKV dynamics.

    This keeps the successful v17 `z + action` path, but gives the model
    separate state/action coordinate systems and a learnable action strength.
    """

    def __init__(
        self,
        embed_dim: int,
        dynamics_dim: int,
        action_scale_init: float = 1.0,
        per_channel_scale: bool = False,
        normalize_streams: bool = True,
    ):
        super().__init__()
        self.normalize_streams = normalize_streams
        self.image = nn.Linear(embed_dim, dynamics_dim)
        self.action = nn.Linear(embed_dim, dynamics_dim)
        self.image_norm = nn.LayerNorm(dynamics_dim, elementwise_affine=False)
        self.action_norm = nn.LayerNorm(dynamics_dim, elementwise_affine=False)
        scale_shape = (dynamics_dim,) if per_channel_scale else ()
        self.action_scale = nn.Parameter(torch.full(scale_shape, float(action_scale_init)))
        self.reset_parameters(embed_dim, dynamics_dim)

    def reset_parameters(self, embed_dim: int, dynamics_dim: int) -> None:
        if embed_dim == dynamics_dim:
            nn.init.eye_(self.image.weight)
            nn.init.eye_(self.action.weight)
            nn.init.zeros_(self.image.bias)
            nn.init.zeros_(self.action.bias)
        else:
            nn.init.xavier_uniform_(self.image.weight)
            nn.init.xavier_uniform_(self.action.weight)
            nn.init.zeros_(self.image.bias)
            nn.init.zeros_(self.action.bias)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        z = self.image(emb)
        u = self.action(act_emb)
        if self.normalize_streams:
            z = self.image_norm(z)
            u = self.action_norm(u)
        return z + self.action_scale.to(dtype=u.dtype) * u


class ScaledAddActionFusion(nn.Module):
    """Minimal v17-style action injection with a learnable action strength.

    When `embed_dim == dynamics_dim`, this is exactly `z + alpha * action` at
    initialization. This keeps the successful v17 coordinate system intact and
    only lets training adapt the action magnitude.
    """

    def __init__(
        self,
        embed_dim: int,
        dynamics_dim: int,
        action_scale_init: float = 1.0,
        per_channel_scale: bool = False,
    ):
        super().__init__()
        self.image = nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim)
        self.action = nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim)
        scale_shape = (dynamics_dim,) if per_channel_scale else ()
        self.action_scale = nn.Parameter(torch.full(scale_shape, float(action_scale_init)))

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        z = self.image(emb)
        u = self.action(act_emb)
        return z + self.action_scale.to(dtype=u.dtype) * u


class ActionHistoryAddFusion(nn.Module):
    """Add current action plus a causal history-action state to latent tokens.

    This keeps the stable v17 input path `z_t + u_t` and adds a small learned
    summary of previous action-block embeddings. The summary is causal within
    the provided context window, so it can be used for both training and MPC
    rollout without peeking at future actions.
    """

    def __init__(
        self,
        embed_dim: int,
        dynamics_dim: int,
        history_size: int = 3,
        include_current: bool = True,
        history_scale_init: float = 0.1,
        per_channel_scale: bool = False,
        hidden_scale: int = 4,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        self.history_size = history_size
        self.include_current = include_current
        self.image = nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim)
        self.action = nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim)
        self.history = nn.Sequential(
            nn.Linear(history_size * dynamics_dim, hidden_scale * dynamics_dim),
            nn.SiLU(),
            nn.Linear(hidden_scale * dynamics_dim, dynamics_dim),
        )
        if zero_init_output:
            nn.init.zeros_(self.history[-1].weight)
            nn.init.zeros_(self.history[-1].bias)
        scale_shape = (dynamics_dim,) if per_channel_scale else ()
        self.history_scale = nn.Parameter(torch.full(scale_shape, float(history_scale_init)))

    @staticmethod
    def _delay(x: torch.Tensor, steps: int) -> torch.Tensor:
        if steps == 0:
            return x
        if steps >= x.size(1):
            return torch.zeros_like(x)
        pad = x.new_zeros(x.size(0), steps, x.size(-1))
        return torch.cat([pad, x[:, :-steps]], dim=1)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        z = self.image(emb)
        u = self.action(act_emb)
        if self.include_current:
            delays = range(self.history_size - 1, -1, -1)
        else:
            delays = range(self.history_size, 0, -1)
        hist = torch.cat([self._delay(u, delay) for delay in delays], dim=-1)
        h = self.history(hist)
        return z + u + self.history_scale.to(dtype=h.dtype) * h


class AdaLNZeroModulation(nn.Module):
    """Zero-initialized adaptive LayerNorm modulation used by LeWM blocks."""

    def __init__(self, cond_dim: int, hidden_dim: int, num_chunks: int = 6):
        super().__init__()
        self.num_chunks = num_chunks
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_chunks * hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.net(cond).chunk(self.num_chunks, dim=-1)

    def zero_init_max_abs(self) -> float:
        layer = self.net[-1]
        return max(layer.weight.detach().abs().max().item(), layer.bias.detach().abs().max().item())


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ActionTransitionHead(nn.Module):
    """Action-conditioned transition head applied after state-only dynamics.

    This keeps the trajectory backbone free of action modulation. The action
    embedding only controls a final residual adapter before the prediction
    projection, so action cannot directly perturb RWKV's recurrent memory.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        hidden_scale: int = 4,
        residual_scale: float = 1.0,
        bounded: bool = False,
        max_shift: float = 0.05,
        max_scale: float = 0.05,
        gate_tanh: bool = False,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.bounded = bounded
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.gate_tanh = gate_tanh
        self.cond_proj = nn.Linear(action_dim, hidden_dim) if action_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.modulation = AdaLNZeroModulation(hidden_dim, hidden_dim, num_chunks=3)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_scale * hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_scale * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        residual_dtype = x.dtype
        x = x.to(dtype=dtype)
        cond = self.cond_proj(action.to(dtype=dtype))
        shift, scale, gate = self.modulation(cond)
        x_norm = self.norm(x)
        if self.bounded:
            x_mod = x_norm * (1 + self.max_scale * torch.tanh(scale)) + self.max_shift * torch.tanh(shift)
        else:
            x_mod = modulate(x_norm, shift, scale)
        if self.gate_tanh:
            gate = torch.tanh(gate)
        out = x + self.residual_scale * gate * self.adapter(x_mod)
        return out.to(dtype=residual_dtype)
