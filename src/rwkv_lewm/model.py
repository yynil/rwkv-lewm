from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .modules import (
    ActionEncoder,
    ActionTransitionHead,
    ActionHistoryAddFusion,
    AdaptiveAddActionFusion,
    LeWMActionEncoder,
    LearnedAddActionFusion,
    MLP,
    SequenceMLP,
    ScaledAddActionFusion,
)
from .rwkv_dynamics import RWKVDynamics, RWKVDynamicsConfig
from .rwkv_v7_cuda_dynamics import RWKV7CudaConfig, RWKV7CudaDynamics


def _norm_from_name(name: str | None) -> type[nn.Module] | None:
    if name is None:
        return nn.BatchNorm1d
    normalized = str(name).lower()
    if normalized in {"batch", "batchnorm", "batchnorm1d", "bn"}:
        return nn.BatchNorm1d
    if normalized in {"layer", "layernorm", "ln"}:
        return nn.LayerNorm
    if normalized in {"none", "identity", "null"}:
        return None
    raise ValueError(f"Unknown norm type: {name}")


class RWKVLeWorldModel(nn.Module):
    """LeWM-style latent world model with recurrent RWKV trajectory dynamics."""

    def __init__(
        self,
        encoder: nn.Module,
        action_encoder: ActionEncoder,
        dynamics: RWKVDynamics,
        projector: nn.Module | None = None,
        fuse: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        transition_head: nn.Module | None = None,
        residual_prediction: bool = True,
        action_modulation: str = "none",
        action_modulation_scope: str = "full",
        action_modulate_ffn: bool = True,
        action_residual_gate: str = "adaln",
        action_residual_gate_scale: float = 0.1,
        kv_max_scale: float = 0.05,
        kv_max_shift: float = 0.05,
        action_adapter_residual_scale: float = 0.1,
        action_adapter_max_scale: float = 0.05,
        action_adapter_max_shift: float = 0.05,
        action_adapter_position: str = "post",
        token_fusion: str = "concat_action",
        use_token_type_embeddings: bool = True,
        use_sequence_pos_embedding: bool = False,
        max_position_embeddings: int = 512,
        normalize_embeddings: bool = True,
        rollout_history_size: int = 3,
        modulate_v_first: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.dynamics = dynamics
        self.projector = projector or nn.Identity()
        self.fuse = fuse or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.transition_head = transition_head
        self.residual_prediction = residual_prediction
        self.action_modulation = action_modulation
        self.action_modulation_scope = action_modulation_scope
        self.action_modulate_ffn = action_modulate_ffn
        self.action_residual_gate = action_residual_gate
        self.action_residual_gate_scale = action_residual_gate_scale
        self.kv_max_scale = kv_max_scale
        self.kv_max_shift = kv_max_shift
        self.action_adapter_residual_scale = action_adapter_residual_scale
        self.action_adapter_max_scale = action_adapter_max_scale
        self.action_adapter_max_shift = action_adapter_max_shift
        self.action_adapter_position = action_adapter_position
        self.token_fusion = token_fusion
        self.use_token_type_embeddings = use_token_type_embeddings
        self.use_sequence_pos_embedding = use_sequence_pos_embedding
        self.max_position_embeddings = max_position_embeddings
        self.normalize_embeddings = normalize_embeddings
        self.rollout_history_size = rollout_history_size
        self.modulate_v_first = modulate_v_first
        dynamics_dim = getattr(dynamics.cfg, "input_dim", None)
        if self.token_fusion == "interleave_action" and self.use_token_type_embeddings:
            if dynamics_dim is None:
                raise ValueError("interleave_action requires dynamics.cfg.input_dim")
            self.token_type_embeddings = nn.Parameter(torch.empty(2, dynamics_dim))
            nn.init.normal_(self.token_type_embeddings, mean=0.0, std=0.02)
        else:
            self.token_type_embeddings = None
        if self.use_sequence_pos_embedding:
            if dynamics_dim is None:
                raise ValueError("sequence position embedding requires dynamics.cfg.input_dim")
            self.sequence_pos_embedding = nn.Parameter(torch.randn(1, max_position_embeddings, dynamics_dim))
        else:
            self.sequence_pos_embedding = None

    @classmethod
    def from_components(
        cls,
        encoder: nn.Module,
        encoder_dim: int,
        action_dim: int,
        embed_dim: int = 192,
        dynamics_dim: int = 384,
        dynamics_layers: int = 6,
        rwkv_ffn_dim: int | None = None,
        dropout: float = 0.1,
        backend: str = "rwkv7_cuda",
        residual_prediction: bool = True,
        action_encoder_type: str = "mlp",
        prediction_head_type: str = "direct",
        projector_norm: str | None = "batchnorm",
        pred_proj_norm: str | None = "batchnorm",
        action_modulation: str = "none",
        action_modulation_scope: str = "full",
        action_modulate_ffn: bool = True,
        action_residual_gate: str = "adaln",
        action_residual_gate_scale: float = 0.1,
        kv_max_scale: float = 0.05,
        kv_max_shift: float = 0.05,
        action_adapter_residual_scale: float = 0.1,
        action_adapter_max_scale: float = 0.05,
        action_adapter_max_shift: float = 0.05,
        action_adapter_position: str = "post",
        token_fusion: str = "concat_action",
        use_token_type_embeddings: bool = True,
        use_sequence_pos_embedding: bool = False,
        max_position_embeddings: int = 512,
        normalize_embeddings: bool = True,
        rollout_history_size: int = 3,
        modulate_v_first: bool = False,
        rwkv_output_init_scale: float = 0.0,
        rwkv_output_head_type: str = "linear",
        action_transition_type: str = "none",
        action_transition_hidden_scale: int = 4,
        action_transition_residual_scale: float = 1.0,
        action_transition_bounded: bool = False,
        action_transition_max_scale: float = 0.05,
        action_transition_max_shift: float = 0.05,
        action_transition_gate_tanh: bool = False,
        adaptive_action_hidden_scale: int = 4,
        adaptive_action_alpha_init: float = 1.0,
        adaptive_action_alpha_delta: float = 0.5,
        adaptive_action_modulate: bool = False,
        adaptive_action_max_scale: float = 0.05,
        adaptive_action_max_shift: float = 0.05,
        learned_action_scale_init: float = 1.0,
        learned_action_per_channel_scale: bool = False,
        learned_action_normalize_streams: bool = True,
        scaled_action_scale_init: float = 1.0,
        scaled_action_per_channel_scale: bool = False,
        action_history_size: int = 3,
        action_history_include_current: bool = True,
        action_history_scale_init: float = 0.1,
        action_history_per_channel_scale: bool = False,
        action_history_hidden_scale: int = 4,
        action_history_zero_init_output: bool = True,
        block_action_transition_type: str = "none",
        block_action_transition_hidden_scale: int = 4,
        block_action_transition_residual_scale: float = 0.1,
        block_action_transition_bounded: bool = True,
        block_action_transition_max_scale: float = 0.05,
        block_action_transition_max_shift: float = 0.05,
        block_action_transition_gate_tanh: bool = True,
        block_action_transition_target: str = "last",
    ) -> "RWKVLeWorldModel":
        if action_encoder_type == "mlp":
            action_encoder = ActionEncoder(action_dim, embed_dim)
        elif action_encoder_type == "lewm":
            action_encoder = LeWMActionEncoder(action_dim, embed_dim)
        else:
            raise ValueError(f"Unknown action_encoder_type: {action_encoder_type}")
        projector = MLP(
            encoder_dim,
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=_norm_from_name(projector_norm),
        )
        if token_fusion == "concat_action":
            fuse_input_dim = 2 * embed_dim
            fuse = nn.Sequential(nn.Linear(fuse_input_dim, dynamics_dim), nn.LayerNorm(dynamics_dim), nn.GELU())
        elif token_fusion == "latent_only":
            fuse_input_dim = embed_dim
            fuse = nn.Sequential(nn.Linear(fuse_input_dim, dynamics_dim), nn.LayerNorm(dynamics_dim), nn.GELU())
        elif token_fusion == "latent_linear":
            fuse_input_dim = embed_dim
            fuse = nn.Linear(fuse_input_dim, dynamics_dim)
        elif token_fusion == "latent_identity":
            if embed_dim != dynamics_dim:
                raise ValueError("latent_identity token_fusion requires embed_dim == dynamics_dim")
            fuse = nn.Identity()
        elif token_fusion == "add_action":
            fuse = nn.ModuleDict(
                {
                    "image": nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim),
                    "action": nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim),
                }
            )
        elif token_fusion == "learned_add_action":
            fuse = LearnedAddActionFusion(
                embed_dim=embed_dim,
                dynamics_dim=dynamics_dim,
                action_scale_init=learned_action_scale_init,
                per_channel_scale=learned_action_per_channel_scale,
                normalize_streams=learned_action_normalize_streams,
            )
        elif token_fusion == "scaled_add_action":
            fuse = ScaledAddActionFusion(
                embed_dim=embed_dim,
                dynamics_dim=dynamics_dim,
                action_scale_init=scaled_action_scale_init,
                per_channel_scale=scaled_action_per_channel_scale,
            )
        elif token_fusion == "action_history_add":
            fuse = ActionHistoryAddFusion(
                embed_dim=embed_dim,
                dynamics_dim=dynamics_dim,
                history_size=action_history_size,
                include_current=action_history_include_current,
                history_scale_init=action_history_scale_init,
                per_channel_scale=action_history_per_channel_scale,
                hidden_scale=action_history_hidden_scale,
                zero_init_output=action_history_zero_init_output,
            )
        elif token_fusion in {"adaptive_add_action_scalar", "adaptive_add_action_channel"}:
            gate_type = "scalar" if token_fusion.endswith("_scalar") else "channel"
            fuse = AdaptiveAddActionFusion(
                embed_dim=embed_dim,
                dynamics_dim=dynamics_dim,
                gate_type=gate_type,
                hidden_scale=adaptive_action_hidden_scale,
                alpha_init=adaptive_action_alpha_init,
                alpha_delta=adaptive_action_alpha_delta,
                modulate_action=adaptive_action_modulate,
                max_scale=adaptive_action_max_scale,
                max_shift=adaptive_action_max_shift,
            )
        elif token_fusion == "interleave_action":
            fuse = nn.ModuleDict(
                {
                    "image": nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim),
                    "action": nn.Identity() if embed_dim == dynamics_dim else nn.Linear(embed_dim, dynamics_dim),
                }
            )
        else:
            raise ValueError(f"Unknown token_fusion: {token_fusion}")
        if prediction_head_type == "direct":
            dynamics_output_dim = embed_dim
            pred_proj = nn.Identity()
            zero_head = True
        elif prediction_head_type == "lewm_pred_proj":
            dynamics_output_dim = dynamics_dim
            pred_proj = SequenceMLP(
                dynamics_dim,
                hidden_dim=2048,
                output_dim=embed_dim,
                norm_fn=_norm_from_name(pred_proj_norm),
            )
            zero_head = False
        else:
            raise ValueError(f"Unknown prediction_head_type: {prediction_head_type}")
        if action_transition_type == "none":
            transition_head = None
        elif action_transition_type == "adaln":
            transition_head = ActionTransitionHead(
                hidden_dim=dynamics_output_dim,
                action_dim=embed_dim,
                hidden_scale=action_transition_hidden_scale,
                residual_scale=action_transition_residual_scale,
                bounded=action_transition_bounded,
                max_scale=action_transition_max_scale,
                max_shift=action_transition_max_shift,
                gate_tanh=action_transition_gate_tanh,
            )
        else:
            raise ValueError(f"Unknown action_transition_type: {action_transition_type}")

        if backend == "rwkv7_cuda":
            dynamics = RWKV7CudaDynamics(
                RWKV7CudaConfig(
                    input_dim=dynamics_dim,
                    hidden_dim=dynamics_dim,
                    output_dim=dynamics_output_dim,
                    num_layers=dynamics_layers,
                    head_size=64,
                    ffn_dim=rwkv_ffn_dim,
                    chunk_len=16,
                    action_cond_dim=embed_dim,
                    action_modulation=action_modulation,
                    action_modulation_scope=action_modulation_scope,
                    action_modulate_ffn=action_modulate_ffn,
                    action_residual_gate=action_residual_gate,
                    action_residual_gate_scale=action_residual_gate_scale,
                    kv_max_scale=kv_max_scale,
                    kv_max_shift=kv_max_shift,
                    action_adapter_residual_scale=action_adapter_residual_scale,
                    action_adapter_max_scale=action_adapter_max_scale,
                    action_adapter_max_shift=action_adapter_max_shift,
                    action_adapter_position=action_adapter_position,
                    project_action_condition=True,
                    modulate_v_first=modulate_v_first,
                    rwkv_output_init_scale=rwkv_output_init_scale,
                    zero_head=zero_head,
                    output_head_type=rwkv_output_head_type,
                    block_action_transition_type=block_action_transition_type,
                    block_action_transition_hidden_scale=block_action_transition_hidden_scale,
                    block_action_transition_residual_scale=block_action_transition_residual_scale,
                    block_action_transition_bounded=block_action_transition_bounded,
                    block_action_transition_max_scale=block_action_transition_max_scale,
                    block_action_transition_max_shift=block_action_transition_max_shift,
                    block_action_transition_gate_tanh=block_action_transition_gate_tanh,
                    block_action_transition_target=block_action_transition_target,
                )
            )
        elif backend == "fallback":
            dynamics = RWKVDynamics(
                RWKVDynamicsConfig(
                    input_dim=dynamics_dim,
                    hidden_dim=dynamics_dim,
                    output_dim=dynamics_output_dim,
                    num_layers=dynamics_layers,
                    dropout=dropout,
                    residual_prediction=True,
                )
            )
        else:
            raise ValueError(f"Unknown dynamics backend: {backend}")
        return cls(
            encoder=encoder,
            action_encoder=action_encoder,
            dynamics=dynamics,
            projector=projector,
            fuse=fuse,
            pred_proj=pred_proj,
            transition_head=transition_head,
            residual_prediction=residual_prediction,
            action_modulation=action_modulation,
            action_modulation_scope=action_modulation_scope,
            action_modulate_ffn=action_modulate_ffn,
            action_residual_gate=action_residual_gate,
            action_residual_gate_scale=action_residual_gate_scale,
            kv_max_scale=kv_max_scale,
            kv_max_shift=kv_max_shift,
            action_adapter_residual_scale=action_adapter_residual_scale,
            action_adapter_max_scale=action_adapter_max_scale,
            action_adapter_max_shift=action_adapter_max_shift,
            action_adapter_position=action_adapter_position,
            token_fusion=token_fusion,
            use_token_type_embeddings=use_token_type_embeddings,
            use_sequence_pos_embedding=use_sequence_pos_embedding,
            max_position_embeddings=max_position_embeddings,
            normalize_embeddings=normalize_embeddings,
            rollout_history_size=rollout_history_size,
            modulate_v_first=modulate_v_first,
        )

    def _encoder_forward(self, pixels: torch.Tensor) -> torch.Tensor:
        try:
            output = self.encoder(pixels, interpolate_pos_encoding=True)
        except TypeError:
            output = self.encoder(pixels)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 3:
            return output[:, 0]
        if output.ndim == 2:
            return output
        raise ValueError(f"Unsupported encoder output shape: {tuple(output.shape)}")

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dtype = next(self.encoder.parameters()).dtype
        pixels = info["pixels"].to(dtype=dtype)
        b = pixels.size(0)
        flat_pixels = rearrange(pixels, "b t ... -> (b t) ...")
        cls = self._encoder_forward(flat_pixels)
        emb = self.projector(cls)
        if self.normalize_embeddings:
            emb = F.layer_norm(emb.float(), (emb.size(-1),)).to(dtype=emb.dtype)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def _uses_current_action_transition(self) -> bool:
        return self.action_modulation == "none" and (
            self.transition_head is not None
            or getattr(self.dynamics, "block_action_transitions", None) is not None
        )

    def make_tokens(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        if self.token_fusion == "concat_action":
            return self.fuse(torch.cat([emb, act_emb], dim=-1))
        if self.token_fusion == "latent_only":
            return self.fuse(emb)
        if self.token_fusion == "latent_linear":
            return self.fuse(emb)
        if self.token_fusion == "latent_identity":
            return self.fuse(emb)
        if self.token_fusion == "add_action":
            return self.fuse["image"](emb) + self.fuse["action"](act_emb)
        if self.token_fusion == "learned_add_action":
            return self.fuse(emb, act_emb)
        if self.token_fusion == "scaled_add_action":
            return self.fuse(emb, act_emb)
        if self.token_fusion == "action_history_add":
            return self.fuse(emb, act_emb)
        if self.token_fusion in {"adaptive_add_action_scalar", "adaptive_add_action_channel"}:
            return self.fuse(emb, act_emb)
        if self.token_fusion == "interleave_action":
            return self.make_interleaved_tokens(emb, act_emb)
        raise ValueError(f"Unknown token_fusion: {self.token_fusion}")

    def add_sequence_pos_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.sequence_pos_embedding is None:
            return tokens
        if tokens.size(1) > self.sequence_pos_embedding.size(1):
            raise ValueError(
                f"sequence length {tokens.size(1)} exceeds max_position_embeddings="
                f"{self.sequence_pos_embedding.size(1)}"
            )
        pos = self.sequence_pos_embedding[:, : tokens.size(1)].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + pos

    def make_interleaved_tokens(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Build z, a, z, a... tokens for RWKV trajectory dynamics."""

        if emb.shape[:2] != act_emb.shape[:2]:
            raise ValueError(
                "interleave_action expects image and action sequences with matching "
                f"(B,T), got emb={tuple(emb.shape)} act_emb={tuple(act_emb.shape)}"
            )
        image_tokens = self.fuse["image"](emb)
        action_tokens = self.fuse["action"](act_emb)
        if self.token_type_embeddings is not None:
            type_emb = self.token_type_embeddings.to(device=image_tokens.device, dtype=image_tokens.dtype)
            image_tokens = image_tokens + type_emb[0]
            action_tokens = action_tokens + type_emb[1]
        tokens = torch.stack((image_tokens, action_tokens), dim=2)
        return rearrange(tokens, "b t m d -> b (t m) d")

    def predict(
        self,
        emb: torch.Tensor,
        act_emb: torch.Tensor,
        state: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.token_fusion == "interleave_action":
            return self.predict_interleaved(emb, act_emb, state)
        tokens = self.make_tokens(emb, act_emb)
        tokens = self.add_sequence_pos_embedding(tokens)
        cond = act_emb if self.action_modulation != "none" else None
        dynamics_out, state = self.dynamics(tokens, state, cond=cond)
        if self.transition_head is not None:
            dynamics_out = self.transition_head(dynamics_out, act_emb)
        delta_or_pred = self.pred_proj(dynamics_out)
        pred = emb + delta_or_pred if self.residual_prediction else delta_or_pred
        return pred, state

    def predict_next_from_context(
        self,
        emb: torch.Tensor,
        current_act_emb: torch.Tensor,
        state: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Predict the next latent from observation history and current action only."""

        if self.token_fusion == "interleave_action":
            raise ValueError("predict_next_from_context is not defined for interleaved action tokens")
        if self.action_modulation != "none":
            raise ValueError("current-action-only prediction requires action_modulation='none'")
        if current_act_emb.ndim == 2:
            current_act_emb = current_act_emb.unsqueeze(1)
        if current_act_emb.size(1) != 1:
            raise ValueError(
                "predict_next_from_context expects one current action embedding, "
                f"got shape {tuple(current_act_emb.shape)}"
            )
        dummy_act = current_act_emb.expand(-1, emb.size(1), -1)
        tokens = self.make_tokens(emb, dummy_act)
        tokens = self.add_sequence_pos_embedding(tokens)
        block_action = (
            current_act_emb
            if getattr(self.dynamics, "block_action_transitions", None) is not None
            else None
        )
        if block_action is None:
            dynamics_out, state = self.dynamics(tokens, state, cond=None)
        else:
            dynamics_out, state = self.dynamics(tokens, state, cond=None, block_action=block_action)
        next_hidden = dynamics_out[:, -1:]
        if self.transition_head is not None:
            next_hidden = self.transition_head(next_hidden, current_act_emb)
        delta_or_pred = self.pred_proj(next_hidden)
        base = emb[:, -1:]
        pred = base + delta_or_pred if self.residual_prediction else delta_or_pred
        return pred, state

    def predict_interleaved(
        self,
        emb: torch.Tensor,
        act_emb: torch.Tensor,
        state: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = self.make_interleaved_tokens(emb, act_emb)
        dynamics_out, state = self.dynamics(tokens, state, cond=None)
        action_outputs = dynamics_out[:, 1::2]
        if self.transition_head is not None:
            action_outputs = self.transition_head(action_outputs, act_emb)
        delta_or_pred = self.pred_proj(action_outputs)
        pred = emb + delta_or_pred if self.residual_prediction else delta_or_pred
        return pred, state

    def predict_step(
        self,
        emb_t: torch.Tensor,
        act_emb_t: torch.Tensor,
        state: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.token_fusion == "interleave_action":
            pred, state = self.predict(emb_t.unsqueeze(1), act_emb_t.unsqueeze(1), state)
            return pred[:, 0], state
        token_t = self.make_tokens(emb_t.unsqueeze(1), act_emb_t.unsqueeze(1)).squeeze(1)
        if self.sequence_pos_embedding is not None:
            token_t = token_t + self.sequence_pos_embedding[:, 0].to(device=token_t.device, dtype=token_t.dtype)
        cond_t = act_emb_t if self.action_modulation != "none" else None
        block_action_t = (
            act_emb_t
            if self.action_modulation == "none"
            and getattr(self.dynamics, "block_action_transitions", None) is not None
            else None
        )
        if block_action_t is None:
            dynamics_out, state = self.dynamics.step(token_t, state, cond_t=cond_t)
        else:
            dynamics_out, state = self.dynamics.step(token_t, state, cond_t=cond_t, block_action_t=block_action_t)
        if self.transition_head is not None:
            dynamics_out = self.transition_head(dynamics_out, act_emb_t)
        delta_or_pred = self.pred_proj(dynamics_out)
        pred = emb_t + delta_or_pred if self.residual_prediction else delta_or_pred
        return pred, state

    def rollout(
        self,
        info: dict[str, torch.Tensor],
        action_sequence: torch.Tensor,
        warmup_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Roll out candidate action sequences in latent space.

        info["pixels"] is (B, S, H, C, W, H) or (B, S, H, ...).
        action_sequence is (B, S, T, action_dim).
        """

        del warmup_steps
        b, s, horizon = action_sequence.shape[:3]
        history = info["pixels"].size(2)
        if horizon <= history:
            raise ValueError(f"action_sequence horizon={horizon} must be > encoded history={history}")

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        history_action = info.get("history_action_blocks")
        history_action_full = None
        if torch.is_tensor(history_action):
            if history_action.ndim == 4:
                if history_action.shape[:3] != (b, s, history):
                    raise ValueError(
                        "history_action_blocks must have shape (B, S, H, action_dim), "
                        f"got {tuple(history_action.shape)} for B={b}, S={s}, H={history}"
                    )
                history_action_full = history_action
                history_action_init = history_action[:, 0]
            elif history_action.ndim == 3:
                if history_action.shape[:2] != (b, history):
                    raise ValueError(
                        "history_action_blocks must have shape (B, H, action_dim), "
                        f"got {tuple(history_action.shape)} for B={b}, H={history}"
                    )
                history_action_full = history_action.unsqueeze(1).expand(b, s, -1, -1)
                history_action_init = history_action
            else:
                raise ValueError(
                    "history_action_blocks must have shape (B, H, action_dim) "
                    f"or (B, S, H, action_dim), got {tuple(history_action.shape)}"
                )
            if history_action_init.size(-1) != action_sequence.size(-1):
                raise ValueError(
                    "history_action_blocks last dim must match candidate action block dim, "
                    f"got {history_action_init.size(-1)} vs {action_sequence.size(-1)}"
                )
            init["action"] = history_action_init
        else:
            init["action"] = action_sequence[:, 0, :history]
        init = self.encode(init)

        emb = init["emb"].unsqueeze(1).expand(b, s, -1, -1)
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        if history_action_full is not None:
            act = history_action_full
        else:
            act = action_sequence[:, :, :history]
        act = rearrange(act, "b s ... -> (b s) ...").clone()
        act_future = rearrange(action_sequence[:, :, history:], "b s ... -> (b s) ...")

        # Match the reference LeWM MPC rollout: repeatedly re-run the dynamics on
        # the most recent context window instead of relying on a one-token state.
        # The vendored RWKV-v7 adapter currently exposes sequence kernels only;
        # using predict_step() would drop the trained 3-step context during MPC.
        context = self.rollout_history_size
        n_steps = horizon - history
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            if self._uses_current_action_transition():
                pred_emb, _ = self.predict_next_from_context(emb[:, -context:], act_emb[:, -1:])
            else:
                pred_emb, _ = self.predict(emb[:, -context:], act_emb[:, -context:])
            emb = torch.cat([emb, pred_emb[:, -1:]], dim=1)
            act = torch.cat([act, act_future[:, t : t + 1]], dim=1)

        act_emb = self.action_encoder(act)
        if self._uses_current_action_transition():
            pred_emb, _ = self.predict_next_from_context(emb[:, -context:], act_emb[:, -1:])
        else:
            pred_emb, _ = self.predict(emb[:, -context:], act_emb[:, -context:])
        emb = torch.cat([emb, pred_emb[:, -1:]], dim=1)

        pred_rollout = emb
        info["predicted_emb"] = rearrange(pred_rollout, "(b s) t d -> b s t d", b=b, s=s)
        return info

    def criterion(self, info: dict[str, torch.Tensor]) -> torch.Tensor:
        pred_emb = info["predicted_emb"]
        goal_emb = info["goal_emb"]
        if goal_emb.ndim == 3:
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb[..., -1:, :])
        return F.mse_loss(pred_emb[..., -1:, :], goal_emb.detach(), reduction="none").sum(
            dim=tuple(range(2, pred_emb.ndim))
        )

    def get_cost(self, info: dict[str, torch.Tensor], action_candidates: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info.items()}
        action_candidates = action_candidates.to(device)

        goal = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for key in list(goal.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        goal = self.encode(goal)
        info["goal_emb"] = goal["emb"].unsqueeze(1)
        info = self.rollout(info, action_candidates)
        return self.criterion(info)


class DummyVisionEncoder(nn.Module):
    """Small encoder for tests; replace with ViT for real training."""

    def __init__(self, image_shape: tuple[int, int, int] = (3, 64, 64), hidden_dim: int = 192):
        super().__init__()
        c, h, w = image_shape
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
        )

    def forward(self, pixels: torch.Tensor, interpolate_pos_encoding: bool = True):
        cls = self.net(pixels)
        return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))
