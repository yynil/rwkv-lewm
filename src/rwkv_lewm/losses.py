import torch

from .modules import SIGReg


def _sliding_teacher_forced_prediction(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Predict every next embedding from the recent true latent/action context."""

    preds = []
    for t in range(emb.size(1) - 1):
        start = max(0, t + 1 - history_size)
        pred_emb, _ = model.predict(emb[:, start : t + 1], act_emb[:, start : t + 1])
        preds.append(pred_emb[:, -1])
    return torch.stack(preds, dim=1)


def _autoregressive_prediction(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    start_size: int,
    horizon: int | None,
) -> torch.Tensor:
    """Match MPC rollout: start from true latents, then feed predictions back."""

    if start_size < 1:
        raise ValueError(f"start_size must be >= 1, got {start_size}")
    if start_size >= emb.size(1):
        raise ValueError(f"start_size={start_size} must be smaller than sequence length {emb.size(1)}")

    max_horizon = emb.size(1) - start_size
    horizon = max_horizon if horizon is None else min(horizon, max_horizon)
    if horizon < 1:
        raise ValueError(f"rollout horizon must be >= 1, got {horizon}")

    pred_rollout = emb[:, :start_size]
    first_action = start_size - 1
    for t in range(horizon):
        step = first_action + t
        ctx_start = max(0, pred_rollout.size(1) - history_size)
        act_start = max(0, step + 1 - history_size)
        pred_emb, _ = model.predict(pred_rollout[:, ctx_start:], act_emb[:, act_start : step + 1])
        pred_rollout = torch.cat([pred_rollout, pred_emb[:, -1:]], dim=1)

    return pred_rollout[:, start_size:]


def rwkv_lewm_loss(
    model,
    batch: dict[str, torch.Tensor],
    sigreg: SIGReg,
    sigreg_weight: float,
    mode: str = "next_step",
    history_size: int = 3,
    num_preds: int = 1,
    rollout_start_size: int = 1,
    rollout_horizon: int | None = None,
    rollout_weight: float = 1.0,
    teacher_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    rollout_pred = None
    rollout_target = None
    teacher_loss = None
    rollout_loss = None
    delta_base = None

    if mode == "next_step":
        pred_emb, _ = model.predict(emb[:, :-1], act_emb[:, :-1])
        target = emb[:, 1:]
        delta_base = emb[:, :-1]
    elif mode == "lewm_context":
        if emb.size(1) != history_size + num_preds:
            raise ValueError(
                "lewm_context loss expects sequence length "
                f"history_size + num_preds = {history_size + num_preds}, "
                f"got {emb.size(1)}"
            )
        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        pred_emb, _ = model.predict(ctx_emb, ctx_act)
        target = emb[:, num_preds:]
        delta_base = ctx_emb
    elif mode == "current_action_context":
        if num_preds != 1:
            raise ValueError("current_action_context currently expects num_preds=1")
        if emb.size(1) != history_size + 1:
            raise ValueError(
                "current_action_context loss expects sequence length "
                f"history_size + 1 = {history_size + 1}, got {emb.size(1)}"
            )
        ctx_emb = emb[:, :history_size]
        current_act = act_emb[:, history_size - 1 : history_size]
        pred_emb, _ = model.predict_next_from_context(ctx_emb, current_act)
        target = emb[:, history_size : history_size + 1]
        delta_base = ctx_emb[:, -1:]
    elif mode == "lewm_context_rollout":
        if emb.size(1) < 2:
            raise ValueError("lewm_context_rollout expects at least 2 frames")
        pred_emb = _sliding_teacher_forced_prediction(model, emb, act_emb, history_size)
        target = emb[:, 1:]
        delta_base = emb[:, :-1]
        rollout_pred = _autoregressive_prediction(
            model,
            emb,
            act_emb,
            history_size=history_size,
            start_size=rollout_start_size,
            horizon=rollout_horizon,
        )
        rollout_target = emb[:, rollout_start_size : rollout_start_size + rollout_pred.size(1)]
    else:
        raise ValueError(f"Unknown loss mode: {mode}")

    pred_f = pred_emb.float()
    target_f = target.float().detach()
    emb_f = emb.float()
    delta_base_f = delta_base.float() if delta_base is not None else emb_f[:, : pred_f.size(1)]

    teacher_loss = (pred_f - target_f).pow(2).mean()
    pred_loss = teacher_loss
    if rollout_pred is not None and rollout_target is not None:
        rollout_loss = (rollout_pred.float() - rollout_target.float().detach()).pow(2).mean()
        pred_loss = teacher_weight * teacher_loss + rollout_weight * rollout_loss
    sigreg_loss = sigreg(emb_f.transpose(0, 1))
    loss = pred_loss + sigreg_weight * sigreg_loss
    metrics = {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "teacher_loss": teacher_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
        "emb_std": emb_f.detach().std(),
        "emb_norm": emb_f.detach().norm(dim=-1).mean(),
        "temporal_mse": (emb_f[:, 1:] - emb_f[:, :-1]).pow(2).mean().detach(),
        "pred_delta_norm": (pred_f - delta_base_f).detach().norm(dim=-1).mean(),
        "emb": emb.detach(),
        "pred_emb": pred_emb.detach(),
    }
    if rollout_loss is not None and rollout_pred is not None:
        rollout_f = rollout_pred.float()
        rollout_target_f = rollout_target.float()
        metrics.update(
            {
                "rollout_loss": rollout_loss.detach(),
                "rollout_final_mse": (
                    rollout_f[:, -1] - rollout_target_f[:, -1].detach()
                ).pow(2).mean().detach(),
                "rollout_final_norm": (
                    rollout_f[:, -1] - rollout_target_f[:, -1].detach()
                ).norm(dim=-1).mean().detach(),
            }
        )
    return metrics
