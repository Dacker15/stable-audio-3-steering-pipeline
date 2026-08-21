import torch


def compute_cfg_diff_shape(
    full_cfg: torch.Tensor,
    retain_cfg: torch.Tensor,
    quantile_low: float = 0.10,
    quantile_high: float = 0.90,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""
    Deterministic, per-frame temporal profile of the CFG-diff (full vs retain), robustly normalized
    to `[0, 1]` per sample.

    This is the "shape" half of Regime A/B's `alpha_t = (alpha_min + (alpha_max - alpha_min) *
    magnitude * shape)`: how strongly the target-removal changes the prediction, per frame, relative
    to that sample's own range. `magnitude` is the other half, either a fixed hyperparameter
    (`compute_cfg_diff_alpha`, Regime A) or a learned per-sample scalar (Regime B).

    Args:
        full_cfg, retain_cfg (`torch.Tensor`): Guided predictions, shape `(batch_size, channels,
            frames)`.
        quantile_low, quantile_high (`float`, *optional*): Percentiles used for the robust
            per-sample normalization of the temporal profile, computed along the frame axis.
        eps (`float`, *optional*): Numerical stabilization for the quantile normalization.

    Returns:
        `torch.Tensor`: `shape` in `[0, 1]`, of shape `(batch_size, 1, frames)`, detached, in
        float32.
    """
    diff = (full_cfg - retain_cfg).detach().float()

    # raw temporal profile: how strongly the target-removal changes the prediction, per frame
    shape = diff.norm(dim=1, keepdim=True)  # (batch, 1, frames)

    # robust per-sample normalization, not plain min-max: a long tail of isolated peaks would
    # otherwise crush the rest of the profile towards zero
    p_lo = shape.quantile(quantile_low, dim=-1, keepdim=True)
    p_hi = shape.quantile(quantile_high, dim=-1, keepdim=True)
    return ((shape - p_lo) / (p_hi - p_lo + eps)).clamp(0.0, 1.0)


def compute_cfg_diff_alpha(
    full_cfg: torch.Tensor,
    retain_cfg: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    magnitude: float,
    quantile_low: float = 0.10,
    quantile_high: float = 0.90,
    eps: float = 1e-6,
    abs_floor: float | None = None,
) -> torch.Tensor:
    r"""
    Deterministic, per-frame `alpha_t`, derived from the norm of the CFG-diff (full vs retain).

    This is "Regime A": a zero-cost, training-free alternative to a learned per-frame steering
    model for `SteeringStableAudioPipeline`. Where a learned model predicts `alpha_t` from the
    latents, this reads it directly off the signal `SteeringDiffusionTransformer` already computes
    at every steered step, the difference between the full-prompt and retain-prompt classifier-free
    guidance predictions.

    Args:
        full_cfg, retain_cfg (`torch.Tensor`): Guided predictions, shape `(batch_size, channels,
            frames)`.
        alpha_min, alpha_max (`float`): Same bounds `"learned"` mode uses, for compatibility.
        magnitude (`float`): Global gain in `[0, 1]`, a fixed hyperparameter rather than a learned
            one in this regime.
        quantile_low, quantile_high (`float`, *optional*): Percentiles used for the robust
            per-sample normalization of the temporal profile, computed along the frame axis.
        eps (`float`, *optional*): Numerical stabilization for the quantile normalization.
        abs_floor (`float` or `None`, *optional*): If given, samples whose overall CFG-diff norm
            (across channels and frames) falls below this floor are pinned to `alpha_min`. The
            per-quantile normalization is relative to each sample: if the CFG-diff is small in
            absolute terms everywhere (e.g. the target was barely present to begin with), it still
            produces a profile that reaches ~1 in its upper half, even though there is nothing
            meaningful to correct. Off by default; validate empirically before enabling.

    Returns:
        `torch.Tensor`: `alpha_t` of shape `(batch_size, 1, frames)`, ready to broadcast over
        `(batch_size, channels, frames)`, in the dtype of `full_cfg`.
    """
    shape = compute_cfg_diff_shape(full_cfg, retain_cfg, quantile_low, quantile_high, eps)

    alpha = alpha_min + (alpha_max - alpha_min) * magnitude * shape

    if abs_floor is not None:
        diff = (full_cfg - retain_cfg).detach().float()
        overall_strength = diff.flatten(1).norm(dim=-1)  # (batch,)
        below_floor = overall_strength < abs_floor
        alpha = torch.where(below_floor[:, None, None], torch.full_like(alpha, alpha_min), alpha)

    return alpha.clamp(alpha_min, alpha_max).to(full_cfg.dtype)
