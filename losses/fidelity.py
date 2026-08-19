import torch


def pingpong_fidelity_loss(alpha_field_records: list[dict]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    r"""
    Penalizes steered x̂0 estimates that drift from their non-steered reference where the CFG-diff
    `shape` profile says nothing should be corrected there.

    For every steered denoising step, `SteeringDiffusionTransformer.forward` already computes
    `full_cfg` (the non-steered, full-prompt x̂0 estimate) and the actually-used `steered_xhat0`
    while building `alpha_field`, so no extra forward pass is needed here: this only reweights and
    compares tensors the pipeline already produced. `weight_t = 1 - shape_t` is highest where
    `shape` is near zero, i.e. where the CFG-diff signal found nothing to correct and any deviation
    from the non-steered trajectory is treated as collateral damage rather than intended suppression.

    Args:
        alpha_field_records (`list[dict]`): The `alpha_field_records` from
            `pipelines.SteeringAudioPipelineOutput` (`steering_mode="cfg_diff_magnitude"`,
            `train=True`), each with `"shape"`, `"steered_xhat0"` and `"fullcfg_xhat0"` tensors of
            shape `(batch_size, 1 or channels, frames)`.

    Returns:
        `tuple[torch.Tensor, list[torch.Tensor]]`: The scalar `l_fid`, averaged over every steered
        step, and the list of per-step losses it was averaged from (for diagnostics: whether the
        deviation concentrates in particular steps).
    """
    if not alpha_field_records:
        raise ValueError(
            "`alpha_field_records` is empty: the pipeline has to be called with `steering_mode="
            "\"cfg_diff_magnitude\"` and `train=True` for this term to have anything to compare"
        )

    per_step = []
    for record in alpha_field_records:
        weight = 1.0 - record["shape"]
        per_step.append((weight * (record["steered_xhat0"] - record["fullcfg_xhat0"]).pow(2)).mean())

    l_fid = torch.stack(per_step).mean()
    return l_fid, per_step
