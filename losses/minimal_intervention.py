import torch


def minimal_intervention_penalty(alpha_field_records: list[dict]) -> torch.Tensor:
    r"""
    Penalizes steering strength itself, so the policy only pays the cost of intervening where the
    hinge terms of `HingeClapLoss` still have gradient to give, instead of steering at full strength
    everywhere the margins allow it.

    Not subtracting `alpha_min` from `alpha_field` before averaging: it is a constant, so it would
    only shift every logged value by the same amount without changing the gradient.

    Args:
        alpha_field_records (`list[dict]`): The `alpha_field_records` from
            `pipelines.SteeringAudioPipelineOutput`, one dict per steered denoising step, each with
            an `"alpha_field"` tensor of shape `(batch_size, 1, frames)`.

    Returns:
        `torch.Tensor`: Scalar mean of `alpha_field` pooled over every steered step and the whole
        batch.
    """
    if not alpha_field_records:
        raise ValueError(
            "`alpha_field_records` is empty: the pipeline has to be called with `steering_mode="
            "\"cfg_diff_magnitude\"` and `train=True` for this term to have anything to penalize"
        )
    return torch.stack([record["alpha_field"].mean() for record in alpha_field_records]).mean()


def linear_warmup(step: int, target: float, warmup_steps: int) -> float:
    r"""
    Linearly ramps a loss weight from `0` to `target` over `warmup_steps` optimizer steps, then holds
    it at `target`.

    Used for `lambda_reg`: introducing the minimal-intervention penalty at full strength from step 1
    risks the policy learning to never activate steering at all, before the hinge terms have had a
    chance to teach it when intervention is actually needed.

    Args:
        step (`int`): The current 1-based training step.
        target (`float`): The weight's value once warmup has completed.
        warmup_steps (`int`): Number of steps to ramp over. `0` disables warmup, i.e. `target` from
            the first step.

    Returns:
        `float`: The warmed-up weight for this step.
    """
    if warmup_steps <= 0:
        return target
    return target * min(1.0, step / warmup_steps)
