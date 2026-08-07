"""Opt-in ACE-Step 1.5 FP32/CUDA endpoint and gradient diagnostics.

These tests load the real SFT DiT, Qwen encoder and VAE.  They are intentionally
excluded from ordinary test runs because the weights are large and the checks
are designed for a CUDA machine such as a Kaggle Tesla T4.

Enable explicitly::

    RUN_ACE_GPU_TESTS=1 pytest -m gpu tests/diagnostics/test_real_ace_step.py
"""

from __future__ import annotations

import math
import os

import pytest


pytestmark = [
    pytest.mark.diagnostic,
    pytest.mark.external_model,
    pytest.mark.gpu,
]


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


RUN_REAL_ACE = _enabled("RUN_ACE_GPU_TESTS")


class _ConstantAlpha:
    """Constructed lazily as an ``nn.Module`` after torch is imported."""

    @staticmethod
    def build(value: float, *, trainable: bool = False):
        import torch
        from torch import nn

        class ConstantAlpha(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                initial = torch.tensor(float(value)).clamp(1e-4, 1.0 - 1e-4)
                logit = torch.logit(initial)
                if trainable:
                    self.logit = nn.Parameter(logit)
                else:
                    self.register_buffer("logit", logit)

            def forward(self, latents, t, target_embed):
                del t, target_embed
                alpha = torch.sigmoid(self.logit)
                # Exact endpoints are useful for trajectory-equivalence checks.
                if not trainable:
                    alpha = latents.new_tensor(value)
                return alpha.expand(latents.shape[0], 1, 1)

        return ConstantAlpha()


@pytest.fixture(scope="module")
def real_ace_setup():
    if not RUN_REAL_ACE:
        pytest.skip("set RUN_ACE_GPU_TESTS=1 to load and execute the real ACE-Step checkpoint")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("RUN_ACE_GPU_TESTS=1 was set, but CUDA is unavailable")

    from pipelines import SteeringAceStepPipeline

    duration = float(os.environ.get("ACE_DIAGNOSTIC_SECONDS", "5.12"))
    prompt = os.environ.get(
        "ACE_DIAGNOSTIC_PROMPT",
        "an energetic jazz quartet with a bright trumpet solo",
    )
    retain_prompt = os.environ.get(
        "ACE_DIAGNOSTIC_RETAIN_PROMPT",
        "an energetic jazz quartet",
    )
    local_only = _enabled("ACE_DIAGNOSTIC_LOCAL_FILES_ONLY")
    pipe = SteeringAceStepPipeline.from_pretrained(
        device="cuda",
        torch_dtype=torch.float32,
        local_files_only=local_only,
        sequential_dit=True,
        offload_text_encoder=True,
        offload_dit_after_generation=True,
    )

    full = pipe.prepare_conditioning(prompt, retain_prompt, audio_length_in_s=duration).cpu()
    # Prepare the retain-only baseline independently.  This catches accidental
    # coupling through joint tokenizer/encoder padding.
    retain_only = pipe.prepare_conditioning(
        retain_prompt,
        retain_prompt,
        audio_length_in_s=duration,
    ).cpu()
    pipe.offload_dit()
    return pipe, full, retain_only


def _denoise_kwargs() -> dict[str, object]:
    return {
        "num_inference_steps": int(os.environ.get("ACE_DIAGNOSTIC_STEPS", "2")),
        "guidance_scale": float(os.environ.get("ACE_DIAGNOSTIC_GUIDANCE", "7.0")),
        "guidance_mode": "apg",
        "shift": 1.0,
        "steering_frac_start": 0.0,
        "steering_frac_end": 1.0,
    }


def test_real_seeded_noise_is_deterministic(real_ace_setup) -> None:
    """Exercise ACE-Step's own prepare_noise path, not a test-only RNG."""

    import torch

    pipe, full, _ = real_ace_setup
    conditioning = full.to(pipe.device, torch.float32)
    first = pipe._prepare_initial_latents(conditioning, 1234, None).detach().cpu()
    repeated = pipe._prepare_initial_latents(conditioning, 1234, None).detach().cpu()
    different = pipe._prepare_initial_latents(conditioning, 1235, None).detach().cpu()
    torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)
    assert not torch.equal(first, different), "different ACE-Step seeds produced identical noise"


def test_real_unsteered_sampler_matches_upstream_generate_audio(real_ace_setup) -> None:
    """The custom disabled-steering trajectory must match upstream SFT generation.

    Qwen/ACE encoder preparation is bypassed in the upstream call because the
    same prepared tensors are supplied to both samplers.  The comparison still
    exercises the official checkpoint's complete APG/Euler loop, learned-null
    branch, flow schedule and seeded ``prepare_noise`` implementation.
    """

    from unittest.mock import patch

    import torch

    pipe, full, _ = real_ace_setup
    conditioning = full.to(pipe.device, torch.float32)
    kwargs = _denoise_kwargs()
    seed = 2468
    pipe._move_model(pipe.device)

    prepared = (
        conditioning.full_hidden_states,
        conditioning.full_attention_mask,
        conditioning.context_latents,
    )
    source_channels = conditioning.latent_shape[-1]
    source = conditioning.context_latents[..., :source_channels]
    chunk_masks = conditioning.context_latents[..., source_channels:]
    is_covers = torch.zeros(conditioning.batch_size, device=pipe.device, dtype=torch.long)

    with patch.object(pipe.model, "prepare_condition", return_value=prepared):
        upstream = pipe.model.generate_audio(
            text_hidden_states=None,
            text_attention_mask=None,
            lyric_hidden_states=None,
            lyric_attention_mask=None,
            refer_audio_acoustic_hidden_states_packed=None,
            refer_audio_order_mask=None,
            src_latents=source,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=pipe.silence_latent.to(pipe.device, torch.float32),
            attention_mask=conditioning.latent_attention_mask,
            seed=seed,
            infer_method="ode",
            infer_steps=int(kwargs["num_inference_steps"]),
            diffusion_guidance_scale=float(kwargs["guidance_scale"]),
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            use_progress_bar=False,
            use_adg=False,
            shift=float(kwargs["shift"]),
            sampler_mode="euler",
            velocity_norm_threshold=0.0,
            velocity_ema_factor=0.0,
            dcw_enabled=False,
        )["target_latents"].detach()

    old_sequential = pipe.sequential_dit
    old_offload = pipe.offload_dit_after_generation
    try:
        # Match upstream's batched conditional/null decoder evaluation.  The
        # T4 training default remains the lower-memory sequential mode.
        pipe.sequential_dit = False
        pipe.offload_dit_after_generation = False
        custom = pipe.denoise(conditioning, seed=seed, **kwargs).audios.detach()
        torch.testing.assert_close(custom, upstream, rtol=2e-4, atol=2e-4)
    finally:
        pipe.sequential_dit = old_sequential
        pipe.offload_dit_after_generation = old_offload
        pipe.offload_dit()


def test_real_fp32_alpha_endpoints_match_unsteered_trajectories(real_ace_setup) -> None:
    """alpha=0 recovers full conditioning; alpha=1 recovers retain conditioning."""

    import torch

    pipe, full, retain_only = real_ace_setup
    generator = torch.Generator(device="cpu").manual_seed(1234)
    initial = torch.randn(full.latent_shape, generator=generator, dtype=torch.float32)
    target = torch.zeros(full.batch_size, 512, dtype=torch.float32)
    kwargs = _denoise_kwargs()

    baseline_full = pipe.denoise(full, initial_latents=initial, **kwargs).audios.detach().cpu()
    alpha_zero = pipe.denoise(
        full,
        initial_latents=initial,
        steering_target_embeds=target,
        steering_model=_ConstantAlpha.build(0.0).cuda(),
        **kwargs,
    ).audios.detach().cpu()
    baseline_retain = pipe.denoise(
        retain_only,
        initial_latents=initial,
        **kwargs,
    ).audios.detach().cpu()
    alpha_one = pipe.denoise(
        full,
        initial_latents=initial,
        steering_target_embeds=target,
        steering_model=_ConstantAlpha.build(1.0).cuda(),
        **kwargs,
    ).audios.detach().cpu()

    for name, tensor in {
        "baseline_full": baseline_full,
        "alpha_zero": alpha_zero,
        "baseline_retain": baseline_retain,
        "alpha_one": alpha_one,
    }.items():
        assert tensor.dtype == torch.float32, f"{name} unexpectedly used {tensor.dtype}"
        assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"

    assert torch.allclose(alpha_zero, baseline_full, rtol=1e-5, atol=1e-5)
    assert torch.allclose(alpha_one, baseline_retain, rtol=1e-5, atol=1e-5)


def test_real_canonical_null_matches_upstream_repetition(real_ace_setup) -> None:
    """One learned-null token must equal the upstream repeated-token branch."""

    import torch

    pipe, full, _ = real_ace_setup
    conditioning = full.to(pipe.device, torch.float32)
    pipe._move_model(pipe.device)
    generator = torch.Generator(device="cpu").manual_seed(99)
    latents = torch.randn(full.latent_shape, generator=generator, dtype=torch.float32).to(pipe.device)
    timestep = torch.tensor(0.5, device=pipe.device, dtype=torch.float32)
    canonical = pipe.model.null_condition_emb.expand(full.batch_size, -1, -1)
    canonical_mask = torch.ones(
        full.batch_size,
        1,
        device=pipe.device,
        dtype=conditioning.full_attention_mask.dtype,
    )
    repeated = pipe.model.null_condition_emb.expand_as(conditioning.full_hidden_states)
    one_token_velocity = pipe._decoder_velocity(
        latents,
        timestep,
        canonical,
        canonical_mask,
        conditioning,
    )
    repeated_velocity = pipe._decoder_velocity(
        latents,
        timestep,
        repeated,
        conditioning.full_attention_mask,
        conditioning,
    )
    pipe.offload_dit()
    torch.testing.assert_close(one_token_velocity, repeated_velocity, rtol=1e-4, atol=1e-4)


@pytest.mark.clap
def test_real_clap_objective_gradient_matches_last_step_finite_difference(real_ace_setup) -> None:
    """The real CLAP objective must give alpha a useful surrogate gradient.

    Steering is active only on the final Euler step.  The latent entering that
    step is consequently identical for the analytic and finite-difference
    evaluations, so the numerical derivative does not include the DiT
    Jacobian intentionally omitted by the supported steering-only surrogate.
    """

    import torch

    from losses import DEFAULT_CLAP_MODEL_ID, DEFAULT_CLAP_REVISION, ClapLoss
    from pipelines import ACE_STEP_SAMPLE_RATE

    pipe, full, _ = real_ace_setup
    device = pipe.device
    local_only = _enabled("ACE_DIAGNOSTIC_LOCAL_FILES_ONLY")
    target_text = os.environ.get("ACE_DIAGNOSTIC_TARGET", "trumpet")
    retain_text = os.environ.get(
        "ACE_DIAGNOSTIC_RETAIN_PROMPT",
        "an energetic jazz quartet",
    )
    retain_weight = float(os.environ.get("ACE_DIAGNOSTIC_RETAIN_WEIGHT", "1.0"))
    if not math.isfinite(retain_weight) or retain_weight < 0.0:
        pytest.fail("ACE_DIAGNOSTIC_RETAIN_WEIGHT must be finite and non-negative")

    # Keep CLAP and the FP32 DiT mutually exclusive on the accelerator.  The
    # VAE and CLAP must coexist only while evaluating the differentiable audio
    # objective, matching the training path's lowest-memory arrangement.
    pipe.offload_dit()
    pipe.offload_vae()
    clap = ClapLoss.from_pretrained(
        DEFAULT_CLAP_MODEL_ID,
        revision=DEFAULT_CLAP_REVISION,
        torch_dtype=torch.float32,
        local_files_only=local_only,
    ).eval()
    if getattr(clap.text_encoder, "supports_gradient_checkpointing", False):
        clap.text_encoder.gradient_checkpointing_enable()

    def move_clap(target_device: torch.device) -> None:
        clap.to(device=target_device, dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    move_clap(device)
    target_embed, retain_embed = (
        clap.encode_text([target_text, retain_text]).detach().cpu().unbind(0)
    )
    move_clap(torch.device("cpu"))

    generator = torch.Generator(device="cpu").manual_seed(4321)
    initial = torch.randn(full.latent_shape, generator=generator, dtype=torch.float32)
    controller = _ConstantAlpha.build(0.35, trainable=True).to(device)
    steps = int(_denoise_kwargs()["num_inference_steps"])
    if steps < 1:
        pytest.fail("ACE_DIAGNOSTIC_STEPS must be at least 1")
    last_step_start = (steps - 1) / steps
    denoise_kwargs = {
        **_denoise_kwargs(),
        "steering_frac_start": last_step_start,
        "steering_frac_end": 1.0,
    }
    target_for_controller = target_embed.unsqueeze(0)

    controller.zero_grad(set_to_none=True)
    output = pipe.denoise(
        full,
        initial_latents=initial,
        steering_target_embeds=target_for_controller,
        steering_model=controller,
        train=True,
        **denoise_kwargs,
    )
    waveform = pipe.decode_latents(
        output.audios,
        num_audio_samples=full.num_audio_samples,
        normalize=True,
    )
    move_clap(device)
    audio_embed = clap.encode_audio(waveform, ACE_STEP_SAMPLE_RATE)
    target_on_device = target_embed.unsqueeze(0).to(device)
    retain_on_device = retain_embed.unsqueeze(0).to(device)
    target_similarity = (audio_embed * target_on_device).sum(dim=-1)
    retain_similarity = (audio_embed * retain_on_device).sum(dim=-1)
    objective = target_similarity.mean() + retain_weight * (1.0 - retain_similarity).mean()
    objective.backward()

    assert torch.isfinite(output.audios).all(), "denoised latents contain NaN/Inf"
    assert torch.isfinite(waveform).all(), "decoded waveform contains NaN/Inf"
    assert torch.isfinite(audio_embed).all(), "CLAP audio embedding contains NaN/Inf"
    assert torch.isfinite(objective), "CLAP target-retain objective contains NaN/Inf"
    assert controller.logit.grad is not None, "alpha did not receive a CLAP gradient"
    assert torch.isfinite(controller.logit.grad).all(), "alpha gradient contains NaN/Inf"

    center_alpha = float(torch.sigmoid(controller.logit.detach()))
    sigmoid_derivative = center_alpha * (1.0 - center_alpha)
    analytic = float(controller.logit.grad.detach()) / sigmoid_derivative
    analytic_objective = float(objective.detach())
    assert abs(analytic) > 1e-8, "CLAP objective produced an effectively zero alpha gradient"

    del output, waveform, audio_embed, objective, target_similarity, retain_similarity
    del target_on_device, retain_on_device
    pipe.offload_vae()
    move_clap(torch.device("cpu"))

    def objective_at(alpha: float) -> float:
        alpha_tensor = torch.tensor(alpha, device=device, dtype=torch.float32)
        with torch.no_grad():
            controller.logit.copy_(torch.logit(alpha_tensor))
            latent_output = pipe.denoise(
                full,
                initial_latents=initial,
                steering_target_embeds=target_for_controller,
                steering_model=controller,
                train=False,
                **denoise_kwargs,
            )
            decoded = pipe.decode_latents(
                latent_output.audios,
                num_audio_samples=full.num_audio_samples,
                normalize=True,
            )
            move_clap(device)
            embedding = clap.encode_audio(decoded, ACE_STEP_SAMPLE_RATE)
            value = (
                (embedding * target_embed.unsqueeze(0).to(device)).sum(dim=-1).mean()
                + retain_weight
                * (
                    1.0
                    - (embedding * retain_embed.unsqueeze(0).to(device)).sum(dim=-1)
                ).mean()
            )
            result = float(value)
        del latent_output, decoded, embedding, value
        pipe.offload_vae()
        move_clap(torch.device("cpu"))
        return result

    epsilon = float(os.environ.get("ACE_DIAGNOSTIC_FD_EPSILON", "0.01"))
    if not 0.0 < epsilon < min(center_alpha, 1.0 - center_alpha):
        pytest.fail("ACE_DIAGNOSTIC_FD_EPSILON must keep alpha +/- epsilon inside (0, 1)")
    minus = objective_at(center_alpha - epsilon)
    plus = objective_at(center_alpha + epsilon)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    assert math.isfinite(analytic_objective)
    assert math.isfinite(minus) and math.isfinite(plus) and math.isfinite(finite_difference)
    assert abs(finite_difference) > 1e-8, "finite difference is effectively zero"
    assert analytic * finite_difference > 0.0, (
        f"analytic and finite-difference gradients disagree in sign: {analytic=} {finite_difference=}"
    )
    tolerance = max(5e-4, 0.2 * max(abs(analytic), abs(finite_difference)))
    assert abs(analytic - finite_difference) <= tolerance, (
        f"analytic and finite-difference gradients disagree: {analytic=} "
        f"{finite_difference=} {tolerance=}"
    )
    descent_value = minus if analytic > 0.0 else plus
    ascent_value = plus if analytic > 0.0 else minus
    assert descent_value < analytic_objective, (
        "a small step opposite the alpha gradient did not reduce the real CLAP objective"
    )
    assert descent_value < ascent_value, (
        "a small step opposite the alpha gradient did not improve the real CLAP objective"
    )

    with torch.no_grad():
        controller.logit.copy_(
            torch.logit(torch.tensor(center_alpha, device=device, dtype=torch.float32))
        )
