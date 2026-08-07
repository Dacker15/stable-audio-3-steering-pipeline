"""Hermetic ACE-Step steering tests with a tiny flow-matching stand-in."""

from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch import nn

from pipelines import AceStepConditioning, SteeringAceStepPipeline, SteeringPredictor


class _ToyDecoder(nn.Module):
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **_: object,
    ) -> object:
        condition = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        time_pattern = torch.linspace(
            -0.4, 0.6, hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )[None, :, None]
        channel_pattern = torch.linspace(
            0.5, 1.1, hidden_states.shape[2], device=hidden_states.device, dtype=hidden_states.dtype
        )[None, None, :]
        # Keep the stand-in velocity independent of x_t in the finite-
        # difference test.  The supported ``steering_only`` gradient
        # deliberately omits the real DiT Jacobian d velocity / d x_t; adding
        # such a term here would compare that surrogate with a different,
        # full-BPTT numerical derivative.
        velocity = torch.zeros_like(hidden_states) + condition * channel_pattern + time_pattern

        class ModelOutputLike:
            def __getitem__(self, index: int) -> torch.Tensor:
                if index != 0:
                    raise IndexError(index)
                return velocity

        return ModelOutputLike()


class _ToyAceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _ToyDecoder()
        self.null_condition_emb = nn.Parameter(torch.zeros(1, 1, 5))
        self.anchor = nn.Parameter(torch.zeros(()))

    def prepare_noise(self, context_latents: torch.Tensor, seed=None) -> torch.Tensor:
        batch, frames, doubled_channels = context_latents.shape
        if isinstance(seed, list):
            seeds = seed
        else:
            seeds = [seed] * batch
        samples = []
        for item_seed in seeds:
            generator = torch.Generator(device=context_latents.device)
            if item_seed is not None:
                generator.manual_seed(int(item_seed))
            samples.append(
                torch.randn(
                    1,
                    frames,
                    doubled_channels // 2,
                    generator=generator,
                    device=context_latents.device,
                    dtype=context_latents.dtype,
                )
            )
        return torch.cat(samples)


class _FixedAlpha(nn.Module):
    def __init__(self, value: float, trainable: bool = False) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value), requires_grad=trainable)

    def forward(self, latents: torch.Tensor, **_: object) -> torch.Tensor:
        return self.value.expand(latents.shape[0], 1, 1)


def _conditioning(batch: int = 1, frames: int = 8, channels: int = 4) -> AceStepConditioning:
    return AceStepConditioning(
        full_hidden_states=torch.ones(batch, 3, 5),
        # A different sequence width verifies that endpoint equivalence does
        # not depend on joint prompt padding.
        retain_hidden_states=torch.full((batch, 5, 5), -0.35),
        full_attention_mask=torch.ones(batch, 3, dtype=torch.bool),
        retain_attention_mask=torch.ones(batch, 5, dtype=torch.bool),
        context_latents=torch.zeros(batch, frames, channels * 2),
        latent_attention_mask=torch.ones(batch, frames),
        num_audio_samples=frames * 16,
    )


def _pipeline() -> SteeringAceStepPipeline:
    return SteeringAceStepPipeline(
        _ToyAceModel(),
        vae=None,
        text_encoder=None,
        text_tokenizer=None,
        silence_latent=torch.zeros(1, 8, 64),
        device="cpu",
        sequential_dit=True,
        offload_dit_after_generation=False,
    )


class AceStepAlphaSteeringTests(unittest.TestCase):
    def test_alpha_zero_matches_unsteered_full_prompt(self) -> None:
        pipeline = _pipeline()
        conditioning = _conditioning()
        common = dict(num_inference_steps=5, guidance_scale=7.0, seed=123, steering_frac_start=0.0, steering_frac_end=1.0)
        baseline = pipeline.denoise(conditioning, **common).audios
        alpha_zero = pipeline.denoise(
            conditioning,
            steering_model=_FixedAlpha(0.0),
            steering_target_embeds=torch.zeros(1, 6),
            **common,
        ).audios
        torch.testing.assert_close(alpha_zero, baseline, rtol=0.0, atol=0.0)

    def test_alpha_one_matches_retain_only_condition(self) -> None:
        pipeline = _pipeline()
        conditioning = _conditioning()
        retain_only = replace(
            conditioning,
            full_hidden_states=conditioning.retain_hidden_states,
            full_attention_mask=conditioning.retain_attention_mask,
        )
        common = dict(num_inference_steps=4, guidance_scale=7.0, seed=9, steering_frac_start=0.0, steering_frac_end=1.0)
        expected = pipeline.denoise(retain_only, **common).audios
        actual = pipeline.denoise(
            conditioning,
            steering_model=_FixedAlpha(1.0),
            steering_target_embeds=torch.zeros(1, 6),
            **common,
        ).audios
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_seeded_generation_is_deterministic_and_has_ace_shape(self) -> None:
        pipeline = _pipeline()
        conditioning = _conditioning(batch=2)
        first = pipeline.denoise(conditioning, num_inference_steps=3, seed=[7, 8]).audios
        second = pipeline.denoise(conditioning, num_inference_steps=3, seed=[7, 8]).audios
        self.assertEqual(first.shape, (2, 8, 4))
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_default_schedule_is_sft_shift_one(self) -> None:
        pipeline = _pipeline()
        conditioning = _conditioning()
        default = pipeline.denoise(conditioning, num_inference_steps=4, seed=21).audios
        explicit = pipeline.denoise(
            conditioning,
            num_inference_steps=4,
            shift=1.0,
            seed=21,
        ).audios
        torch.testing.assert_close(default, explicit, rtol=0.0, atol=0.0)

    def test_batched_setting_falls_back_for_independent_prompt_widths(self) -> None:
        conditioning = _conditioning()
        sequential = _pipeline()
        batched = _pipeline()
        batched.sequential_dit = False
        kwargs = dict(
            steering_target_embeds=torch.zeros(1, 6),
            steering_model=_FixedAlpha(0.4),
            num_inference_steps=3,
            steering_frac_start=0.0,
            steering_frac_end=1.0,
            seed=22,
        )
        expected = sequential.denoise(conditioning, **kwargs).audios
        actual = batched.denoise(conditioning, **kwargs).audios
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_alpha_gradient_is_finite_nonzero_and_matches_finite_difference(self) -> None:
        pipeline = _pipeline()
        conditioning = _conditioning()
        steering = _FixedAlpha(0.25, trainable=True)
        kwargs = dict(
            conditioning=conditioning,
            steering_target_embeds=torch.zeros(1, 6),
            steering_model=steering,
            num_inference_steps=3,
            guidance_scale=7.0,
            steering_frac_start=0.0,
            steering_frac_end=1.0,
            initial_latents=torch.full((1, 8, 4), 0.2),
            train=True,
        )
        loss = pipeline.denoise(**kwargs).audios.square().mean()
        loss.backward()
        analytic = steering.value.grad.item()
        self.assertTrue(torch.isfinite(steering.value.grad))
        self.assertGreater(abs(analytic), 1e-6)

        epsilon = 1e-3
        with torch.no_grad():
            steering.value.fill_(0.25 + epsilon)
            plus = pipeline.denoise(**{**kwargs, "train": False}).audios.square().mean()
            steering.value.fill_(0.25 - epsilon)
            minus = pipeline.denoise(**{**kwargs, "train": False}).audios.square().mean()
        finite_difference = ((plus - minus) / (2 * epsilon)).item()
        self.assertAlmostEqual(analytic, finite_difference, delta=max(2e-3, abs(finite_difference) * 0.02))

    def test_one_optimizer_step_updates_real_predictor(self) -> None:
        torch.manual_seed(4)
        pipeline = _pipeline()
        predictor = SteeringPredictor(
            latent_channels=4,
            target_embed_dim=6,
            block_out_channels=(8, 16),
            layers_per_block=1,
            cond_embed_dim=16,
            use_attention=False,
            norm_num_groups=4,
        )
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
        before = predictor.head[-1].weight.detach().clone()
        output = pipeline.denoise(
            _conditioning(),
            steering_target_embeds=torch.randn(1, 6),
            steering_model=predictor,
            num_inference_steps=3,
            guidance_scale=7.0,
            steering_frac_start=0.0,
            steering_frac_end=1.0,
            seed=13,
            train=True,
        )
        loss = output.audios.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, predictor.head[-1].weight.detach()))

    def test_thinking_and_dcw_are_explicitly_rejected(self) -> None:
        pipeline = _pipeline()
        pipeline.prepare_conditioning = lambda *args, **kwargs: _conditioning()  # type: ignore[method-assign]
        with self.assertRaisesRegex(ValueError, "thinking=False"):
            pipeline("prompt", thinking=True)
        with self.assertRaisesRegex(ValueError, "dcw_enabled=False"):
            pipeline("prompt", dcw_enabled=True)


if __name__ == "__main__":
    unittest.main()
