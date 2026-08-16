import importlib.util
import math
import unittest
from types import SimpleNamespace


if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers", "transformers")):
    raise unittest.SkipTest("PyTorch, diffusers, and transformers are required for steering runtime tests")

import torch

from pipelines import ACE_STEP_MODEL_ID, FixedAlphaSteering, SteeringAceStepPipeline, SteeringPredictor
from scripts.evaluate import resolve_generation_config
from scripts.train import normalize_accumulated_gradients


class TargetSpecificSteeringTests(unittest.TestCase):
    def test_apg_interpolation_endpoints_and_midpoint(self) -> None:
        full = torch.tensor([5.0])
        retain = torch.tensor([9.0])

        outputs = [
            SteeringAceStepPipeline._interpolate_guided_predictions(full, retain, torch.tensor([alpha]))
            for alpha in (0.0, 0.5, 1.0)
        ]

        self.assertTrue(torch.equal(outputs[0], torch.tensor([5.0])))
        self.assertTrue(torch.equal(outputs[1], torch.tensor([7.0])))
        self.assertTrue(torch.equal(outputs[2], torch.tensor([9.0])))

    def test_non_sft_checkpoints_are_rejected_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "Extract/Lego/Complete"):
            SteeringAceStepPipeline.from_pretrained("ACE-Step/acestep-v15-base", device="cpu")

    def test_retain_prompt_validation_and_batch_expansion(self) -> None:
        self.assertEqual(
            SteeringAceStepPipeline._prepare_retain_prompt("  drums and piano  ", 2),
            ["drums and piano", "drums and piano"],
        )
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            SteeringAceStepPipeline._prepare_retain_prompt(["drums", "  "], 2)
        with self.assertRaisesRegex(ValueError, "batch size"):
            SteeringAceStepPipeline._prepare_retain_prompt(["drums"], 2)

    def test_pipeline_is_pinned_to_official_sft_checkpoint(self) -> None:
        self.assertEqual(ACE_STEP_MODEL_ID, "ACE-Step/acestep-v15-sft")

    def test_published_silence_latent_layout_is_canonicalized(self) -> None:
        published = torch.arange(64 * 25, dtype=torch.float32).reshape(1, 64, 25)

        canonical = SteeringAceStepPipeline._canonicalize_silence_latent(published)

        self.assertEqual(canonical.shape, (1, 25, 64))
        self.assertTrue(torch.equal(canonical, published.transpose(1, 2)))

    def test_sft_prompt_uses_the_official_instruction_sections(self) -> None:
        formatted = SteeringAceStepPipeline._format_sft_prompt("drums and piano", 10.0)

        self.assertEqual(
            formatted,
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n"
            "# Caption\ndrums and piano\n\n"
            "# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n"
            "- duration: 10 seconds\n<|endoftext|>\n",
        )

    def test_audio_postprocessing_normalizes_each_clip_to_minus_one_dbfs(self) -> None:
        audio = torch.tensor([[[0.25, -0.5]], [[2.0, -4.0]]])

        normalized = SteeringAceStepPipeline._normalize_audio(audio)

        expected_peak = 10.0 ** (-1.0 / 20.0)
        peaks = normalized.abs().flatten(1).amax(dim=1)
        self.assertTrue(torch.allclose(peaks, torch.full_like(peaks, expected_peak)))

    def test_default_alpha_bias_is_fifteen_percent_of_range(self) -> None:
        predictor = SteeringPredictor(
            latent_channels=2,
            target_embed_dim=3,
            block_out_channels=(4,),
            layers_per_block=1,
            cond_embed_dim=8,
            use_attention=False,
            norm_num_groups=1,
        )
        initialized_alpha = torch.sigmoid(predictor.head[-1].bias).item()
        self.assertTrue(math.isclose(initialized_alpha, 0.15, rel_tol=0.0, abs_tol=1e-6))

    def test_predictor_accepts_a_singleton_height_axis(self) -> None:
        r"""
        ACE-Step latents become `(batch, 64, 1, frames)`; the pipeline adds the height axis
        the encoder needs, so the predictor has to survive a height of 1 through every downsample.
        """
        predictor = SteeringPredictor(
            latent_channels=4,
            target_embed_dim=3,
            block_out_channels=(4, 8),
            layers_per_block=1,
            cond_embed_dim=8,
            num_attention_heads=2,
            norm_num_groups=1,
        )
        latents = torch.randn(2, 4, 1, 17)
        alpha = predictor(latents=latents, t=torch.tensor([500.0]), target_embed=torch.randn(2, 3))

        self.assertEqual(alpha.shape, (2, 1, 1, 1))
        self.assertTrue(torch.all((alpha >= 0.0) & (alpha <= 1.0)))

    def test_selection_generation_settings_are_inherited_and_conflicts_rejected(self) -> None:
        args = SimpleNamespace(
            num_inference_steps=None,
            audio_length_in_s=None,
            cfg_scale=None,
            shift=None,
            steering_frac_start=None,
            steering_frac_end=None,
        )
        checkpoint_args = {"steering_frac_start": 0.2, "steering_frac_end": 0.7}
        baseline = {
            "num_inference_steps": 50,
            "audio_length_in_s": 10.0,
            "cfg_scale": 7.0,
            "shift": 1.0,
        }

        resolved = resolve_generation_config(args, checkpoint_args, baseline)

        self.assertEqual(resolved["num_inference_steps"], 50)
        self.assertEqual(resolved["steering_frac_start"], 0.2)
        self.assertEqual(resolved["shift"], 1.0)

        args.cfg_scale = 5.0
        with self.assertRaisesRegex(ValueError, "does not match saved baseline"):
            resolve_generation_config(args, checkpoint_args, baseline)

    def test_fixed_alpha_controller_returns_one_value_per_sample(self) -> None:
        controller = FixedAlphaSteering(0.0)
        alpha = controller(torch.randn(3, 4, 1, 8), torch.tensor([1.0]), torch.randn(3, 2))
        self.assertEqual(alpha.shape, (3, 1, 1, 1))
        self.assertTrue(torch.equal(alpha, torch.zeros_like(alpha)))

    def test_accumulated_gradients_are_normalized_by_exact_sample_count(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        # Two batch means with sizes 2 and 1. Accumulating their sample sums should produce
        # (2 * 2 + 1 * 8) / 3 = 4 after exact sample normalization.
        (parameter * 2.0 * 2).backward()
        (parameter * 8.0 * 1).backward()
        normalize_accumulated_gradients([parameter], sample_count=3)
        self.assertTrue(torch.isclose(parameter.grad, torch.tensor(4.0)))


if __name__ == "__main__":
    unittest.main()
