import importlib.util
import math
import unittest


if importlib.util.find_spec("torch") is None or importlib.util.find_spec("diffusers") is None:
    raise unittest.SkipTest("PyTorch and diffusers are required for steering runtime tests")

import torch

from pipelines import SteeringMusicLDMPipeline, SteeringPredictor


class TargetSpecificSteeringTests(unittest.TestCase):
    def test_cfg_interpolation_endpoints_and_midpoint(self) -> None:
        uncond = torch.tensor([1.0])
        full = torch.tensor([3.0])
        retain = torch.tensor([5.0])

        outputs = [
            SteeringMusicLDMPipeline._interpolate_cfg_predictions(
                uncond, full, retain, guidance_scale=2.0, alpha_t=torch.tensor([alpha])
            )
            for alpha in (0.0, 0.5, 1.0)
        ]

        self.assertTrue(torch.equal(outputs[0], torch.tensor([5.0])))
        self.assertTrue(torch.equal(outputs[1], torch.tensor([7.0])))
        self.assertTrue(torch.equal(outputs[2], torch.tensor([9.0])))

    def test_retain_prompt_validation_and_batch_expansion(self) -> None:
        self.assertEqual(
            SteeringMusicLDMPipeline._prepare_retain_prompt("  drums and piano  ", 2),
            ["drums and piano", "drums and piano"],
        )
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            SteeringMusicLDMPipeline._prepare_retain_prompt(["drums", "  "], 2)
        with self.assertRaisesRegex(ValueError, "batch size"):
            SteeringMusicLDMPipeline._prepare_retain_prompt(["drums"], 2)

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


if __name__ == "__main__":
    unittest.main()
