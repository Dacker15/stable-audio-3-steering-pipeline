import importlib.util
import math
import unittest


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is required for SteeringPredictor tests")

import torch

from pipelines import SteeringPredictor


class SteeringPredictorTests(unittest.TestCase):
    @staticmethod
    def _predictor(**overrides) -> SteeringPredictor:
        config = {
            "latent_channels": 4,
            "target_embed_dim": 6,
            "block_out_channels": (8, 16),
            "layers_per_block": 1,
            "cond_embed_dim": 16,
            "num_attention_heads": 4,
            "use_attention": True,
            "norm_num_groups": 4,
        }
        config.update(overrides)
        return SteeringPredictor(**config).cpu()

    def test_accepts_ace_step_channel_last_latents_and_returns_broadcast_alpha(self) -> None:
        predictor = self._predictor().eval()
        latents = torch.randn(3, 17, 4, dtype=torch.float32)
        target_embed = torch.randn(3, 6, dtype=torch.float32)

        with torch.no_grad():
            scalar_t_alpha = predictor(latents, torch.tensor(0.75), target_embed)
            batched_t_alpha = predictor(latents, torch.tensor([0.1, 0.5, 0.9]), target_embed)

        self.assertEqual(scalar_t_alpha.shape, (3, 1, 1))
        self.assertEqual(batched_t_alpha.shape, (3, 1, 1))
        self.assertEqual(scalar_t_alpha.dtype, latents.dtype)
        self.assertTrue(torch.all((scalar_t_alpha >= 0.0) & (scalar_t_alpha <= 1.0)))
        self.assertTrue(torch.all((batched_t_alpha >= 0.0) & (batched_t_alpha <= 1.0)))

    def test_validates_latent_target_and_timestep_shapes(self) -> None:
        predictor = self._predictor(use_attention=False)
        latents = torch.randn(2, 11, 4)
        target_embed = torch.randn(2, 6)

        with self.assertRaisesRegex(ValueError, "batch_size, time, latent_channels"):
            predictor(torch.randn(2, 4, 3, 5), 0.5, target_embed)
        with self.assertRaisesRegex(ValueError, "expects 4"):
            predictor(torch.randn(2, 11, 5), 0.5, target_embed)
        with self.assertRaisesRegex(ValueError, "batch size"):
            predictor(latents, 0.5, torch.randn(1, 6))
        with self.assertRaisesRegex(ValueError, "expects 6"):
            predictor(latents, 0.5, torch.randn(2, 7))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            predictor(latents, torch.ones(2, 1), target_embed)
        with self.assertRaisesRegex(ValueError, "batch size"):
            predictor(latents, torch.ones(3), target_embed)

    def test_alpha_initialization_and_configurable_bounds(self) -> None:
        torch.manual_seed(0)
        predictor = self._predictor(alpha_min=-0.5, alpha_max=1.5).eval()
        normalized_bias = torch.sigmoid(predictor.head[-1].bias).item()
        expected_alpha = -0.5 + 0.15 * (1.5 - -0.5)

        self.assertTrue(math.isclose(normalized_bias, 0.15, rel_tol=0.0, abs_tol=1e-6))
        with torch.no_grad():
            alpha = predictor(torch.randn(4, 13, 4), 0.25, torch.randn(4, 6))
        self.assertTrue(torch.all(alpha >= -0.5))
        self.assertTrue(torch.all(alpha <= 1.5))
        self.assertTrue(torch.allclose(alpha, torch.full_like(alpha, expected_alpha), atol=0.02, rtol=0.0))

        with self.assertRaisesRegex(ValueError, "smaller"):
            self._predictor(alpha_min=1.0, alpha_max=1.0)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            self._predictor(alpha_min=0.0, alpha_max=1.0, alpha_init=1.0)

    def test_gradient_is_finite_nonzero_and_reaches_inputs_and_encoder(self) -> None:
        torch.manual_seed(1)
        predictor = self._predictor().train()
        latents = torch.randn(2, 15, 4, requires_grad=True)
        target_embed = torch.randn(2, 6, requires_grad=True)

        alpha = predictor(latents, torch.tensor([0.2, 0.8]), target_embed)
        loss = (alpha * torch.tensor([[[1.0]], [[-0.4]]])).sum()
        loss.backward()

        gradients = (latents.grad, target_embed.grad, predictor.conv_in.weight.grad, predictor.head[-1].weight.grad)
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)


if __name__ == "__main__":
    unittest.main()
