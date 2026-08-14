import importlib.util
import unittest


REQUIRED = ("torch", "torchaudio", "transformers")
if any(importlib.util.find_spec(module) is None for module in REQUIRED):
    raise unittest.SkipTest("PyTorch, torchaudio and transformers are required for CLAP resampling tests")

import torch

from losses.clap import ClapLoss


class ClapResampleTests(unittest.TestCase):
    def test_resample_is_differentiable_and_has_expected_length(self) -> None:
        waveform = torch.randn(2, 4_410, requires_grad=True)

        resampled = ClapLoss._resample(waveform, 44_100, 48_000)
        resampled.square().mean().backward()

        self.assertEqual(resampled.shape, (2, 4_800))
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())
        self.assertGreater(float(waveform.grad.abs().sum()), 0.0)

    def test_equal_rates_are_identity_and_invalid_rates_are_rejected(self) -> None:
        waveform = torch.randn(1, 32)
        self.assertIs(ClapLoss._resample(waveform, 48_000, 48_000), waveform)
        with self.assertRaisesRegex(ValueError, "sampling rates must be positive"):
            ClapLoss._resample(waveform, 0, 48_000)


if __name__ == "__main__":
    unittest.main()
