import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    raise unittest.SkipTest("PyTorch and transformers are required for CLAP audio tests")

import numpy as np
import torch
from torch import nn

from losses.clap import ClapLoss, audio_to_mono


class _FakeClapModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def get_audio_features(self, input_features: torch.Tensor, is_longer=None) -> torch.Tensor:
        del is_longer
        mean = input_features.mean(dim=(1, 2, 3))
        energy = input_features.square().mean(dim=(1, 2, 3))
        return torch.stack((mean, energy), dim=-1) * self.anchor


def _fake_feature_extractor(sampling_rate: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_rate=sampling_rate,
        fft_window_size=4,
        hop_length=2,
        nb_max_samples=16,
        mel_filters_slaney=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


class AudioToMonoTests(unittest.TestCase):
    def test_mono_is_returned_unchanged(self) -> None:
        waveform = torch.randn(2, 8, requires_grad=True)

        result = audio_to_mono(waveform)

        self.assertIs(result, waveform)

    def test_channel_downmix_is_differentiable(self) -> None:
        waveform = torch.tensor(
            [[[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]]],
            requires_grad=True,
        )

        mono = audio_to_mono(waveform)
        mono.sum().backward()

        self.assertTrue(torch.equal(mono, torch.tensor([[2.0, 4.0, 6.0]])))
        self.assertTrue(torch.equal(waveform.grad, torch.full_like(waveform, 0.5)))

    def test_invalid_or_empty_shapes_are_rejected(self) -> None:
        invalid = (
            torch.zeros(8),
            torch.zeros(1, 1, 1, 8),
            torch.zeros(0, 8),
            torch.zeros(1, 0),
            torch.zeros(1, 0, 8),
        )

        for waveform in invalid:
            with self.subTest(shape=tuple(waveform.shape)):
                with self.assertRaises(ValueError):
                    audio_to_mono(waveform)


class ClapAudioTests(unittest.TestCase):
    def _loss(self, sampling_rate: int = 16) -> ClapLoss:
        return ClapLoss(_FakeClapModel(), object(), _fake_feature_extractor(sampling_rate))

    def test_encode_audio_accepts_channel_first_and_matches_explicit_downmix(self) -> None:
        loss = self._loss()
        stereo = torch.randn(2, 2, 16, requires_grad=True)

        stereo_embedding = loss.encode_audio(stereo, sampling_rate=16)
        mono_embedding = loss.encode_audio(stereo.mean(dim=1), sampling_rate=16)
        stereo_embedding.sum().backward()

        self.assertEqual(stereo_embedding.shape, (2, 2))
        self.assertTrue(torch.allclose(stereo_embedding, mono_embedding))
        self.assertIsNotNone(stereo.grad)
        self.assertTrue(torch.isfinite(stereo.grad).all())

    def test_resampling_is_differentiable_and_handles_short_audio(self) -> None:
        loss = self._loss()
        waveform = torch.linspace(-1.0, 1.0, 8).unsqueeze(0).requires_grad_()

        resampled = loss._resample(waveform, orig_rate=16, target_rate=8)
        resampled.square().sum().backward()

        self.assertEqual(resampled.shape, (1, 4))
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())
        self.assertGreater(waveform.grad.abs().sum().item(), 0.0)

    def test_polyphase_resampling_handles_44100_to_48000_without_zero_stuffing(self) -> None:
        loss = self._loss(sampling_rate=48_000)
        waveform = torch.randn(1, 4_410, requires_grad=True)

        resampled = loss._resample(waveform, orig_rate=44_100, target_rate=48_000)
        resampled.square().mean().backward()

        self.assertEqual(resampled.shape, (1, 4_800))
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())
        self.assertGreater(waveform.grad.abs().sum().item(), 0.0)

    def test_invalid_sampling_rates_are_rejected(self) -> None:
        loss = self._loss()
        waveform = torch.zeros(1, 16)

        for rate in (0, -1, 16.0, True):
            with self.subTest(rate=rate):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    loss.encode_audio(waveform, sampling_rate=rate)

    def test_audio_longer_than_complete_clap_window_is_rejected(self) -> None:
        loss = self._loss()

        # The fake processor's exact 16-sample window is accepted; one sample
        # more must fail rather than silently excluding the tail.
        exact = loss.encode_audio(torch.zeros(1, 16), sampling_rate=16)
        self.assertEqual(exact.shape, (1, 2))
        with self.assertRaisesRegex(ValueError, "fixed input window"):
            loss.encode_audio(torch.zeros(1, 17), sampling_rate=16)

    def test_from_pretrained_loads_a_separate_matching_clap_bundle(self) -> None:
        model = _FakeClapModel()
        tokenizer = object()
        feature_extractor = _fake_feature_extractor()

        with (
            patch("losses.clap.ClapModel.from_pretrained", return_value=model) as load_model,
            patch("losses.clap.RobertaTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
            patch(
                "losses.clap.ClapFeatureExtractor.from_pretrained",
                return_value=feature_extractor,
            ) as load_feature_extractor,
        ):
            loss = ClapLoss.from_pretrained(
                "local/clap", reduction="none", local_files_only=True, torch_dtype=torch.float32
            )

        load_model.assert_called_once_with(
            "local/clap", local_files_only=True, torch_dtype=torch.float32
        )
        load_tokenizer.assert_called_once_with("local/clap", local_files_only=True)
        load_feature_extractor.assert_called_once_with("local/clap", local_files_only=True)
        self.assertIs(loss.text_encoder, model)
        self.assertIs(loss.tokenizer, tokenizer)
        self.assertIs(loss.feature_extractor, feature_extractor)
        self.assertEqual(loss.reduction, "none")
        self.assertFalse(next(loss.text_encoder.parameters()).requires_grad)


if __name__ == "__main__":
    unittest.main()
