import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from utils.audio import save_waveform, waveform_to_pcm16


class AudioHelpersTests(unittest.TestCase):
    def test_pcm_conversion_interleaves_stereo(self) -> None:
        stereo = torch.tensor([[1.0, 0.0], [-1.0, 0.5]])
        pcm = waveform_to_pcm16(stereo)
        self.assertEqual(pcm.shape, (2, 2))
        np.testing.assert_array_equal(pcm[0], np.array([32767, -32767], dtype=np.int16))

    def test_wav_header_preserves_stereo_and_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_waveform(Path(directory) / "sample.wav", torch.zeros(1, 2, 48), 48_000)
            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getframerate(), 48_000)
                self.assertEqual(wav_file.getnframes(), 48)


if __name__ == "__main__":
    unittest.main()
