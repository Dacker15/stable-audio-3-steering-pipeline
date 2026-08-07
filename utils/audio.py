"""Small, dependency-free WAV helpers used by training and evaluation."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch


def waveform_to_pcm16(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert mono/stereo channel-first audio to interleaved signed PCM16."""

    if torch.is_tensor(waveform):
        waveform = waveform.detach().float().cpu().numpy()
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] not in (1, 2) or array.shape[1] == 0:
        raise ValueError(
            "`waveform` must have shape [samples], [channels, samples], or "
            f"[1, channels, samples] with one/two channels; got {array.shape}"
        )
    return np.rint(np.clip(array.T, -1.0, 1.0) * 32767.0).astype("<i2")


def save_waveform(path: str | Path, waveform: torch.Tensor | np.ndarray, sample_rate: int) -> Path:
    """Write a recoverable PCM16 WAV while preserving ACE-Step stereo output."""

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("`sample_rate` must be a positive integer")
    pcm = waveform_to_pcm16(waveform)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return destination.resolve()


__all__ = ["save_waveform", "waveform_to_pcm16"]
