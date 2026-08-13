import wave
from pathlib import Path

import numpy as np


def save_waveform(path: Path, waveform: np.ndarray, sampling_rate: int) -> None:
    r"""
    Writes float audio as clipped signed 16-bit PCM, using only the standard library.

    Args:
        path (`Path`): Destination `.wav` file. Parent directories are created.
        waveform (`np.ndarray`): Audio of shape `(channels, samples)`, or `(samples,)` for mono.
        sampling_rate (`int`): Sampling rate to write into the header.
    """
    waveform = np.asarray(waveform)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim != 2:
        raise ValueError(f"`waveform` has to be of shape `(channels, samples)` but has {waveform.ndim} dimensions")

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    frames = np.ascontiguousarray(pcm.T)  # (samples, channels), interleaved for wave's frame layout
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(frames.tobytes())
