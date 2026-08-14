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


def load_waveform(path: Path) -> tuple[np.ndarray, int]:
    r"""Reads the 16-bit PCM WAV format written by :func:`save_waveform`.

    Returns audio as float32 with shape ``(channels, samples)`` and its sampling rate.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16-bit PCM but has sample width {wav_file.getsampwidth()} bytes")
        if wav_file.getcomptype() != "NONE":
            raise ValueError(f"{path} must be uncompressed PCM but uses {wav_file.getcomptype()!r}")
        channels = wav_file.getnchannels()
        sampling_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    pcm = np.frombuffer(frames, dtype="<i2")
    if channels < 1 or pcm.size % channels != 0:
        raise ValueError(f"{path} has an invalid interleaved channel layout")
    waveform = pcm.reshape(-1, channels).T.astype(np.float32) / 32767.0
    return waveform, int(sampling_rate)
