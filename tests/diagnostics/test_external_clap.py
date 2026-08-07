"""Opt-in semantic sanity check for the external CLAP model.

The test deliberately uses user-supplied real recordings rather than a synthetic
waveform: a synthetic harmonic signal is not a reliable proxy for an instrument
class.  Nothing is downloaded or decoded during normal unit-test collection.

Run with, for example::

    RUN_CLAP_DIAGNOSTIC=1 \
    CLAP_TRUMPET_AUDIO=/path/to/trumpet.wav \
    CLAP_NON_TRUMPET_AUDIO=/path/to/non_trumpet.wav \
    pytest -m clap tests/diagnostics/test_external_clap.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.diagnostic,
    pytest.mark.external_model,
    pytest.mark.clap,
]


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _required_audio_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} must point to a real audio file when RUN_CLAP_DIAGNOSTIC=1")
    path = Path(value).expanduser()
    if not path.is_file():
        pytest.fail(f"{variable} does not point to a file: {path}")
    return path


def _load_audio(path: Path):
    import soundfile as sf
    import torch

    samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if samples.shape[0] == 0:
        pytest.fail(f"Audio fixture is empty: {path}")
    # soundfile is sample-first; ClapLoss accepts channel-first [B, C, S].
    return torch.from_numpy(samples.T).unsqueeze(0), int(sample_rate)


@pytest.mark.skipif(
    not _enabled("RUN_CLAP_DIAGNOSTIC"),
    reason="set RUN_CLAP_DIAGNOSTIC=1 and provide the two audio fixture paths",
)
def test_external_clap_ranks_trumpet_above_non_trumpet() -> None:
    """A trumpet recording must match ``trumpet`` better than a control clip."""

    import torch

    from losses import DEFAULT_CLAP_REVISION, ClapLoss

    trumpet_path = _required_audio_path("CLAP_TRUMPET_AUDIO")
    control_path = _required_audio_path("CLAP_NON_TRUMPET_AUDIO")
    model_id = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
    revision = os.environ.get("CLAP_REVISION", DEFAULT_CLAP_REVISION)
    device = torch.device(
        os.environ.get("CLAP_DIAGNOSTIC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )

    clap = ClapLoss.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float32
    ).to(device).eval()
    target = clap.encode_text(["a music recording featuring a trumpet"]).to(device)

    def similarity(path: Path) -> float:
        waveform, sample_rate = _load_audio(path)
        with torch.no_grad():
            embedding = clap.encode_audio(waveform.to(device), sample_rate)
            return float((embedding * target).sum(dim=-1).item())

    trumpet_similarity = similarity(trumpet_path)
    control_similarity = similarity(control_path)
    margin = float(os.environ.get("CLAP_DIAGNOSTIC_MARGIN", "0.0"))

    assert trumpet_similarity > control_similarity + margin, (
        f"CLAP semantic diagnostic failed: trumpet={trumpet_similarity:.6f}, "
        f"control={control_similarity:.6f}, required_margin={margin:.6f}"
    )
