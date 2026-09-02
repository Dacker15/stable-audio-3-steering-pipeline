r"""
LPAPS: a perceptual audio distance analogous to LPIPS, computed in the feature space of a VGG16-style
classifier ("VGGish-ish") trained from scratch on VGGSound. Ported from Iashin & Rahtu, 2021 (*Taming
Visually Guided Sound Generation*), `github.com/v-iashin/SpecVQGAN`
(`specvqgan/modules/losses/lpaps.py`, `specvqgan/modules/losses/vggishish/model.py`).

VGGish-ish is trained on VGGSound (environmental/video sound events), not music. LPAPS here is a
reasonable approximation of preserved perceptual structure, not a metric calibrated for music — see
`docs/metriche-valutazione-steering.md`.

Deliberate deviation from the upstream reference, both to fix and to avoid reproducing a known
checkpoint-loading bug (see `docs/metriche-valutazione-steering.md`): the released
`vggishish16.pt` is not a bare state_dict, it is a training checkpoint with top-level keys
`loss`/`metrics`/`epoch`/`optimizer`/`model`, and the backbone's weights only live under
`checkpoint["model"]`. The upstream reference implementation calls
`load_state_dict(torch.load(ckpt), strict=False)` on the *raw top-level dict* in both places it
loads this checkpoint — a complete key-namespace mismatch (100% of keys land in
missing/unexpected) that `strict=False` silently swallows instead of raising. This module unwraps
`checkpoint["model"]` before loading, verified empirically against the actual downloaded
checkpoint, and raises loudly if the resulting load isn't a clean match. It also fixes the five
per-scale linear calibration layers ("NetLinLayer" in the reference) to a deterministic uniform
channel average rather than leaving them randomly initialized, since this checkpoint ships no
pretrained weights for them at all (upstream's own linear-layer load is likewise a pure no-op,
for the same top-level-dict reason).
"""

import hashlib
from pathlib import Path

import numpy as np
import requests
import torch
from torch import nn

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "musicldm-steering-pipeline" / "lpaps"

_VGGISHISH_URL = "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vggishish16.pt"
_VGGISHISH_MD5 = "197040c524a07ccacf7715d7080a80bd"
_MELSPEC_STATS_URL = (
    "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/"
    "train_means_stds_melspec_10s_22050hz.txt"
)
_MELSPEC_STATS_MD5 = "f449c6fd0e248936c16f6d22492bb625"

# specvqgan/modules/losses/vggishish/model.py: VGG16-style conv stack (no batchnorm), 'MP' entries
# are MaxPool2d(kernel=2, stride=2), everything else is Conv2d(kernel=3, padding=1, stride=1) + ReLU.
_VGGISHISH_CONV_LAYERS: tuple[int | str, ...] = (
    64, 64, "MP", 128, 128, "MP", 256, 256, 256, "MP", 512, 512, 512, "MP", 512, 512, 512,
)
_VGGISHISH_NUM_CLASSES = 309  # VGGSound class count; the head is unused but kept so checkpoint
# state_dict keys match exactly (an empty `unexpected_keys` on load requires this shape).

# Index ranges of `VGGishish.features` matching relu1_2/relu2_2/relu3_3/relu4_3/relu5_3, from
# specvqgan/modules/losses/lpaps.py's `vggishish16` wrapper class.
_FEATURE_SLICES = ((0, 4), (4, 9), (9, 16), (16, 23), (23, 30))
_FEATURE_CHANNELS = (64, 128, 256, 512, 512)

# feature_extraction/extract_mel_spectrogram.py's `TRANSFORMS` pipeline, minus the fixed-length
# `TrimSpec(860)` step: VGGish-ish pools adaptively (`AdaptiveAvgPool2d((5, 10))`), so it does not
# require a fixed number of mel frames, and base/method clips here are not guaranteed to already be
# frame-identical the way SpecVQGAN's own fixed-duration training clips were.
MEL_SAMPLING_RATE = 22050
_MEL_N_FFT = 1024
_MEL_HOP_LENGTH = 256
_MEL_N_MELS = 80
_MEL_FMIN = 125
_MEL_FMAX = 7600


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_download(url: str, expected_md5: str, path: Path) -> Path:
    if not path.is_file() or _md5(path) != expected_md5:
        print(f"Downloading {url} to {path}")
        _download(url, path)
        actual_md5 = _md5(path)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"downloaded {path} has MD5 {actual_md5!r}, expected {expected_md5!r}")
    return path


class _VGGishishBackbone(nn.Module):
    """Ported from `specvqgan/modules/losses/vggishish/model.py`. Only `.features` (the conv
    stack) is used by LPAPS; `avgpool`/`classifier` exist solely so the pretrained checkpoint's
    state_dict keys match exactly, keeping `load_state_dict`'s missing/unexpected key check clean."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for value in _VGGISHISH_CONV_LAYERS:
            if value == "MP":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, value, kernel_size=3, padding=1, stride=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = value
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 10))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 5 * 10, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, _VGGISHISH_NUM_CLASSES),
        )


class LPAPS(nn.Module):
    r"""Perceptual distance between two waveforms in VGGish-ish feature space (lower = more similar).

    Both waveforms are converted to a mono 22050 Hz mel-spectrogram, passed through the frozen
    backbone, and compared at five feature scales (L2-normalized per channel, squared difference,
    uniformly-weighted channel average, spatial average), summed across scales.
    """

    def __init__(self, checkpoint_path: Path, mean_std_path: Path):
        super().__init__()
        backbone = _VGGishishBackbone()
        # MD5-verified by `_cached_download` before this runs, so the pickle is trusted; the
        # checkpoint predates `weights_only=True` becoming the torch>=2.6 default, contains plain
        # numpy scalars, and its "optimizer" entry needs `omegaconf` installed to unpickle.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # This is the actual bug the spec requires guarding against: the checkpoint's top-level
        # dict has keys `loss`/`metrics`/`epoch`/`optimizer`/`model`, and the model's own weights
        # only live under `checkpoint["model"]`. Passing the raw top-level dict straight to
        # `load_state_dict(..., strict=False)`, as the upstream reference implementation does, is
        # a complete key-namespace mismatch (100% unexpected/missing keys) that is silently
        # swallowed rather than raised.
        state_dict = checkpoint["model"]
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"LPAPS backbone checkpoint {checkpoint_path} did not load cleanly: "
                f"missing_keys={missing}, unexpected_keys={unexpected}. Refusing to silently score "
                "with an incorrectly initialized perceptual network (see the strict=False checkpoint "
                "bug documented in docs/metriche-valutazione-steering.md)."
            )
        backbone.requires_grad_(False)
        backbone.eval()

        children = list(backbone.features.children())
        self.slices = nn.ModuleList(
            nn.Sequential(*children[start:end]) for start, end in _FEATURE_SLICES
        )

        # No pretrained weights exist for these in the checkpoint (see module docstring), so each
        # is fixed to a deterministic uniform channel average rather than left randomly initialized.
        self.linear_layers = nn.ModuleList()
        for channels in _FEATURE_CHANNELS:
            layer = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
            layer.weight.data.fill_(1.0 / channels)
            layer.requires_grad_(False)
            self.linear_layers.append(layer)

        means, stds = np.loadtxt(mean_std_path, dtype=np.float32).T
        # stats are given for a [0, 1]-normalized mel-spectrogram, rescaled here to [-1, 1]
        means = 2.0 * means - 1.0
        stds = 2.0 * stds
        self.register_buffer("_shift", torch.from_numpy(means)[None, None, :, None])
        self.register_buffer("_scale", torch.from_numpy(stds)[None, None, :, None])

        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, cache_dir: Path | None = None) -> "LPAPS":
        cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        checkpoint_path = _cached_download(_VGGISHISH_URL, _VGGISHISH_MD5, cache_dir / "vggishish16.pt")
        mean_std_path = _cached_download(
            _MELSPEC_STATS_URL, _MELSPEC_STATS_MD5, cache_dir / "train_means_stds_melspec_10s_22050hz.txt"
        )
        return cls(checkpoint_path, mean_std_path)

    def _melspectrogram(self, waveform: np.ndarray, sampling_rate: int) -> np.ndarray:
        import librosa

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("waveform must contain mono samples or have shape (channels, samples)")
        if sampling_rate != MEL_SAMPLING_RATE:
            waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=MEL_SAMPLING_RATE)

        magnitude = np.abs(librosa.stft(waveform, n_fft=_MEL_N_FFT, hop_length=_MEL_HOP_LENGTH))
        mel_basis = librosa.filters.mel(
            sr=MEL_SAMPLING_RATE, n_fft=_MEL_N_FFT, fmin=_MEL_FMIN, fmax=_MEL_FMAX, n_mels=_MEL_N_MELS
        )
        mel = mel_basis @ magnitude
        # feature_extraction/extract_mel_spectrogram.py's TRANSFORMS: floor, log10, then an affine
        # remap of the resulting dB-like range into [0, 1]
        mel = np.maximum(1e-5, mel)
        mel = np.log10(mel) * 20.0 - 20.0 + 100.0
        return np.clip(mel / 100.0, 0.0, 1.0)

    def distance(self, waveform_a: np.ndarray, waveform_b: np.ndarray, sampling_rate: int) -> float:
        mel_a = self._melspectrogram(waveform_a, sampling_rate)
        mel_b = self._melspectrogram(waveform_b, sampling_rate)
        # base and method clips are generated from independently resolved audio_length_in_s, so
        # their frame counts can differ by a handful of frames; align to the shorter one
        num_frames = min(mel_a.shape[-1], mel_b.shape[-1])
        mel_a, mel_b = mel_a[:, :num_frames], mel_b[:, :num_frames]

        device = self._shift.device
        tensor_a = torch.from_numpy(mel_a).to(device)[None, None]
        tensor_b = torch.from_numpy(mel_b).to(device)[None, None]

        with torch.inference_mode():
            scaled_a = (tensor_a - self._shift) / self._scale
            scaled_b = (tensor_b - self._shift) / self._scale

            total = torch.zeros((1, 1), device=device)
            for slice_module, linear_layer in zip(self.slices, self.linear_layers):
                scaled_a = slice_module(scaled_a)
                scaled_b = slice_module(scaled_b)
                norm_a = scaled_a / (scaled_a.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10)
                norm_b = scaled_b / (scaled_b.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10)
                diff = (norm_a - norm_b).pow(2)
                total = total + linear_layer(diff).mean(dim=(2, 3))

        return float(total.item())
