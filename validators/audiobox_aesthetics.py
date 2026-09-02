r"""
No-reference audio quality scoring via Meta's Audiobox Aesthetics (Tjandra et al., 2025,
arXiv 2502.05139): a WavLM-based encoder with four MLP heads predicting Content Enjoyment (CE),
Content Usefulness (CU), Production Complexity (PC) and Production Quality (PQ). Unlike LPAPS, this
needs no paired reference audio, so it is meaningful on `base` as well as on a steering method.
"""

import numpy as np

DEFAULT_MODEL_ID = "facebook/audiobox-aesthetics"


class AudioboxAestheticsScorer:
    """Wraps `audiobox_aesthetics.infer.AesPredictor`, matching the constructor/`score()` idiom of
    `validators.instrument_classification.AudioSetInstrumentClassifier`."""

    def __init__(self, device: str = "cpu"):
        import torch
        from audiobox_aesthetics.infer import initialize_predictor

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA audiobox-aesthetics device requested but CUDA is unavailable")

        # `initialize_predictor` auto-selects cuda > mps > cpu with no device argument of its own;
        # the predictor's `device`/`model` are plain public attributes, so pin them explicitly to
        # stay consistent with the rest of the evaluation run instead of relying on autodetection.
        self.predictor = initialize_predictor()
        self.predictor.device = resolved_device
        self.predictor.model.to(resolved_device)
        self.device = resolved_device
        self.model_id = DEFAULT_MODEL_ID

    def score(self, waveform: np.ndarray, sampling_rate: int) -> dict[str, float]:
        import torch

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        if waveform.ndim != 2 or waveform.size == 0:
            raise ValueError("waveform must contain samples or have shape (channels, samples)")

        tensor = torch.from_numpy(np.ascontiguousarray(waveform))
        result = self.predictor.forward([{"path": tensor, "sample_rate": sampling_rate}])[0]

        ce, cu, pc, pq = float(result["CE"]), float(result["CU"]), float(result["PC"]), float(result["PQ"])
        return {"ce": ce, "cu": cu, "pc": pc, "pq": pq, "mean": (ce + cu + pc + pq) / 4.0}
