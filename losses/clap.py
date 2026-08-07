import math
from numbers import Integral

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ClapFeatureExtractor, ClapModel, RobertaTokenizer


DEFAULT_CLAP_MODEL_ID = "laion/clap-htsat-unfused"
DEFAULT_CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
DEFAULT_CLAP_MAX_LENGTH_S = 10.0


def audio_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    r"""Convert a batched waveform to mono without detaching it from autograd.

    Args:
        waveform (`torch.Tensor`): Mono audio of shape `(batch_size, num_samples)` or channel-first
            audio of shape `(batch_size, num_channels, num_samples)`.

    Returns:
        `torch.Tensor`: Mono audio of shape `(batch_size, num_samples)`. Channel-first audio is
        downmixed by taking the arithmetic mean over its channels.
    """
    if not torch.is_tensor(waveform):
        raise TypeError(f"`waveform` has to be a torch.Tensor but is {type(waveform)}")
    if waveform.ndim not in (2, 3):
        raise ValueError(
            "`waveform` has to have shape `(batch_size, num_samples)` or "
            f"`(batch_size, num_channels, num_samples)`, but has shape {tuple(waveform.shape)}"
        )
    if waveform.shape[0] == 0:
        raise ValueError("`waveform` must contain at least one batch item")
    if waveform.shape[-1] == 0:
        raise ValueError("`waveform` must contain at least one audio sample")

    if waveform.ndim == 3:
        if waveform.shape[1] == 0:
            raise ValueError("`waveform` must contain at least one audio channel")
        waveform = waveform.mean(dim=1)

    return waveform


class ClapLoss(nn.Module):
    r"""
    Cosine-distance loss in the CLAP audio-text embedding space.

    Given CLAP audio embeddings of a generated waveform and a text describing the target concept,
    returns `1 - cosine_similarity(audio_embeds, text_embeds)`, so that minimizing the loss pulls
    the generated audio towards the text.

    Both towers are frozen: gradients only flow back through the waveform that produced
    `audio_embeds`. Use `encode_audio` to obtain those embeddings differentiably.

    Args:
        text_encoder (`~transformers.ClapModel`): Frozen, external CLAP model used to embed both
            text and audio. It must be independent from the generative model's text encoder.
        tokenizer (`~transformers.RobertaTokenizer`): Tokenizer matching `text_encoder`.
        feature_extractor (`~transformers.ClapFeatureExtractor`): Feature extractor matching
            `text_encoder`. Only its parameters are used, so that `encode_audio` can reproduce it
            with differentiable ops; it is never called.
        reduction (`str`, *optional*, defaults to `"mean"`): How to reduce the per-item losses. One
            of `"mean"`, `"sum"` or `"none"`.
    """

    def __init__(
        self,
        text_encoder: ClapModel,
        tokenizer: RobertaTokenizer,
        feature_extractor: ClapFeatureExtractor,
        reduction: str = "mean",
    ):
        super().__init__()

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"`reduction` has to be one of 'mean', 'sum' or 'none' but is {reduction}")

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.reduction = reduction

        # both towers only produce embeddings, they are never updated
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # `ClapFeatureExtractor` builds these with numpy, so mirror them as buffers to keep
        # `encode_audio` on the autograd graph. `mel_filters_slaney` is the bank the reference
        # implementation uses for every truncation mode except `"fusion"`, and CLAP's
        # `window_function(n, "hann")` is `np.hanning(n + 1)[:n]`, i.e. a periodic Hann window
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(feature_extractor.mel_filters_slaney).float(),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(feature_extractor.fft_window_size, periodic=True),
            persistent=False,
        )

        # Built on first use, since the source rate is known only per call.
        # Each cache entry is a CPU/double polyphase bank plus integer sample
        # offsets; no zero-stuffed waveform is ever materialized.
        self._resample_kernels: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        reduction: str = "mean",
        **model_kwargs,
    ) -> "ClapLoss":
        r"""Load all parts of an external CLAP loss from one pretrained model identifier.

        Args:
            model_id (`str`): Hugging Face model identifier or local directory containing a CLAP
                model, matching tokenizer and matching feature extractor.
            reduction (`str`, *optional*, defaults to `"mean"`): Loss reduction passed to the
                constructor.
            **model_kwargs: Additional keyword arguments passed to `ClapModel.from_pretrained`.
                Common repository-loading arguments such as `cache_dir`, `revision`, `token` and
                `local_files_only` are also forwarded to the tokenizer and feature extractor.

        Returns:
            `ClapLoss`: A loss whose audio and text towers share the requested CLAP embedding
            space. The model is frozen by the constructor.
        """
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("`model_id` has to be a non-empty string")

        shared_loading_keys = ("cache_dir", "force_download", "local_files_only", "revision", "token")
        shared_loading_kwargs = {
            key: model_kwargs[key] for key in shared_loading_keys if key in model_kwargs
        }
        text_encoder = ClapModel.from_pretrained(model_id, **model_kwargs)
        tokenizer = RobertaTokenizer.from_pretrained(model_id, **shared_loading_kwargs)
        feature_extractor = ClapFeatureExtractor.from_pretrained(model_id, **shared_loading_kwargs)
        return cls(text_encoder, tokenizer, feature_extractor, reduction=reduction)

    @staticmethod
    def _validate_sampling_rate(sampling_rate: int, name: str) -> int:
        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, Integral) or sampling_rate <= 0:
            raise ValueError(f"`{name}` has to be a positive integer but is {sampling_rate!r}")
        return int(sampling_rate)

    @staticmethod
    def _build_resample_kernel(up: int, down: int, half_width: int = 32) -> torch.Tensor:
        r"""
        Builds the windowed-sinc lowpass of a rational `up / down` resampler.

        The filter runs at the zero-stuffed rate and has to suppress everything above the lower of
        the two Nyquist frequencies: the images that zero-stuffing creates when upsampling, and the
        content that would alias when decimating.

        Args:
            up (`int`): Zero-stuffing factor, i.e. the numerator of the rate ratio.
            down (`int`): Decimation factor, i.e. the denominator of the rate ratio.
            half_width (`int`, *optional*, defaults to 32): Number of periods of the sinc kept on
                each side of its centre. Wider is sharper; 32 gives a conservative transition band
                for common audio sampling-rate ratios.

        Returns:
            `torch.Tensor`: Kernel of shape `(2 * half_width * max(up, down) + 1,)`, odd-length so
            the group delay is an integer number of samples.
        """
        stride = max(up, down)
        num_taps = 2 * half_width * stride + 1

        positions = torch.arange(num_taps, dtype=torch.float64) - (num_taps - 1) / 2
        kernel = torch.sinc(positions / stride)
        kernel = kernel * torch.hamming_window(num_taps, periodic=False, dtype=torch.float64)

        # the `up - 1` inserted zeros divide the passband gain by `up`, so normalize it back
        return kernel / kernel.sum() * up

    @classmethod
    def _build_polyphase_bank(
        cls,
        up: int,
        down: int,
        half_width: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor the windowed-sinc filter into ``up`` compact phases."""

        kernel = cls._build_resample_kernel(up, down, half_width)
        center = kernel.numel() // 2
        # A common offset range keeps phase selection vectorizable. Entries
        # outside the prototype filter are exactly zero.
        radius = math.ceil((center + up - 1) / up) + 1
        offsets = torch.arange(-radius, radius + 1, dtype=torch.long)
        phases = torch.arange(up, dtype=torch.long).unsqueeze(1)
        prototype_indices = center + offsets.unsqueeze(0) * up - phases
        valid = (prototype_indices >= 0) & (prototype_indices < kernel.numel())
        bank = torch.zeros(up, offsets.numel(), dtype=torch.float64)
        bank[valid] = kernel[prototype_indices[valid]]
        return offsets, bank

    def _resample(self, waveform: torch.Tensor, orig_rate: int, target_rate: int) -> torch.Tensor:
        r"""
        Band-limited rational resampling, differentiably with respect to `waveform`.

        Args:
            waveform (`torch.Tensor`): Waveform of shape `(batch_size, num_samples)`.
            orig_rate (`int`): Sampling rate of `waveform`.
            target_rate (`int`): Sampling rate to resample to.

        Returns:
            `torch.Tensor`: Waveform of shape `(batch_size, num_samples * up // down)`.
        """
        orig_rate = self._validate_sampling_rate(orig_rate, "orig_rate")
        target_rate = self._validate_sampling_rate(target_rate, "target_rate")

        if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[-1] == 0:
            raise ValueError(
                "`waveform` passed to `_resample` has to be non-empty and have shape "
                f"`(batch_size, num_samples)`, but has shape {tuple(waveform.shape)}"
            )

        divisor = math.gcd(orig_rate, target_rate)
        up, down = target_rate // divisor, orig_rate // divisor

        if up == 1 and down == 1:
            return waveform

        if (up, down) not in self._resample_kernels:
            self._resample_kernels[(up, down)] = self._build_polyphase_bank(up, down)
        offsets_cpu, bank_cpu = self._resample_kernels[(up, down)]
        offsets = offsets_cpu.to(device=waveform.device)
        bank = bank_cpu.to(device=waveform.device, dtype=waveform.dtype)

        _, num_samples = waveform.shape
        output_samples = (num_samples * up + down - 1) // down
        # Chunking bounds the temporary [batch, output, taps] gather. For the
        # common 44.1 -> 48 kHz case this replaces a 160x zero-stuffed signal
        # with roughly 65 source samples per output point.
        chunk_size = 16_384
        output_chunks = []
        for start in range(0, output_samples, chunk_size):
            stop = min(start + chunk_size, output_samples)
            output_index = torch.arange(start, stop, device=waveform.device, dtype=torch.long)
            positions = output_index * down
            centers = torch.div(positions, up, rounding_mode="floor")
            phases = torch.remainder(positions, up)
            sample_indices = centers.unsqueeze(1) + offsets.unsqueeze(0)
            valid = (sample_indices >= 0) & (sample_indices < num_samples)
            safe_indices = sample_indices.clamp(0, num_samples - 1)
            samples = waveform[:, safe_indices]
            coefficients = bank[phases] * valid.to(bank.dtype)
            output_chunks.append((samples * coefficients.unsqueeze(0)).sum(dim=-1))
        return torch.cat(output_chunks, dim=-1)

    def encode_audio(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        r"""
        Embeds `waveform` with the CLAP audio tower, differentiably with respect to `waveform`.

        Reproduces the pinned `ClapFeatureExtractor` repeat-padding, STFT and mel preprocessing for
        inputs within its fixed window, using `torch` ops so the waveform stays on the graph.
        Longer inputs are rejected instead of applying the reference processor's nondeterministic
        `truncation="rand_trunc"` policy.

        Resampling is band-limited (`_resample`). A cheaper interpolation can leave signal-dependent
        spectral images in the upper part of CLAP's mel bank and corrupt the embedding.

        Args:
            waveform (`torch.Tensor`): Mono waveform of shape `(batch_size, num_samples)` or
                channel-first waveform of shape `(batch_size, num_channels, num_samples)`. Multiple
                channels are downmixed differentiably before resampling.
            sampling_rate (`int`): Positive sampling rate of `waveform` in Hz.

        Returns:
            `torch.Tensor`: Normalized audio embeddings of shape `(batch_size, embed_dim)`.
        """
        waveform = audio_to_mono(waveform)
        sampling_rate = self._validate_sampling_rate(sampling_rate, "sampling_rate")
        target_rate = self._validate_sampling_rate(
            self.feature_extractor.sampling_rate, "feature_extractor.sampling_rate"
        )

        # `torch.stft` needs float32; the tower's own dtype is matched at its input in step 5
        waveform = waveform.to(self.window.dtype)

        # 1. resample to the rate the mel filters were built for
        waveform = self._resample(waveform, sampling_rate, target_rate)

        # 2. repeat-pad to the fixed window the audio tower expects. the reference
        # implementation tiles `floor(max_samples / num_samples)` times and zero-pads the remainder,
        # rather than tiling once more and truncating, so a partial final repeat becomes silence
        num_samples = waveform.shape[-1]
        max_samples = self.feature_extractor.nb_max_samples
        if num_samples > max_samples:
            max_seconds = max_samples / target_rate
            raise ValueError(
                "CLAP audio exceeds its fixed input window: "
                f"received {num_samples} samples at {target_rate} Hz, but this processor accepts "
                f"at most {max_samples} samples ({max_seconds:g} seconds). "
                "Crop deliberately before calling encode_audio or use a documented window aggregation policy."
            )
        if num_samples < max_samples:
            waveform = waveform.repeat(1, max_samples // num_samples)
            waveform = F.pad(waveform, (0, max_samples - waveform.shape[-1]))
        else:
            # Exact-window input needs no padding or crop.  Longer inputs are
            # rejected above so no part of a generated clip is silently
            # omitted from the differentiable objective or evaluation metric.
            waveform = waveform[:, :max_samples]

        # 3. power spectrogram. `real ** 2 + imag ** 2` instead of `abs() ** 2` because `abs()` is
        # not differentiable at zero magnitude and would produce NaN gradients on silent bins
        stft = torch.stft(
            waveform,
            n_fft=self.feature_extractor.fft_window_size,
            hop_length=self.feature_extractor.hop_length,
            win_length=self.feature_extractor.fft_window_size,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.real.pow(2) + stft.imag.pow(2)

        # 4. mel projection and conversion to decibels, i.e. `10 * log10(power / reference)` with the
        # reference of 1.0 and the floor of 1e-10 the reference implementation uses
        mel = (self.mel_filters.transpose(0, 1) @ power).clamp(min=1e-10)
        log_mel = 10.0 * torch.log10(mel)

        # 5. `(batch_size, 1, num_frames, num_mel_bins)`, as returned by the feature extractor. the
        # spectrogram is always computed in float32, so match the tower's dtype only at its input
        input_features = log_mel.transpose(-1, -2).unsqueeze(1)
        input_features = input_features.to(next(self.text_encoder.parameters()).dtype)

        # `is_longer` is only read when the audio tower was trained with feature fusion, which
        # `laion/clap-htsat-unfused` was not
        audio_embeds = self.text_encoder.get_audio_features(input_features=input_features, is_longer=None)

        # transformers >= 5 returns a `BaseModelOutputWithPooling`, earlier versions a plain tensor
        if not torch.is_tensor(audio_embeds):
            audio_embeds = audio_embeds.pooler_output

        return F.normalize(audio_embeds, dim=-1)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        r"""
        Embeds `text` with the CLAP text tower and L2-normalizes the result.

        Args:
            text (`list[str]`): The texts to embed.

        Returns:
            `torch.Tensor`: Normalized text embeddings of shape `(len(text), embed_dim)`.
        """
        device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeds = self.text_encoder.get_text_features(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            )

        # transformers >= 5 returns a `BaseModelOutputWithPooling`, earlier versions a plain tensor
        if not torch.is_tensor(text_embeds):
            text_embeds = text_embeds.pooler_output

        return F.normalize(text_embeds, dim=-1)

    def forward(self, audio_embeds: torch.Tensor, text: str | list[str]) -> torch.Tensor:
        r"""
        Args:
            audio_embeds (`torch.Tensor`): CLAP audio embeddings of shape `(batch_size, embed_dim)`.
            text (`str` or `list[str]`): The target concept(s). A single `str` is used for the whole
                batch, a list must match the batch size of `audio_embeds`.

        Returns:
            `torch.Tensor`: `1 - cosine_similarity`, reduced according to `reduction`. A scalar
            unless `reduction` is `"none"`, in which case the shape is `(batch_size,)`.
        """
        if audio_embeds.ndim != 2:
            raise ValueError(
                f"`audio_embeds` has to be of shape `(batch_size, embed_dim)` but has {audio_embeds.ndim} dimensions"
            )

        batch_size = audio_embeds.shape[0]

        if isinstance(text, str):
            texts = [text] * batch_size
        elif isinstance(text, list):
            if len(text) == 1:
                texts = text * batch_size
            elif len(text) != batch_size:
                raise ValueError(
                    f"`text` has batch size {len(text)}, but `audio_embeds` has batch size {batch_size}. Please make"
                    " sure that passed `text` matches the batch size of `audio_embeds`."
                )
            else:
                texts = text
        else:
            raise ValueError(f"`text` has to be of type `str` or `list` but is {type(text)}")

        text_embeds = self.encode_text(texts).to(device=audio_embeds.device, dtype=audio_embeds.dtype)

        cosine_similarity = (F.normalize(audio_embeds, dim=-1) * text_embeds).sum(dim=-1)
        loss = 1.0 - cosine_similarity

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


__all__ = [
    "DEFAULT_CLAP_MAX_LENGTH_S",
    "DEFAULT_CLAP_MODEL_ID",
    "DEFAULT_CLAP_REVISION",
    "ClapLoss",
    "audio_to_mono",
]
