"""Differentiable target-specific steering for ACE-Step 1.5 SFT.

The official SFT checkpoint predicts a flow-matching velocity over temporal
latents shaped ``[batch, frames, 64]``.  At an active steering step this
module evaluates the same frozen DiT at the same ``x_t`` for the full prompt,
the retain prompt, and ACE-Step's learned null condition.  It interpolates the
two *conditional velocities* and then applies ACE-Step's native APG once::

    v_cond = v_full + alpha_t * (v_retain - v_full)
    v = APG(v_cond, v_null, guidance_scale)
    x_next = x_t - (t_current - t_next) * v

Consequently ``alpha=0`` is the full-prompt trajectory and ``alpha=1`` is the
retain-prompt trajectory.  Qwen embeddings are used only by ACE-Step; the
target conditioning supplied to the steering predictor is a separate frozen
CLAP text embedding.

The high-level upstream generation/decode helpers are intentionally not used:
they run under ``no_grad``/``inference_mode`` and detach the VAE output.  The
DiT is frozen and evaluated under ``no_grad`` here (the documented
``steering_only`` surrogate gradient), while alpha interpolation, Euler
integration, VAE decode, stereo downmix and CLAP remain differentiable.
"""

from __future__ import annotations

import gc
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


ACE_STEP_MODEL_ID = "ACE-Step/acestep-v15-sft"
ACE_STEP_MODEL_REVISION = "c410d249e71ea9385a7b586865e65b1473e1098d"
ACE_STEP_COMPONENTS_ID = "ACE-Step/Ace-Step1.5"
# Short revisions are accepted by the Hub and keep the component bundle used
# by this project reproducible without downloading the optional 5 Hz LM.
ACE_STEP_COMPONENTS_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
ACE_STEP_SAMPLE_RATE = 48_000
ACE_STEP_LATENT_RATE = 25
ACE_STEP_LATENT_CHANNELS = 64

STEERING_MODE = "ace_step_apg_cond_velocity_lerp_canonical_null_v2"
GRADIENT_MODE = "steering_only_surrogate_v1"

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


class MomentumBuffer:
    """ACE-Step APG momentum state, reset for every generated trajectory."""

    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average: torch.Tensor | int = 0

    def update(self, value: torch.Tensor) -> torch.Tensor:
        self.running_average = value + self.momentum * self.running_average
        return self.running_average


def _project(
    value: torch.Tensor,
    direction: torch.Tensor,
    dims: tuple[int, ...] = (1,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``value`` into components parallel/orthogonal to ``direction``.

    This mirrors the APG implementation shipped with the official SFT
    checkpoint.  The projection is evaluated in float64 as upstream does, then
    cast back so FP32 generation stays numerically stable on a Tesla T4.
    """

    dtype = value.dtype
    value64 = value.double()
    direction64 = F.normalize(direction.double(), dim=dims)
    parallel = (value64 * direction64).sum(dim=dims, keepdim=True) * direction64
    orthogonal = value64 - parallel
    return parallel.to(dtype), orthogonal.to(dtype)


def adaptive_projected_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer | None = None,
    *,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Apply the native ACE-Step SFT Adaptive Projected Guidance (APG)."""

    update = pred_cond - pred_uncond
    if momentum_buffer is not None:
        update = momentum_buffer.update(update)

    if norm_threshold > 0:
        update_norm = update.norm(p=2, dim=dims, keepdim=True)
        scale = torch.minimum(torch.ones_like(update_norm), norm_threshold / update_norm.clamp_min(1e-12))
        update = update * scale

    parallel, orthogonal = _project(update, pred_cond, dims)
    normalized_update = orthogonal + eta * parallel
    return pred_cond + (guidance_scale - 1.0) * normalized_update


def classifier_free_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Linear CFG, exposed only as an explicit diagnostic alternative to APG."""

    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


@dataclass
class AceStepConditioning:
    """Frozen ACE-Step conditioning tensors for one prompt/retain batch."""

    full_hidden_states: torch.Tensor
    retain_hidden_states: torch.Tensor
    full_attention_mask: torch.Tensor
    retain_attention_mask: torch.Tensor
    context_latents: torch.Tensor
    latent_attention_mask: torch.Tensor
    num_audio_samples: int

    @property
    def batch_size(self) -> int:
        return self.context_latents.shape[0]

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        batch, frames, context_channels = self.context_latents.shape
        if context_channels % 2:
            raise ValueError("ACE-Step context channels must be source+mask pairs")
        return batch, frames, context_channels // 2

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "AceStepConditioning":
        def move(tensor: torch.Tensor) -> torch.Tensor:
            if torch.is_floating_point(tensor):
                return tensor.to(device=device, dtype=dtype)
            return tensor.to(device=device)

        return replace(
            self,
            full_hidden_states=move(self.full_hidden_states),
            retain_hidden_states=move(self.retain_hidden_states),
            full_attention_mask=move(self.full_attention_mask),
            retain_attention_mask=move(self.retain_attention_mask),
            context_latents=move(self.context_latents),
            latent_attention_mask=move(self.latent_attention_mask),
        )

    def cpu(self) -> "AceStepConditioning":
        return self.to("cpu", self.context_latents.dtype)


@dataclass
class SteeringAudioPipelineOutput:
    """Generated ACE-Step waveform/latents plus the predicted alpha schedule."""

    audios: np.ndarray | torch.Tensor
    alpha_records: list[tuple[float, float]]
    initial_latents: torch.Tensor | None = None


def _as_text_batch(value: str | Sequence[str], name: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise TypeError(f"`{name}` must be a string or a sequence of strings")
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"`{name}` must contain non-empty strings")
    return [item.strip() for item in values]


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class SteeringAceStepPipeline:
    """ACE-Step 1.5 SFT flow sampler with target-specific alpha steering."""

    def __init__(
        self,
        model: torch.nn.Module,
        vae: torch.nn.Module | None,
        text_encoder: torch.nn.Module | None,
        text_tokenizer: Any | None,
        silence_latent: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        model_id: str = ACE_STEP_MODEL_ID,
        model_revision: str = ACE_STEP_MODEL_REVISION,
        components_id: str = ACE_STEP_COMPONENTS_ID,
        components_revision: str = ACE_STEP_COMPONENTS_REVISION,
        sequential_dit: bool = True,
        offload_text_encoder: bool = True,
        offload_dit_after_generation: bool = True,
        cache_conditioning: bool = True,
        conditioning_cache_size: int = 8,
    ):
        if dtype != torch.float32:
            raise ValueError(
                "This project requires ACE-Step in float32. FP16 produced NaN on Tesla T4 and is intentionally rejected."
            )
        if silence_latent.ndim != 3 or silence_latent.shape[-1] != ACE_STEP_LATENT_CHANNELS:
            raise ValueError(
                "`silence_latent` must have shape [1, frames, 64], got "
                f"{tuple(silence_latent.shape)}"
            )
        if conditioning_cache_size < 0:
            raise ValueError("`conditioning_cache_size` must be non-negative")

        self.model = model.requires_grad_(False).eval()
        self.vae = vae.requires_grad_(False).eval() if vae is not None else None
        self.text_encoder = text_encoder.requires_grad_(False).eval() if text_encoder is not None else None
        self.text_tokenizer = text_tokenizer
        self.silence_latent = silence_latent.detach().to("cpu", dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_id = model_id
        self.model_revision = model_revision
        self.components_id = components_id
        self.components_revision = components_revision
        self.sequential_dit = sequential_dit
        self.offload_text_encoder = offload_text_encoder
        self.offload_dit_after_generation = offload_dit_after_generation
        self.conditioning_cache_size = int(conditioning_cache_size)
        self.cache_conditioning = bool(cache_conditioning and self.conditioning_cache_size > 0)
        self._conditioning_cache: OrderedDict[tuple[Any, ...], AceStepConditioning] = OrderedDict()
        self._progress_bar_disabled = True

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = ACE_STEP_MODEL_ID,
        *,
        revision: str = ACE_STEP_MODEL_REVISION,
        components_id: str = ACE_STEP_COMPONENTS_ID,
        components_revision: str = ACE_STEP_COMPONENTS_REVISION,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
        attention_implementation: str = "eager",
        local_files_only: bool = False,
        **pipeline_kwargs: Any,
    ) -> "SteeringAceStepPipeline":
        """Load the exact official SFT remote-code checkpoint and its Qwen/VAE assets.

        The optional 5 Hz LM is never downloaded: ``thinking=False`` is a hard
        invariant of this project.
        """

        if torch_dtype != torch.float32:
            raise ValueError("`torch_dtype` must be torch.float32 for the supported ACE-Step training path")

        from diffusers.models import AutoencoderOobleck
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        load_kwargs = {
            "revision": revision,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch_dtype,
            "attn_implementation": attention_implementation,
        }
        try:
            model = AutoModel.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            # Transformers 4.56 accepts ``dtype`` while earlier compatible
            # versions use ``torch_dtype``.
            load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
            model = AutoModel.from_pretrained(model_id, **load_kwargs)

        component_kwargs = {
            "revision": components_revision,
            "local_files_only": local_files_only,
        }
        vae = AutoencoderOobleck.from_pretrained(
            components_id,
            subfolder="vae",
            torch_dtype=torch_dtype,
            **component_kwargs,
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            components_id,
            subfolder="Qwen3-Embedding-0.6B",
            trust_remote_code=True,
            **component_kwargs,
        )
        text_encoder = AutoModel.from_pretrained(
            components_id,
            subfolder="Qwen3-Embedding-0.6B",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            **component_kwargs,
        )

        silence_path = hf_hub_download(
            repo_id=model_id,
            filename="silence_latent.pt",
            revision=revision,
            local_files_only=local_files_only,
        )
        silence_latent = torch.load(silence_path, map_location="cpu", weights_only=True)
        if silence_latent.ndim != 3:
            raise ValueError(f"Unexpected silence latent shape: {tuple(silence_latent.shape)}")
        if silence_latent.shape[-1] != ACE_STEP_LATENT_CHANNELS:
            silence_latent = silence_latent.transpose(1, 2)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(
            model,
            vae,
            text_encoder,
            text_tokenizer,
            silence_latent,
            device=resolved_device,
            dtype=torch_dtype,
            model_id=model_id,
            model_revision=revision,
            components_id=components_id,
            components_revision=components_revision,
            **pipeline_kwargs,
        )

    def set_progress_bar_config(self, *, disable: bool = True, **_: Any) -> None:
        self._progress_bar_disabled = disable

    def clear_conditioning_cache(self) -> None:
        self._conditioning_cache.clear()

    def _empty_cache(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _move_model(self, device: str | torch.device) -> None:
        target = torch.device(device)
        if _module_device(self.model) != target:
            self.model.to(device=target, dtype=self.dtype)
            self._empty_cache()

    def _move_vae(self, device: str | torch.device) -> None:
        if self.vae is None:
            raise RuntimeError("The VAE was not supplied; latent output is available but waveform decode is not")
        target = torch.device(device)
        if _module_device(self.vae) != target:
            self.vae.to(device=target, dtype=torch.float32)
            self._empty_cache()

    def offload_vae(self) -> None:
        if self.vae is not None:
            self.vae.to("cpu", dtype=torch.float32)
            self._empty_cache()

    def offload_dit(self) -> None:
        self.model.to("cpu", dtype=self.dtype)
        self._empty_cache()

    def _silence_slice(self, frames: int, batch_size: int, device: torch.device) -> torch.Tensor:
        available = self.silence_latent.shape[1]
        repeats = math.ceil(frames / available)
        sliced = self.silence_latent.repeat(1, repeats, 1)[:, :frames]
        return sliced.expand(batch_size, -1, -1).clone().to(device=device, dtype=self.dtype)

    @staticmethod
    def _metadata(duration: float) -> str:
        return (
            "- bpm: N/A\n"
            "- timesignature: N/A\n"
            "- keyscale: N/A\n"
            f"- duration: {int(duration)} seconds\n"
        )

    @classmethod
    def _format_caption(cls, caption: str, duration: float) -> str:
        return SFT_GEN_PROMPT.format(DEFAULT_DIT_INSTRUCTION, caption, cls._metadata(duration))

    @staticmethod
    def _format_lyrics(lyrics: str, language: str) -> str:
        return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    def _encode_qwen(self, texts: list[str], *, lyrics: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.text_tokenizer is None:
            raise RuntimeError("Qwen text encoder/tokenizer are required to prepare ACE-Step conditioning")

        self.text_encoder.to(device=self.device, dtype=self.dtype)
        max_length = 2048 if lyrics else 256
        tokens = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        # Keep encoder masks numeric.  The upstream ACE-Step sequence packer
        # sorts concatenated masks and some CUDA/T4 builds do not implement
        # ``argsort`` for bool tensors.
        attention_mask = tokens.attention_mask.to(self.device, dtype=torch.int32)
        with torch.no_grad():
            if lyrics:
                if hasattr(self.text_encoder, "embed_tokens"):
                    hidden = self.text_encoder.embed_tokens(input_ids)
                else:
                    hidden = self.text_encoder.get_input_embeddings()(input_ids)
            else:
                try:
                    encoded = self.text_encoder(input_ids=input_ids, lyric_attention_mask=None)
                except TypeError:
                    encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
        return hidden.to(dtype=self.dtype), attention_mask

    def prepare_conditioning(
        self,
        prompt: str | Sequence[str],
        retain_prompt: str | Sequence[str],
        *,
        audio_length_in_s: float = 10.0,
        lyrics: str = "[Instrumental]",
        vocal_language: str = "en",
    ) -> AceStepConditioning:
        prompts = _as_text_batch(prompt, "prompt")
        retains = _as_text_batch(retain_prompt, "retain_prompt")
        if len(prompts) != len(retains):
            if len(retains) == 1:
                retains *= len(prompts)
            else:
                raise ValueError("`retain_prompt` batch size must match `prompt`")
        if audio_length_in_s <= 0:
            raise ValueError("`audio_length_in_s` must be positive")

        cache_key = (tuple(prompts), tuple(retains), float(audio_length_in_s), lyrics, vocal_language)
        cached = self._conditioning_cache.get(cache_key)
        if cached is not None:
            self._conditioning_cache.move_to_end(cache_key)
            self._move_model(self.device)
            return cached.to(self.device, self.dtype)

        # A T4 cannot safely hold Qwen and the ~2B-parameter DiT together in
        # true FP32.  A cache miss needs Qwen first, so evict a DiT left on the
        # accelerator by a previous call before encoding the two captions.
        if _module_device(self.model).type != "cpu":
            self.offload_dit()

        batch_size = len(prompts)
        formatted_full = [self._format_caption(text, audio_length_in_s) for text in prompts]
        formatted_retain = [self._format_caption(text, audio_length_in_s) for text in retains]
        lyric_text = self._format_lyrics(lyrics, vocal_language)
        # Encode each prompt family independently.  This prevents the full
        # caption from changing the retain caption's native sequence width (or
        # vice versa) through batch padding.
        full_text_hidden, full_text_mask = self._encode_qwen(formatted_full)
        same_prompt_family = formatted_full == formatted_retain
        if same_prompt_family:
            retain_text_hidden, retain_text_mask = full_text_hidden, full_text_mask
        else:
            retain_text_hidden, retain_text_mask = self._encode_qwen(formatted_retain)
        lyric_hidden, lyric_mask = self._encode_qwen([lyric_text] * batch_size, lyrics=True)

        if self.offload_text_encoder:
            self.text_encoder.to("cpu", dtype=torch.float32)
            self._empty_cache()

        self._move_model(self.device)
        num_audio_samples = round(audio_length_in_s * ACE_STEP_SAMPLE_RATE)
        requested_frames = max(1, num_audio_samples // (ACE_STEP_SAMPLE_RATE // ACE_STEP_LATENT_RATE))
        frames = max(128, requested_frames)
        # Upstream uses a 30-second (750 latent frame) silent reference even
        # when the requested generation itself is shorter.
        reference = self._silence_slice(750, batch_size, self.device)
        reference_order = torch.arange(batch_size, device=self.device, dtype=torch.long)
        with torch.no_grad():
            full_hidden, full_mask = self.model.encoder(
                text_hidden_states=full_text_hidden.to(self.device, self.dtype),
                text_attention_mask=full_text_mask.to(self.device),
                lyric_hidden_states=lyric_hidden.to(self.device, self.dtype),
                lyric_attention_mask=lyric_mask.to(self.device),
                refer_audio_acoustic_hidden_states_packed=reference,
                refer_audio_order_mask=reference_order,
            )
            if same_prompt_family:
                retain_hidden, retain_mask = full_hidden, full_mask
            else:
                retain_hidden, retain_mask = self.model.encoder(
                    text_hidden_states=retain_text_hidden.to(self.device, self.dtype),
                    text_attention_mask=retain_text_mask.to(self.device),
                    lyric_hidden_states=lyric_hidden.to(self.device, self.dtype),
                    lyric_attention_mask=lyric_mask.to(self.device),
                    refer_audio_acoustic_hidden_states_packed=reference,
                    refer_audio_order_mask=reference_order,
                )

        source = self._silence_slice(frames, batch_size, self.device)
        chunk_mask = torch.ones_like(source)
        conditioning = AceStepConditioning(
            full_hidden_states=full_hidden,
            retain_hidden_states=retain_hidden,
            full_attention_mask=full_mask,
            retain_attention_mask=retain_mask,
            context_latents=torch.cat([source, chunk_mask], dim=-1),
            latent_attention_mask=torch.cat(
                [
                    torch.ones(batch_size, requested_frames, device=self.device, dtype=self.dtype),
                    torch.zeros(batch_size, frames - requested_frames, device=self.device, dtype=self.dtype),
                ],
                dim=1,
            ),
            num_audio_samples=num_audio_samples,
        )
        if self.cache_conditioning:
            self._conditioning_cache[cache_key] = conditioning.cpu()
            self._conditioning_cache.move_to_end(cache_key)
            while len(self._conditioning_cache) > self.conditioning_cache_size:
                self._conditioning_cache.popitem(last=False)
        return conditioning

    def _decoder_velocity(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_mask: torch.Tensor,
        conditioning: AceStepConditioning,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        t_batch = timestep.expand(batch_size)
        with torch.no_grad():
            output = self.model.decoder(
                hidden_states=latents.detach(),
                timestep=t_batch,
                timestep_r=t_batch,
                attention_mask=conditioning.latent_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=hidden_mask,
                context_latents=conditioning.context_latents,
                use_cache=False,
                past_key_values=None,
            )
        if torch.is_tensor(output):
            return output
        try:
            velocity = output[0]
        except (KeyError, IndexError, TypeError) as error:
            raise TypeError("ACE-Step decoder returned an unsupported output container") from error
        if not torch.is_tensor(velocity):
            raise TypeError("ACE-Step decoder output[0] must be a tensor")
        return velocity

    def _predict_branches(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: AceStepConditioning,
        *,
        include_retain: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch_size = conditioning.batch_size
        # The learned null parameter is one token.  Upstream repeats that
        # identical token to the conditional width, but the pinned decoder has
        # no encoder-position encoding and ignores encoder_attention_mask in
        # eager mode.  A canonical one-token null is therefore length
        # invariant and lets independently packed full/retain conditions share
        # exactly one APG null branch.
        null_hidden = self.model.null_condition_emb.expand(batch_size, -1, -1)
        null_mask = torch.ones(
            batch_size,
            1,
            device=null_hidden.device,
            dtype=conditioning.full_attention_mask.dtype,
        )

        same_width = (
            conditioning.full_hidden_states.shape[1]
            == conditioning.retain_hidden_states.shape[1]
        )
        if self.sequential_dit or (include_retain and not same_width):
            full = self._decoder_velocity(
                latents, timestep, conditioning.full_hidden_states, conditioning.full_attention_mask, conditioning
            )
            retain = (
                self._decoder_velocity(
                    latents,
                    timestep,
                    conditioning.retain_hidden_states,
                    conditioning.retain_attention_mask,
                    conditioning,
                )
                if include_retain
                else None
            )
            null = self._decoder_velocity(
                latents, timestep, null_hidden, null_mask, conditioning
            )
            return full, retain, null

        branches = [conditioning.full_hidden_states]
        masks = [conditioning.full_attention_mask]
        if include_retain:
            branches.append(conditioning.retain_hidden_states)
            masks.append(conditioning.retain_attention_mask)
        # Batched execution requires one shared sequence width.  Repetition of
        # the canonical null token is mathematically equivalent here; if full
        # and retain widths differ we already fell back to sequential passes.
        null_hidden = null_hidden.expand(-1, conditioning.full_hidden_states.shape[1], -1)
        null_mask = null_mask.expand(-1, conditioning.full_hidden_states.shape[1])
        branches.append(null_hidden)
        masks.append(null_mask)
        repeats = len(branches)
        expanded = replace(
            conditioning,
            context_latents=torch.cat([conditioning.context_latents] * repeats),
            latent_attention_mask=torch.cat([conditioning.latent_attention_mask] * repeats),
        )
        velocity = self._decoder_velocity(
            torch.cat([latents] * repeats),
            timestep,
            torch.cat(branches),
            torch.cat(masks),
            expanded,
        )
        chunks = velocity.chunk(repeats)
        if include_retain:
            return chunks[0], chunks[1], chunks[2]
        return chunks[0], None, chunks[1]

    def _prepare_initial_latents(
        self,
        conditioning: AceStepConditioning,
        seed: int | Sequence[int] | None,
        initial_latents: torch.Tensor | None,
    ) -> torch.Tensor:
        expected_shape = conditioning.latent_shape
        if initial_latents is not None:
            if tuple(initial_latents.shape) != expected_shape:
                raise ValueError(
                    f"`initial_latents` must have shape {expected_shape}, got {tuple(initial_latents.shape)}"
                )
            return initial_latents.to(self.device, self.dtype).clone()
        if hasattr(self.model, "prepare_noise"):
            normalized_seed = list(seed) if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)) else seed
            return self.model.prepare_noise(conditioning.context_latents, normalized_seed)

        batch, frames, channels = expected_shape
        seeds: Iterable[int | None]
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            if len(seed) != batch:
                raise ValueError("A seed sequence must match the batch size")
            seeds = seed
        else:
            seeds = [seed] * batch
        samples = []
        for item_seed in seeds:
            generator = None
            if item_seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(item_seed))
            samples.append(
                torch.randn(1, frames, channels, device=self.device, dtype=self.dtype, generator=generator)
            )
        return torch.cat(samples)

    def denoise(
        self,
        conditioning: AceStepConditioning,
        *,
        steering_target_embeds: torch.Tensor | None = None,
        steering_model: torch.nn.Module | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_mode: str = "apg",
        shift: float = 1.0,
        steering_frac_start: float = 0.3,
        steering_frac_end: float = 0.8,
        seed: int | Sequence[int] | None = None,
        initial_latents: torch.Tensor | None = None,
        train: bool = False,
        callback: Callable[[int, float, torch.Tensor], None] | None = None,
    ) -> SteeringAudioPipelineOutput:
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be at least 1")
        if shift <= 0:
            raise ValueError("`shift` must be positive")
        if not math.isfinite(guidance_scale) or guidance_scale < 1.0:
            raise ValueError("`guidance_scale` must be finite and at least 1.0")
        if not 0 <= steering_frac_start < steering_frac_end <= 1:
            raise ValueError("steering fractions must satisfy 0 <= start < end <= 1")
        if guidance_mode not in {"apg", "cfg"}:
            raise ValueError("`guidance_mode` must be 'apg' or 'cfg'")

        self._move_model(self.device)
        conditioning = conditioning.to(self.device, self.dtype)
        steering_enabled = steering_model is not None
        if steering_enabled:
            if steering_target_embeds is None:
                raise ValueError("External CLAP `steering_target_embeds` are required when steering is enabled")
            if steering_target_embeds.ndim != 2 or steering_target_embeds.shape[0] != conditioning.batch_size:
                raise ValueError("`steering_target_embeds` must have shape [batch, clap_dim]")
            steering_target_embeds = steering_target_embeds.to(self.device)

        latents = self._prepare_initial_latents(conditioning, seed, initial_latents)
        initial = latents.detach().clone()
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        if shift != 1.0:
            timesteps = shift * timesteps / (1.0 + (shift - 1.0) * timesteps)

        momentum = MomentumBuffer()
        records: list[tuple[float, float]] = []
        for step_index, (t_current, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
            fraction = step_index / num_inference_steps
            active = steering_enabled and steering_frac_start <= fraction < steering_frac_end
            full, retain, null = self._predict_branches(
                latents, t_current, conditioning, include_retain=active
            )

            if active:
                if train:
                    alpha = torch.utils.checkpoint.checkpoint(
                        lambda x, t, embed: steering_model(latents=x, t=t, target_embed=embed),
                        latents,
                        t_current,
                        steering_target_embeds,
                        use_reentrant=False,
                    )
                else:
                    with torch.no_grad():
                        alpha = steering_model(
                            latents=latents,
                            t=t_current,
                            target_embed=steering_target_embeds,
                        )
                if alpha.ndim == 1:
                    alpha = alpha[:, None, None]
                elif alpha.ndim == 2:
                    alpha = alpha[:, :, None]
                if tuple(alpha.shape) != (latents.shape[0], 1, 1):
                    raise ValueError("The steering model must return one alpha with shape [batch, 1, 1]")
                if not torch.isfinite(alpha).all():
                    raise FloatingPointError("The steering model returned NaN/Inf alpha")
                conditional = full + alpha * (retain - full)
                records.append((float(t_current), float(alpha.detach().float().mean())))
            else:
                conditional = full

            if guidance_scale > 1.0:
                if guidance_mode == "apg":
                    velocity = adaptive_projected_guidance(
                        conditional,
                        null,
                        guidance_scale,
                        momentum,
                        dims=(1,),
                    )
                else:
                    velocity = classifier_free_guidance(conditional, null, guidance_scale)
            else:
                velocity = conditional

            dt = (t_current - t_next).reshape(1, 1, 1)
            latents = latents - dt * velocity
            if callback is not None:
                callback(step_index, float(t_current), latents)

        if self.offload_dit_after_generation:
            self.offload_dit()
        return SteeringAudioPipelineOutput(audios=latents, alpha_records=records, initial_latents=initial)

    def decode_latents(
        self,
        latents: torch.Tensor,
        *,
        num_audio_samples: int | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Decode ``[B,T,64]`` to differentiable stereo ``[B,2,S]`` audio."""

        self._move_vae(latents.device)
        decoded = self.vae.decode(latents.transpose(1, 2).contiguous().to(torch.float32))
        waveform = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        if num_audio_samples is not None:
            waveform = waveform[..., :num_audio_samples]
        waveform = waveform.float()
        if normalize:
            peak = waveform.abs().amax(dim=(-2, -1), keepdim=True)
            waveform = waveform / peak.clamp(min=1.0)
        return waveform

    def __call__(
        self,
        prompt: str | Sequence[str],
        *,
        retain_prompt: str | Sequence[str] | None = None,
        steering_target_embeds: torch.Tensor | None = None,
        steering_model: torch.nn.Module | None = None,
        audio_length_in_s: float = 10.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        guidance_mode: str = "apg",
        shift: float = 1.0,
        steering_frac_start: float = 0.3,
        steering_frac_end: float = 0.8,
        seed: int | Sequence[int] | None = None,
        initial_latents: torch.Tensor | None = None,
        output_type: str = "pt",
        train: bool = False,
        thinking: bool = False,
        dcw_enabled: bool = False,
        callback: Callable[[int, float, torch.Tensor], None] | None = None,
    ) -> SteeringAudioPipelineOutput:
        if thinking:
            raise ValueError("This project fixes `thinking=False`; the optional 5 Hz LM is not loaded")
        if dcw_enabled:
            raise ValueError("This project fixes `dcw_enabled=False` for a differentiable, auditable sampler")
        prompts = _as_text_batch(prompt, "prompt")
        if retain_prompt is None:
            if steering_model is not None:
                raise ValueError("`retain_prompt` is required when steering is enabled")
            retains = prompts
        else:
            retains = _as_text_batch(retain_prompt, "retain_prompt")
        conditioning = self.prepare_conditioning(
            prompts,
            retains,
            audio_length_in_s=audio_length_in_s,
        )
        output = self.denoise(
            conditioning,
            steering_target_embeds=steering_target_embeds,
            steering_model=steering_model,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_mode=guidance_mode,
            shift=shift,
            steering_frac_start=steering_frac_start,
            steering_frac_end=steering_frac_end,
            seed=seed,
            initial_latents=initial_latents,
            train=train,
            callback=callback,
        )
        if output_type == "latent":
            return output
        waveform = self.decode_latents(
            output.audios,
            num_audio_samples=conditioning.num_audio_samples,
        )
        if output_type == "pt":
            return replace(output, audios=waveform)
        if output_type == "np":
            return replace(output, audios=waveform.detach().cpu().numpy())
        raise ValueError("`output_type` must be 'latent', 'pt' or 'np'")


__all__ = [
    "ACE_STEP_COMPONENTS_ID",
    "ACE_STEP_COMPONENTS_REVISION",
    "ACE_STEP_LATENT_CHANNELS",
    "ACE_STEP_LATENT_RATE",
    "ACE_STEP_MODEL_ID",
    "ACE_STEP_MODEL_REVISION",
    "ACE_STEP_SAMPLE_RATE",
    "GRADIENT_MODE",
    "STEERING_MODE",
    "AceStepConditioning",
    "MomentumBuffer",
    "SteeringAceStepPipeline",
    "SteeringAudioPipelineOutput",
    "adaptive_projected_guidance",
    "classifier_free_guidance",
]
