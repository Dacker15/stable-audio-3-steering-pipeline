"""Differentiable concept steering for the official ACE-Step 1.5 SFT checkpoint.

The pipeline intentionally accepts only ``ACE-Step/acestep-v15-sft``.  In particular it does not
load the Base checkpoint, whose Extract/Lego/Complete tasks would confound the learned steering
mechanism evaluated by this project.  ACE-Step's DiT stays frozen; gradients flow only through the
predicted interpolation coefficient, the Euler trajectory, and the shared Oobleck VAE decoder.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.utils.checkpoint
from diffusers import AutoencoderOobleck
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer


ACE_STEP_MODEL_ID = "ACE-Step/acestep-v15-sft"
ACE_STEP_COMPONENTS_ID = "ACE-Step/Ace-Step1.5"
ACE_STEP_TEXT_ENCODER_SUBFOLDER = "Qwen3-Embedding-0.6B"
STEERING_MODE = "ace_step_1_5_sft_full_to_retain_apg_v1"

DEFAULT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
EMPTY_LYRICS = "# Languages\nen\n\n# Lyric\n<|endoftext|>"
PREDICTOR_TIMESTEP_SCALE = 1000.0
SAMPLE_RATE = 48_000
LATENT_FRAMES_PER_SECOND = 25
MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = 600.0


@dataclass
class SteeringAudioPipelineOutput:
    """Audio (or final latents) and the mean alpha predicted at each steered step."""

    audios: Any
    alpha_records: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class _Conditioning:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    context_latents: torch.Tensor


class SteeringAceStepPipeline:
    """ACE-Step 1.5 SFT text-to-music pipeline with learned full/retain APG interpolation."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        text_encoder: torch.nn.Module,
        tokenizer,
        vae: AutoencoderOobleck,
        silence_latent: torch.Tensor,
        model_name: str = ACE_STEP_MODEL_ID,
    ):
        self.backbone = backbone
        self.transformer = backbone.decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.silence_latent = self._canonicalize_silence_latent(silence_latent)
        self.model_name = model_name

        # Use the checkpoint's own APG implementation.  This avoids subtly changing the guidance
        # formula or momentum defaults relative to the official SFT inference path.
        modeling_module = importlib.import_module(backbone.__class__.__module__)
        self._apg_forward = modeling_module.apg_forward
        self._momentum_buffer_type = modeling_module.MomentumBuffer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ACE_STEP_MODEL_ID,
        device: str | torch.device | None = None,
        model_half: bool = True,
    ) -> "SteeringAceStepPipeline":
        """Load the official 2B SFT DiT plus its shared text encoder and Oobleck VAE.

        ``model_half=True`` means BF16 on CUDA.  Float16 is deliberately never used because the
        checkpoint is BF16 and the unrolled training trajectory is numerically sensitive.
        CPU and MPS use float32.
        """
        if model_name != ACE_STEP_MODEL_ID:
            raise ValueError(
                f"This project is pinned to {ACE_STEP_MODEL_ID!r}, but received {model_name!r}. "
                "The Base checkpoint is intentionally rejected because it includes "
                "Extract/Lego/Complete; Turbo is rejected because it has no CFG branch."
            )

        resolved_device = cls._resolve_device(device)
        model_dtype = torch.bfloat16 if model_half and resolved_device.type == "cuda" else torch.float32

        backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            attn_implementation="sdpa",
        ).to(resolved_device)
        tokenizer = AutoTokenizer.from_pretrained(
            ACE_STEP_COMPONENTS_ID,
            subfolder=ACE_STEP_TEXT_ENCODER_SUBFOLDER,
        )
        text_encoder = AutoModel.from_pretrained(
            ACE_STEP_COMPONENTS_ID,
            subfolder=ACE_STEP_TEXT_ENCODER_SUBFOLDER,
            torch_dtype=model_dtype,
            attn_implementation="sdpa",
        ).to(resolved_device)
        # The decoder participates in the training graph, so keep it in float32.
        vae = AutoencoderOobleck.from_pretrained(
            ACE_STEP_COMPONENTS_ID,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(resolved_device)

        silence_path = hf_hub_download(model_name, "silence_latent.pt")
        silence_latent = torch.load(silence_path, map_location="cpu", weights_only=True)

        pipeline = cls(backbone, text_encoder, tokenizer, vae, silence_latent, model_name=model_name)
        pipeline.freeze_backbone()
        return pipeline

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is not None and str(device) != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _canonicalize_silence_latent(value: Any) -> torch.Tensor:
        if isinstance(value, dict):
            tensors = [item for item in value.values() if torch.is_tensor(item)]
            if len(tensors) != 1:
                raise ValueError("silence_latent.pt must contain exactly one tensor")
            value = tensors[0]
        if not torch.is_tensor(value):
            raise TypeError(f"silence_latent.pt did not contain a tensor: {type(value)}")
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(f"silence latent must be rank 3, got shape {tuple(value.shape)}")
        # The published file is [1, 64, T], while the DiT consumes [B, T, 64].
        if value.shape[1] == 64 and value.shape[-1] != 64:
            value = value.transpose(1, 2)
        if value.shape[-1] != 64:
            raise ValueError(f"silence latent must have 64 channels, got shape {tuple(value.shape)}")
        return value.contiguous().float()

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def io_channels(self) -> int:
        return int(self.backbone.config.audio_acoustic_hidden_dim)

    @property
    def downsampling_ratio(self) -> int:
        return SAMPLE_RATE // LATENT_FRAMES_PER_SECOND

    @property
    def max_duration_in_s(self) -> float:
        return MAX_DURATION_SECONDS

    def freeze_backbone(self) -> None:
        for module in (self.backbone, self.text_encoder, self.vae):
            module.eval()
            module.requires_grad_(False)

    @staticmethod
    def _prepare_prompt_list(prompt: str | list[str], name: str = "prompt") -> list[str]:
        values = [prompt] if isinstance(prompt, str) else list(prompt)
        if not values:
            raise ValueError(f"`{name}` must contain at least one prompt")
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` at index {index} must be a non-empty string")
        return [value.strip() for value in values]

    @classmethod
    def _prepare_retain_prompt(cls, retain_prompt: str | list[str] | None, batch_size: int) -> list[str]:
        if retain_prompt is None:
            raise ValueError("`retain_prompt` must be provided when steering is enabled")
        values = cls._prepare_prompt_list(retain_prompt, "retain_prompt")
        if len(values) == 1 and isinstance(retain_prompt, str):
            return values * batch_size
        if len(values) != batch_size:
            raise ValueError(
                f"`retain_prompt` has batch size {len(values)}, but `prompt` has batch size {batch_size}"
            )
        return values

    @staticmethod
    def _format_sft_prompt(caption: str, audio_length_in_s: float) -> str:
        metadata = (
            "- bpm: N/A\n"
            "- timesignature: N/A\n"
            "- keyscale: N/A\n"
            f"- duration: {int(round(audio_length_in_s))} seconds\n"
        )
        return (
            f"# Instruction\n{DEFAULT_INSTRUCTION}\n\n"
            f"# Caption\n{caption}\n\n"
            f"# Metas\n{metadata}<|endoftext|>\n"
        )

    def _tokenize(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in tokens.items()}

    def _encode_conditioning(self, captions: list[str], audio_length_in_s: float, latent_length: int) -> _Conditioning:
        formatted = [self._format_sft_prompt(caption, audio_length_in_s) for caption in captions]
        text_tokens = self._tokenize(formatted, max_length=256)
        lyric_tokens = self._tokenize([EMPTY_LYRICS] * len(captions), max_length=2048)

        silence = self.silence_latent.to(device=self.device, dtype=self.model_dtype)
        if latent_length > silence.shape[1]:
            raise ValueError(
                f"requested {latent_length} latent frames, but the published silence latent has only "
                f"{silence.shape[1]} frames"
            )
        source_latents = silence[:, :latent_length].expand(len(captions), -1, -1)
        timbre_frames = min(int(self.backbone.config.timbre_fix_frame), silence.shape[1])
        timbre_latents = silence[:, :timbre_frames].expand(len(captions), -1, -1)
        reference_order = torch.arange(len(captions), device=self.device, dtype=torch.long)

        with torch.no_grad():
            text_hidden = self.text_encoder(
                input_ids=text_tokens["input_ids"],
            ).last_hidden_state
            lyric_hidden = self.text_encoder.get_input_embeddings()(lyric_tokens["input_ids"])
            encoder_hidden, encoder_mask = self.backbone.encoder(
                text_hidden_states=text_hidden.to(self.model_dtype),
                text_attention_mask=text_tokens["attention_mask"].bool(),
                lyric_hidden_states=lyric_hidden.to(self.model_dtype),
                lyric_attention_mask=lyric_tokens["attention_mask"].bool(),
                refer_audio_acoustic_hidden_states_packed=timbre_latents,
                refer_audio_order_mask=reference_order,
            )

        # Text-to-music uses silence as source context.  An all-one chunk mask marks every frame as
        # generatable, matching the official no-source input preparation.
        chunk_mask = torch.ones_like(source_latents)
        context = torch.cat([source_latents, chunk_mask], dim=-1)
        return _Conditioning(encoder_hidden, encoder_mask, context)

    def _prepare_latents(
        self,
        batch_size: int,
        latent_length: int,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        shape = (batch_size, latent_length, self.io_channels)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"`generator` has batch size {len(generator)}, but `prompt` has batch size {batch_size}"
                )
            latents = torch.cat(
                [
                    torch.randn((1, *shape[1:]), generator=item, device=item.device, dtype=torch.float32)
                    for item in generator
                ],
                dim=0,
            )
        else:
            noise_device = self.device if generator is None else generator.device
            latents = torch.randn(shape, generator=generator, device=noise_device, dtype=torch.float32)
        return latents.to(self.device)

    @staticmethod
    def _interpolate_guided_predictions(
        full_guided: torch.Tensor,
        retain_guided: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        return full_guided + alpha_t * (retain_guided - full_guided)

    @staticmethod
    def _predict_alpha(
        steering_model: torch.nn.Module,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        target_embed: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        # Predictor convolutions expect [B, C, H, W]; ACE latents are [B, T, 64].
        predictor_latents = latents.transpose(1, 2).unsqueeze(-2)
        predictor_timestep = timestep * PREDICTOR_TIMESTEP_SCALE
        if train:
            alpha = torch.utils.checkpoint.checkpoint(
                lambda hidden, step_t, embed: steering_model(latents=hidden, t=step_t, target_embed=embed),
                predictor_latents,
                predictor_timestep,
                target_embed,
                use_reentrant=False,
            )
        else:
            with torch.no_grad():
                alpha = steering_model(
                    latents=predictor_latents,
                    t=predictor_timestep,
                    target_embed=target_embed,
                )
        return alpha.view(-1, 1, 1)

    def _decoder_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: _Conditioning,
    ) -> torch.Tensor:
        output = self.transformer(
            hidden_states=latents.to(self.model_dtype),
            timestep=timestep.to(self.model_dtype),
            timestep_r=timestep.to(self.model_dtype),
            attention_mask=torch.ones(
                latents.shape[0], latents.shape[1], device=self.device, dtype=self.model_dtype
            ),
            encoder_hidden_states=conditioning.hidden_states.to(self.model_dtype),
            encoder_attention_mask=conditioning.attention_mask,
            context_latents=conditioning.context_latents.to(self.model_dtype),
            use_cache=False,
        )
        return output[0].float()

    def decode_latents(
        self,
        latents: torch.Tensor,
        audio_length_in_s: float | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Decode ``[B, T, 64]`` ACE latents differentiably to 48 kHz stereo audio."""
        audio = self.vae.decode(latents.transpose(1, 2).float()).sample
        if audio_length_in_s is not None:
            audio = audio[..., : int(round(audio_length_in_s * self.sample_rate))]
        if normalize:
            audio = self._normalize_audio(audio)
        return audio

    @staticmethod
    def _normalize_audio(audio: torch.Tensor) -> torch.Tensor:
        """Match ACE-Step's anti-clipping pass followed by per-sample -1 dBFS peak normalization."""

        peak = audio.abs().flatten(1).amax(dim=1).view(-1, 1, 1)
        audio = audio / peak.clamp_min(1.0)
        peak = audio.abs().flatten(1).amax(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        return audio * ((10.0 ** (-1.0 / 20.0)) / peak)

    def __call__(
        self,
        prompt: str | list[str],
        retain_prompt: str | list[str] | None = None,
        target_embed: torch.Tensor | None = None,
        steering_model: torch.nn.Module | None = None,
        steering_frac_start: float = 0.0,
        steering_frac_end: float = 1.0,
        num_inference_steps: int = 50,
        audio_length_in_s: float = 10.0,
        cfg_scale: float = 7.0,
        shift: float = 1.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        train: bool = False,
    ) -> SteeringAudioPipelineOutput | tuple[Any, list[tuple[int, float]]]:
        if output_type not in {"np", "pt", "latent"}:
            raise ValueError(f"`output_type` must be 'np', 'pt', or 'latent', got {output_type!r}")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be at least 1")
        if not MIN_DURATION_SECONDS <= audio_length_in_s <= self.max_duration_in_s:
            raise ValueError(
                f"`audio_length_in_s` must be in [{MIN_DURATION_SECONDS}, {self.max_duration_in_s}], "
                f"got {audio_length_in_s}"
            )
        if cfg_scale < 1.0:
            raise ValueError(f"`cfg_scale` must be at least 1.0, got {cfg_scale}")
        if shift <= 0.0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        if not 0.0 <= steering_frac_start < steering_frac_end <= 1.0:
            raise ValueError("steering fractions must satisfy 0 <= start < end <= 1")

        prompts = self._prepare_prompt_list(prompt)
        batch_size = len(prompts)
        steering_enabled = steering_model is not None
        if steering_enabled and cfg_scale <= 1.0:
            raise ValueError("ACE-Step steering requires `cfg_scale > 1.0`")
        if steering_enabled:
            if target_embed is None or target_embed.ndim != 2 or target_embed.shape[0] != batch_size:
                shape = None if target_embed is None else tuple(target_embed.shape)
                raise ValueError(
                    f"`target_embed` must have shape [batch_size, dim] with batch size {batch_size}; got {shape}"
                )
            retain_prompts = self._prepare_retain_prompt(retain_prompt, batch_size)
        else:
            retain_prompts = None

        latent_length = int(math.ceil(audio_length_in_s * LATENT_FRAMES_PER_SECOND))
        if latents is None:
            latents = self._prepare_latents(batch_size, latent_length, generator)
        else:
            expected = (batch_size, latent_length, self.io_channels)
            if tuple(latents.shape) != expected:
                raise ValueError(f"`latents` must have shape {expected}, got {tuple(latents.shape)}")
            latents = latents.to(device=self.device, dtype=torch.float32)

        all_captions = prompts if retain_prompts is None else prompts + retain_prompts
        conditioning = self._encode_conditioning(all_captions, audio_length_in_s, latent_length)
        full_condition = _Conditioning(
            conditioning.hidden_states[:batch_size],
            conditioning.attention_mask[:batch_size],
            conditioning.context_latents[:batch_size],
        )
        retain_condition = None
        if retain_prompts is not None:
            retain_condition = _Conditioning(
                conditioning.hidden_states[batch_size:],
                conditioning.attention_mask[batch_size:],
                conditioning.context_latents[batch_size:],
            )

        null_hidden = self.backbone.null_condition_emb.expand_as(full_condition.hidden_states)
        null_condition = _Conditioning(
            null_hidden,
            full_condition.attention_mask,
            full_condition.context_latents,
        )

        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=torch.float32)
        if shift != 1.0:
            timesteps = shift * timesteps / (1.0 + (shift - 1.0) * timesteps)

        full_momentum = self._momentum_buffer_type()
        retain_momentum = self._momentum_buffer_type()
        records: list[tuple[int, float]] = []

        for step, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            step_fraction = step / num_inference_steps
            active = steering_enabled and steering_frac_start <= step_fraction < steering_frac_end
            step_t = t_curr.expand(batch_size)

            with torch.no_grad():
                if cfg_scale > 1.0:
                    conditions = [full_condition, null_condition]
                    if active:
                        conditions.append(retain_condition)
                    count = len(conditions)
                    batch_condition = _Conditioning(
                        torch.cat([item.hidden_states for item in conditions], dim=0),
                        torch.cat([item.attention_mask for item in conditions], dim=0),
                        torch.cat([item.context_latents for item in conditions], dim=0),
                    )
                    velocity = self._decoder_forward(
                        torch.cat([latents] * count, dim=0),
                        t_curr.expand(batch_size * count),
                        batch_condition,
                    )
                    chunks = velocity.chunk(count)
                    pred_full, pred_uncond = chunks[:2]
                    full_guided = self._apg_forward(
                        pred_cond=pred_full,
                        pred_uncond=pred_uncond,
                        guidance_scale=cfg_scale,
                        momentum_buffer=full_momentum,
                        dims=[1],
                    ).float()
                    if active:
                        retain_guided = self._apg_forward(
                            pred_cond=chunks[2],
                            pred_uncond=pred_uncond,
                            guidance_scale=cfg_scale,
                            momentum_buffer=retain_momentum,
                            dims=[1],
                        ).float()
                else:
                    full_guided = self._decoder_forward(latents, step_t, full_condition)

            if active:
                alpha_t = self._predict_alpha(
                    steering_model,
                    latents,
                    step_t,
                    target_embed,
                    train=train,
                )
                records.append((step, float(alpha_t.detach().float().mean())))
                velocity = self._interpolate_guided_predictions(full_guided, retain_guided, alpha_t)
            else:
                velocity = full_guided

            latents = latents - velocity * (t_curr - t_next)

        if output_type == "latent":
            audio: Any = latents
        else:
            audio = self.decode_latents(latents, audio_length_in_s=audio_length_in_s)
            if output_type == "np":
                audio = audio.detach().float().cpu().numpy()

        if not return_dict:
            return audio, records
        return SteeringAudioPipelineOutput(audios=audio, alpha_records=records)
