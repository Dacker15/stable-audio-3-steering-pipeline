from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import MusicLDMPipeline
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput
from diffusers.utils import is_torch_xla_available

STEERING_MODE = "full_to_retain_v1"

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


@dataclass
class SteeringAudioPipelineOutput(AudioPipelineOutput):
    r"""
    Output class for `SteeringMusicLDMPipeline`.

    Args:
        audios (`np.ndarray` or `torch.Tensor`): Generated audio, or latents when `output_type="latent"`.
        alpha_records (`list[tuple[float, float]]`): `(timestep, alpha_t)` pairs recorded at every
            steering-active denoising step, in denoising order (noisiest to cleanest).
    """

    alpha_records: list[tuple[float, float]] = None


class SteeringMusicLDMPipeline(MusicLDMPipeline):
    r"""
    `MusicLDMPipeline` variant that steers denoising away from a target concept over a sub-range of
    the denoising trajectory. An external `steering_model` predicts a per-step scalar `alpha_t`.
    Inside the steering window, `alpha_t` interpolates between classifier-free guidance for the
    full prompt and classifier-free guidance for a `retain_prompt` with the target removed. Thus
    `alpha_t = 0` reproduces the ordinary full-prompt generation and `alpha_t = 1` follows the
    retain prompt. Outside the window, generation is identical to the base `MusicLDMPipeline`.
    """

    @contextmanager
    def _tensor_text_features(self):
        r"""
        Makes `text_encoder.get_text_features` return a plain tensor for the duration of the context.

        `transformers >= 5` returns a `BaseModelOutputWithPooling`, while both the inherited
        `_encode_prompt` and `_encode_steering_target` expect the tensor that earlier versions
        returned. `MusicLDMPipeline` is deprecated upstream (`_last_supported_version = "0.33.1"`),
        so this shim patches the call site instead of duplicating `_encode_prompt`.
        """
        was_patched = "get_text_features" in self.text_encoder.__dict__
        original_get_text_features = self.text_encoder.get_text_features

        def get_text_features(*args, **kwargs):
            text_features = original_get_text_features(*args, **kwargs)
            if not torch.is_tensor(text_features):
                text_features = text_features.pooler_output
            return text_features

        self.text_encoder.get_text_features = get_text_features
        try:
            yield
        finally:
            if was_patched:
                self.text_encoder.get_text_features = original_get_text_features
            else:
                del self.text_encoder.get_text_features

    def _encode_steering_target(self, steering_target, batch_size, num_waveforms_per_prompt, device, dtype):
        if isinstance(steering_target, str):
            steering_targets = [steering_target] * batch_size
        elif isinstance(steering_target, list):
            if len(steering_target) != batch_size:
                raise ValueError(
                    f"`steering_target` has batch size {len(steering_target)}, but `prompt` has batch size"
                    f" {batch_size}. Please make sure that passed `steering_target` matches the batch size of"
                    " `prompt`."
                )
            steering_targets = steering_target
        else:
            raise ValueError(f"`steering_target` has to be of type `str` or `list` but is {type(steering_target)}")

        with torch.no_grad(), self._tensor_text_features():
            text_inputs = self.tokenizer(
                steering_targets,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            target_embed = self.text_encoder.get_text_features(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            )
        target_embed = target_embed.to(dtype=dtype, device=device)

        # duplicate target embeddings for each generation per prompt, using mps friendly method
        seq_len = target_embed.shape[1]
        target_embed = target_embed.repeat(1, num_waveforms_per_prompt)
        target_embed = target_embed.view(batch_size * num_waveforms_per_prompt, seq_len)
        return target_embed

    @staticmethod
    def _prepare_retain_prompt(retain_prompt, batch_size):
        def validate(value, index=None):
            location = "" if index is None else f" at index {index}"
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`retain_prompt`{location} has to be a non-empty string but is {value!r}")
            return value.strip()

        if isinstance(retain_prompt, str):
            return [validate(retain_prompt)] * batch_size
        if isinstance(retain_prompt, list):
            if len(retain_prompt) != batch_size:
                raise ValueError(
                    f"`retain_prompt` has batch size {len(retain_prompt)}, but `prompt` has batch size"
                    f" {batch_size}. Please make sure that passed `retain_prompt` matches the batch size of"
                    " `prompt`."
                )
            return [validate(value, index) for index, value in enumerate(retain_prompt)]
        raise ValueError(f"`retain_prompt` has to be of type `str` or `list` but is {type(retain_prompt)}")

    @staticmethod
    def _interpolate_cfg_predictions(
        noise_pred_uncond,
        noise_pred_full,
        noise_pred_retain,
        guidance_scale,
        alpha_t,
    ):
        noise_pred_full_cfg = noise_pred_uncond + guidance_scale * (noise_pred_full - noise_pred_uncond)
        noise_pred_retain_cfg = noise_pred_uncond + guidance_scale * (noise_pred_retain - noise_pred_uncond)
        return noise_pred_full_cfg + alpha_t * (noise_pred_retain_cfg - noise_pred_full_cfg)

    def __call__(
        self,
        prompt: str | list[str] = None,
        audio_length_in_s: float | None = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 2.0,
        negative_prompt: str | list[str] | None = None,
        num_waveforms_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        return_dict: bool = True,
        callback: Callable[[int, int, torch.Tensor], None] | None = None,
        callback_steps: int | None = 1,
        cross_attention_kwargs: dict[str, Any] | None = None,
        output_type: str | None = "np",
        steering_target: str | list[str] = None,
        retain_prompt: str | list[str] = None,
        steering_model: torch.nn.Module = None,
        steering_frac_start: float = 0.0,
        steering_frac_end: float = 1.0,
        train: bool = False,
    ):
        r"""
        Extends `MusicLDMPipeline.__call__` with concept steering.

        Args:
            steering_target (`str` or `list[str]`): The concept(s) to suppress and the target
                conditioning passed to `steering_model`. If a list, it must match the batch size of
                `prompt`.
            retain_prompt (`str` or `list[str]`): Prompt(s) describing what generation should preserve,
                normally the full prompt with `steering_target` removed. Inside the steering window,
                `alpha_t` interpolates from full-prompt CFG (`alpha_t = 0`) to retain-prompt CFG
                (`alpha_t = 1`). If a list, it must match the batch size of `prompt`.
            steering_model (`torch.nn.Module`): Model called as `steering_model(latents=..., t=..., target_embed=...)` to predict `alpha_t`.
            steering_frac_start (`float`, *optional*, defaults to 0.0): Fraction of the denoising loop (by step index) where steering begins.
            steering_frac_end (`float`, *optional*, defaults to 1.0): Fraction of the denoising loop (by step index) where steering ends.
            train (`bool`, *optional*, defaults to `False`): If `True`, gradients are enabled for the `steering_model` call.
        """
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        steering_enabled = do_classifier_free_guidance and steering_model is not None

        if steering_enabled:
            if steering_target is None:
                raise ValueError("`steering_target` must be provided when `steering_model` is enabled")
            retain_prompts = self._prepare_retain_prompt(retain_prompt, batch_size)
        else:
            retain_prompts = None

        # 3. Encode the full prompt, retain prompt and steering target
        with torch.no_grad(), self._tensor_text_features():
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_waveforms_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
            if steering_enabled:
                # Only the conditional retain embeddings are needed. During steering they share
                # the same unconditional/negative prediction already encoded for the full prompt.
                retain_prompt_embeds = self._encode_prompt(
                    retain_prompts,
                    device,
                    num_waveforms_per_prompt,
                    False,
                    None,
                ).to(device=device, dtype=prompt_embeds.dtype)

        if steering_enabled:
            target_embed = self._encode_steering_target(
                steering_target,
                batch_size,
                num_waveforms_per_prompt,
                device,
                prompt_embeds.dtype,
            )
            unconditional_prompt_embeds, full_prompt_embeds = prompt_embeds.chunk(2)
            steering_prompt_embeds = torch.cat(
                [unconditional_prompt_embeds, full_prompt_embeds, retain_prompt_embeds], dim=0
            )
        else:
            target_embed = None
            steering_prompt_embeds = None

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        records: list[tuple[float, float]] = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                step_frac = i / num_inference_steps
                steering_active = steering_enabled and steering_frac_start <= step_frac < steering_frac_end

                # During an active steering step the UNet evaluates the common negative condition,
                # the full prompt and the retain prompt in one batch. Outside the window it keeps
                # the ordinary two-way CFG batch.
                if steering_active:
                    latent_model_input = torch.cat([latents] * 3)
                    step_prompt_embeds = steering_prompt_embeds
                elif do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    step_prompt_embeds = prompt_embeds
                else:
                    latent_model_input = latents
                    step_prompt_embeds = prompt_embeds
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        class_labels=step_prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    if steering_active:
                        noise_pred_uncond, noise_pred_full, noise_pred_retain = noise_pred.chunk(3)

                        if train:
                            # the predictor runs once per steered step and every one of those
                            # activations would be held until the backward pass; recomputing them
                            # instead is what makes an unrolled trajectory of this length fit.
                            #
                            # both details below are load-bearing. `use_reentrant=False`: on the
                            # first steered step `latents` carries no grad history, and the
                            # reentrant implementation would silently return no gradient for the
                            # predictor's own parameters. `t` passed as an argument rather than
                            # captured: the recomputation runs during the backward pass, by which
                            # point a captured loop variable holds the *last* timestep, and every
                            # recomputed step would be conditioned on the wrong one
                            alpha_t = torch.utils.checkpoint.checkpoint(
                                lambda hidden_states, timestep, embed: steering_model(
                                    latents=hidden_states, t=timestep, target_embed=embed
                                ),
                                latents,
                                t,
                                target_embed,
                                use_reentrant=False,
                            )
                        else:
                            with torch.no_grad():
                                alpha_t = steering_model(latents=latents, t=t, target_embed=target_embed)
                        records.append((float(t), float(alpha_t.detach().float().mean())))
                        noise_pred = self._interpolate_cfg_predictions(
                            noise_pred_uncond,
                            noise_pred_full,
                            noise_pred_retain,
                            guidance_scale,
                            alpha_t,
                        )
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        self.maybe_free_model_hooks()

        # 8. Post-processing
        with torch.no_grad():
            if output_type == "latent":
                audio = latents
            else:
                latents = 1 / self.vae.config.scaling_factor * latents
                mel_spectrogram = self.vae.decode(latents).sample
                audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
                audio = audio[:, :original_waveform_length]

                # 9. Automatic scoring
                if num_waveforms_per_prompt > 1 and prompt is not None:
                    audio = self.score_waveforms(
                        text=prompt,
                        audio=audio,
                        num_waveforms_per_prompt=num_waveforms_per_prompt,
                        device=device,
                        dtype=prompt_embeds.dtype,
                    )

        if output_type == "np":
            audio = audio.numpy()

        if not return_dict:
            return (audio, records)

        return SteeringAudioPipelineOutput(audios=audio, alpha_records=records)
