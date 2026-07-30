from typing import Any, Callable

import numpy as np
import torch
from diffusers import MusicLDMPipeline
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput
from diffusers.utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class SteeringMusicLDMPipeline(MusicLDMPipeline):
    r"""
    `MusicLDMPipeline` variant that steers denoising towards/away from a target concept
    over a sub-range of the denoising trajectory, using an external `steering_model`
    that predicts a per-step scalar `alpha_t`. Inside the steering window, the CFG
    guidance scale is replaced by `(1.0 - 2.0 * alpha_t) * guidance_scale`; outside the
    window, generation is identical to the base `MusicLDMPipeline`.
    """

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

        with torch.no_grad():
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
        steering_model: torch.nn.Module = None,
        steering_frac_start: float = 0.0,
        steering_frac_end: float = 1.0,
        train: bool = False,
    ):
        r"""
        Extends `MusicLDMPipeline.__call__` with concept steering.

        Args:
            steering_target (`str` or `list[str]`): The concept(s) the `steering_model` should steer generation towards/away from. If a list, must match the batch size of `prompt`.
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

        # 3. Encode input prompt and steering target
        with torch.no_grad():
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_waveforms_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

        target_embed = self._encode_steering_target(
            steering_target,
            batch_size,
            num_waveforms_per_prompt,
            device,
            prompt_embeds.dtype,
        )

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
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        class_labels=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    step_frac = i / num_inference_steps
                    steering_active = steering_frac_start <= step_frac < steering_frac_end

                    if steering_active:
                        with torch.set_grad_enabled(train):
                            alpha_t = steering_model(latents=latents, t=t, target_embed=target_embed)
                        effective_guidance_scale = (1.0 - 2.0 * alpha_t) * guidance_scale
                    else:
                        effective_guidance_scale = guidance_scale

                    noise_pred = noise_pred_uncond + effective_guidance_scale * (noise_pred_text - noise_pred_uncond)

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
            if not output_type == "latent":
                latents = 1 / self.vae.config.scaling_factor * latents
                mel_spectrogram = self.vae.decode(latents).sample
            else:
                return AudioPipelineOutput(audios=latents)

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
            return (audio,)

        return AudioPipelineOutput(audios=audio)
