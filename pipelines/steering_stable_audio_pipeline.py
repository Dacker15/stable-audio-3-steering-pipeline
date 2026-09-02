r"""
Concept steering for Stable Audio 3.

Stable Audio 3 is not supported by diffusers, so there is no pipeline class to subclass. Its
classifier free guidance lives one level deeper, inside the model itself:
`stable_audio_3.models.dit.DiffusionTransformer.forward` builds the `[conditional, unconditional]`
batch, converts both predictions into denoised space, combines them and converts the result back.
The sampler never sees the two branches. Steering therefore overrides *that* component,
`SteeringDiffusionTransformer`, and `SteeringStableAudioPipeline` reimplements the parts of
`StableAudioModel.generate` and `sample_diffusion` needed to drive it: both are decorated with
`torch.inference_mode()` / `torch.no_grad()`, which would detach the trajectory the predictor is
trained through.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.utils.checkpoint
from stable_audio_3 import StableAudioModel
from stable_audio_3.data.utils import (
    compute_effective_seq_len_from_conditioning,
    create_padding_mask_from_lengths,
)
from stable_audio_3.inference.sampling import build_schedule
from stable_audio_3.models.dit import DiffusionTransformer
from stable_audio_3.models.lora import has_lora

from pipelines.cfg_diff_alpha import compute_cfg_diff_alpha, compute_cfg_diff_shape

# Stamped into `MagnitudePredictor` checkpoints trained against `"cfg_diff_magnitude"` mode:
# a predictor is only meaningful for the backbone and the guidance formula it was fitted against,
# so the evaluator rejects a checkpoint whose stamp doesn't match.
STEERING_MODE_MAGNITUDE = "sa3_cfg_diff_magnitude_v1"

# Learned steering models that embed the timestep with `diffusers`' `Timesteps` (e.g.
# `MagnitudePredictor`) have frequencies calibrated for the 0-1000 integer timesteps of a discrete
# scheduler. A rectified flow sigma lives in `[0, 1]`, where every one of those frequencies collapses
# towards `sin = 0, cos = 1` and the model would be effectively blind to where it is on the
# trajectory. Rescaling restores the range the embedding was designed for without touching the model.
PREDICTOR_TIMESTEP_SCALE = 1000.0


@dataclass
class SteeringAudioPipelineOutput:
    r"""
    Output class for `SteeringStableAudioPipeline`.

    Args:
        audios (`np.ndarray` or `torch.Tensor`): Generated audio of shape `(batch_size, channels,
            samples)`, or latents of shape `(batch_size, latent_channels, frames)` when
            `output_type="latent"`.
        alpha_records (`list[tuple[int, float]]`): `(step, alpha_t)` pairs recorded at every
            steering-active denoising step, in denoising order (noisiest to cleanest). These carry
            the loop step index rather than the timestep, because a rectified flow sigma is a
            continuous float and, with per-element schedules, differs across the batch.
        padding_mask (`torch.Tensor` or `None`): Boolean mask of shape `(batch_size, frames)` marking
            the latent frames that carry audio rather than padding. Needed to decode latents outside
            the pipeline, see `SteeringStableAudioPipeline.decode_latents`.
        alpha_field_records (`list[dict]`): One dict per steering-active denoising step, populated
            only in `"cfg_diff_magnitude"` mode (empty for `"learned"` and `"cfg_diff"`). Each holds
            the loop `step` index plus `alpha_field`, `magnitude`, `shape`, `steered_xhat0` and
            `fullcfg_xhat0`; graph-connected when the call was made with `train=True`, plain tensors
            otherwise.
    """

    audios: Any
    alpha_records: list[tuple[int, float]] = field(default_factory=list)
    padding_mask: torch.Tensor | None = None
    alpha_field_records: list[dict] = field(default_factory=list)


@dataclass
class SteeringState:
    r"""
    Everything `SteeringDiffusionTransformer.forward` needs that the sampler cannot pass it.

    `DiTWrapper.forward` forwards its `**kwargs` all the way down into the transformer, so unknown
    keyword arguments do not stop at the guidance layer; the steering inputs travel as state on the
    module instead. The step index has to be mutated per iteration anyway, which keyword arguments
    could not express.
    """

    retain_cross_attn_cond: torch.Tensor
    steering_model: torch.nn.Module | None
    target_embed: torch.Tensor
    num_steps: int
    frac_start: float
    frac_end: float
    train: bool = False
    step: int | None = None
    records: list[tuple[int, float]] = field(default_factory=list)
    # `mode="cfg_diff"`: deterministic, per-frame `alpha_t` from the CFG-diff norm, computed by
    # `compute_cfg_diff_alpha` instead of calling `steering_model`, sharing `alpha_min`/`alpha_max`
    # with `"learned"` mode's own bounds for compatibility between modes. `magnitude` is the fixed
    # float below.
    # `mode="cfg_diff_magnitude"`: same deterministic `shape`, but `magnitude` is a learned
    # per-sample scalar from `steering_model` (a `MagnitudePredictor`) instead of the fixed float.
    mode: Literal["learned", "cfg_diff", "cfg_diff_magnitude"] = "learned"
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    magnitude: float = 0.6
    quantile_low: float = 0.10
    quantile_high: float = 0.90
    # Populated only in `"cfg_diff_magnitude"` mode, see `SteeringAudioPipelineOutput.alpha_field_records`.
    alpha_field_records: list[dict] = field(default_factory=list)

    def is_active(self, cfg_scale: float) -> bool:
        if self.step is None or cfg_scale == 1.0:
            return False
        step_frac = self.step / self.num_steps
        return self.frac_start <= step_frac < self.frac_end


class SteeringDiffusionTransformer(DiffusionTransformer):
    r"""
    `DiffusionTransformer` variant that steers denoising away from a target concept over a sub-range
    of the denoising trajectory.

    Inside the steering window the transformer evaluates three conditionings in one batch — the full
    prompt, the shared negative (or null) condition and a `retain_prompt` with the target removed —
    and interpolates between the two resulting classifier-free guidance predictions by a per-step
    scalar `alpha_t` predicted by an external `steering_model`. `alpha_t = 0` reproduces the ordinary
    full-prompt generation and `alpha_t = 1` follows the retain prompt. Outside the window, and
    whenever no steering state is installed, behaviour is identical to the base
    `DiffusionTransformer`.

    Stable Audio 3 builds its transformer inside `factory.create_diffusion_cond_from_config` from a
    JSON config, so there is no constructor hook to pass a subclass through. `install` therefore
    rebinds `__class__` on the already loaded instance.
    """

    steering: SteeringState | None = None

    @classmethod
    def install(cls, transformer: DiffusionTransformer) -> "SteeringDiffusionTransformer":
        if isinstance(transformer, cls):
            return transformer
        if not isinstance(transformer, DiffusionTransformer):
            raise TypeError(f"`transformer` has to be a `DiffusionTransformer` but is {type(transformer)}")
        transformer.__class__ = cls
        transformer.steering = None
        return transformer

    @staticmethod
    def uninstall(transformer: "SteeringDiffusionTransformer") -> DiffusionTransformer:
        transformer.steering = None
        transformer.__class__ = DiffusionTransformer
        return transformer

    @staticmethod
    def _cfg_predictions(
        pred_uncond,
        pred_full,
        pred_retain,
        guidance_scale,
        cfg_diff_full=None,
        cfg_diff_retain=None,
    ):
        r"""
        Computes the full-prompt and retain-prompt guided predictions, in the denoised space Stable
        Audio 3 guides in. `alpha_t` — from either steering mode — interpolates between the two.

        With the default `cfg_diff_*`, i.e. plain classifier free guidance, each is
        `uncond + guidance_scale * (cond - uncond)`. Adaptive projected guidance replaces the
        difference with its projected counterpart, which is why they can be passed in.
        """
        if cfg_diff_full is None:
            cfg_diff_full = pred_full - pred_uncond
        if cfg_diff_retain is None:
            cfg_diff_retain = pred_retain - pred_uncond

        full_cfg = pred_full + (guidance_scale - 1.0) * cfg_diff_full
        retain_cfg = pred_retain + (guidance_scale - 1.0) * cfg_diff_retain
        return full_cfg, retain_cfg

    def _guidance_diff(self, cond_denoised, uncond_denoised, apg_scale, padding_mask, cfg_norm_threshold):
        r"""Mirrors the norm-clipping and adaptive projected guidance of the base transformer."""
        diff = cond_denoised - uncond_denoised

        if cfg_norm_threshold > 0:
            if padding_mask is not None:
                diff_norm = (diff * padding_mask.unsqueeze(1).float()).norm(p=2, dim=[-1, -2], keepdim=True)
            else:
                diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
            diff = diff * torch.minimum(torch.ones_like(diff), cfg_norm_threshold / diff_norm)

        if apg_scale == 0.0:
            return diff

        _, diff_orthogonal = self.apg_project(diff, cond_denoised, padding_mask=padding_mask)
        if apg_scale == 1.0:
            return diff_orthogonal
        return apg_scale * diff_orthogonal + (1 - apg_scale) * diff

    def _predict_alpha(self, state: SteeringState, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""
        Returns `alpha_t` of shape `(batch_size, 1, 1)`, broadcasting over Stable Audio 3's
        `(batch_size, channels, frames)` latents.

        `"learned"` mode adds a singleton height axis here rather than requiring every steering model
        to do it, so 2D-convolutional-encoder-style models can consume the latents as a spectrogram
        of height 1 without any special-casing elsewhere.
        """
        hidden_states = latents.unsqueeze(-2)
        timestep = t * PREDICTOR_TIMESTEP_SCALE

        if state.train:
            # the predictor runs once per steered step and every one of those activations would be
            # held until the backward pass; recomputing them instead is what makes an unrolled
            # trajectory of this length fit.
            #
            # both details below are load-bearing. `use_reentrant=False`: on the first steered step
            # `latents` carries no grad history, and the reentrant implementation would silently
            # return no gradient for the predictor's own parameters. `timestep` passed as an
            # argument rather than captured: the recomputation runs during the backward pass, by
            # which point a captured loop variable holds the *last* timestep, and every recomputed
            # step would be conditioned on the wrong one
            alpha = torch.utils.checkpoint.checkpoint(
                lambda hidden, step_t, embed: state.steering_model(latents=hidden, t=step_t, target_embed=embed),
                hidden_states,
                timestep,
                state.target_embed,
                use_reentrant=False,
            )
        else:
            with torch.no_grad():
                alpha = state.steering_model(latents=hidden_states, t=timestep, target_embed=state.target_embed)

        return alpha.view(-1, 1, 1)

    def _predict_magnitude(self, state: SteeringState, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""
        Returns `magnitude` of shape `(batch_size, 1, 1)`, the learned gain on the deterministic
        CFG-diff `shape` profile.

        Same checkpointing rationale as `_predict_alpha`, but `MagnitudePredictor` reads the latents
        directly (no convolutional encoder, so no singleton height axis to add).
        """
        timestep = t * PREDICTOR_TIMESTEP_SCALE

        if state.train:
            magnitude = torch.utils.checkpoint.checkpoint(
                lambda latents_, step_t, embed: state.steering_model(latents=latents_, t=step_t, target_embed=embed),
                latents,
                timestep,
                state.target_embed,
                use_reentrant=False,
            )
        else:
            with torch.no_grad():
                magnitude = state.steering_model(latents=latents, t=timestep, target_embed=state.target_embed)

        return magnitude.view(-1, 1, 1)

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval=(0, 1),
        lora_interval=(0, 1),
        lora_layer_filter="",
        lora_configs=None,
        causal=False,
        scale_phi=0.0,
        cfg_norm_threshold=0.0,
        apg_scale=1.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs,
    ):
        state = self.steering
        # `return_info` and `exit_layer_ix` are diagnostic paths of the base implementation that the
        # steered batch has no meaning for, so they fall back to it as well.
        if state is None or return_info or exit_layer_ix is not None or not state.is_active(cfg_scale):
            return super().forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                negative_cross_attn_cond=negative_cross_attn_cond,
                negative_cross_attn_mask=negative_cross_attn_mask,
                input_concat_cond=input_concat_cond,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
                global_embed=global_embed,
                negative_global_embed=negative_global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                cfg_scale=cfg_scale,
                cfg_dropout_prob=cfg_dropout_prob,
                cfg_interval=cfg_interval,
                lora_interval=lora_interval,
                lora_layer_filter=lora_layer_filter,
                lora_configs=lora_configs,
                causal=causal,
                scale_phi=scale_phi,
                cfg_norm_threshold=cfg_norm_threshold,
                apg_scale=apg_scale,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs,
            )

        if cross_attn_cond is None:
            raise ValueError("steering needs a cross-attention conditioned model, but `cross_attn_cond` is None")
        if has_lora(self):
            # the base implementation enables and disables LoRA per sigma interval, which the steered batch would silently skip
            raise RuntimeError("steering does not support LoRA adapters")

        model_dtype = next(self.parameters()).dtype
        # keep `t` in float32 for the same reason the base implementation does: the logsnr transform
        # amplifies half precision error enormously near t = 1
        t = t.float()
        latents = x.float()

        def triple(tensor):
            return None if tensor is None else torch.cat([tensor] * 3, dim=0)

        cross_attn_cond = cross_attn_cond.to(model_dtype)
        null_embed = torch.zeros_like(cross_attn_cond)
        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)
            if negative_cross_attn_mask is not None:
                negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
                negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
        else:
            negative_cross_attn_cond = null_embed

        batch_cross_attn_cond = torch.cat(
            [cross_attn_cond, negative_cross_attn_cond, state.retain_cross_attn_cond.to(model_dtype)], dim=0
        )

        # The unconditional branch is shared by both guidance computations, exactly as it is by the
        # base implementation's single one, so the retain prompt costs one extra forward, not two.
        with torch.no_grad():
            batch_output = self._forward(
                torch.cat([latents.to(model_dtype)] * 3, dim=0),
                triple(t),
                cross_attn_cond=batch_cross_attn_cond,
                # the base implementation discards this mask unconditionally, see the flash attention
                # note in `DiffusionTransformer.forward`
                cross_attn_cond_mask=None,
                mask=triple(mask),
                input_concat_cond=triple(input_concat_cond),
                local_add_cond=triple(local_add_cond),
                modular_local_cond=(
                    None if modular_local_cond is None else {k: triple(v) for k, v in modular_local_cond.items()}
                ),
                global_embed=triple(global_embed),
                prepend_cond=triple(prepend_cond),
                prepend_cond_mask=triple(prepend_cond_mask),
                padding_mask=triple(padding_mask),
                **kwargs,
            )

        # Everything from here on stays in float32 and outside `no_grad`: the transformer's outputs
        # are constants with respect to the predictor, but `alpha_t` is not, and this is the only
        # place its gradient enters the trajectory.
        #
        # Float32 is not only about the gradient. The final division by `sigma` undoes a subtraction
        # of the same magnitude, so as the loop approaches `sigma = 0` the base implementation's half
        # precision arithmetic cancels catastrophically. With `alpha_t = 0` this branch reproduces
        # the base implementation to a relative 1e-5 when both run in float32, but differs from it by
        # a relative 1e-2 when the transformer is loaded in half precision — that gap is the base
        # implementation's own error, not this one's.
        full_output, uncond_output, retain_output = (chunk.float() for chunk in torch.chunk(batch_output, 3, dim=0))

        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha_bar = torch.cos(t * math.pi / 2)
            base = latents * alpha_bar[:, None, None]
        elif self.diffusion_objective in ("rectified_flow", "rf_denoiser"):
            sigma = t
            base = latents
        else:
            raise ValueError(f"unsupported diffusion objective {self.diffusion_objective!r}")

        sigma = sigma[:, None, None]
        full_denoised = base - full_output * sigma
        uncond_denoised = base - uncond_output * sigma
        retain_denoised = base - retain_output * sigma

        full_cfg, retain_cfg = self._cfg_predictions(
            uncond_denoised,
            full_denoised,
            retain_denoised,
            cfg_scale,
            cfg_diff_full=self._guidance_diff(
                full_denoised, uncond_denoised, apg_scale, padding_mask, cfg_norm_threshold
            ),
            cfg_diff_retain=self._guidance_diff(
                retain_denoised, uncond_denoised, apg_scale, padding_mask, cfg_norm_threshold
            ),
        )

        shape_t = None
        magnitude_t = None
        if state.mode == "learned":
            alpha_t = self._predict_alpha(state, latents, t)
        elif state.mode == "cfg_diff":
            alpha_t = compute_cfg_diff_alpha(
                full_cfg,
                retain_cfg,
                state.alpha_min,
                state.alpha_max,
                state.magnitude,
                quantile_low=state.quantile_low,
                quantile_high=state.quantile_high,
            )
        else:  # "cfg_diff_magnitude"
            magnitude_t = self._predict_magnitude(state, latents, t)
            shape_t = compute_cfg_diff_shape(full_cfg, retain_cfg, state.quantile_low, state.quantile_high)
            alpha_t = (state.alpha_min + (state.alpha_max - state.alpha_min) * magnitude_t * shape_t).clamp(
                state.alpha_min, state.alpha_max
            )
        state.records.append((state.step, float(alpha_t.detach().float().mean())))

        steered_denoised = full_cfg + alpha_t * (retain_cfg - full_cfg)

        if state.mode == "cfg_diff_magnitude":
            # collected whether or not `state.train`, so validation (which calls with `train=False`)
            # can still compute `l_reg`/`l_fid`/diagnostics; graph-connected for `alpha_field`/
            # `steered_xhat0` only when there is a graph to connect to, i.e. training
            state.alpha_field_records.append(
                {
                    "step": state.step,
                    "alpha_field": alpha_t,
                    "magnitude": magnitude_t.detach(),
                    "shape": shape_t.detach(),
                    "steered_xhat0": steered_denoised,
                    "fullcfg_xhat0": full_cfg.detach(),
                }
            )

        output = (base - steered_denoised) / sigma

        if scale_phi != 0.0:
            output = scale_phi * (output * (full_output.std(dim=1, keepdim=True) / output.std(dim=1, keepdim=True))) + (
                1 - scale_phi
            ) * output

        return output


class SteeringStableAudioPipeline:
    r"""
    Runs Stable Audio 3 with a `SteeringDiffusionTransformer` in place of its stock guidance.

    `StableAudioModel` is a plain object rather than an `nn.Module`, and its `from_pretrained` is a
    static method that hard-codes its own constructor, so this composes it instead of subclassing it.

    Only the `-base` checkpoints are accepted. The post-trained ones are distilled and ignore
    `cfg_scale` entirely, and steering is applied inside the classifier free guidance branch, so on
    those checkpoints the predictor would receive no gradient and change nothing.
    """

    def __init__(self, model: StableAudioModel, model_name: str | None = None):
        self.model = model
        self.model_name = model_name
        self.diffusion = model.model  # ConditionedDiffusionModelWrapper
        self.dit_wrapper = self.diffusion.model  # DiTWrapper, what the samplers call
        self.transformer = SteeringDiffusionTransformer.install(self.dit_wrapper.model)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "medium-base",
        device: str | torch.device | None = None,
        model_half: bool = True,
    ) -> "SteeringStableAudioPipeline":
        r"""
        Args:
            model_name (`str`, *optional*, defaults to `"medium-base"`): A Stable Audio 3 checkpoint.
                Has to be one of the `-base` ones.
            device (`str` or `torch.device`, *optional*): Defaults to cuda, then mps, then cpu.
            model_half (`bool`, *optional*, defaults to `True`): Whether to load the transformer in
                half precision. The autoencoder is kept in float32 either way, since the training
                gradient reaches `alpha_t` through its decoder.
        """
        if not model_name.endswith("-base"):
            raise ValueError(
                f"`model_name` has to be a `-base` checkpoint but is {model_name!r}. The post-trained checkpoints are"
                " distilled and ignore classifier free guidance, which is where steering is applied, so the steering"
                " model would have no effect on them."
            )

        model = StableAudioModel.from_pretrained(model_name, device=device, model_half=model_half)
        pipeline = cls(model, model_name=model_name)

        # The latent trajectory, the guidance algebra and the decoder all stay in float32: the
        # gradient reaches `alpha_t` through the Euler recurrence and then through the autoencoder,
        # and in half precision that chain is not reliable enough to optimize against.
        pipeline.diffusion.pretransform.float()

        # `AudioAutoencoder.decode` wraps its own inner pretransform in `torch.no_grad()` unless this
        # flag is set, which would silently detach the waveform from the latents and leave the
        # training loss with nothing to backpropagate through.
        inner_pretransform = pipeline.diffusion.pretransform.model.pretransform
        if inner_pretransform is not None:
            inner_pretransform.enable_grad = True

        return pipeline

    @property
    def device(self) -> torch.device:
        return torch.device(self.model.device)

    @property
    def sample_rate(self) -> int:
        return int(self.diffusion.sample_rate)

    @property
    def io_channels(self) -> int:
        return int(self.diffusion.io_channels)

    @property
    def downsampling_ratio(self) -> int:
        return int(self.diffusion.pretransform.downsampling_ratio)

    @property
    def max_duration_in_s(self) -> float:
        return float(self.model.model_config["sample_size"]) / self.sample_rate

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

    def _encode_conditioning(self, prompts, audio_length_in_s, mask, masked_input, negative=False):
        r"""Runs the T5Gemma conditioner and lays the result out the way the transformer expects."""
        conditioning = [{"prompt": prompt, "seconds_total": audio_length_in_s} for prompt in prompts]
        tensors = self.diffusion.conditioner(conditioning, str(self.device))
        # inpainting conditioning is always present in these configs; steering never inpaints, so it
        # is the same all-zero mask the stock pipeline builds for plain text-to-audio
        tensors["inpaint_mask"] = [mask]
        tensors["inpaint_masked_input"] = [masked_input]
        return self.diffusion.get_conditioning_inputs(tensors, negative=negative)

    def _prepare_latents(self, batch_size, num_frames, generator):
        shape = (batch_size, self.io_channels, num_frames)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"`generator` has batch size {len(generator)}, but `prompt` has batch size {batch_size}."
                )
            latents = torch.cat(
                [torch.randn((1, *shape[1:]), generator=g, device=g.device, dtype=torch.float32) for g in generator]
            )
        else:
            device = self.device if generator is None else generator.device
            latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        return latents.to(self.device)

    def decode_latents(
        self,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        audio_length_in_s: float | None = None,
        chunked: bool = False,
        clamp: bool = False,
    ) -> torch.Tensor:
        r"""
        Decodes latents to a waveform of shape `(batch_size, channels, samples)`, differentiably.

        Args:
            latents (`torch.Tensor`): Latents of shape `(batch_size, latent_channels, frames)`.
            padding_mask (`torch.Tensor`, *optional*): The mask returned by `__call__`. The frames it
                marks as padding decode to garbage and are zeroed, as the stock sampler does.
            audio_length_in_s (`float`, *optional*): Truncates the waveform to this many seconds.
            chunked (`bool`, *optional*, defaults to `False`): Whether to decode in overlapping
                chunks. Cheaper in VRAM, but the stitching is unnecessary for short clips.
            clamp (`bool`, *optional*, defaults to `False`): Whether to clamp to `[-1, 1]`. Left off
                for training, where a clamp would zero the gradient of every saturated sample.
        """
        pretransform = self.diffusion.pretransform
        audio = pretransform.decode(latents.to(next(pretransform.parameters()).dtype), chunked=chunked)

        if padding_mask is not None:
            audio_mask = padding_mask.unsqueeze(1).repeat_interleave(self.downsampling_ratio, dim=-1)
            if audio_mask.shape[-1] > audio.shape[-1]:
                audio_mask = audio_mask[..., : audio.shape[-1]]
            elif audio_mask.shape[-1] < audio.shape[-1]:
                audio_mask = torch.nn.functional.pad(audio_mask, (0, audio.shape[-1] - audio_mask.shape[-1]))
            audio = audio * audio_mask.to(audio.dtype)

        if clamp:
            audio = audio.clamp(-1, 1)
        if audio_length_in_s is not None:
            audio = audio[:, :, : int(audio_length_in_s * self.sample_rate)]
        return audio

    def __call__(
        self,
        prompt: str | list[str],
        retain_prompt: str | list[str] | None = None,
        target_embed: torch.Tensor | None = None,
        steering_model: torch.nn.Module | None = None,
        steering_mode: Literal["learned", "cfg_diff", "cfg_diff_magnitude"] = "learned",
        steering_frac_start: float = 0.0,
        steering_frac_end: float = 1.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        alpha_magnitude: float = 0.6,
        alpha_shape_quantile_low: float = 0.10,
        alpha_shape_quantile_high: float = 0.90,
        num_inference_steps: int = 50,
        audio_length_in_s: float = 10.0,
        cfg_scale: float = 7.0,
        negative_prompt: str | list[str] | None = None,
        apg_scale: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        duration_padding_sec: float = 6.0,
        chunked_decode: bool = False,
        output_type: str = "np",
        return_dict: bool = True,
        train: bool = False,
    ):
        r"""
        Args:
            prompt (`str` or `list[str]`): The prompt(s) to generate.
            retain_prompt (`str` or `list[str]`): Prompt(s) describing what generation should
                preserve, normally the full prompt with the steered concept removed. Inside the
                steering window `alpha_t` interpolates from full-prompt guidance (`alpha_t = 0`) to
                retain-prompt guidance (`alpha_t = 1`).
            target_embed (`torch.Tensor`): CLAP embedding of the steered concept, of shape
                `(batch_size, target_embed_dim)`, passed straight to `steering_model`. Stable Audio 3
                conditions on T5Gemma and has no CLAP tower of its own, so the caller supplies this;
                `losses.ClapLoss.encode_text` produces it in the same space the training loss scores.
            steering_model (`torch.nn.Module`): Required when `steering_mode` is `"learned"` or
                `"cfg_diff_magnitude"`. Called as `steering_model(latents=..., t=...,
                target_embed=...)` to predict `alpha_t` (`"learned"`) or `magnitude`
                (`"cfg_diff_magnitude"`, a `MagnitudePredictor`). Has to be `None` when
                `steering_mode` is `"cfg_diff"`, since that mode needs no learned model.
            steering_mode (`"learned"`, `"cfg_diff"` or `"cfg_diff_magnitude"`, *optional*, defaults
                to `"learned"`): `"learned"` calls `steering_model` for the full per-frame `alpha_t`,
                as before. `"cfg_diff"` derives a deterministic, per-frame `alpha_t` from the norm of
                the CFG-diff between the full-prompt and retain-prompt guidance
                (`pipelines.compute_cfg_diff_alpha`), at zero training cost. `"cfg_diff_magnitude"`
                keeps that same deterministic per-frame `shape`
                (`pipelines.compute_cfg_diff_shape`) but replaces the fixed `alpha_magnitude` gain
                with a single learned scalar per sample from `steering_model`. Either way the
                steering window and CFG algebra are identical; only the source of `alpha_t` differs.
            steering_frac_start (`float`, *optional*, defaults to 0.0): Fraction of the denoising loop
                (by step index) where steering begins.
            steering_frac_end (`float`, *optional*, defaults to 1.0): Fraction of the denoising loop
                (by step index) where steering ends.
            alpha_min, alpha_max (`float`, *optional*, defaults to 0.0 and 1.0): Bounds of `alpha_t`
                in `"cfg_diff"`/`"cfg_diff_magnitude"` mode. Unused in `"learned"` mode, where the
                bounds live in `steering_model` instead.
            alpha_magnitude (`float`, *optional*, defaults to 0.6): Global gain in `[0, 1]` for
                `"cfg_diff"` mode, a fixed hyperparameter rather than a learned one. Unused in
                `"learned"` and `"cfg_diff_magnitude"` mode, where the gain is learned instead.
            alpha_shape_quantile_low, alpha_shape_quantile_high (`float`, *optional*, defaults to
                0.10 and 0.90): Percentiles used by `"cfg_diff"`/`"cfg_diff_magnitude"` mode to
                normalize the temporal profile per sample. Unused in `"learned"` mode.
            cfg_scale (`float`, *optional*, defaults to 7.0): Classifier free guidance scale. Has to
                differ from 1.0 for steering to happen at all.
            apg_scale (`float`, *optional*, defaults to 0.0): 0.0 is plain classifier free guidance,
                1.0 is full adaptive projected guidance. The default differs from Stable Audio 3's
                own, which is 1.0, so that `alpha_t` interpolates between two ordinary guidance
                predictions.
            output_type (`str`, *optional*, defaults to `"np"`): One of `"np"`, `"pt"` or
                `"latent"`. `"latent"` returns the raw trajectory endpoint with its graph intact,
                which is what training differentiates through.
            train (`bool`, *optional*, defaults to `False`): If `True`, gradients are enabled for the
                `steering_model` call and its activations are recomputed rather than stored.
        """
        if output_type not in ("np", "pt", "latent"):
            raise ValueError(f"`output_type` has to be one of 'np', 'pt' or 'latent' but is {output_type!r}")
        if num_inference_steps < 1:
            raise ValueError(f"`num_inference_steps` has to be at least 1 but is {num_inference_steps}")
        if not 0.0 < audio_length_in_s <= self.max_duration_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be in (0, {self.max_duration_in_s:.1f}] but is {audio_length_in_s}"
            )
        if not 0.0 <= steering_frac_start < steering_frac_end <= 1.0:
            raise ValueError(
                "`steering_frac_start` and `steering_frac_end` have to satisfy `0.0 <= start < end <= 1.0` but are"
                f" {steering_frac_start} and {steering_frac_end}"
            )

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(prompts)
        device = self.device

        if steering_mode not in ("learned", "cfg_diff", "cfg_diff_magnitude"):
            raise ValueError(
                f"`steering_mode` has to be 'learned', 'cfg_diff' or 'cfg_diff_magnitude' but is {steering_mode!r}"
            )
        if steering_mode == "cfg_diff" and steering_model is not None:
            raise ValueError(
                "`steering_model` has to be `None` when `steering_mode` is 'cfg_diff': that mode derives"
                " `alpha_t` deterministically from the CFG-diff and needs no learned model, so no steering model"
                " should even be instantiated for it."
            )

        steering_requested = steering_model is not None or steering_mode == "cfg_diff"
        steering_enabled = cfg_scale != 1.0 and steering_requested
        if steering_requested and cfg_scale == 1.0:
            raise ValueError(
                "`cfg_scale` has to differ from 1.0 for steering to be applied, because Stable Audio 3 skips its"
                " guidance branch entirely at 1.0 and steering would never be applied."
            )

        if steering_enabled:
            if target_embed is None:
                raise ValueError("`target_embed` must be provided when `steering_model` is enabled")
            if target_embed.ndim != 2 or target_embed.shape[0] != batch_size:
                raise ValueError(
                    f"`target_embed` has to be of shape `(batch_size, target_embed_dim)` with batch size {batch_size}"
                    f" but has shape {tuple(target_embed.shape)}"
                )
            retain_prompts = self._prepare_retain_prompt(retain_prompt, batch_size)
        else:
            retain_prompts = None

        # 1. Sizes. Stable Audio 3 generates a padded window and trims it afterwards, so the latent
        # length follows the same rounding the stock pipeline applies.
        conditioning = [{"prompt": p, "seconds_total": audio_length_in_s} for p in prompts]
        sample_size = int(self.model.model_config["sample_size"])
        audio_sample_size = self.model._adapt_sample_size(conditioning, sample_size, duration_padding_sec)
        num_frames = audio_sample_size // self.downsampling_ratio

        # 2. Latents
        if latents is None:
            latents = self._prepare_latents(batch_size, num_frames, generator)
        else:
            latents = latents.to(device=device, dtype=torch.float32)

        # 3. Conditioning, including the all-zero inpainting inputs the model always expects
        inpaint_mask = torch.zeros((batch_size, 1, num_frames), device=device)
        inpaint_masked_input = torch.zeros((batch_size, self.io_channels, num_frames), device=device)

        conditioning_inputs = self._encode_conditioning(
            prompts, audio_length_in_s, inpaint_mask, inpaint_masked_input
        )
        if negative_prompt is not None:
            negative_prompts = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else list(negative_prompt)
            if len(negative_prompts) != batch_size:
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompts)}, but `prompt` has batch size"
                    f" {batch_size}."
                )
            negative_inputs = self._encode_conditioning(
                negative_prompts, audio_length_in_s, inpaint_mask, inpaint_masked_input, negative=True
            )
        else:
            negative_inputs = {}

        retain_cross_attn_cond = None
        if steering_enabled:
            # only the conditional retain embedding is needed: during steering it shares the same
            # unconditional prediction already computed for the full prompt
            retain_cross_attn_cond = self._encode_conditioning(
                retain_prompts, audio_length_in_s, inpaint_mask, inpaint_masked_input
            )["cross_attn_cond"]

        model_dtype = next(self.transformer.parameters()).dtype
        conditioning_inputs = {
            key: value.type(model_dtype) if torch.is_tensor(value) else value
            for key, value in conditioning_inputs.items()
        }

        # 4. Schedule and padding mask, both derived from the requested duration exactly as
        # `sample_diffusion` derives them
        effective_seq_len = compute_effective_seq_len_from_conditioning(
            conditioning, self.sample_rate, self.downsampling_ratio, device
        )
        padding_mask = None
        if effective_seq_len is not None:
            headroom_tokens = int(duration_padding_sec * self.sample_rate / self.downsampling_ratio)
            valid_lengths = (effective_seq_len + headroom_tokens).clamp(max=num_frames).long()
            padding_mask = create_padding_mask_from_lengths(valid_lengths, num_frames)

        sigmas = build_schedule(
            steps=num_inference_steps,
            sigma_max=1.0,
            dist_shift=self.diffusion.sampling_dist_shift,
            effective_seq_len=effective_seq_len,
            fallback_seq_len=num_frames,
            include_endpoint=True,
            device=device,
        ).to(device)

        model_kwargs = {
            **conditioning_inputs,
            **negative_inputs,
            "cfg_scale": cfg_scale,
            "batch_cfg": True,
            "apg_scale": apg_scale,
            "padding_mask": padding_mask,
        }

        # 5. Denoising loop. This is `sample_discrete_euler` without the `torch.no_grad()` its caller
        # applies, and with the loop index handed to the steering state. Euler is the only sampler
        # supported here: Stable Audio 3 defaults to it for rectified flow anyway, and the
        # higher-order ones call the model several times per step, which the step-indexed steering
        # window could not describe.
        state = None
        if steering_enabled:
            state = SteeringState(
                retain_cross_attn_cond=retain_cross_attn_cond,
                steering_model=steering_model,
                target_embed=target_embed,
                num_steps=num_inference_steps,
                frac_start=steering_frac_start,
                frac_end=steering_frac_end,
                train=train,
                mode=steering_mode,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                magnitude=alpha_magnitude,
                quantile_low=alpha_shape_quantile_low,
                quantile_high=alpha_shape_quantile_high,
            )
        self.transformer.steering = state

        try:
            per_element_schedule = sigmas.dim() == 2
            for step in range(num_inference_steps):
                if per_element_schedule:
                    t_curr = sigmas[:, step].float()
                    dt = (sigmas[:, step + 1].float() - t_curr).view(-1, 1, 1)
                else:
                    t_curr = sigmas[step].float() * torch.ones((batch_size,), device=device)
                    dt = sigmas[step + 1].float() - sigmas[step].float()

                if state is not None:
                    state.step = step

                velocity = self.dit_wrapper(latents, t_curr, **model_kwargs)
                latents = latents + dt * velocity.float()
        finally:
            self.transformer.steering = None

        records = list(state.records) if state is not None else []
        alpha_field_records = list(state.alpha_field_records) if state is not None else []

        # 6. Post-processing
        if output_type == "latent":
            audio = latents
        else:
            audio = self.decode_latents(
                latents,
                padding_mask=padding_mask,
                audio_length_in_s=audio_length_in_s,
                chunked=chunked_decode,
                clamp=True,
            )
            if output_type == "np":
                audio = audio.detach().float().cpu().numpy()

        if not return_dict:
            return (audio, records)

        return SteeringAudioPipelineOutput(
            audios=audio, alpha_records=records, padding_mask=padding_mask, alpha_field_records=alpha_field_records
        )
