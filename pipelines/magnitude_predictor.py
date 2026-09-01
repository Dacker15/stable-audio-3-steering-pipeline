import math

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


class MagnitudePredictor(nn.Module):
    r"""
    Predicts a single scalar `magnitude` in `[0, 1]` per sample, the one learned degree of freedom in
    `SteeringStableAudioPipeline`'s `"cfg_diff_magnitude"` mode:
    `alpha_field = alpha_min + (alpha_max - alpha_min) * magnitude * shape`, where `shape` stays
    entirely deterministic (`pipelines.compute_cfg_diff_shape`, the same profile `"cfg_diff"` mode
    uses with a fixed `magnitude`).

    Unlike a learned model that predicts a full per-frame `alpha_t` and therefore reads the latents'
    spatial structure with a convolutional encoder, this only has to produce one number per sample:
    the latents are pooled into whole-clip statistics up front and the network never looks at their
    frame-by-frame structure. Conditioning: the current timestep, the CLAP embedding of the steering
    target, and the latents' own scale (which spans orders of magnitude along the trajectory, so the
    pooled per-channel statistics are normalized by it before the head reads them, with the log-scale
    fed back in as an explicit feature).

    Args:
        latent_channels (`int`, *optional*, defaults to 8): Number of latent channels, i.e.
            `io_channels` of the pipeline.
        target_embed_dim (`int`, *optional*, defaults to 512): Dimension of `target_embed`, i.e. the
            CLAP `projection_dim` of `losses.ClapLoss`.
        cond_embed_dim (`int`, *optional*, defaults to 128): Dimension of the conditioning vector and
            width of the readout MLP.
        hidden_dim (`int`, *optional*, defaults to 128): Width of the readout MLP's hidden layer.
        magnitude_init (`float`, *optional*, defaults to 0.15): Value every sample is initialized to.
            Must lie strictly inside `(0, 1)`. Starts with a modest steering strength without placing
            the sigmoid close to saturation.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        target_embed_dim: int = 512,
        cond_embed_dim: int = 128,
        hidden_dim: int = 128,
        magnitude_init: float = 0.15,
    ):
        super().__init__()

        if not 0.0 < magnitude_init < 1.0:
            raise ValueError(f"`magnitude_init` has to lie strictly inside (0, 1) but is {magnitude_init}")

        # 1. conditioning: timestep, steering target and the latent scale share one embedding space
        self.time_proj = Timesteps(cond_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(cond_embed_dim, cond_embed_dim)
        self.target_embedding = nn.Linear(target_embed_dim, cond_embed_dim)
        self.scale_embedding = nn.Linear(1, cond_embed_dim)

        # 2. readout over the pooled per-channel statistics plus the conditioning vector
        stats_dim = 2 * latent_channels
        self.head = nn.Sequential(
            nn.LayerNorm(stats_dim + cond_embed_dim),
            nn.Linear(stats_dim + cond_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Start every sample at ~`magnitude_init`. The weight is near-zero rather than exactly zero
        # so that gradients still reach the rest of the network on the very first optimizer step.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.constant_(self.head[-1].bias, math.log(magnitude_init / (1.0 - magnitude_init)))

    def forward(self, latents: torch.Tensor, t: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            latents (`torch.Tensor`): Intermediate latents of shape `(batch_size, latent_channels,
                frames)`.
            t (`torch.Tensor`): The current timestep. Either a scalar shared by the whole batch or a
                tensor of shape `(batch_size,)`.
            target_embed (`torch.Tensor`): CLAP embedding of the steering target, of shape
                `(batch_size, target_embed_dim)`.

        Returns:
            `torch.Tensor`: `magnitude` in `[0, 1]`, of shape `(batch_size, 1, 1)` so that it
            broadcasts over the frame axis of `compute_cfg_diff_shape`'s output, in the dtype of
            `latents`.
        """
        if latents.ndim != 3:
            raise ValueError(
                f"`latents` has to be of shape `(batch_size, latent_channels, frames)` but has {latents.ndim}"
                " dimensions"
            )
        if target_embed.ndim != 2:
            raise ValueError(
                f"`target_embed` has to be of shape `(batch_size, target_embed_dim)` but has {target_embed.ndim}"
                " dimensions"
            )

        batch_size = latents.shape[0]

        if target_embed.shape[0] != batch_size:
            raise ValueError(
                f"`target_embed` has batch size {target_embed.shape[0]}, but `latents` has batch size {batch_size}."
                " Please make sure that `target_embed` is not duplicated for classifier free guidance."
            )

        output_dtype = latents.dtype
        dtype = next(self.parameters()).dtype

        latents = latents.to(dtype)
        target_embed = target_embed.to(dtype)

        # 1. per-channel statistics, pooled over frames, normalized by the sample's own overall
        # scale so the head sees numbers of order 1 regardless of where on the trajectory this is
        overall_std = torch.std(latents.flatten(1), dim=-1, correction=0)
        log_scale = torch.log(overall_std + 1e-6).unsqueeze(-1)
        channel_std, channel_mean = torch.std_mean(latents, dim=-1, correction=0)  # (batch, channels) each
        stats = torch.cat([channel_mean, channel_std], dim=-1) / (overall_std.unsqueeze(-1) + 1e-6)

        # 2. conditioning
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=latents.device)
        timesteps = torch.atleast_1d(t).to(latents.device)
        if timesteps.shape[0] == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.shape[0] != batch_size:
            raise ValueError(
                f"`t` has batch size {timesteps.shape[0]}, but `latents` has batch size {batch_size}. Please make"
                " sure that `t` is either a scalar or matches the batch size of `latents`."
            )

        cond = (
            self.time_embedding(self.time_proj(timesteps).to(dtype))
            + self.target_embedding(target_embed)
            + self.scale_embedding(log_scale)
        )

        # 3. map the statistics to a single bounded scalar per sample
        magnitude = torch.sigmoid(self.head(torch.cat([stats, cond], dim=-1)))

        return magnitude.view(-1, 1, 1).to(output_dtype)
