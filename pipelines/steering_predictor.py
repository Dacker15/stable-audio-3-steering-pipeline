import math

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


def _num_groups(num_channels: int, max_num_groups: int) -> int:
    # `nn.GroupNorm` requires `num_channels` to be divisible by `num_groups`
    return max(math.gcd(num_channels, max_num_groups), 1)


class FiLMResnetBlock2D(nn.Module):
    r"""
    Pre-norm residual block whose activations are modulated by a conditioning vector through FiLM
    (feature-wise linear modulation), i.e. a per-channel `scale` and `shift` predicted from `cond`.

    This is what lets a single set of weights behave differently at different points of the
    denoising trajectory: the timestep and the steering target enter every block, instead of only
    being concatenated to the final readout.

    Args:
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        cond_embed_dim (`int`): Dimension of the conditioning vector the FiLM parameters are predicted from.
        dropout (`float`, *optional*, defaults to 0.0): Dropout applied before the second convolution.
        norm_num_groups (`int`, *optional*, defaults to 32): Upper bound on the number of groups used by the
            `GroupNorm` layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_embed_dim: int,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(_num_groups(in_channels, norm_num_groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.film = nn.Linear(cond_embed_dim, 2 * out_channels)

        self.norm2 = nn.GroupNorm(_num_groups(out_channels, norm_num_groups), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor`): Input of shape `(batch_size, in_channels, height, width)`.
            cond (`torch.Tensor`): Conditioning vector of shape `(batch_size, cond_embed_dim)`.

        Returns:
            `torch.Tensor`: Output of shape `(batch_size, out_channels, height, width)`.
        """
        residual = self.skip(hidden_states)

        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))

        scale, shift = self.film(F.silu(cond)).chunk(2, dim=-1)
        hidden_states = hidden_states * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        hidden_states = self.conv2(self.dropout(F.silu(self.norm2(hidden_states))))

        return hidden_states + residual


class SelfAttentionBlock2D(nn.Module):
    r"""
    Residual self-attention over the flattened spatial grid, used at the encoder bottleneck to give
    the readout global (whole-clip) context that stacked 3x3 convolutions cannot reach.

    Args:
        channels (`int`): Number of input and output channels.
        num_attention_heads (`int`, *optional*, defaults to 4): Number of attention heads. Must divide `channels`.
        norm_num_groups (`int`, *optional*, defaults to 32): Upper bound on the number of groups used by the
            `GroupNorm` layer.
    """

    def __init__(self, channels: int, num_attention_heads: int = 4, norm_num_groups: int = 32):
        super().__init__()

        if channels % num_attention_heads != 0:
            raise ValueError(
                f"`channels` has to be divisible by `num_attention_heads` but got channels={channels} and"
                f" num_attention_heads={num_attention_heads}"
            )

        self.norm = nn.GroupNorm(_num_groups(channels, norm_num_groups), channels)
        self.attention = nn.MultiheadAttention(channels, num_attention_heads, batch_first=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor`): Input of shape `(batch_size, channels, height, width)`.

        Returns:
            `torch.Tensor`: Output of the same shape as `hidden_states`.
        """
        batch_size, channels, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states).view(batch_size, channels, height * width).transpose(1, 2)
        hidden_states, _ = self.attention(hidden_states, hidden_states, hidden_states, need_weights=False)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, channels, height, width)

        return hidden_states + residual


class SteeringPredictor(nn.Module):
    r"""
    Predicts the per-step steering strength `alpha_t` consumed by `SteeringMusicLDMPipeline`.

    The pipeline replaces the classifier-free guidance scale inside the steering window with
    `(1.0 - 2.0 * alpha_t) * guidance_scale`, so with the default range `alpha_t = 0` reproduces
    plain CFG, `alpha_t = 0.5` disables guidance and `alpha_t = 1` fully inverts it.

    Architecture: a FiLM-conditioned convolutional encoder over the intermediate latents with
    multi-scale statistical pooling. Convolutions read local time-frequency structure, FiLM injects
    the timestep and the CLAP target embedding into every block, a bottleneck self-attention block
    adds global context, and mean/std pooling at every resolution summarizes how much structure the
    latents currently carry before a small MLP maps those statistics to a single scalar.

    The latents are instance-normalized because their magnitude spans orders of magnitude along the
    trajectory; the discarded scale is fed back as an explicit conditioning feature so no
    information is lost.

    Args:
        latent_channels (`int`, *optional*, defaults to 8): Number of latent channels, i.e. `unet.config.in_channels`
            of the pipeline. The spatial size of the latents is never assumed, so a single instance works for any
            `audio_length_in_s`.
        target_embed_dim (`int`, *optional*, defaults to 512): Dimension of `target_embed`, i.e. the CLAP
            `projection_dim` of the pipeline's `text_encoder`.
        block_out_channels (`tuple[int, ...]`, *optional*, defaults to `(64, 128, 256)`): Output channels of each
            encoder level. Every level but the last is followed by a stride-2 downsample.
        layers_per_block (`int`, *optional*, defaults to 2): Number of `FiLMResnetBlock2D`s per encoder level.
        cond_embed_dim (`int`, *optional*, defaults to 256): Dimension of the conditioning vector and width of the
            readout MLP.
        num_attention_heads (`int`, *optional*, defaults to 4): Number of heads of the bottleneck attention.
        use_attention (`bool`, *optional*, defaults to `True`): Whether to use the bottleneck self-attention block.
        dropout (`float`, *optional*, defaults to 0.0): Dropout used inside the residual blocks.
        norm_num_groups (`int`, *optional*, defaults to 32): Upper bound on the number of `GroupNorm` groups.
        alpha_min (`float`, *optional*, defaults to 0.0): Lower bound of the predicted `alpha_t`.
        alpha_max (`float`, *optional*, defaults to 1.0): Upper bound of the predicted `alpha_t`.
        alpha_init (`float` or `None`, *optional*): Value every sample is initialized to, so that an untrained
            predictor reproduces the unsteered pipeline. Must lie strictly inside
            `(alpha_min, alpha_max)`. Defaults to `alpha_min + 0.01 * (alpha_max - alpha_min)`.
        normalize_latents (`bool`, *optional*, defaults to `True`): Whether to instance-normalize the latents before
            the encoder.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        target_embed_dim: int = 512,
        block_out_channels: tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 2,
        cond_embed_dim: int = 256,
        num_attention_heads: int = 4,
        use_attention: bool = True,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        alpha_init: float | None = None,
        normalize_latents: bool = True,
    ):
        super().__init__()

        if len(block_out_channels) == 0:
            raise ValueError("`block_out_channels` has to contain at least one element but is empty")

        if alpha_min >= alpha_max:
            raise ValueError(f"`alpha_min` has to be smaller than `alpha_max` but got {alpha_min} >= {alpha_max}")

        alpha_range = alpha_max - alpha_min

        if alpha_init is None:
            # just inside `alpha_min`, i.e. base CFG behaviour. deliberately not the midpoint of the
            # range, which for the default range would start training with `guidance_scale = 0`
            alpha_init = alpha_min + 0.01 * alpha_range
        elif not alpha_min < alpha_init < alpha_max:
            raise ValueError(
                f"`alpha_init` has to lie strictly between `alpha_min` and `alpha_max` but got {alpha_init}, which is"
                f" outside ({alpha_min}, {alpha_max})"
            )

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.normalize_latents = normalize_latents

        # 1. conditioning: timestep, steering target and the latent scale share one embedding space
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], cond_embed_dim)
        self.target_embedding = nn.Linear(target_embed_dim, cond_embed_dim)
        self.scale_embedding = nn.Linear(1, cond_embed_dim)
        self.cond_mixer = nn.Sequential(nn.SiLU(), nn.Linear(cond_embed_dim, cond_embed_dim))

        # 2. encoder
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            blocks = nn.ModuleList()
            for _ in range(layers_per_block):
                blocks.append(FiLMResnetBlock2D(channels, out_channels, cond_embed_dim, dropout, norm_num_groups))
                channels = out_channels
            self.down_blocks.append(blocks)

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block:
                self.downsamplers.append(nn.Identity())
            else:
                # a stride-2 padded conv maps any spatial size to `ceil(size / 2) >= 1`, so short
                # clips and non-default `audio_length_in_s` need no special casing
                self.downsamplers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))

        # 3. bottleneck
        self.mid_block_1 = FiLMResnetBlock2D(channels, channels, cond_embed_dim, dropout, norm_num_groups)
        if use_attention:
            self.mid_attention = SelfAttentionBlock2D(channels, num_attention_heads, norm_num_groups)
        else:
            self.mid_attention = None
        self.mid_block_2 = FiLMResnetBlock2D(channels, channels, cond_embed_dim, dropout, norm_num_groups)

        # 4. readout over the pooled multi-scale statistics plus the conditioning vector
        stats_dim = 2 * (sum(block_out_channels) + channels) + cond_embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, 1),
        )

        # start every sample at ~`alpha_init` so an untrained predictor reproduces the unsteered
        # pipeline. the weight is near-zero rather than exactly zero so that gradients still reach
        # the encoder on the very first optimizer step
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        normalized_alpha_init = (alpha_init - alpha_min) / alpha_range
        nn.init.constant_(self.head[-1].bias, math.log(normalized_alpha_init / (1.0 - normalized_alpha_init)))

    @staticmethod
    def _pool(hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Pools a feature map into per-channel statistics: the mean captures the average activation,
        the standard deviation how much spatial structure the level responds to.

        Args:
            hidden_states (`torch.Tensor`): Feature map of shape `(batch_size, channels, height, width)`.

        Returns:
            `torch.Tensor`: Statistics of shape `(batch_size, 2 * channels)`.
        """
        std, mean = torch.std_mean(hidden_states.flatten(2), dim=-1, correction=0)
        return torch.cat([mean, std], dim=-1)

    def forward(self, latents: torch.Tensor, t: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            latents (`torch.Tensor`): Intermediate latents of shape `(batch_size, latent_channels, height, width)`.
            t (`torch.Tensor`): The current timestep. Either a scalar shared by the whole batch or a tensor of shape
                `(batch_size,)`.
            target_embed (`torch.Tensor`): CLAP embedding of the steering target, of shape
                `(batch_size, target_embed_dim)`.

        Returns:
            `torch.Tensor`: `alpha_t` in `[alpha_min, alpha_max]`, of shape `(batch_size, 1, 1, 1)` so that it
            broadcasts over the batch axis of the noise prediction, in the dtype of `latents`.
        """
        if latents.ndim != 4:
            raise ValueError(
                "`latents` has to be of shape `(batch_size, latent_channels, height, width)` but has"
                f" {latents.ndim} dimensions"
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

        # 1. the latent magnitude spans orders of magnitude along the trajectory: normalize it away
        # for a well conditioned encoder, but keep it as an explicit conditioning feature
        latent_std, latent_mean = torch.std_mean(latents.flatten(1), dim=-1, correction=0)
        log_scale = torch.log(latent_std + 1e-6).unsqueeze(-1)
        if self.normalize_latents:
            latents = (latents - latent_mean.view(-1, 1, 1, 1)) / (latent_std.view(-1, 1, 1, 1) + 1e-6)

        # 2. conditioning
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=latents.device)
        timesteps = torch.atleast_1d(t).to(latents.device)
        if timesteps.shape[0] == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.shape[0] != batch_size:
            raise ValueError(
                f"`t` has batch size {timesteps.shape[0]}, but `latents` has batch size {batch_size}. Please make sure"
                " that `t` is either a scalar or matches the batch size of `latents`."
            )

        cond = (
            self.time_embedding(self.time_proj(timesteps).to(dtype))
            + self.target_embedding(target_embed)
            + self.scale_embedding(log_scale)
        )
        cond = cond + self.cond_mixer(cond)

        # 3. encode the latents, pooling statistics at every resolution
        hidden_states = self.conv_in(latents)

        stats = []
        for blocks, downsampler in zip(self.down_blocks, self.downsamplers, strict=True):
            for block in blocks:
                hidden_states = block(hidden_states, cond)
            stats.append(self._pool(hidden_states))
            hidden_states = downsampler(hidden_states)

        hidden_states = self.mid_block_1(hidden_states, cond)
        if self.mid_attention is not None:
            hidden_states = self.mid_attention(hidden_states)
        hidden_states = self.mid_block_2(hidden_states, cond)
        stats.append(self._pool(hidden_states))

        # 4. map the statistics to a single bounded scalar per sample
        alpha = torch.sigmoid(self.head(torch.cat([*stats, cond], dim=-1)))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha

        return alpha.view(-1, 1, 1, 1).to(output_dtype)
