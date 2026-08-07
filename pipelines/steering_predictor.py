import math

import torch
import torch.nn.functional as F
from torch import nn


def _num_groups(num_channels: int, max_num_groups: int) -> int:
    """Return the largest valid GroupNorm divisor up to ``max_num_groups``."""
    return max(math.gcd(num_channels, max_num_groups), 1)


class _SinusoidalTimesteps(nn.Module):
    """Parameter-free sinusoidal embedding matching ACE-Step's flow-timestep scale."""

    def __init__(self, embedding_dim: int, max_period: int = 10_000, scale: float = 1_000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)

        exponent = -math.log(self.max_period) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        ) / half_dim
        frequencies = torch.exp(exponent)
        # ACE-Step's native scheduler uses continuous t in [0, 1] and its DiT multiplies t by
        # 1000 before sinusoidal projection. Mirroring that scale keeps adjacent denoising steps
        # distinguishable to this controller as well.
        phase = (timesteps.float() * self.scale).unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

        if self.embedding_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FiLMResnetBlock1D(nn.Module):
    r"""Pre-normalized residual Conv1d block modulated by a conditioning vector."""

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
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film = nn.Linear(cond_embed_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels, norm_num_groups), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(hidden_states)
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))

        scale, shift = self.film(F.silu(cond)).chunk(2, dim=-1)
        hidden_states = hidden_states * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        hidden_states = self.conv2(self.dropout(F.silu(self.norm2(hidden_states))))
        return hidden_states + residual


class SelfAttentionBlock1D(nn.Module):
    r"""Residual self-attention over the temporal axis of an ACE-Step latent."""

    def __init__(self, channels: int, num_attention_heads: int = 4, norm_num_groups: int = 32):
        super().__init__()
        if num_attention_heads <= 0:
            raise ValueError(f"`num_attention_heads` has to be positive but got {num_attention_heads}")
        if channels % num_attention_heads != 0:
            raise ValueError(
                f"`channels` has to be divisible by `num_attention_heads` but got channels={channels} and"
                f" num_attention_heads={num_attention_heads}"
            )

        self.norm = nn.GroupNorm(_num_groups(channels, norm_num_groups), channels)
        self.attention = nn.MultiheadAttention(channels, num_attention_heads, batch_first=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states).transpose(1, 2)
        hidden_states, _ = self.attention(hidden_states, hidden_states, hidden_states, need_weights=False)
        return hidden_states.transpose(1, 2) + residual


class SteeringPredictor(nn.Module):
    r"""Predict a target-specific steering strength for ACE-Step 1.5.

    ACE-Step exposes its 1-D VAE/DiT latents as ``(batch, time, channels)``. The predictor uses
    temporal convolutions (after an internal channel-first transpose), FiLM conditioning in every
    residual block, and optional self-attention over time. The target conditioning remains a frozen
    CLAP text embedding; it is not compared with ACE-Step's Qwen conditioning embeddings.

    The public constructor intentionally retains the previous ``SteeringPredictor`` API. The
    ``latent_channels`` default is updated to ACE-Step 1.5's 64 latent features.

    Args:
        latent_channels: Size of the final axis of the ACE-Step latent. Defaults to 64.
        target_embed_dim: Size of the external CLAP target embedding.
        block_out_channels: Output channels at each temporal encoder level.
        layers_per_block: Number of FiLM residual blocks at each level.
        cond_embed_dim: Width of the shared timestep/target/scale conditioning.
        num_attention_heads: Number of bottleneck attention heads.
        use_attention: Whether to apply bottleneck temporal attention.
        dropout: Dropout probability in the residual blocks.
        norm_num_groups: Maximum number of GroupNorm groups.
        alpha_min: Inclusive lower output bound.
        alpha_max: Inclusive upper output bound.
        alpha_init: Initial output value. If omitted, starts 15% through the configured range.
        normalize_latents: Whether to normalize every sample before the temporal encoder.
    """

    def __init__(
        self,
        latent_channels: int = 64,
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

        if latent_channels <= 0:
            raise ValueError(f"`latent_channels` has to be positive but got {latent_channels}")
        if target_embed_dim <= 0:
            raise ValueError(f"`target_embed_dim` has to be positive but got {target_embed_dim}")
        if len(block_out_channels) == 0:
            raise ValueError("`block_out_channels` has to contain at least one element but is empty")
        if any(channels <= 0 for channels in block_out_channels):
            raise ValueError(f"all `block_out_channels` have to be positive but got {block_out_channels}")
        if layers_per_block <= 0:
            raise ValueError(f"`layers_per_block` has to be positive but got {layers_per_block}")
        if cond_embed_dim <= 0:
            raise ValueError(f"`cond_embed_dim` has to be positive but got {cond_embed_dim}")
        if norm_num_groups <= 0:
            raise ValueError(f"`norm_num_groups` has to be positive but got {norm_num_groups}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"`dropout` has to lie in [0, 1] but got {dropout}")
        if not math.isfinite(alpha_min) or not math.isfinite(alpha_max) or alpha_min >= alpha_max:
            raise ValueError(f"`alpha_min` has to be smaller than `alpha_max` but got {alpha_min} >= {alpha_max}")

        alpha_range = alpha_max - alpha_min
        if alpha_init is None:
            alpha_init = alpha_min + 0.15 * alpha_range
        elif not math.isfinite(alpha_init) or not alpha_min < alpha_init < alpha_max:
            raise ValueError(
                f"`alpha_init` has to lie strictly between `alpha_min` and `alpha_max` but got {alpha_init}, which is"
                f" outside ({alpha_min}, {alpha_max})"
            )

        self.latent_channels = latent_channels
        self.target_embed_dim = target_embed_dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.normalize_latents = normalize_latents

        time_embed_dim = block_out_channels[0]
        self.time_proj = _SinusoidalTimesteps(time_embed_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embed_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        self.target_embedding = nn.Linear(target_embed_dim, cond_embed_dim)
        self.scale_embedding = nn.Linear(1, cond_embed_dim)
        self.cond_mixer = nn.Sequential(nn.SiLU(), nn.Linear(cond_embed_dim, cond_embed_dim))

        self.conv_in = nn.Conv1d(latent_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        channels = block_out_channels[0]
        for index, out_channels in enumerate(block_out_channels):
            blocks = nn.ModuleList()
            for _ in range(layers_per_block):
                blocks.append(FiLMResnetBlock1D(channels, out_channels, cond_embed_dim, dropout, norm_num_groups))
                channels = out_channels
            self.down_blocks.append(blocks)
            self.downsamplers.append(
                nn.Identity()
                if index == len(block_out_channels) - 1
                else nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
            )

        self.mid_block_1 = FiLMResnetBlock1D(channels, channels, cond_embed_dim, dropout, norm_num_groups)
        self.mid_attention = (
            SelfAttentionBlock1D(channels, num_attention_heads, norm_num_groups) if use_attention else None
        )
        self.mid_block_2 = FiLMResnetBlock1D(channels, channels, cond_embed_dim, dropout, norm_num_groups)

        stats_dim = 2 * (sum(block_out_channels) + channels) + cond_embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, 1),
        )

        # A tiny, non-zero final weight preserves the desired initial value while allowing gradients
        # to reach the encoder on the first optimization step.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        normalized_alpha_init = (alpha_init - alpha_min) / alpha_range
        nn.init.constant_(self.head[-1].bias, math.log(normalized_alpha_init / (1.0 - normalized_alpha_init)))

    @staticmethod
    def _pool(hidden_states: torch.Tensor) -> torch.Tensor:
        std, mean = torch.std_mean(hidden_states, dim=-1, correction=0)
        return torch.cat([mean, std], dim=-1)

    def forward(self, latents: torch.Tensor, t: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        r"""Return bounded ``alpha_t`` with shape ``(batch, 1, 1)``.

        Args:
            latents: ACE-Step intermediate latents shaped ``(batch, time, latent_channels)``.
            t: A scalar flow timestep shared by the batch or a vector shaped ``(batch,)``.
            target_embed: External CLAP target text embeddings shaped ``(batch, target_embed_dim)``.
        """
        if latents.ndim != 3:
            raise ValueError(
                "`latents` has to be of shape `(batch_size, time, latent_channels)` but has"
                f" {latents.ndim} dimensions"
            )
        if latents.shape[0] == 0 or latents.shape[1] == 0:
            raise ValueError("`latents` batch and time dimensions have to be non-empty")
        if latents.shape[-1] != self.latent_channels:
            raise ValueError(
                f"`latents` has {latents.shape[-1]} channels, but this predictor expects {self.latent_channels}"
            )
        if not latents.is_floating_point():
            raise ValueError(f"`latents` has to use a floating-point dtype but got {latents.dtype}")
        if target_embed.ndim != 2:
            raise ValueError(
                f"`target_embed` has to be of shape `(batch_size, target_embed_dim)` but has {target_embed.ndim}"
                " dimensions"
            )

        batch_size = latents.shape[0]
        if target_embed.shape[0] != batch_size:
            raise ValueError(
                f"`target_embed` has batch size {target_embed.shape[0]}, but `latents` has batch size {batch_size}."
                " Please make sure that `target_embed` is not duplicated for classifier-free guidance."
            )
        if target_embed.shape[-1] != self.target_embed_dim:
            raise ValueError(
                f"`target_embed` has feature size {target_embed.shape[-1]}, but this predictor expects"
                f" {self.target_embed_dim}"
            )

        output_dtype = latents.dtype
        dtype = next(self.parameters()).dtype
        latents = latents.to(dtype=dtype)
        target_embed = target_embed.to(device=latents.device, dtype=dtype)

        latent_std, latent_mean = torch.std_mean(latents.flatten(1), dim=-1, correction=0)
        log_scale = torch.log(latent_std + 1e-6).unsqueeze(-1)
        if self.normalize_latents:
            latents = (latents - latent_mean[:, None, None]) / (latent_std[:, None, None] + 1e-6)

        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=latents.device)
        else:
            t = t.to(latents.device)
        if t.ndim == 0:
            timesteps = t.reshape(1)
        elif t.ndim == 1:
            timesteps = t
        else:
            raise ValueError(f"`t` has to be a scalar or one-dimensional tensor but has {t.ndim} dimensions")

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
        cond = cond + self.cond_mixer(cond)

        # ACE-Step stores channels last; Conv1d and GroupNorm operate with channels first.
        hidden_states = self.conv_in(latents.transpose(1, 2))
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

        alpha = torch.sigmoid(self.head(torch.cat([*stats, cond], dim=-1)))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha
        return alpha.reshape(batch_size, 1, 1).to(output_dtype)
