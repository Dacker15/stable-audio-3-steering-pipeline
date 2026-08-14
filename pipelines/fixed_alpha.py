import torch
from torch import nn


class FixedAlphaSteering(nn.Module):
    r"""Pipeline-compatible controller returning one constant alpha per sample."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, latents: torch.Tensor, t: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        del t, target_embed
        return torch.full(
            (latents.shape[0], 1, 1, 1),
            self.alpha,
            device=latents.device,
            dtype=latents.dtype,
        )
