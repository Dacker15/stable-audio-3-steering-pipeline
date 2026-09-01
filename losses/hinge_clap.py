import torch
import torch.nn.functional as F
from torch import nn

from losses.clap import ClapLoss


class HingeClapLoss(nn.Module):
    r"""
    Margin hinge contrast between the CLAP audio embedding and the steering target/retain prompts.

    Ordinary `ClapLoss` used as `loss = -target_distance + retain_weight * retain_distance` has no
    saturation point: it always rewards moving the target similarity down and the retain similarity
    up, however far either already is. That is enough for a policy with a single fixed degree of
    freedom (a constant `magnitude`), but one where `magnitude` is learned per sample collapses
    towards "steer at maximum strength always" instead of learning when to hold back. Replacing the
    two
    linear terms with hinges gives each one a point past which it stops contributing gradient:
    `l_target` once the target similarity has dropped below `margin_target`, `l_retain` once the
    retain similarity has risen above `margin_retain`.

    `margin_target`/`margin_retain` have no principled default: they are set by listening to a batch
    of fixed-`magnitude` generations and reading off the `s_t`/`s_r` this loss also reports, at the
    similarity level judged "good enough". `outputs/07_cfg_diff_evaluation/results.csv`'s
    `target_similarity`/`retain_similarity` columns (method `cfg_diff`) are a starting sample.

    Args:
        clap_loss (`~losses.ClapLoss`): Only its `encode_text` is used, to embed `targets`/`retains`
            in the same space `audio_embeds` was produced in.
        margin_target, margin_retain (`float`): Cosine similarity thresholds in `[-1, 1]`. See above.
        retain_weight (`float`, *optional*, defaults to 1.0): Weight of the retain term, same
            rationale as `ClapLoss`-based training: pure suppression can be solved by degrading the
            audio, and this term is what rules that out.
    """

    def __init__(self, clap_loss: ClapLoss, margin_target: float, margin_retain: float, retain_weight: float = 1.0):
        super().__init__()

        if not -1.0 <= margin_target <= 1.0:
            raise ValueError(f"`margin_target` has to be a cosine similarity in [-1, 1] but is {margin_target}")
        if not -1.0 <= margin_retain <= 1.0:
            raise ValueError(f"`margin_retain` has to be a cosine similarity in [-1, 1] but is {margin_retain}")
        if retain_weight < 0.0:
            raise ValueError(f"`retain_weight` has to be non-negative but is {retain_weight}")

        self.clap_loss = clap_loss
        self.margin_target = margin_target
        self.margin_retain = margin_retain
        self.retain_weight = retain_weight

    def forward(self, audio_embeds: torch.Tensor, targets: list[str], retains: list[str]) -> dict[str, torch.Tensor]:
        r"""
        Args:
            audio_embeds (`torch.Tensor`): CLAP audio embeddings of shape `(batch_size, embed_dim)`.
            targets (`list[str]`): The steering targets, one per sample.
            retains (`list[str]`): The retain prompts, one per sample.

        Returns:
            `dict[str, torch.Tensor]`: `l_clap` is the term to back-propagate through. The rest is
            for logging/diagnostics only: the per-batch means `l_target`, `l_retain`, `s_t`, `s_r`
            (tensors) and the fraction of samples whose margin is already satisfied
            (`target_satisfied_frac`, `retain_satisfied_frac`, `both_satisfied_frac`, plain Python
            floats).
        """
        if audio_embeds.ndim != 2:
            raise ValueError(
                f"`audio_embeds` has to be of shape `(batch_size, embed_dim)` but has {audio_embeds.ndim} dimensions"
            )
        batch_size = audio_embeds.shape[0]
        if len(targets) != batch_size or len(retains) != batch_size:
            raise ValueError(
                f"`targets` and `retains` have to have length {batch_size} (`audio_embeds`'s batch size) but have"
                f" {len(targets)} and {len(retains)}"
            )

        target_embeds = self.clap_loss.encode_text(targets).to(device=audio_embeds.device, dtype=audio_embeds.dtype)
        retain_embeds = self.clap_loss.encode_text(retains).to(device=audio_embeds.device, dtype=audio_embeds.dtype)

        s_t = F.cosine_similarity(audio_embeds, target_embeds, dim=-1)
        s_r = F.cosine_similarity(audio_embeds, retain_embeds, dim=-1)

        l_target_per_sample = F.relu(s_t - self.margin_target)
        l_retain_per_sample = F.relu(self.margin_retain - s_r)

        l_target = l_target_per_sample.mean()
        l_retain = l_retain_per_sample.mean()
        l_clap = l_target + self.retain_weight * l_retain

        with torch.no_grad():
            target_satisfied = l_target_per_sample == 0.0
            retain_satisfied = l_retain_per_sample == 0.0
            target_satisfied_frac = float(target_satisfied.float().mean())
            retain_satisfied_frac = float(retain_satisfied.float().mean())
            both_satisfied_frac = float((target_satisfied & retain_satisfied).float().mean())

        return {
            "l_clap": l_clap,
            "l_target": l_target,
            "l_retain": l_retain,
            "s_t": s_t.mean(),
            "s_r": s_r.mean(),
            "target_satisfied_frac": target_satisfied_frac,
            "retain_satisfied_frac": retain_satisfied_frac,
            "both_satisfied_frac": both_satisfied_frac,
        }
