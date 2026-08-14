r"""
Trains a `SteeringPredictor` to suppress a concept in `SteeringStableAudioPipeline` generations.

Each training step generates a full batch of waveforms with steering enabled, embeds them with the
CLAP audio tower and scores them against two texts:

* the steering target, whose cosine distance is *maximized*, so the predictor learns a per-timestep
  `alpha_t` that moves the denoising trajectory from full-prompt guidance towards retain-prompt
  guidance;
* the retain prompt supplied by the dataset (or derived with `utils.strip_target` for legacy
  two-column CSVs), whose cosine distance is *minimized*, so everything else the prompt asks for
  survives the suppression.

The retain term is what separates removing the concept from degrading the audio: silence and noise
are both far from the target, and only the retain term tells them apart from a faithful rendition of
the rest of the prompt.

Gradients reach the predictor because the pipeline is called with `output_type="latent"`, which
returns the trajectory endpoint before any decoding happens, and because the pipeline reimplements
Stable Audio 3's sampling loop without the `torch.no_grad()` the library applies to it. The
diffusion transformer itself still runs under `no_grad`, so the gradient flows to each `alpha_t`
through the Euler recurrence only. That is a first-order approximation of the true gradient and is
what keeps the unrolled trajectory affordable.

Preferred example:
    uv run python scripts/train.py \
        --train-selection outputs/trumpet-train-baselines \
        --validation-selection outputs/trumpet-validation-baselines \
        --batch-size 2 --output outputs/trumpet-target-specific --epochs 5

Both manifests are filtered to their preassigned target-valid prompt/seed pairs. The legacy
``--dataset`` input remains available for compatibility, without held-out validation.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

# non-interactive backend: this script only saves figures to disk and never calls `plt.show()`, and
# the default TkAgg backend on Windows raises a spurious "main thread is not in main loop" from
# tkinter's Image.__del__ when the garbage collector runs between epochs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from losses import ClapLoss
from pipelines import STEERING_MODE, SteeringPredictor, SteeringStableAudioPipeline
from utils import PromptTargetDataset, collate_prompt_target
from utils.instrument_classification import BaselineSelection, load_baseline_selection

# static light-surface artifacts: chrome and ink from the reference palette, the first two
# categorical slots for the target/retain pair and a validated 5-step ordinal blue ramp for the
# per-epoch alpha lines
PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": "#2a78d6",
    "series_2": "#eb6834",
}
ORDINAL_BLUE = ("#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")


def steered_step_indices(num_inference_steps: int, steering_frac_start: float, steering_frac_end: float) -> list[int]:
    r"""
    Returns the 0-based denoising-loop step indices where steering is active, mirroring the exact
    condition evaluated in `SteeringState.is_active`.
    """
    return [
        step
        for step in range(num_inference_steps)
        if steering_frac_start <= step / num_inference_steps < steering_frac_end
    ]


def mean_alpha_by_step(records: list[tuple[int, float]]) -> list[tuple[int, float]]:
    r"""
    Returns:
        `list[tuple[int, float]]`: `(step, mean_alpha)` pairs, one per denoising-loop step inside the
        steering window, ordered from `steering_frac_start` to `steering_frac_end`.
    """
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for step, alpha in records:
        totals[step] += alpha
        counts[step] += 1
    # the pipeline records the loop step index directly, which already rises from noisy to clean
    return [(step, totals[step] / counts[step]) for step in sorted(totals)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_pair_rows(selection: BaselineSelection) -> list[dict]:
    r"""Builds training rows exclusively from target-valid selection records."""
    return [
        {
            "pair_id": pair.pair_id,
            "prompt": str(pair.record["prompt"]),
            "target": str(pair.record["target"]),
            "retain_prompt": str(pair.record["retain_prompt"]),
            "seed": int(pair.record["seed"]),
        }
        for pair in selection.pairs
    ]


def collate_selected_pairs(batch: list[dict]) -> tuple[list[str], list[str], list[str], list[int], list[str]]:
    return (
        [row["prompt"] for row in batch],
        [row["target"] for row in batch],
        [row["retain_prompt"] for row in batch],
        [int(row["seed"]) for row in batch],
        [row["pair_id"] for row in batch],
    )


def _same_setting(left, right) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9))
    return left == right


def validate_selection_configuration(selection: BaselineSelection, args: argparse.Namespace, name: str) -> None:
    if selection.config.get("baseline_mode") != "paired_alpha0":
        raise ValueError(
            f"{name} selection must use baseline_mode='paired_alpha0'; regenerate it with the current selector"
        )

    generation = selection.config.get("generation", {})
    expected = {
        "model": args.model,
        "model_half": not args.no_half,
        "num_inference_steps": args.num_inference_steps,
        "audio_length_in_s": args.audio_length_in_s,
        "cfg_scale": args.cfg_scale,
        "apg_scale": args.apg_scale,
        "negative_prompt": args.negative_prompt,
        "chunked_decode": args.chunked_decode,
        "steering_frac_start": args.steering_frac_start,
        "steering_frac_end": args.steering_frac_end,
    }
    for key, expected_value in expected.items():
        if key not in generation:
            raise ValueError(f"{name} selection generation config is missing {key!r}")
        if not _same_setting(generation[key], expected_value):
            raise ValueError(
                f"{name} selection {key}={generation[key]!r} does not match training value {expected_value!r}"
            )


def validate_train_validation_selections(
    train_selection: BaselineSelection,
    validation_selection: BaselineSelection,
    args: argparse.Namespace,
) -> None:
    validate_selection_configuration(train_selection, args, "training")
    validate_selection_configuration(validation_selection, args, "validation")

    train_classifier = train_selection.config["classifier"]
    validation_classifier = validation_selection.config["classifier"]
    for key in (
        "model",
        "requested_revision",
        "resolved_revision",
        "sampling_rate",
        "window_seconds",
        "window_aggregation",
        "detection_threshold",
    ):
        if not _same_setting(train_classifier.get(key), validation_classifier.get(key)):
            raise ValueError(
                f"training and validation selections use different classifier setting {key!r}: "
                f"{train_classifier.get(key)!r} != {validation_classifier.get(key)!r}"
            )
    if train_selection.config["instrument_vocabulary"] != validation_selection.config["instrument_vocabulary"]:
        raise ValueError("training and validation selections must use the same instrument vocabulary")

    train_dataset = Path(train_selection.config["dataset"]).resolve()
    validation_dataset = Path(validation_selection.config["dataset"]).resolve()
    if train_dataset == validation_dataset:
        raise ValueError("training and validation selections must come from different dataset files")

    train_seeds = {int(pair.record["seed"]) for pair in train_selection.pairs}
    validation_seeds = {int(pair.record["seed"]) for pair in validation_selection.pairs}
    overlapping_seeds = sorted(train_seeds & validation_seeds)
    if overlapping_seeds:
        raise ValueError(
            "training and validation selections must use disjoint seed values; overlapping seeds include "
            f"{overlapping_seeds[:5]}"
        )

    train_targets = {str(pair.record["target"]) for pair in train_selection.pairs}
    validation_targets = {str(pair.record["target"]) for pair in validation_selection.pairs}
    unseen_targets = sorted(validation_targets - train_targets)
    if unseen_targets:
        raise ValueError(f"validation selection contains targets unseen in training: {unseen_targets}")

    def group_ids(selection: BaselineSelection) -> set[str]:
        return {
            str(pair.record.get("source_metadata", {}).get("group_id", "")).strip()
            for pair in selection.pairs
            if str(pair.record.get("source_metadata", {}).get("group_id", "")).strip()
        }

    overlapping_groups = sorted(group_ids(train_selection) & group_ids(validation_selection))
    if overlapping_groups:
        raise ValueError(
            "training and validation selections contain overlapping semantic group_id values: "
            f"{overlapping_groups[:5]}"
        )


def selection_metadata(selection: BaselineSelection) -> dict:
    pair_ids = "\n".join(pair.pair_id for pair in selection.pairs).encode("utf-8")
    records = [pair.record for pair in selection.pairs]
    retain_applicable = [record for record in records if record["all_retain_valid"] is not None]
    return {
        "root": str(selection.root),
        "records": str(selection.records_path),
        "config_sha256": sha256_file(selection.root / "config.json"),
        "records_sha256": sha256_file(selection.records_path),
        "selected_pair_ids_sha256": hashlib.sha256(pair_ids).hexdigest(),
        "dataset": str(Path(selection.config["dataset"]).resolve()),
        "dataset_sha256": sha256_file(Path(selection.config["dataset"])),
        "baseline_mode": selection.config["baseline_mode"],
        "num_target_valid_pairs": len(records),
        "num_prompts": len({int(record["sample_id"]) for record in records}),
        "target_counts": dict(Counter(str(record["target"]) for record in records)),
        "num_pairs_with_retain_instruments": len(retain_applicable),
        "num_pairs_all_retain_valid": sum(bool(record["all_retain_valid"]) for record in retain_applicable),
        "num_pairs_with_invalid_retain": sum(
            record["all_retain_valid"] is False for record in retain_applicable
        ),
        "seed_min": min(int(record["seed"]) for record in records),
        "seed_max": max(int(record["seed"]) for record in records),
        "first_seed": selection.config.get("first_seed"),
        "generation": selection.config["generation"],
        "classifier": selection.config["classifier"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SteeringPredictor to suppress a concept in Stable Audio 3 generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data and output")
    source = data.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="CSV with prompt,target and preferably an explicit retain_prompt column",
    )
    source.add_argument(
        "--train-selection",
        type=Path,
        default=None,
        help="target-valid paired-alpha0 baseline selection for the training split",
    )
    data.add_argument(
        "--validation-selection",
        type=Path,
        default=None,
        help="target-valid paired-alpha0 selection for a held-out validation split with disjoint seeds",
    )
    data.add_argument("--output", type=Path, default=Path("outputs"), help="folder for weights and plots")
    data.add_argument("--batch-size", type=int, default=1, help="prompts generated per forward pass")
    data.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="use only the first N dataset rows or target-valid training pairs",
    )
    data.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="use only the first N target-valid validation pairs",
    )

    steering = parser.add_argument_group("steering")
    steering.add_argument("--steering-frac-start", type=float, default=0.3, help="fraction of the loop steering starts")
    steering.add_argument("--steering-frac-end", type=float, default=0.8, help="fraction of the loop steering ends")
    steering.add_argument("--alpha-min", type=float, default=0.0, help="lower bound of the predicted alpha_t")
    steering.add_argument("--alpha-max", type=float, default=1.0, help="upper bound of the predicted alpha_t")
    steering.add_argument(
        "--alpha-init",
        type=float,
        default=0.15,
        help="initial alpha_t: 0 follows the full prompt and 1 follows the retain prompt",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=5)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-2)
    optim.add_argument("--grad-accum-steps", type=int, default=4, help="batches accumulated per optimizer step")
    optim.add_argument("--max-grad-norm", type=float, default=1.0)
    optim.add_argument(
        "--retain-weight",
        type=float,
        default=1.0,
        help=(
            "weight of the retain term, which pulls the audio towards the prompt with the target removed. pure"
            " suppression can be solved by degrading the audio rather than by removing the target concept from it,"
            " and this term is what rules that solution out. 0.0 disables it and optimizes suppression alone"
        ),
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument(
        "--model",
        type=str,
        default="medium-base",
        help="Stable Audio 3 checkpoint, has to be one of the `-base` ones because the post-trained ones ignore CFG",
    )
    generation.add_argument("--num-inference-steps", type=int, default=50, help="denoising steps per generation")
    generation.add_argument(
        "--audio-length-in-s",
        type=float,
        default=10.0,
        help=(
            "the default maps exactly onto the 10s window of CLAP's audio tower, so the whole clip is"
            " scored and nothing is generated that the loss cannot see"
        ),
    )
    generation.add_argument("--cfg-scale", type=float, default=7.0, help="must exceed 1.0 to enable steering")
    generation.add_argument(
        "--apg-scale",
        type=float,
        default=0.0,
        help=(
            "0.0 is plain classifier free guidance, 1.0 is Stable Audio 3's own adaptive projected guidance."
            " alpha_t interpolates between two guidance predictions, so the plain one is the default here"
        ),
    )
    generation.add_argument("--negative-prompt", type=str, default=None)
    generation.add_argument(
        "--chunked-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="decode in overlapping chunks; must match paired-alpha0 selections",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument(
        "--no-half",
        action="store_true",
        help="load the diffusion transformer in float32; the autoencoder and the steering algebra always are",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"`--batch-size` has to be at least 1 but is {args.batch_size}")
    if args.grad_accum_steps < 1:
        raise ValueError(f"`--grad-accum-steps` has to be at least 1 but is {args.grad_accum_steps}")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError(f"`--max-samples` has to be at least 1 but is {args.max_samples}")
    if args.max_validation_samples is not None and args.max_validation_samples < 1:
        raise ValueError(
            f"`--max-validation-samples` has to be at least 1 but is {args.max_validation_samples}"
        )
    if args.train_selection is not None and args.validation_selection is None:
        raise ValueError("`--validation-selection` is required with `--train-selection`")
    if args.dataset is not None and args.validation_selection is not None:
        raise ValueError("`--validation-selection` can only be used with `--train-selection`")
    if args.retain_weight < 0.0:
        # a negative weight would push the audio away from the rest of the prompt as well, i.e. ask
        # for degraded audio outright
        raise ValueError(f"`--retain-weight` has to be non-negative but is {args.retain_weight}")
    if args.alpha_min >= args.alpha_max:
        raise ValueError(
            f"`--alpha-min` has to be smaller than `--alpha-max` but got {args.alpha_min} >= {args.alpha_max}"
        )
    if not args.alpha_min < args.alpha_init < args.alpha_max:
        raise ValueError(
            f"`--alpha-init` has to lie strictly inside ({args.alpha_min}, {args.alpha_max}) but is"
            f" {args.alpha_init}"
        )
    if not 0.0 <= args.steering_frac_start < args.steering_frac_end <= 1.0:
        raise ValueError(
            "`--steering-frac-start` and `--steering-frac-end` have to satisfy"
            f" `0.0 <= start < end <= 1.0` but are {args.steering_frac_start} and {args.steering_frac_end}"
        )
    if args.cfg_scale <= 1.0:
        # the transformer only calls the steering model inside its classifier free guidance branch
        raise ValueError(
            f"`--cfg-scale` has to be greater than 1.0 for steering to be applied but is {args.cfg_scale}."
            " With a lower value Stable Audio 3 skips classifier free guidance and never calls the steering model,"
            " so the predictor would receive no gradient."
        )
    if not 0.0 <= args.apg_scale <= 1.0:
        raise ValueError(f"`--apg-scale` has to be in [0.0, 1.0] but is {args.apg_scale}")
    if not args.model.endswith("-base"):
        raise ValueError(
            f"`--model` has to be a `-base` checkpoint but is {args.model!r}. The post-trained checkpoints are"
            " distilled and ignore classifier free guidance, which is where steering is applied."
        )

    num_steered_steps = len(
        steered_step_indices(args.num_inference_steps, args.steering_frac_start, args.steering_frac_end)
    )
    if num_steered_steps == 0:
        raise ValueError(
            f"The steering window [{args.steering_frac_start}, {args.steering_frac_end}) contains no step of the"
            f" {args.num_inference_steps} denoising steps, so the predictor would receive no gradient. Widen the"
            " window or raise `--num-inference-steps`."
        )
    args.num_steered_steps = num_steered_steps

    return args


def style_axes(ax: plt.Axes) -> None:
    r"""Applies the recessive chrome shared by every plot: hairline solid grid and muted axes."""
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(PALETTE["ink_secondary"])
    ax.yaxis.label.set_color(PALETTE["ink_secondary"])
    ax.title.set_color(PALETTE["ink"])


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, facecolor=PALETTE["surface"], bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history: dict, path: Path) -> None:
    r"""
    Writes the optimized loss and both CLAP cosine similarities, one point per training iteration.

    Two stacked panels rather than one plot with two y-scales: the loss and the similarities have
    unrelated ranges, and overlaying them on separate scales would invent a relationship. The two
    similarities do share a panel, because they share the cosine scale and reading one against the
    other *is* the suppression/retain trade-off. Plotting `history["iterations"]` rather than the
    epoch means in `history["epochs"]` keeps per-step noise and outliers visible instead of smoothing
    them away. Dashed vertical lines mark where each epoch ends, using `history["epochs"]`'
    `last_step`.
    """
    iterations = history["iterations"]
    steps = [record["step"] for record in iterations]
    # the last epoch's boundary coincides with the plot's right edge, so it would only draw a
    # line on top of the axis spine
    epoch_boundaries = [record["last_step"] for record in history["epochs"][:-1]]

    fig, (ax_loss, ax_cos) = plt.subplots(2, 1, figsize=(9, 7.5), facecolor=PALETTE["surface"])

    def draw_epoch_boundaries(ax: plt.Axes, label: bool) -> None:
        for epoch_record, boundary in zip(history["epochs"][:-1], epoch_boundaries, strict=True):
            ax.axvline(boundary, color=PALETTE["muted"], linewidth=0.8, linestyle="--", zorder=1)
            if label:
                ax.annotate(
                    f"epoch {epoch_record['epoch']}",
                    (boundary, 1.0),
                    xycoords=ax.get_xaxis_transform(),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=PALETTE["muted"],
                    ha="left",
                )

    def draw_series(ax: plt.Axes, key: str, color: str, label: str | None = None) -> None:
        values = [record[key] for record in iterations]
        ax.plot(steps, values, color=color, linewidth=1.2, label=label)
        # direct-label only the endpoint, so the value is readable without a tooltip
        ax.annotate(
            f"{values[-1]:.3f}",
            (steps[-1], values[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=9,
            color=PALETTE["ink"],
        )

    # labels only on the top panel, so the two panels' epoch numbers don't repeat right on top of
    # each other
    draw_epoch_boundaries(ax_loss, label=True)
    draw_epoch_boundaries(ax_cos, label=False)

    draw_series(ax_loss, "loss", PALETTE["series"])
    draw_series(ax_cos, "target_similarity", PALETTE["series"], "target — suppressed")
    draw_series(ax_cos, "retain_similarity", PALETTE["series_2"], "retain prompt — preserved")

    validation_epochs = [record for record in history["epochs"] if record.get("validation") is not None]
    if validation_epochs:
        validation_steps = [record["last_step"] for record in validation_epochs]
        ax_loss.plot(
            validation_steps,
            [record["validation"]["loss"] for record in validation_epochs],
            color=PALETTE["series_2"],
            marker="o",
            linewidth=1.5,
            label="validation loss",
        )
        ax_cos.plot(
            validation_steps,
            [record["validation"]["target_similarity"] for record in validation_epochs],
            color=PALETTE["series"],
            marker="o",
            linestyle="--",
            linewidth=1.2,
            label="validation target",
        )
        ax_cos.plot(
            validation_steps,
            [record["validation"]["retain_similarity"] for record in validation_epochs],
            color=PALETTE["series_2"],
            marker="o",
            linestyle="--",
            linewidth=1.2,
            label="validation retain",
        )
        ax_loss.legend(frameon=False, fontsize=9, loc="best")

    for ax, title, ylabel in (
        (ax_loss, "Training loss — the suppression and retain terms combined", "loss"),
        (ax_cos, "CLAP cosine similarity, target against retain prompt", "cosine similarity"),
    ):
        ax.set_title(title, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.margins(y=0.15)
        style_axes(ax)

    legend = ax_cos.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    fig.tight_layout()
    save_figure(fig, path)


def plot_alpha_schedule(history: dict, path: Path, alpha_min: float, alpha_max: float) -> None:
    r"""
    Writes the mean predicted `alpha_t` against the denoising-loop step, one line per epoch.

    This is the load-bearing diagnostic: it shows whether the predictor learned a schedule that
    varies along the trajectory or collapsed to a constant.

    The x-axis is the 0-based step index inside the steering window: `steering_frac_start` maps to
    the lowest step plotted and `steering_frac_end` to the highest, matching the condition evaluated
    in `SteeringState.is_active`. Step index rises from noisy to clean as the loop advances, so,
    unlike a raw-timestep axis, it needs no inversion to read left to right.

    Epochs are an ordered quantity, so the lines use a single-hue ordinal ramp rather than
    categorical hues. At most `len(ORDINAL_BLUE)` epochs are drawn, evenly spaced and always
    including the first and the last; `history.json` carries every epoch.
    """
    epochs = history["epochs"]
    args = history["args"]
    steps = steered_step_indices(args["num_inference_steps"], args["steering_frac_start"], args["steering_frac_end"])

    if len(epochs) <= len(ORDINAL_BLUE):
        selected = epochs
    else:
        stride = (len(epochs) - 1) / (len(ORDINAL_BLUE) - 1)
        selected = [epochs[round(index * stride)] for index in range(len(ORDINAL_BLUE))]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["surface"])

    # `alpha=0.5` is the midpoint between full-prompt and retain-prompt classifier-free guidance.
    if alpha_min < 0.5 < alpha_max:
        ax.axhline(0.5, color=PALETTE["muted"], linewidth=1.0, linestyle="--", zorder=1)
        ax.annotate(
            "α = 0.5 — full/retain midpoint",
            (0.995, 0.5),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 5),
            ha="right",
            fontsize=8,
            color=PALETTE["muted"],
        )

    # spread the drawn epochs across the whole ramp rather than crowding one end, so the lightness
    # gaps stay visible for any epoch count. a lone line is not an ordinal encoding at all
    if len(selected) == 1:
        colors = [PALETTE["series"]]
    else:
        step_size = (len(ORDINAL_BLUE) - 1) / (len(selected) - 1)
        colors = [ORDINAL_BLUE[round(index * step_size)] for index in range(len(selected))]

    for record, color in zip(selected, colors, strict=True):
        step_values = [step for step, _ in record["alpha_by_step"]]
        alphas = [alpha for _, alpha in record["alpha_by_step"]]
        ax.plot(step_values, alphas, color=color, linewidth=2.0, label=f"epoch {record['epoch']}", zorder=2)

    ax.set_title("Predicted steering strength across the denoising trajectory", fontsize=11, loc="left", pad=10)
    ax.set_xlabel("denoising step (noisy → clean)")
    ax.set_ylabel("mean α")
    ax.set_xlim(steps[0], steps[-1])
    margin = 0.04 * (alpha_max - alpha_min)
    ax.set_ylim(alpha_min - margin, alpha_max + margin)
    style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    fig.tight_layout()
    save_figure(fig, path)


def normalize_accumulated_gradients(parameters, sample_count: int) -> None:
    r"""Turns accumulated sums of per-sample gradients into an exact sample mean."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive when normalizing accumulated gradients")
    scale = 1.0 / sample_count
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def target_embedding_batch(target_embeds: dict[str, torch.Tensor], targets: list[str]) -> torch.Tensor:
    return torch.cat([target_embeds[target] for target in targets], dim=0)


@torch.inference_mode()
def run_validation(
    *,
    pipe: SteeringStableAudioPipeline,
    predictor: SteeringPredictor,
    clap_loss: ClapLoss,
    dataloader: DataLoader,
    target_embeds: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict:
    r"""Evaluates fixed, target-valid validation pairs and their explicit held-out seeds."""
    predictor.eval()
    total_loss = 0.0
    total_target_similarity = 0.0
    total_retain_similarity = 0.0
    total_samples = 0
    alpha_records: list[tuple[int, float]] = []

    for batch_index, (prompts, targets, retains, seeds, _) in enumerate(dataloader):
        generators = [torch.Generator().manual_seed(int(seed)) for seed in seeds]
        output = pipe(
            prompt=prompts,
            retain_prompt=retains,
            target_embed=target_embedding_batch(target_embeds, targets),
            steering_model=predictor,
            steering_frac_start=args.steering_frac_start,
            steering_frac_end=args.steering_frac_end,
            train=False,
            num_inference_steps=args.num_inference_steps,
            audio_length_in_s=args.audio_length_in_s,
            cfg_scale=args.cfg_scale,
            negative_prompt=args.negative_prompt,
            apg_scale=args.apg_scale,
            generator=generators,
            chunked_decode=args.chunked_decode,
            output_type="pt",
        )
        waveform = output.audios
        audio_embeds = clap_loss.encode_audio(waveform.mean(dim=1), pipe.sample_rate)
        target_distance = clap_loss(audio_embeds, targets)
        retain_distance = clap_loss(audio_embeds, retains)
        loss = -target_distance + args.retain_weight * retain_distance

        batch_size = len(prompts)
        total_samples += batch_size
        total_loss += float(loss) * batch_size
        total_target_similarity += (1.0 - float(target_distance)) * batch_size
        total_retain_similarity += (1.0 - float(retain_distance)) * batch_size
        # The pipeline records a batch mean at each step. Repeating it by batch size makes the
        # aggregate sample-weighted when the final validation batch is smaller.
        alpha_records.extend(output.alpha_records * batch_size)
        print(
            f"validation batch {batch_index + 1}/{len(dataloader)} "
            f"loss {float(loss):+.4f} target_cos {1.0 - float(target_distance):+.4f} "
            f"retain_cos {1.0 - float(retain_distance):+.4f}",
            flush=True,
        )

    return {
        "num_pairs": total_samples,
        "loss": total_loss / total_samples,
        "target_similarity": total_target_similarity / total_samples,
        "retain_similarity": total_retain_similarity / total_samples,
        "alpha_by_step": mean_alpha_by_step(alpha_records),
    }


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train_selection = None
    validation_selection = None
    validation_loader = None
    if args.train_selection is not None:
        train_selection = load_baseline_selection(args.train_selection, max_pairs=args.max_samples)
        validation_selection = load_baseline_selection(
            args.validation_selection, max_pairs=args.max_validation_samples
        )
        validate_train_validation_selections(train_selection, validation_selection, args)
        training_rows = selected_pair_rows(train_selection)
        validation_rows = selected_pair_rows(validation_selection)
        dataloader = DataLoader(
            training_rows,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_selected_pairs,
        )
        validation_loader = DataLoader(
            validation_rows,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_selected_pairs,
        )
        data_metadata = {
            "mode": "target_valid_baseline_selections",
            "training": selection_metadata(train_selection),
            "validation": selection_metadata(validation_selection),
            "seed_sets_disjoint": True,
        }
        example_target = training_rows[0]["target"]
        example_retain = training_rows[0]["retain_prompt"]
        print(
            f"Loaded {len(training_rows)} target-valid training pairs and {len(validation_rows)} target-valid "
            f"validation pairs -> {len(dataloader)}/{len(validation_loader)} batches"
        )
    else:
        dataset = PromptTargetDataset(args.dataset, args.max_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_prompt_target,
        )
        data_metadata = {
            "mode": "legacy_dataset",
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256_file(args.dataset),
            "num_prompts": len(dataset),
            "validation": None,
        }
        _, example_target, example_retain = dataset.rows[0]
        print(f"Loaded {len(dataset)} prompts from {args.dataset} -> {len(dataloader)} batches per epoch")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (output_dir / "data_metadata.json").write_text(
        json.dumps(data_metadata, indent=2, default=str), encoding="utf-8"
    )

    # Show the effective retain prompt, whether explicit or derived, before an expensive first step.
    print(f"Retain weight {args.retain_weight}, target {example_target!r}, example retain prompt: {example_retain!r}")

    # The latent trajectory, the guidance algebra and the autoencoder stay in float32 whatever the
    # transformer runs in: the gradient reaches `alpha_t` through the Euler recurrence, the decoder
    # and the CLAP tower, and in half precision that chain returns a gradient whose sign disagrees
    # with a finite-difference check of the same loss more often than not.
    print(f"Loading {args.model} on {device} with {'float32' if args.no_half else 'half'} precision")
    pipe = SteeringStableAudioPipeline.from_pretrained(args.model, device=device, model_half=not args.no_half)
    pipe.diffusion.requires_grad_(False)
    pipe.diffusion.eval()

    # the predictor casts its inputs and its output to the latents' dtype itself, so it keeps
    # working unchanged whichever precision the transformer above was loaded in
    predictor_config = {
        "latent_channels": pipe.io_channels,
        "target_embed_dim": None,  # filled in below, once CLAP is loaded
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_init": args.alpha_init,
    }

    # Stable Audio 3 conditions on T5Gemma, so CLAP is loaded on its own. It scores the loss and
    # also provides the `target_embed` the predictor is conditioned on, which keeps both in one space
    clap_loss = ClapLoss.from_pretrained().to(device)
    predictor_config["target_embed_dim"] = clap_loss.embed_dim

    if train_selection is not None:
        training_targets = {row["target"] for row in training_rows}
        validation_targets = {row["target"] for row in validation_rows}
        all_targets = training_targets | validation_targets
    else:
        training_targets = {target for _, target, _ in dataset.rows}
        validation_targets = set()
        all_targets = training_targets
    target_embeds = {target: clap_loss.encode_text([target]) for target in all_targets}

    predictor = SteeringPredictor(**predictor_config).to(device)
    num_parameters = sum(parameter.numel() for parameter in predictor.parameters())
    print(f"SteeringPredictor: {num_parameters / 1e6:.2f}M parameters, config {predictor_config}")

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"Steering {args.num_steered_steps}/{args.num_inference_steps} steps in"
        f" [{args.steering_frac_start}, {args.steering_frac_end}), {args.audio_length_in_s}s clips"
    )

    history: dict = {
        "args": json.loads(json.dumps(vars(args), default=str)),
        "data": data_metadata,
        "checkpoint_selection_metric": "validation_loss" if validation_loader is not None else "training_loss_legacy",
        "iterations": [],
        "epochs": [],
    }
    best_metric = float("inf")
    step = 0

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss_sum = 0.0
        epoch_target_similarity_sum = 0.0
        epoch_retain_similarity_sum = 0.0
        epoch_sample_count = 0
        epoch_alpha_records: list[tuple[int, float]] = []
        accumulated_samples = 0

        for batch_index, batch in enumerate(dataloader):
            if train_selection is not None:
                prompts, targets, retains, sample_seeds, pair_ids = batch
                generator = [torch.Generator().manual_seed(int(seed)) for seed in sample_seeds]
            else:
                prompts, targets, retains = batch
                pair_ids = []
                # Legacy mode preserves the historical fresh-noise schedule. Selection mode instead
                # uses the explicit target-valid seed attached to every pair.
                generator = torch.Generator().manual_seed(args.seed + step + 1)
            step += 1

            output = pipe(
                prompt=prompts,
                retain_prompt=retains,
                target_embed=target_embedding_batch(target_embeds, targets),
                steering_model=predictor,
                steering_frac_start=args.steering_frac_start,
                steering_frac_end=args.steering_frac_end,
                train=True,
                num_inference_steps=args.num_inference_steps,
                audio_length_in_s=args.audio_length_in_s,
                cfg_scale=args.cfg_scale,
                negative_prompt=args.negative_prompt,
                apg_scale=args.apg_scale,
                generator=generator,
                output_type="latent",
            )
            latents = output.audios
            # the pipeline's own post-processing, minus the clamp, which would zero the gradient of
            # every saturated sample
            waveform = pipe.decode_latents(
                latents,
                padding_mask=output.padding_mask,
                audio_length_in_s=args.audio_length_in_s,
                chunked=args.chunked_decode,
            )

            # Stable Audio 3 is stereo while CLAP's audio tower is mono, so the channels are summed
            # to a single mid signal before scoring
            audio_embeds = clap_loss.encode_audio(waveform.mean(dim=1), pipe.sample_rate)

            # `ClapLoss` returns `1 - cosine_similarity`, so negating it maximizes the distance to
            # the target, i.e. suppresses the concept the prompt asks for
            target_distance = clap_loss(audio_embeds, targets)
            loss = -target_distance

            # the same quantity against the prompt without the target, this time minimized, so the
            # audio keeps everything the prompt asks for besides the concept being suppressed
            retain_distance = clap_loss(audio_embeds, retains) if args.retain_weight > 0.0 else None
            if retain_distance is not None:
                loss = loss + args.retain_weight * retain_distance

            batch_size = len(prompts)
            # As in validation, each pipeline alpha record is a batch mean. Weight it by the number
            # of examples represented by that batch for an exact epoch-level mean.
            epoch_alpha_records.extend(output.alpha_records * batch_size)
            # Accumulate sums, then divide by the exact number of samples just before stepping. This
            # keeps the final partial accumulation group and a smaller last batch correctly scaled.
            (loss * batch_size).backward()
            accumulated_samples += batch_size

            grad_norm = None
            is_last_batch = batch_index == len(dataloader) - 1
            if (batch_index + 1) % args.grad_accum_steps == 0 or is_last_batch:
                normalize_accumulated_gradients(predictor.parameters(), accumulated_samples)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0

            loss_value = float(loss.detach())
            target_similarity = 1.0 - float(target_distance.detach())
            # measured even when the term is switched off, since it is the diagnostic that tells
            # concept removal apart from plain audio degradation
            with torch.no_grad():
                retain_similarity = (
                    1.0 - float(retain_distance.detach())
                    if retain_distance is not None
                    else 1.0 - float(clap_loss(audio_embeds.detach(), retains))
                )
            epoch_sample_count += batch_size
            epoch_loss_sum += loss_value * batch_size
            epoch_target_similarity_sum += target_similarity * batch_size
            epoch_retain_similarity_sum += retain_similarity * batch_size
            history["iterations"].append(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_value,
                    "target_similarity": target_similarity,
                    "retain_similarity": retain_similarity,
                    "grad_norm": grad_norm,
                    "batch_size": batch_size,
                    "pair_ids": pair_ids,
                }
            )
            print(
                f"epoch {epoch}/{args.epochs} batch {batch_index + 1}/{len(dataloader)}"
                f" loss {loss_value:+.4f} target_cos {target_similarity:+.4f} retain_cos {retain_similarity:+.4f}"
                f" grad_norm {'-' if grad_norm is None else f'{grad_norm:.3e}'}",
                flush=True,
            )

        epoch_loss = epoch_loss_sum / epoch_sample_count
        epoch_record = {
            "epoch": epoch,
            "last_step": step,
            "num_pairs": epoch_sample_count,
            "loss": epoch_loss,
            "target_similarity": epoch_target_similarity_sum / epoch_sample_count,
            "retain_similarity": epoch_retain_similarity_sum / epoch_sample_count,
            "alpha_by_step": mean_alpha_by_step(epoch_alpha_records),
            "validation": None,
        }
        print(f"epoch {epoch}/{args.epochs} mean training loss {epoch_loss:+.4f}")

        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = run_validation(
                pipe=pipe,
                predictor=predictor,
                clap_loss=clap_loss,
                dataloader=validation_loader,
                target_embeds=target_embeds,
                args=args,
            )
            epoch_record["validation"] = validation_metrics
            print(
                f"epoch {epoch}/{args.epochs} validation loss {validation_metrics['loss']:+.4f} "
                f"target_cos {validation_metrics['target_similarity']:+.4f} "
                f"retain_cos {validation_metrics['retain_similarity']:+.4f}"
            )

        history["epochs"].append(epoch_record)
        metric_name = "validation_loss" if validation_metrics is not None else "training_loss_legacy"
        metric_value = validation_metrics["loss"] if validation_metrics is not None else epoch_loss

        checkpoint = {
            "checkpoint_schema_version": 2,
            "state_dict": predictor.state_dict(),
            "config": predictor_config,
            "steering_mode": STEERING_MODE,
            "model": args.model,
            "model_half": not args.no_half,
            "args": history["args"],
            "data": data_metadata,
            "generation": {
                "model": args.model,
                "model_half": not args.no_half,
                "num_inference_steps": args.num_inference_steps,
                "audio_length_in_s": args.audio_length_in_s,
                "cfg_scale": args.cfg_scale,
                "apg_scale": args.apg_scale,
                "negative_prompt": args.negative_prompt,
                "chunked_decode": args.chunked_decode,
                "steering_frac_start": args.steering_frac_start,
                "steering_frac_end": args.steering_frac_end,
            },
            "targets": sorted(all_targets),
            "training_targets": sorted(training_targets),
            "validation_targets": sorted(validation_targets),
            "optimizer": {
                "name": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_accum_steps": args.grad_accum_steps,
                "max_grad_norm": args.max_grad_norm,
                "accumulation_reduction": "exact_sample_mean",
            },
            "epoch": epoch,
            "loss": epoch_loss,
            "training_metrics": {
                key: epoch_record[key]
                for key in ("num_pairs", "loss", "target_similarity", "retain_similarity", "alpha_by_step")
            },
            "validation_metrics": validation_metrics,
            "checkpoint_selection_metric": metric_name,
            "checkpoint_selection_value": metric_value,
            "torch_version": torch.__version__,
        }
        torch.save(checkpoint, output_dir / "steering_predictor.pt")
        if metric_value < best_metric:
            best_metric = metric_value
            torch.save(checkpoint, output_dir / "steering_predictor_best.pt")

        # written every epoch so an interrupted run keeps its numbers, and doubles as the
        # machine-readable twin of the two plots
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_loss_curve(history, output_dir / "loss_curve.png")
        plot_alpha_schedule(history, output_dir / "alpha_schedule.png", args.alpha_min, args.alpha_max)

    metric_label = "validation loss" if validation_loader is not None else "training loss (legacy mode)"
    print(f"\nDone. Best {metric_label} {best_metric:+.4f}. Artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
