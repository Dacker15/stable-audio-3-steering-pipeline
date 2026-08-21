r"""
Trains a `MagnitudePredictor` to steer `SteeringStableAudioPipeline` with a learned, per-sample gain
on Regime A's deterministic CFG-diff shape ("Regime B", `steering_mode="cfg_diff_magnitude"`).

Unlike a learned model that predicts a full per-frame `alpha_t`, this script keeps the temporal
profile (`pipelines.compute_cfg_diff_shape`) entirely deterministic and learns only its scalar gain,
`magnitude`. `alpha_min`, `alpha_max` and the shape quantiles stay fixed, matching Regime A.

The loss combines three terms (see `spec-loss-regime-b.md`):

* a margin hinge contrast (`losses.HingeClapLoss`) between the CLAP audio embedding and the
  steering target (maximized past `--margin-target`) and retain prompt (minimized past
  `--margin-retain`), rather than an unbounded contrast that always rewards steering harder;
* a minimal-intervention penalty (`losses.minimal_intervention_penalty`) on `alpha_field` itself,
  weighted by `--lambda-reg` after a linear warmup (`--lambda-reg-warmup-steps`), so the predictor
  only pays the cost of intervening where the hinge terms still have gradient to give;
* a fidelity loss (`losses.pingpong_fidelity_loss`) that penalizes the steered x̂0 estimate for
  drifting from the non-steered one wherever the CFG-diff shape found nothing to correct.

`--margin-target`, `--margin-retain`, `--lambda-reg`, `--lambda-reg-warmup-steps` and `--lambda-fid`
have no principled default and are required. `outputs/07_cfg_diff_evaluation/results.csv`'s
`target_similarity`/`retain_similarity` columns (method `cfg_diff`) are Regime A output on a fixed
`magnitude`, a starting sample for picking margins by listening to a batch of generations at
different similarity levels.

Example:
    uv run python scripts/train_regime_b.py --dataset datasets/trumpet_simple_splits/train.csv \
        --validation-dataset datasets/trumpet_simple_splits/val.csv --batch-size 2 \
        --output outputs/trumpet-regime-b --epochs 5 --margin-target 0.30 --margin-retain 0.40 \
        --lambda-reg 0.01 --lambda-reg-warmup-steps 200 --lambda-fid 0.1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

# non-interactive backend: this script only saves figures to disk and never calls `plt.show()`, and
# the default TkAgg backend on Windows raises a spurious "main thread is not in main loop" from
# tkinter's Image.__del__ when the garbage collector runs between epochs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from losses import ClapLoss, HingeClapLoss, linear_warmup, minimal_intervention_penalty, pingpong_fidelity_loss
from pipelines import STEERING_MODE_MAGNITUDE, MagnitudePredictor, SteeringStableAudioPipeline
from utils import PromptTargetDataset, collate_prompt_target

# static light-surface artifacts: chrome and ink from the reference palette, the first two
# categorical slots for the target/retain pair and a validated 5-step ordinal blue ramp for the
# per-epoch magnitude lines
PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": "#2a78d6",
    "series_2": "#eb6834",
    "series_3": "#39875b",
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


def mean_value_by_step(records: list[tuple[int, float]]) -> list[tuple[int, float]]:
    r"""
    Returns:
        `list[tuple[int, float]]`: `(step, mean_value)` pairs, one per denoising-loop step inside the
        steering window, ordered from `steering_frac_start` to `steering_frac_end`.
    """
    totals: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for step, value in records:
        totals[step] += value
        counts[step] += 1
    # the pipeline records the loop step index directly, which already rises from noisy to clean
    return [(step, totals[step] / counts[step]) for step in sorted(totals)]


def magnitude_step_records(alpha_field_records: list[dict]) -> list[tuple[int, float]]:
    r"""Extracts `(step, mean magnitude)` pairs from one batch's `output.alpha_field_records`."""
    return [(int(record["step"]), float(record["magnitude"].mean())) for record in alpha_field_records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a MagnitudePredictor for Regime B (cfg_diff_magnitude): a learned per-sample gain on Regime"
            " A's deterministic CFG-diff shape."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data and output")
    data.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CSV with prompt,target and preferably an explicit retain_prompt column",
    )
    data.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help="CSV with the same schema as --dataset, held out and evaluated once per epoch",
    )
    data.add_argument("--output", type=Path, default=Path("outputs"), help="folder for weights and plots")
    data.add_argument("--batch-size", type=int, default=1, help="prompts generated per forward pass")
    data.add_argument("--max-samples", type=int, default=None, help="use only the first N rows of the dataset")
    data.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="use only the first N rows of --validation-dataset",
    )

    steering = parser.add_argument_group("steering")
    steering.add_argument("--steering-frac-start", type=float, default=0.3, help="fraction of the loop steering starts")
    steering.add_argument("--steering-frac-end", type=float, default=0.8, help="fraction of the loop steering ends")
    steering.add_argument(
        "--alpha-min", type=float, default=0.0, help="lower bound of alpha_field, fixed as in Regime A"
    )
    steering.add_argument(
        "--alpha-max", type=float, default=5.0, help="upper bound of alpha_field, fixed as in Regime A"
    )
    steering.add_argument(
        "--quantile-low",
        type=float,
        default=0.10,
        help="low percentile used to normalize the CFG-diff shape profile per sample, fixed as in Regime A",
    )
    steering.add_argument(
        "--quantile-high",
        type=float,
        default=0.90,
        help="high percentile used to normalize the CFG-diff shape profile per sample, fixed as in Regime A",
    )
    steering.add_argument(
        "--magnitude-init",
        type=float,
        default=0.15,
        help="initial predicted magnitude, must lie strictly inside (0, 1)",
    )

    loss = parser.add_argument_group(
        "loss",
        description=(
            "No principled defaults for these five: see outputs/07_cfg_diff_evaluation/results.csv"
            " (target_similarity/retain_similarity columns, method cfg_diff) as a starting sample for the margins."
        ),
    )
    loss.add_argument(
        "--margin-target",
        type=float,
        required=True,
        help="cosine similarity to the target below which l_target stops giving gradient",
    )
    loss.add_argument(
        "--margin-retain",
        type=float,
        required=True,
        help="cosine similarity to the retain prompt above which l_retain stops giving gradient",
    )
    loss.add_argument(
        "--retain-weight",
        type=float,
        default=1.0,
        help="weight of l_retain relative to l_target inside l_clap",
    )
    loss.add_argument(
        "--lambda-reg",
        type=float,
        required=True,
        help="target weight of the minimal-intervention penalty, reached after --lambda-reg-warmup-steps",
    )
    loss.add_argument(
        "--lambda-reg-warmup-steps",
        type=int,
        required=True,
        help="optimizer steps to linearly ramp lambda_reg from 0 to its target value; 0 disables warmup",
    )
    loss.add_argument(
        "--lambda-fid",
        type=float,
        required=True,
        help="weight of the ping-pong fidelity loss",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=5)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-2)
    optim.add_argument("--grad-accum-steps", type=int, default=4, help="batches accumulated per optimizer step")
    optim.add_argument("--max-grad-norm", type=float, default=1.0)

    generation = parser.add_argument_group("generation")
    generation.add_argument(
        "--model",
        type=str,
        default="small-music-base",
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

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument(
        "--no-half",
        default=True,
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
    if args.retain_weight < 0.0:
        raise ValueError(f"`--retain-weight` has to be non-negative but is {args.retain_weight}")
    if args.alpha_min >= args.alpha_max:
        raise ValueError(
            f"`--alpha-min` has to be smaller than `--alpha-max` but got {args.alpha_min} >= {args.alpha_max}"
        )
    if not 0.0 <= args.quantile_low < args.quantile_high <= 1.0:
        raise ValueError(
            "`--quantile-low` and `--quantile-high` have to satisfy `0.0 <= low < high <= 1.0` but are"
            f" {args.quantile_low} and {args.quantile_high}"
        )
    if not 0.0 < args.magnitude_init < 1.0:
        raise ValueError(f"`--magnitude-init` has to lie strictly inside (0, 1) but is {args.magnitude_init}")
    if not -1.0 <= args.margin_target <= 1.0:
        raise ValueError(f"`--margin-target` has to be a cosine similarity in [-1, 1] but is {args.margin_target}")
    if not -1.0 <= args.margin_retain <= 1.0:
        raise ValueError(f"`--margin-retain` has to be a cosine similarity in [-1, 1] but is {args.margin_retain}")
    if args.lambda_reg < 0.0:
        raise ValueError(f"`--lambda-reg` has to be non-negative but is {args.lambda_reg}")
    if args.lambda_reg_warmup_steps < 0:
        raise ValueError(
            f"`--lambda-reg-warmup-steps` has to be non-negative but is {args.lambda_reg_warmup_steps}"
        )
    if args.lambda_fid < 0.0:
        raise ValueError(f"`--lambda-fid` has to be non-negative but is {args.lambda_fid}")
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

    Two stacked panels rather than one plot with two y-scales: loss and similarities have unrelated
    ranges, so they get separate panels, while `s_t`/`s_r` share the cosine scale and reading one
    against the other is the suppression/retain trade-off itself.
    """
    iterations = history["iterations"]
    steps = [record["step"] for record in iterations]
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
        ax.annotate(
            f"{values[-1]:.3f}",
            (steps[-1], values[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=9,
            color=PALETTE["ink"],
        )

    draw_epoch_boundaries(ax_loss, label=True)
    draw_epoch_boundaries(ax_cos, label=False)

    draw_series(ax_loss, "loss", PALETTE["series"])
    draw_series(ax_cos, "s_t", PALETTE["series"], "target — suppressed")
    draw_series(ax_cos, "s_r", PALETTE["series_2"], "retain prompt — preserved")

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
            [record["validation"]["s_t"] for record in validation_epochs],
            color=PALETTE["series"],
            marker="o",
            linestyle="--",
            linewidth=1.2,
            label="validation target",
        )
        ax_cos.plot(
            validation_steps,
            [record["validation"]["s_r"] for record in validation_epochs],
            color=PALETTE["series_2"],
            marker="o",
            linestyle="--",
            linewidth=1.2,
            label="validation retain",
        )
        ax_loss.legend(frameon=False, fontsize=9, loc="best")

    for ax, title, ylabel in (
        (ax_loss, "Training loss — l_clap + lambda_reg * l_reg + lambda_fid * l_fid", "loss"),
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


def plot_loss_terms(history: dict, path: Path) -> None:
    r"""Writes `l_clap`, `lambda_reg * l_reg` and `lambda_fid * l_fid` separately (Task 4 logging)."""
    iterations = history["iterations"]
    steps = [record["step"] for record in iterations]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["surface"])
    for key, label, color in (
        ("l_clap", "l_clap", PALETTE["series"]),
        ("weighted_l_reg", "lambda_reg * l_reg", PALETTE["series_2"]),
        ("weighted_l_fid", "lambda_fid * l_fid", PALETTE["series_3"]),
    ):
        values = [record[key] for record in iterations]
        ax.plot(steps, values, color=color, linewidth=1.2, label=label)

    ax.set_title("Loss terms, weighted as summed into the training loss", fontsize=11, loc="left", pad=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss term")
    style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    fig.tight_layout()
    save_figure(fig, path)


def plot_magnitude_schedule(history: dict, path: Path) -> None:
    r"""
    Writes the mean predicted `magnitude` against the denoising-loop step, one line per epoch.

    Load-bearing diagnostic: shows whether the predictor learned a schedule that varies along the
    trajectory or collapsed to a constant. Unlike `alpha_field`, `magnitude` always lives in `[0, 1]`
    regardless of `alpha_min`/`alpha_max`, so the y-axis is fixed rather than derived from the CLI
    bounds.
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

    if len(selected) == 1:
        colors = [PALETTE["series"]]
    else:
        step_size = (len(ORDINAL_BLUE) - 1) / (len(selected) - 1)
        colors = [ORDINAL_BLUE[round(index * step_size)] for index in range(len(selected))]

    for record, color in zip(selected, colors, strict=True):
        step_values = [step for step, _ in record["magnitude_by_step"]]
        magnitudes = [magnitude for _, magnitude in record["magnitude_by_step"]]
        ax.plot(step_values, magnitudes, color=color, linewidth=2.0, label=f"epoch {record['epoch']}", zorder=2)

    ax.set_title("Predicted magnitude across the denoising trajectory", fontsize=11, loc="left", pad=10)
    ax.set_xlabel("denoising step (noisy → clean)")
    ax.set_ylabel("mean magnitude")
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(-0.04, 1.04)
    style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    fig.tight_layout()
    save_figure(fig, path)


def compute_losses(
    output,
    audio_embeds: torch.Tensor,
    targets: list[str],
    retains: list[str],
    hinge_loss: HingeClapLoss,
    lambda_reg: float,
    lambda_fid: float,
) -> tuple[torch.Tensor, dict]:
    r"""Shared by the training step and `run_validation`: combines the three Task 1-3 loss terms."""
    hinge = hinge_loss(audio_embeds, targets, retains)
    l_reg = minimal_intervention_penalty(output.alpha_field_records)
    l_fid, l_fid_per_step = pingpong_fidelity_loss(output.alpha_field_records)

    weighted_l_reg = lambda_reg * l_reg
    weighted_l_fid = lambda_fid * l_fid
    loss = hinge["l_clap"] + weighted_l_reg + weighted_l_fid

    magnitude_values = torch.cat([record["magnitude"].flatten() for record in output.alpha_field_records])

    metrics = {
        **hinge,
        "l_reg": l_reg,
        "l_fid": l_fid,
        "l_fid_per_step": [float(value.detach()) for value in l_fid_per_step],
        "weighted_l_reg": weighted_l_reg,
        "weighted_l_fid": weighted_l_fid,
        "magnitude_min": float(magnitude_values.min()),
        "magnitude_mean": float(magnitude_values.mean()),
        "magnitude_max": float(magnitude_values.max()),
    }
    return loss, metrics


@torch.inference_mode()
def run_validation(
    *,
    pipe: SteeringStableAudioPipeline,
    predictor: MagnitudePredictor,
    clap_loss: ClapLoss,
    hinge_loss: HingeClapLoss,
    dataloader: DataLoader,
    args: argparse.Namespace,
) -> dict:
    r"""Evaluates the held-out `--validation-dataset` once, with a fixed per-batch seed."""
    predictor.eval()
    total_loss = 0.0
    total_s_t = 0.0
    total_s_r = 0.0
    total_samples = 0
    magnitude_records: list[tuple[int, float]] = []

    for batch_index, (prompts, targets, retains, seeds) in enumerate(dataloader):
        if seeds[0] is not None:
            generator = [torch.Generator().manual_seed(seed) for seed in seeds]
        else:
            generator = torch.Generator().manual_seed(args.seed + 1_000_000 + batch_index)

        output = pipe(
            prompt=prompts,
            retain_prompt=retains,
            target_embed=clap_loss.encode_text(targets),
            steering_model=predictor,
            steering_mode="cfg_diff_magnitude",
            steering_frac_start=args.steering_frac_start,
            steering_frac_end=args.steering_frac_end,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_shape_quantile_low=args.quantile_low,
            alpha_shape_quantile_high=args.quantile_high,
            train=False,
            num_inference_steps=args.num_inference_steps,
            audio_length_in_s=args.audio_length_in_s,
            cfg_scale=args.cfg_scale,
            apg_scale=args.apg_scale,
            generator=generator,
            output_type="latent",
        )
        latents = output.audios
        waveform = pipe.decode_latents(
            latents, padding_mask=output.padding_mask, audio_length_in_s=args.audio_length_in_s
        )
        audio_embeds = clap_loss.encode_audio(waveform.mean(dim=1), pipe.sample_rate)

        loss, metrics = compute_losses(
            output, audio_embeds, targets, retains, hinge_loss, args.lambda_reg, args.lambda_fid
        )

        batch_size = len(prompts)
        total_samples += batch_size
        total_loss += float(loss) * batch_size
        total_s_t += float(metrics["s_t"]) * batch_size
        total_s_r += float(metrics["s_r"]) * batch_size
        magnitude_records.extend(magnitude_step_records(output.alpha_field_records))
        print(
            f"validation batch {batch_index + 1}/{len(dataloader)} loss {float(loss):+.4f} s_t {float(metrics['s_t']):+.4f} "
            f"s_r {float(metrics['s_r']):+.4f} magnitude_mean {metrics['magnitude_mean']:.4f}",
            flush=True,
        )

    return {
        "num_pairs": total_samples,
        "loss": total_loss / total_samples,
        "s_t": total_s_t / total_samples,
        "s_r": total_s_r / total_samples,
        "magnitude_by_step": mean_value_by_step(magnitude_records),
    }


def main() -> None:
    args = parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    dataset = PromptTargetDataset(args.dataset, args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_prompt_target,
    )
    print(f"Loaded {len(dataset)} prompts from {args.dataset} -> {len(dataloader)} batches per epoch")

    validation_loader = None
    if args.validation_dataset is not None:
        validation_dataset = PromptTargetDataset(args.validation_dataset, args.max_validation_samples)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_prompt_target,
        )
        print(
            f"Loaded {len(validation_dataset)} validation prompts from {args.validation_dataset} -> "
            f"{len(validation_loader)} batches"
        )

    _, example_target, example_retain, _ = dataset.rows[0]
    print(
        f"Retain weight {args.retain_weight}, margin_target {args.margin_target}, margin_retain"
        f" {args.margin_retain}, target {example_target!r}, example retain prompt: {example_retain!r}"
    )

    print(f"Loading {args.model} on {device} with {'float32' if args.no_half else 'half'} precision")
    pipe = SteeringStableAudioPipeline.from_pretrained(args.model, device=device, model_half=not args.no_half)
    pipe.diffusion.requires_grad_(False)
    pipe.diffusion.eval()

    predictor_config = {
        "latent_channels": pipe.io_channels,
        "target_embed_dim": None,  # filled in below, once CLAP is loaded
        "magnitude_init": args.magnitude_init,
    }

    clap_loss = ClapLoss.from_pretrained().to(device)
    predictor_config["target_embed_dim"] = clap_loss.embed_dim

    predictor = MagnitudePredictor(**predictor_config).to(device)
    num_parameters = sum(parameter.numel() for parameter in predictor.parameters())
    print(f"MagnitudePredictor: {num_parameters / 1e6:.2f}M parameters, config {predictor_config}")

    hinge_loss = HingeClapLoss(
        clap_loss, margin_target=args.margin_target, margin_retain=args.margin_retain, retain_weight=args.retain_weight
    ).to(device)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"Steering {args.num_steered_steps}/{args.num_inference_steps} steps in"
        f" [{args.steering_frac_start}, {args.steering_frac_end}), {args.audio_length_in_s}s clips,"
        f" alpha_field in [{args.alpha_min}, {args.alpha_max}]"
    )

    history: dict = {
        "args": json.loads(json.dumps(vars(args), default=str)),
        "checkpoint_selection_metric": "validation_loss" if validation_loader is not None else "training_loss",
        "iterations": [],
        "epochs": [],
    }
    best_metric = float("inf")
    step = 0

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_losses: list[float] = []
        epoch_s_t: list[float] = []
        epoch_s_r: list[float] = []
        epoch_magnitude_records: list[tuple[int, float]] = []

        for batch_index, (prompts, targets, retains, seeds) in enumerate(dataloader):
            step += 1
            if seeds[0] is not None:
                generator = [torch.Generator().manual_seed(seed) for seed in seeds]
            else:
                generator = torch.Generator().manual_seed(args.seed + step)

            output = pipe(
                prompt=prompts,
                retain_prompt=retains,
                target_embed=clap_loss.encode_text(targets),
                steering_model=predictor,
                steering_mode="cfg_diff_magnitude",
                steering_frac_start=args.steering_frac_start,
                steering_frac_end=args.steering_frac_end,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                alpha_shape_quantile_low=args.quantile_low,
                alpha_shape_quantile_high=args.quantile_high,
                train=True,
                num_inference_steps=args.num_inference_steps,
                audio_length_in_s=args.audio_length_in_s,
                cfg_scale=args.cfg_scale,
                apg_scale=args.apg_scale,
                generator=generator,
                output_type="latent",
            )
            latents = output.audios
            epoch_magnitude_records.extend(magnitude_step_records(output.alpha_field_records))

            waveform = pipe.decode_latents(
                latents, padding_mask=output.padding_mask, audio_length_in_s=args.audio_length_in_s
            )
            audio_embeds = clap_loss.encode_audio(waveform.mean(dim=1), pipe.sample_rate)

            lambda_reg_t = linear_warmup(step, args.lambda_reg, args.lambda_reg_warmup_steps)
            loss, metrics = compute_losses(
                output, audio_embeds, targets, retains, hinge_loss, lambda_reg_t, args.lambda_fid
            )

            (loss / args.grad_accum_steps).backward()

            grad_norm = None
            is_last_batch = batch_index == len(dataloader) - 1
            if (batch_index + 1) % args.grad_accum_steps == 0 or is_last_batch:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.detach())
            s_t = float(metrics["s_t"])
            s_r = float(metrics["s_r"])
            epoch_losses.append(loss_value)
            epoch_s_t.append(s_t)
            epoch_s_r.append(s_r)
            history["iterations"].append(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_value,
                    "s_t": s_t,
                    "s_r": s_r,
                    "l_clap": float(metrics["l_clap"].detach()),
                    "l_target": float(metrics["l_target"].detach()),
                    "l_retain": float(metrics["l_retain"].detach()),
                    "l_reg": float(metrics["l_reg"].detach()),
                    "l_fid": float(metrics["l_fid"].detach()),
                    "l_fid_per_step": metrics["l_fid_per_step"],
                    "weighted_l_reg": float(metrics["weighted_l_reg"].detach()),
                    "weighted_l_fid": float(metrics["weighted_l_fid"].detach()),
                    "lambda_reg": lambda_reg_t,
                    "target_satisfied_frac": metrics["target_satisfied_frac"],
                    "retain_satisfied_frac": metrics["retain_satisfied_frac"],
                    "both_satisfied_frac": metrics["both_satisfied_frac"],
                    "magnitude_min": metrics["magnitude_min"],
                    "magnitude_mean": metrics["magnitude_mean"],
                    "magnitude_max": metrics["magnitude_max"],
                    "grad_norm": grad_norm,
                }
            )
            print(
                f"epoch {epoch}/{args.epochs} batch {batch_index + 1}/{len(dataloader)}"
                f" loss {loss_value:+.4f} s_t {s_t:+.4f} s_r {s_r:+.4f} lambda_reg {lambda_reg_t:.4g}"
                f" magnitude[min/mean/max] {metrics['magnitude_min']:.3f}/{metrics['magnitude_mean']:.3f}/"
                f"{metrics['magnitude_max']:.3f} both_satisfied {metrics['both_satisfied_frac']:.2f}"
                f" grad_norm {'-' if grad_norm is None else f'{grad_norm:.3e}'}",
                flush=True,
            )

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_record = {
            "epoch": epoch,
            "last_step": step,
            "loss": epoch_loss,
            "s_t": sum(epoch_s_t) / len(epoch_s_t),
            "s_r": sum(epoch_s_r) / len(epoch_s_r),
            "magnitude_by_step": mean_value_by_step(epoch_magnitude_records),
            "validation": None,
        }
        print(f"epoch {epoch}/{args.epochs} mean loss {epoch_loss:+.4f}")

        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = run_validation(
                pipe=pipe,
                predictor=predictor,
                clap_loss=clap_loss,
                hinge_loss=hinge_loss,
                dataloader=validation_loader,
                args=args,
            )
            epoch_record["validation"] = validation_metrics
            print(
                f"epoch {epoch}/{args.epochs} validation loss {validation_metrics['loss']:+.4f} "
                f"s_t {validation_metrics['s_t']:+.4f} s_r {validation_metrics['s_r']:+.4f}"
            )

        history["epochs"].append(epoch_record)
        metric_value = validation_metrics["loss"] if validation_metrics is not None else epoch_loss

        checkpoint = {
            "state_dict": predictor.state_dict(),
            "config": predictor_config,
            "steering_mode": STEERING_MODE_MAGNITUDE,
            "model": args.model,
            "args": history["args"],
            "epoch": epoch,
            "loss": epoch_loss,
            "validation_metrics": validation_metrics,
            "checkpoint_selection_metric": history["checkpoint_selection_metric"],
            "checkpoint_selection_value": metric_value,
        }
        torch.save(checkpoint, output_dir / "magnitude_predictor.pt")
        if metric_value < best_metric:
            best_metric = metric_value
            torch.save(checkpoint, output_dir / "magnitude_predictor_best.pt")

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_loss_curve(history, output_dir / "loss_curve.png")
        plot_loss_terms(history, output_dir / "loss_terms.png")
        plot_magnitude_schedule(history, output_dir / "magnitude_schedule.png")

    metric_label = "validation loss" if validation_loader is not None else "training loss"
    print(f"\nDone. Best {metric_label} {best_metric:+.4f}. Artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
