r"""
Trains a `MagnitudePredictor` to steer `SteeringStableAudioPipeline` with a learned, per-sample gain
on the deterministic CFG-diff shape (`steering_mode="cfg_diff_magnitude"`).

Unlike a learned model that predicts a full per-frame `alpha_t`, this script keeps the temporal
profile (`pipelines.compute_cfg_diff_shape`) entirely deterministic and learns only its scalar gain,
`magnitude`. `alpha_min`, `alpha_max` and the shape quantiles stay fixed, matching the deterministic
`"cfg_diff"` mode, which uses a fixed `magnitude` hyperparameter instead of a learned one.

The loss combines two terms:

* a margin hinge contrast (`losses.HingeClapLoss`) between the CLAP audio embedding and the
  steering target (maximized past `margin_target`) and retain prompt (minimized past
  `margin_retain`), rather than an unbounded contrast that always rewards steering harder;
* a fidelity loss (`losses.pingpong_fidelity_loss`) that penalizes the steered x̂0 estimate for
  drifting from the non-steered one wherever the CFG-diff shape found nothing to correct.

Every steering/loss/optimization hyperparameter above is read from `--experiments-csv`, one row
per experiment, rather than from individual CLI flags: this script trains every row in one process
(reusing the Stable Audio 3 and CLAP loads across experiments) and writes back a copy of that CSV
with a `best_model_path` column, repo-root-relative, pointing at each experiment's best checkpoint
(or `FAILED: <error>` if that experiment's training run raised — the sweep continues past a failed
row). `outputs/07_cfg_diff_evaluation/results.csv`'s `target_similarity`/`retain_similarity` columns
(method `cfg_diff`) are output from the fixed-`magnitude` mode, a starting sample for picking margins
by listening to a batch of generations at different similarity levels.

Example:
    uv run scripts/train.py --dataset datasets/trumpet_simple_splits/train.csv \
        --validation-dataset datasets/trumpet_simple_splits/val.csv --batch-size 2 \
        --output outputs/trumpet-learned --experiments-csv experiments/learned_sweep.csv

`experiments/learned_sweep.csv_sweep.csv` must contain exactly these columns: `name`,
`steering_frac_start`, `steering_frac_end`, `alpha_min`, `alpha_max`, `quantile_low`,
`quantile_high`, `magnitude_init`, `margin_target`, `margin_retain`, `retain_weight`,
`lambda_fid`, `epochs`, `lr`, `weight_decay`, `grad_accum_steps`, `max_grad_norm`.
"""

import argparse
import csv
import json
import traceback
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

from losses import ClapLoss, HingeClapLoss, pingpong_fidelity_loss
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

# The experiments CSV's required columns, in write-back order. `read_experiments_csv` rejects any
# CSV whose header isn't exactly this set, and `main` writes the output CSV with these columns plus
# `best_model_path` appended.
EXPERIMENT_CSV_FIELDS = [
    "name",
    "steering_frac_start",
    "steering_frac_end",
    "alpha_min",
    "alpha_max",
    "quantile_low",
    "quantile_high",
    "magnitude_init",
    "margin_target",
    "margin_retain",
    "retain_weight",
    "lambda_fid",
    "epochs",
    "lr",
    "weight_decay",
    "grad_accum_steps",
    "max_grad_norm",
]
EXPERIMENT_FLOAT_FIELDS = [
    "steering_frac_start",
    "steering_frac_end",
    "alpha_min",
    "alpha_max",
    "quantile_low",
    "quantile_high",
    "magnitude_init",
    "margin_target",
    "margin_retain",
    "retain_weight",
    "lambda_fid",
    "lr",
    "weight_decay",
    "max_grad_norm",
]
EXPERIMENT_INT_FIELDS = ["epochs", "grad_accum_steps"]
# `evaluate.py` always runs a "base" and "cfg_diff" method and names its fixed-alpha methods
# `fixed_alpha_*`; an experiment name colliding with one of those would silently merge results when
# the output CSV is fed into evaluate.py later.
RESERVED_METHOD_NAMES = {"base", "cfg_diff"}


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


def read_experiments_csv(path: Path) -> list[dict[str, str]]:
    r"""
    Reads and validates the sweep's experiments CSV.

    Enforces the header is exactly `EXPERIMENT_CSV_FIELDS` (no missing, no extra columns) and that
    every row's `name` is non-empty, filesystem-safe, unique, and not reserved by `evaluate.py`'s
    built-in methods. Returns the raw string rows; numeric parsing and range validation happens in
    `parse_experiment_params`.
    """
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        missing = [field for field in EXPERIMENT_CSV_FIELDS if field not in fieldnames]
        extra = [field for field in fieldnames if field not in EXPERIMENT_CSV_FIELDS]
        if missing or extra:
            raise ValueError(
                f"{path} has to contain exactly the columns {EXPERIMENT_CSV_FIELDS}, but"
                f"{f' is missing {missing}' if missing else ''}{f' has unexpected columns {extra}' if extra else ''}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} contains no experiment rows")

    seen_names: set[str] = set()
    for row_number, row in enumerate(rows, start=2):  # header is row 1
        name = row["name"].strip()
        if not name:
            raise ValueError(f"row {row_number}: 'name' cannot be empty")
        if "/" in name or "\\" in name:
            raise ValueError(f"row {row_number}: 'name' {name!r} cannot contain path separators")
        if name in RESERVED_METHOD_NAMES or name.startswith("fixed_alpha_"):
            raise ValueError(
                f"row {row_number}: 'name' {name!r} is reserved for evaluate.py's built-in methods"
                " (base, cfg_diff, fixed_alpha_*)"
            )
        if name in seen_names:
            raise ValueError(f"row {row_number}: duplicate experiment name {name!r}")
        seen_names.add(name)
        row["name"] = name

    return rows


def parse_experiment_params(row: dict[str, str], row_number: int, num_inference_steps: int) -> dict:
    r"""
    Converts one experiment CSV row to typed values and validates it, mirroring the range checks the
    single-run CLI used to perform on `--margin-target`, `--alpha-min`, etc. Raises `ValueError` with
    the experiment's name and CSV row number on any invalid value.
    """
    name = row["name"]
    label = f"experiment {name!r} (row {row_number})"
    params: dict = {"name": name}
    try:
        for field in EXPERIMENT_FLOAT_FIELDS:
            params[field] = float(row[field])
        for field in EXPERIMENT_INT_FIELDS:
            params[field] = int(row[field])
    except ValueError as error:
        raise ValueError(f"{label}: could not parse numeric field: {error}") from error

    if params["retain_weight"] < 0.0:
        raise ValueError(f"{label}: retain_weight has to be non-negative but is {params['retain_weight']}")
    if params["alpha_min"] >= params["alpha_max"]:
        raise ValueError(
            f"{label}: alpha_min has to be smaller than alpha_max but got"
            f" {params['alpha_min']} >= {params['alpha_max']}"
        )
    if not 0.0 <= params["quantile_low"] < params["quantile_high"] <= 1.0:
        raise ValueError(
            f"{label}: quantile_low and quantile_high have to satisfy 0.0 <= low < high <= 1.0 but are"
            f" {params['quantile_low']} and {params['quantile_high']}"
        )
    if not 0.0 < params["magnitude_init"] < 1.0:
        raise ValueError(f"{label}: magnitude_init has to lie strictly inside (0, 1) but is {params['magnitude_init']}")
    if not -1.0 <= params["margin_target"] <= 1.0:
        raise ValueError(f"{label}: margin_target has to be a cosine similarity in [-1, 1] but is {params['margin_target']}")
    if not -1.0 <= params["margin_retain"] <= 1.0:
        raise ValueError(f"{label}: margin_retain has to be a cosine similarity in [-1, 1] but is {params['margin_retain']}")
    if params["lambda_fid"] < 0.0:
        raise ValueError(f"{label}: lambda_fid has to be non-negative but is {params['lambda_fid']}")
    if not 0.0 <= params["steering_frac_start"] < params["steering_frac_end"] <= 1.0:
        raise ValueError(
            f"{label}: steering_frac_start and steering_frac_end have to satisfy 0.0 <= start < end <= 1.0"
            f" but are {params['steering_frac_start']} and {params['steering_frac_end']}"
        )
    if params["epochs"] < 1:
        raise ValueError(f"{label}: epochs has to be at least 1 but is {params['epochs']}")
    if params["grad_accum_steps"] < 1:
        raise ValueError(f"{label}: grad_accum_steps has to be at least 1 but is {params['grad_accum_steps']}")

    num_steered_steps = len(
        steered_step_indices(num_inference_steps, params["steering_frac_start"], params["steering_frac_end"])
    )
    if num_steered_steps == 0:
        raise ValueError(
            f"{label}: the steering window [{params['steering_frac_start']}, {params['steering_frac_end']}) contains"
            f" no step of the {num_inference_steps} denoising steps, so the predictor would receive no gradient."
            " Widen the window or raise --num-inference-steps."
        )
    params["num_steered_steps"] = num_steered_steps

    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a batch of MagnitudePredictor experiments (steering_mode=cfg_diff_magnitude): a learned"
            " per-sample gain on the deterministic CFG-diff shape, one experiment per row of"
            " --experiments-csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data and output")
    data.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CSV with prompt,target and preferably an explicit retain_prompt column, shared by every experiment",
    )
    data.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help="CSV with the same schema as --dataset, held out and evaluated once per epoch, shared by every experiment",
    )
    data.add_argument("--output", type=Path, default=Path("outputs"), help="sweep root folder; each experiment gets a subfolder named after its 'name' column")
    data.add_argument("--batch-size", type=int, default=1, help="prompts generated per forward pass")
    data.add_argument("--max-samples", type=int, default=None, help="use only the first N rows of the dataset")
    data.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="use only the first N rows of --validation-dataset",
    )

    experiments = parser.add_argument_group("experiments")
    experiments.add_argument(
        "--experiments-csv",
        type=Path,
        required=True,
        help=f"CSV with one row per experiment, columns exactly {EXPERIMENT_CSV_FIELDS}",
    )

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
    runtime.add_argument("--seed", type=int, default=42, help="reset before every experiment, so each is reproducible independent of sweep order")
    runtime.add_argument(
        "--no-half",
        default=True,
        action="store_true",
        help="load the diffusion transformer in float32; the autoencoder and the steering algebra always are",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"`--batch-size` has to be at least 1 but is {args.batch_size}")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError(f"`--max-samples` has to be at least 1 but is {args.max_samples}")
    if args.max_validation_samples is not None and args.max_validation_samples < 1:
        raise ValueError(
            f"`--max-validation-samples` has to be at least 1 but is {args.max_validation_samples}"
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
        (ax_loss, "Training loss — l_clap + lambda_fid * l_fid", "loss"),
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
    r"""Writes `l_clap` and `lambda_fid * l_fid` separately (Task 4 logging)."""
    iterations = history["iterations"]
    steps = [record["step"] for record in iterations]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["surface"])
    for key, label, color in (
        ("l_clap", "l_clap", PALETTE["series"]),
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
    lambda_fid: float,
) -> tuple[torch.Tensor, dict]:
    r"""Shared by the training step and `run_validation`: combines the two Task 1/3 loss terms."""
    hinge = hinge_loss(audio_embeds, targets, retains)
    l_fid, l_fid_per_step = pingpong_fidelity_loss(output.alpha_field_records)

    weighted_l_fid = lambda_fid * l_fid
    loss = hinge["l_clap"] + weighted_l_fid

    magnitude_values = torch.cat([record["magnitude"].flatten() for record in output.alpha_field_records])

    metrics = {
        **hinge,
        "l_fid": l_fid,
        "l_fid_per_step": [float(value.detach()) for value in l_fid_per_step],
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

        loss, metrics = compute_losses(output, audio_embeds, targets, retains, hinge_loss, args.lambda_fid)

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


def run_experiment(
    *,
    pipe: SteeringStableAudioPipeline,
    clap_loss: ClapLoss,
    dataloader: DataLoader,
    validation_loader: DataLoader | None,
    combined_args: argparse.Namespace,
    output_dir: Path,
    device: str,
) -> dict:
    r"""
    Trains one experiment end to end: builds a fresh `MagnitudePredictor`/optimizer from
    `combined_args` (the shared CLI args merged with one CSV row's steering/loss/optimization
    values), runs the same epoch loop the single-run CLI used to run in `main`, and writes
    checkpoints/plots into `output_dir`. `pipe`/`clap_loss` are loaded once and shared across every
    experiment in the sweep.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(combined_args), indent=2, default=str), encoding="utf-8")

    distinct_targets = sorted({row[1] for row in dataloader.dataset.rows})
    _, _, example_retain, _ = dataloader.dataset.rows[0]
    target_label = (
        repr(distinct_targets[0])
        if len(distinct_targets) == 1
        else f"{len(distinct_targets)} distinct targets {distinct_targets}"
    )
    print(
        f"[{combined_args.name}] Retain weight {combined_args.retain_weight}, margin_target {combined_args.margin_target},"
        f" margin_retain {combined_args.margin_retain}, target(s) {target_label}, example retain prompt:"
        f" {example_retain!r}"
    )

    predictor_config = {
        "latent_channels": pipe.io_channels,
        "target_embed_dim": clap_loss.embed_dim,
        "magnitude_init": combined_args.magnitude_init,
    }
    predictor = MagnitudePredictor(**predictor_config).to(device)
    num_parameters = sum(parameter.numel() for parameter in predictor.parameters())
    print(f"[{combined_args.name}] MagnitudePredictor: {num_parameters / 1e6:.2f}M parameters, config {predictor_config}")

    hinge_loss = HingeClapLoss(
        clap_loss,
        margin_target=combined_args.margin_target,
        margin_retain=combined_args.margin_retain,
        retain_weight=combined_args.retain_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=combined_args.lr, weight_decay=combined_args.weight_decay)

    print(
        f"[{combined_args.name}] Steering {combined_args.num_steered_steps}/{combined_args.num_inference_steps} steps in"
        f" [{combined_args.steering_frac_start}, {combined_args.steering_frac_end}), {combined_args.audio_length_in_s}s clips,"
        f" alpha_field in [{combined_args.alpha_min}, {combined_args.alpha_max}]"
    )

    history: dict = {
        "args": json.loads(json.dumps(vars(combined_args), default=str)),
        "checkpoint_selection_metric": "validation_loss" if validation_loader is not None else "training_loss",
        "iterations": [],
        "epochs": [],
    }
    best_metric = float("inf")
    best_model_path = output_dir / "magnitude_predictor_best.pt"
    step = 0

    for epoch in range(1, combined_args.epochs + 1):
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
                generator = torch.Generator().manual_seed(combined_args.seed + step)

            output = pipe(
                prompt=prompts,
                retain_prompt=retains,
                target_embed=clap_loss.encode_text(targets),
                steering_model=predictor,
                steering_mode="cfg_diff_magnitude",
                steering_frac_start=combined_args.steering_frac_start,
                steering_frac_end=combined_args.steering_frac_end,
                alpha_min=combined_args.alpha_min,
                alpha_max=combined_args.alpha_max,
                alpha_shape_quantile_low=combined_args.quantile_low,
                alpha_shape_quantile_high=combined_args.quantile_high,
                train=True,
                num_inference_steps=combined_args.num_inference_steps,
                audio_length_in_s=combined_args.audio_length_in_s,
                cfg_scale=combined_args.cfg_scale,
                apg_scale=combined_args.apg_scale,
                generator=generator,
                output_type="latent",
            )
            latents = output.audios
            epoch_magnitude_records.extend(magnitude_step_records(output.alpha_field_records))

            waveform = pipe.decode_latents(
                latents, padding_mask=output.padding_mask, audio_length_in_s=combined_args.audio_length_in_s
            )
            audio_embeds = clap_loss.encode_audio(waveform.mean(dim=1), pipe.sample_rate)

            loss, metrics = compute_losses(
                output, audio_embeds, targets, retains, hinge_loss, combined_args.lambda_fid
            )

            (loss / combined_args.grad_accum_steps).backward()

            grad_norm = None
            is_last_batch = batch_index == len(dataloader) - 1
            if (batch_index + 1) % combined_args.grad_accum_steps == 0 or is_last_batch:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(predictor.parameters(), combined_args.max_grad_norm))
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
                    "l_fid": float(metrics["l_fid"].detach()),
                    "l_fid_per_step": metrics["l_fid_per_step"],
                    "weighted_l_fid": float(metrics["weighted_l_fid"].detach()),
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
                f"[{combined_args.name}] epoch {epoch}/{combined_args.epochs} batch {batch_index + 1}/{len(dataloader)}"
                f" loss {loss_value:+.4f} s_t {s_t:+.4f} s_r {s_r:+.4f}"
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
        print(f"[{combined_args.name}] epoch {epoch}/{combined_args.epochs} mean loss {epoch_loss:+.4f}")

        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = run_validation(
                pipe=pipe,
                predictor=predictor,
                clap_loss=clap_loss,
                hinge_loss=hinge_loss,
                dataloader=validation_loader,
                args=combined_args,
            )
            epoch_record["validation"] = validation_metrics
            print(
                f"[{combined_args.name}] epoch {epoch}/{combined_args.epochs} validation loss {validation_metrics['loss']:+.4f} "
                f"s_t {validation_metrics['s_t']:+.4f} s_r {validation_metrics['s_r']:+.4f}"
            )

        history["epochs"].append(epoch_record)
        metric_value = validation_metrics["loss"] if validation_metrics is not None else epoch_loss

        checkpoint = {
            "state_dict": predictor.state_dict(),
            "config": predictor_config,
            "steering_mode": STEERING_MODE_MAGNITUDE,
            "model": combined_args.model,
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
            torch.save(checkpoint, best_model_path)

        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_loss_curve(history, output_dir / "loss_curve.png")
        plot_loss_terms(history, output_dir / "loss_terms.png")
        plot_magnitude_schedule(history, output_dir / "magnitude_schedule.png")

    metric_label = "validation loss" if validation_loader is not None else "training loss"
    print(f"[{combined_args.name}] Done. Best {metric_label} {best_metric:+.4f}. Artifacts in {output_dir.resolve()}")

    return {"best_model_path": best_model_path, "best_metric": best_metric}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    raw_rows = read_experiments_csv(args.experiments_csv)
    experiments = [
        parse_experiment_params(row, row_number, args.num_inference_steps)
        for row_number, row in enumerate(raw_rows, start=2)
    ]
    print(f"Loaded {len(experiments)} experiments from {args.experiments_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    print(f"Loading {args.model} on {device} with {'float32' if args.no_half else 'half'} precision")
    pipe = SteeringStableAudioPipeline.from_pretrained(args.model, device=device, model_half=not args.no_half)
    pipe.diffusion.requires_grad_(False)
    pipe.diffusion.eval()

    clap_loss = ClapLoss.from_pretrained().to(device)

    output_fieldnames = EXPERIMENT_CSV_FIELDS + ["best_model_path"]
    output_rows: list[dict] = [{field: raw_row[field] for field in EXPERIMENT_CSV_FIELDS} for raw_row in raw_rows]
    results_path = output_root / "experiments.csv"

    def write_results() -> None:
        with results_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    num_succeeded = 0
    num_failed = 0
    for index, experiment_params in enumerate(experiments):
        name = experiment_params["name"]
        print(f"\n=== Experiment {index + 1}/{len(experiments)}: {name} ===", flush=True)
        combined_args = argparse.Namespace(**{**vars(args), **experiment_params})
        experiment_output_dir = output_root / name
        torch.manual_seed(args.seed)
        try:
            if experiment_output_dir.exists() and any(experiment_output_dir.iterdir()):
                raise FileExistsError(f"experiment output {experiment_output_dir} is not empty")
            result = run_experiment(
                pipe=pipe,
                clap_loss=clap_loss,
                dataloader=dataloader,
                validation_loader=validation_loader,
                combined_args=combined_args,
                output_dir=experiment_output_dir,
                device=device,
            )
            output_rows[index]["best_model_path"] = (
                result["best_model_path"].resolve().relative_to(repo_root).as_posix()
            )
            num_succeeded += 1
        except Exception as error:
            traceback.print_exc()
            output_rows[index]["best_model_path"] = f"FAILED: {error}"
            num_failed += 1
        write_results()

    print(
        f"\nDone. {num_succeeded}/{len(experiments)} experiments succeeded, {num_failed} failed."
        f" Results written to {results_path.resolve()}"
    )


if __name__ == "__main__":
    main()
