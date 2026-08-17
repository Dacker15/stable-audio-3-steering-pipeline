r"""
Evaluates a trained ``SteeringPredictor`` against the unsteered Stable Audio 3 baseline.

The evaluator runs up to 4 independent steps against the same paired noise: an always-on ``base``
reference (constant ``alpha=0``, i.e. plain full-prompt CFG), the ``predictor`` pipeline (needs a
trained checkpoint), the ``fixed-alpha`` pipeline (constant-alpha controls) and the ``cfg-diff``
pipeline (deterministic, training-free). Each of the latter three has its own prefixed CLI
parameters, so passing e.g. ``--cfg-diff-cfg-scale`` never affects the ``predictor`` pipeline's
generation settings and vice versa.

Example smoke test (using a checkpoint produced by ``scripts/train.py``):

    uv run python scripts/evaluate.py \
        --dataset datasets/trumpet_simple_splits/validation.csv \
        --predictor-checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt \
        --output outputs/eval-smoke --max-samples 4 --num-seeds 1 --predictor-num-inference-steps 20

Example final run with fixed-alpha controls:

    uv run python scripts/evaluate.py \
        --dataset datasets/trumpet_simple_splits/test.csv \
        --predictor-checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt \
        --output outputs/eval-final --num-seeds 5 \
        --fixed-alpha-values 0.25 0.5 0.75 1.0

Example cfg-diff-only run (no trained checkpoint required):

    uv run python scripts/evaluate.py \
        --dataset datasets/trumpet_simple_splits/test.csv \
        --output outputs/eval-cfg-diff-only --num-seeds 3 --fixed-alpha-values

Omitting --predictor-checkpoint drops the "learned" method, since it needs trained predictor
weights; --fixed-alpha-values with no values then leaves "base" and "cfg_diff" as the evaluated
methods.
"""

import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from losses import ClapLoss
from pipelines import STEERING_MODE, SteeringPredictor, SteeringStableAudioPipeline, FixedAlphaSteering
from utils import PromptTargetDataset, save_waveform


RESULT_FIELDS = [
    "sample_id",
    "seed",
    "method",
    "prompt",
    "target",
    "retain_prompt",
    "target_similarity",
    "prompt_similarity",
    "retain_similarity",
    "rms",
    "peak",
    "silence_ratio",
    "clipping_ratio",
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
    "retain_dominant_ratio",
]
PLOT_COLORS = ("#2a78d6", "#d56b25", "#39875b", "#845ec2", "#b64c66", "#6b6b6b")

# Shared by every pipeline's generation-parameter resolution: the predictor pipeline falls back to
# these only after the checkpoint's own stored training args; the other pipelines fall back to these
# directly, since they never depend on a checkpoint.
GENERATION_DEFAULTS: dict[str, float | int] = {
    "num_inference_steps": 50,
    "audio_length_in_s": 10.0,
    "cfg_scale": 7.0,
    "apg_scale": 0.0,
    "steering_frac_start": 0.3,
    "steering_frac_end": 0.8,
}

# `alpha_min`/`alpha_max`/`alpha_magnitude`/quantiles are meaningless outside `steering_mode="cfg_diff"`
# (see `SteeringStableAudioPipeline.__call__`'s docstring), so this is what "base" and "predictor" pass
# through: harmless placeholders, never read by `mode="learned"`.
INERT_ALPHA_BOUNDS = (0.0, 1.0, 0.0, 0.10, 0.90)


def _flag(prefix: str, name: str) -> str:
    return f"--{prefix.replace('_', '-')}-{name.replace('_', '-')}"


def _add_generation_args(group: argparse._ArgumentGroup, prefix: str, fallback_note: str) -> None:
    flag_prefix = prefix.replace("_", "-")
    group.add_argument(
        f"--{flag_prefix}-num-inference-steps", dest=f"{prefix}_num_inference_steps",
        type=int, default=None, help=f"{fallback_note}, else {GENERATION_DEFAULTS['num_inference_steps']}",
    )
    group.add_argument(
        f"--{flag_prefix}-audio-length-in-s", dest=f"{prefix}_audio_length_in_s",
        type=float, default=None, help=f"{fallback_note}, else {GENERATION_DEFAULTS['audio_length_in_s']}",
    )
    group.add_argument(
        f"--{flag_prefix}-cfg-scale", dest=f"{prefix}_cfg_scale",
        type=float, default=None, help=f"{fallback_note}, else {GENERATION_DEFAULTS['cfg_scale']}",
    )
    group.add_argument(
        f"--{flag_prefix}-apg-scale", dest=f"{prefix}_apg_scale",
        type=float, default=None, help=f"{fallback_note}, else {GENERATION_DEFAULTS['apg_scale']}",
    )
    group.add_argument(
        f"--{flag_prefix}-steering-frac-start", dest=f"{prefix}_steering_frac_start",
        type=float, default=None, help=f"{fallback_note}, else {GENERATION_DEFAULTS['steering_frac_start']}",
    )
    group.add_argument(
        f"--{flag_prefix}-steering-frac-end", dest=f"{prefix}_steering_frac_end",
        type=float, default=None, help=f"{fallback_note}, else {GENERATION_DEFAULTS['steering_frac_end']}",
    )


def _validate_generation_overrides(parser: argparse.ArgumentParser, args: argparse.Namespace, prefix: str) -> None:
    num_inference_steps = getattr(args, f"{prefix}_num_inference_steps")
    if num_inference_steps is not None and num_inference_steps < 1:
        parser.error(f"{_flag(prefix, 'num_inference_steps')} must be at least 1")
    audio_length_in_s = getattr(args, f"{prefix}_audio_length_in_s")
    if audio_length_in_s is not None and audio_length_in_s <= 0.0:
        parser.error(f"{_flag(prefix, 'audio_length_in_s')} must be positive")
    cfg_scale = getattr(args, f"{prefix}_cfg_scale")
    if cfg_scale is not None and cfg_scale <= 1.0:
        parser.error(f"{_flag(prefix, 'cfg_scale')} must exceed 1.0 because steering is applied inside the CFG branch")
    apg_scale = getattr(args, f"{prefix}_apg_scale")
    if apg_scale is not None and not 0.0 <= apg_scale <= 1.0:
        parser.error(f"{_flag(prefix, 'apg_scale')} must be in [0.0, 1.0]")
    frac_start = getattr(args, f"{prefix}_steering_frac_start")
    frac_end = getattr(args, f"{prefix}_steering_frac_end")
    if frac_start is not None and frac_end is not None and not 0.0 <= frac_start < frac_end <= 1.0:
        parser.error(
            f"{_flag(prefix, 'steering_frac_start')} and {_flag(prefix, 'steering_frac_end')} must satisfy"
            " 0.0 <= start < end <= 1.0"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SteeringPredictor with paired Stable Audio 3 generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="evaluation CSV with prompt,target and preferably an explicit retain_prompt column",
    )
    data.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    data.add_argument("--max-samples", type=int, default=None, help="evaluate only the first N prompts")
    data.add_argument(
        "--model",
        default=None,
        help=(
            "Stable Audio 3 `-base` checkpoint; defaults to the predictor checkpoint's own model when"
            " --predictor-checkpoint is given, else to small-music-base"
        ),
    )

    runtime = parser.add_argument_group("noise and runtime")
    runtime.add_argument("--num-seeds", type=int, default=1, help="independent noises generated per prompt")
    runtime.add_argument("--seed", type=int, default=1000, help="first evaluation seed")
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    runtime.add_argument(
        "--no-half",
        default=True,
        action="store_true",
        help="load the diffusion transformer in float32; the autoencoder and the steering algebra always are",
    )
    runtime.add_argument("--save-audio", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--bootstrap-samples", type=int, default=10_000)
    runtime.add_argument("--silence-threshold", type=float, default=1e-4)
    runtime.add_argument("--clipping-threshold", type=float, default=0.999)

    predictor = parser.add_argument_group(
        "predictor pipeline",
        description=(
            "The 'learned' method: SteeringPredictor loaded from --predictor-checkpoint, predicting alpha_t from"
            " the latents at every steered step. Skipped entirely when --predictor-checkpoint is omitted."
        ),
    )
    predictor.add_argument(
        "--predictor-checkpoint",
        type=Path,
        default=None,
        help="checkpoint produced by scripts/train.py; omit to skip the predictor pipeline",
    )
    _add_generation_args(predictor, "predictor", "defaults to checkpoint setting")

    fixed_alpha = parser.add_argument_group(
        "fixed-alpha pipeline",
        description="Constant-alpha controls: one evaluated method per --fixed-alpha-values entry. Skipped when empty.",
    )
    fixed_alpha.add_argument(
        "--fixed-alpha-values",
        type=float,
        nargs="*",
        default=[1.0],
        metavar="ALPHA",
        help="constant alpha values to evaluate, one method each (default contains 1.0); pass with no values to skip this pipeline",
    )
    # fixed_alpha.add_argument(
    #     "--fixed-alpha-min", type=float, default=0.0, help="lower bound every --fixed-alpha-values entry must satisfy"
    # )
    # fixed_alpha.add_argument(
    #     "--fixed-alpha-max", type=float, default=5.0, help="upper bound every --fixed-alpha-values entry must satisfy"
    # )
    _add_generation_args(fixed_alpha, "fixed_alpha", "defaults to hard-coded default")

    cfg_diff = parser.add_argument_group(
        "cfg-diff pipeline",
        description=(
            "Regime A: a zero-cost, training-free 'cfg_diff' method, always evaluated, deriving a deterministic"
            " per-frame alpha_t from the CFG-diff norm instead of calling a steering model."
        ),
    )
    cfg_diff.add_argument("--cfg-diff-alpha-min", type=float, default=0.0, help="lower bound of cfg_diff's alpha_t")
    cfg_diff.add_argument("--cfg-diff-alpha-max", type=float, default=5.0, help="upper bound of cfg_diff's alpha_t")
    cfg_diff.add_argument(
        "--cfg-diff-alpha-magnitude", type=float, default=0.6, help="global gain in [0, 1] for cfg_diff's alpha_t"
    )
    cfg_diff.add_argument(
        "--cfg-diff-alpha-quantile-low",
        type=float,
        default=0.10,
        help="low percentile used by cfg_diff to normalize its temporal profile per sample",
    )
    cfg_diff.add_argument(
        "--cfg-diff-alpha-quantile-high",
        type=float,
        default=0.90,
        help="high percentile used by cfg_diff to normalize its temporal profile per sample",
    )
    _add_generation_args(cfg_diff, "cfg_diff", "defaults to hard-coded default")

    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")
    if args.silence_threshold < 0.0:
        parser.error("--silence-threshold cannot be negative")
    if args.clipping_threshold <= 0.0:
        parser.error("--clipping-threshold must be positive")

    if args.cfg_diff_alpha_min >= args.cfg_diff_alpha_max:
        parser.error("--cfg-diff-alpha-min must be smaller than --cfg-diff-alpha-max")
    if not 0.0 <= args.cfg_diff_alpha_magnitude <= 1.0:
        parser.error("--cfg-diff-alpha-magnitude must be in [0, 1]")
    if not 0.0 <= args.cfg_diff_alpha_quantile_low < args.cfg_diff_alpha_quantile_high <= 1.0:
        parser.error(
            "--cfg-diff-alpha-quantile-low and --cfg-diff-alpha-quantile-high must satisfy"
            " 0.0 <= low < high <= 1.0"
        )

    for prefix in ("predictor", "fixed_alpha", "cfg_diff"):
        _validate_generation_overrides(parser, args, prefix)

    return args


def _finalize_generation_config(resolved: dict[str, float | int]) -> dict[str, float | int]:
    resolved = dict(resolved)
    resolved["num_inference_steps"] = int(resolved["num_inference_steps"])
    for name in ("audio_length_in_s", "cfg_scale", "apg_scale", "steering_frac_start", "steering_frac_end"):
        resolved[name] = float(resolved[name])

    if resolved["num_inference_steps"] < 1:
        raise ValueError("num_inference_steps must be at least 1")
    if resolved["audio_length_in_s"] <= 0.0:
        raise ValueError("audio_length_in_s must be positive")
    if resolved["cfg_scale"] <= 1.0:
        raise ValueError("cfg_scale must exceed 1.0 because steering is applied inside the CFG branch")
    if not 0.0 <= resolved["apg_scale"] <= 1.0:
        raise ValueError("apg_scale must be in [0.0, 1.0]")
    start = resolved["steering_frac_start"]
    end = resolved["steering_frac_end"]
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("steering fractions must satisfy 0.0 <= start < end <= 1.0")

    active_steps = sum(
        start <= step / resolved["num_inference_steps"] < end
        for step in range(resolved["num_inference_steps"])
    )
    if active_steps == 0:
        raise ValueError("the selected steering window contains no denoising step")
    resolved["num_steered_steps"] = active_steps
    return resolved


def resolve_predictor_generation_config(args: argparse.Namespace, checkpoint_args: dict) -> dict[str, float | int]:
    """CLI overrides checkpoint training settings; hard-coded defaults are the last fallback."""

    resolved = {}
    for name, fallback in GENERATION_DEFAULTS.items():
        cli_value = getattr(args, f"predictor_{name}")
        resolved[name] = cli_value if cli_value is not None else checkpoint_args.get(name, fallback)
    return _finalize_generation_config(resolved)


def resolve_generation_config(args: argparse.Namespace, prefix: str) -> dict[str, float | int]:
    """CLI overrides the hard-coded defaults directly; this pipeline never depends on a checkpoint."""

    resolved = {}
    for name, fallback in GENERATION_DEFAULTS.items():
        cli_value = getattr(args, f"{prefix}_{name}")
        resolved[name] = cli_value if cli_value is not None else fallback
    return _finalize_generation_config(resolved)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return device


def prepare_output_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Evaluation output {path} is not empty. Choose another --output.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fixed_alpha_name(alpha: float) -> str:
    text = f"{alpha:g}".replace("-", "minus_").replace(".", "p")
    return f"fixed_alpha_{text}"


def waveform_metrics(waveform: np.ndarray, silence_threshold: float, clipping_threshold: float) -> dict[str, float]:
    """Signal sanity checks. Stereo input is downmixed first, so silence is silence in both channels."""

    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    waveform = waveform.reshape(-1)
    if waveform.size == 0:
        raise ValueError("generated waveform is empty")
    if not np.isfinite(waveform).all():
        raise ValueError("generated waveform contains NaN or infinite values")
    absolute = np.abs(waveform)
    return {
        "rms": float(np.sqrt(np.mean(np.square(waveform)))),
        "peak": float(absolute.max()),
        "silence_ratio": float(np.mean(absolute < silence_threshold)),
        "clipping_ratio": float(np.mean(absolute >= clipping_threshold)),
    }


def alpha_metrics(records: list[tuple[float, float]]) -> dict[str, float | None]:
    if not records:
        return {
            "alpha_mean": None,
            "alpha_std": None,
            "alpha_min": None,
            "alpha_max": None,
            "retain_dominant_ratio": None,
        }
    values = np.asarray([alpha for _, alpha in records], dtype=np.float64)
    return {
        "alpha_mean": float(values.mean()),
        "alpha_std": float(values.std()),
        "alpha_min": float(values.min()),
        "alpha_max": float(values.max()),
        "retain_dominant_ratio": float(np.mean(values > 0.5)),
    }


def _mean_ci_by_prompt(
    rows: list[dict],
    value_fn,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Averages seeds within prompt, then bootstraps prompts as the independent units."""

    by_prompt: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_prompt[int(row["sample_id"])].append(float(value_fn(row)))
    prompt_values = np.asarray([np.mean(values) for values in by_prompt.values()], dtype=np.float64)
    mean = float(prompt_values.mean())
    indices = rng.integers(0, len(prompt_values), size=(bootstrap_samples, len(prompt_values)))
    bootstrap_means = prompt_values[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {"mean": mean, "ci95_low": float(low), "ci95_high": float(high)}


def summarize(
    rows: list[dict],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    base_by_key = {(int(row["sample_id"]), int(row["seed"])): row for row in by_method["base"]}
    summary = {"methods": {}}

    for method, method_rows in by_method.items():
        target = _mean_ci_by_prompt(method_rows, lambda row: row["target_similarity"], bootstrap_samples, rng)
        prompt = _mean_ci_by_prompt(method_rows, lambda row: row["prompt_similarity"], bootstrap_samples, rng)
        retain = _mean_ci_by_prompt(method_rows, lambda row: row["retain_similarity"], bootstrap_samples, rng)
        method_summary = {
            "num_generations": len(method_rows),
            "target_similarity": target,
            "prompt_similarity": prompt,
            "retain_similarity": retain,
            "rms_mean": float(np.mean([row["rms"] for row in method_rows])),
            "peak_mean": float(np.mean([row["peak"] for row in method_rows])),
            "silence_ratio_mean": float(np.mean([row["silence_ratio"] for row in method_rows])),
            "clipping_ratio_mean": float(np.mean([row["clipping_ratio"] for row in method_rows])),
        }

        if method != "base":
            paired = []
            for row in method_rows:
                key = (int(row["sample_id"]), int(row["seed"]))
                base = base_by_key[key]
                paired.append(
                    {
                        **row,
                        "target_gain": float(base["target_similarity"] - row["target_similarity"]),
                        "prompt_change": float(row["prompt_similarity"] - base["prompt_similarity"]),
                        "retain_change": float(row["retain_similarity"] - base["retain_similarity"]),
                    }
                )
            method_summary["target_suppression_gain"] = _mean_ci_by_prompt(
                paired, lambda row: row["target_gain"], bootstrap_samples, rng
            )
            method_summary["prompt_similarity_change"] = _mean_ci_by_prompt(
                paired, lambda row: row["prompt_change"], bootstrap_samples, rng
            )
            method_summary["retain_similarity_change"] = _mean_ci_by_prompt(
                paired, lambda row: row["retain_change"], bootstrap_samples, rng
            )

            prompt_gains: dict[int, list[float]] = defaultdict(list)
            for row in paired:
                prompt_gains[int(row["sample_id"])].append(row["target_gain"])
            method_summary["fraction_prompts_with_target_reduction"] = float(
                np.mean([np.mean(values) > 0.0 for values in prompt_gains.values()])
            )

        alpha_values = [row["alpha_mean"] for row in method_rows if row["alpha_mean"] is not None]
        if alpha_values:
            method_summary["alpha_mean"] = float(np.mean(alpha_values))
            method_summary["retain_dominant_ratio_mean"] = float(
                np.mean([row["retain_dominant_ratio"] for row in method_rows])
            )
        summary["methods"][method] = method_summary
    return summary


def plot_target_similarity(summary: dict, path: Path) -> None:
    methods = list(summary["methods"])
    means = [summary["methods"][method]["target_similarity"]["mean"] for method in methods]
    lows = [
        mean - summary["methods"][method]["target_similarity"]["ci95_low"]
        for method, mean in zip(methods, means, strict=True)
    ]
    highs = [
        summary["methods"][method]["target_similarity"]["ci95_high"] - mean
        for method, mean in zip(methods, means, strict=True)
    ]

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(methods)), 4.8))
    x = np.arange(len(methods))
    colors = [PLOT_COLORS[index % len(PLOT_COLORS)] for index in range(len(methods))]
    ax.bar(x, means, color=colors, alpha=0.85)
    ax.errorbar(x, means, yerr=np.asarray([lows, highs]), fmt="none", color="#222222", capsize=4)
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set_ylabel("CLAP cosine similarity to target")
    ax.set_title("Target similarity (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(summary: dict, path: Path) -> None:
    methods = [method for method in summary["methods"] if method != "base"]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color="#999999", linewidth=1)
    ax.axvline(0.0, color="#999999", linewidth=1)
    for index, method in enumerate(methods):
        method_summary = summary["methods"][method]
        x = method_summary["target_suppression_gain"]["mean"]
        # the retain prompt, not the full prompt: the full prompt still contains the target, so
        # improving on it partly means failing to suppress
        y = method_summary["retain_similarity_change"]["mean"]
        ax.scatter(x, y, s=70, color=PLOT_COLORS[index % len(PLOT_COLORS)])
        ax.annotate(method, (x, y), xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("Target suppression gain vs base (higher is better)")
    ax.set_ylabel("Retain-prompt similarity change vs base (higher is better)")
    ax.set_title("Suppression/fidelity trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alpha_schedules(alpha_runs: list[dict], path: Path) -> None:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in alpha_runs:
        for step, alpha in run["records"]:
            grouped[run["method"]][int(step)].append(float(alpha))
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for index, (method, by_step) in enumerate(grouped.items()):
        # the pipeline records the loop step index, which already rises from noisy to clean
        steps = sorted(by_step)
        values = [float(np.mean(by_step[step])) for step in steps]
        ax.plot(steps, values, label=method, color=PLOT_COLORS[index % len(PLOT_COLORS)], linewidth=2)
    ax.axhline(0.5, color="#999999", linewidth=1, linestyle="--", label="full/retain midpoint")
    ax.set_xlabel("Denoising step (noisy to clean)")
    ax.set_ylabel("Mean alpha")
    ax.set_title("Steering schedules")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _same_dataset_as_training(eval_path: Path, checkpoint_args: dict) -> bool:
    training_path = checkpoint_args.get("dataset")
    if training_path is None:
        return False
    try:
        return eval_path.resolve() == Path(training_path).resolve()
    except OSError:
        return False


def run_pipeline(
    pipe: SteeringStableAudioPipeline,
    clap: ClapLoss,
    dataset: PromptTargetDataset,
    target_embeds: dict[str, torch.Tensor],
    methods: dict[str, tuple[nn.Module | None, str]],
    generation_config: dict[str, float | int],
    alpha_bounds: tuple[float, float, float, float, float],
    args: argparse.Namespace,
    device: torch.device,
    sampling_rate: int,
    writer: "csv.DictWriter",
    csv_file,
    rows: list[dict],
    alpha_runs: list[dict],
    progress: dict[str, int],
    output_dir: Path,
) -> None:
    """Runs one pipeline's methods over the full dataset/seed grid, writing paired rows.

    `methods` maps a method name to `(steering_model, steering_mode)`, following
    `SteeringStableAudioPipeline.__call__`'s contract: `steering_model=None` requires
    `steering_mode="cfg_diff"`, anything else uses `steering_mode="learned"`. `alpha_bounds` is
    `(alpha_min, alpha_max, alpha_magnitude, quantile_low, quantile_high)`, meaningful only for
    `"cfg_diff"` methods and otherwise an inert pass-through.
    """

    alpha_min, alpha_max, alpha_magnitude, quantile_low, quantile_high = alpha_bounds

    # Batch size is intentionally one. The current pipeline logs alpha averaged over the batch;
    # evaluating one prompt at a time keeps every saved schedule attributable to one prompt.
    for sample_id, (prompt, target, retain_prompt, row_seed) in enumerate(dataset.rows):
        for seed_index in range(args.num_seeds):
            sample_seed = (
                row_seed + seed_index
                if row_seed is not None
                else args.seed + seed_index * len(dataset) + sample_id
            )
            for method_name, (steering_model, steering_mode) in methods.items():
                # Recreate the generator for every method so paired runs receive identical noise.
                generator = torch.Generator().manual_seed(sample_seed)
                with torch.inference_mode():
                    output = pipe(
                        prompt=prompt,
                        retain_prompt=retain_prompt,
                        target_embed=target_embeds[target],
                        steering_model=steering_model,
                        steering_mode=steering_mode,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        alpha_magnitude=alpha_magnitude,
                        alpha_shape_quantile_low=quantile_low,
                        alpha_shape_quantile_high=quantile_high,
                        steering_frac_start=generation_config["steering_frac_start"],
                        steering_frac_end=generation_config["steering_frac_end"],
                        train=False,
                        num_inference_steps=generation_config["num_inference_steps"],
                        audio_length_in_s=generation_config["audio_length_in_s"],
                        cfg_scale=generation_config["cfg_scale"],
                        apg_scale=generation_config["apg_scale"],
                        generator=generator,
                        output_type="pt",
                    )
                    # `(1, channels, samples)`; CLAP's audio tower is mono, so the channels are
                    # summed to a mid signal for scoring while the saved file stays stereo
                    waveform_tensor = output.audios.to(device)
                    audio_embeds = clap.encode_audio(waveform_tensor.mean(dim=1), sampling_rate)
                    target_similarity = 1.0 - float(clap(audio_embeds, target)[0])
                    prompt_similarity = 1.0 - float(clap(audio_embeds, prompt)[0])
                    retain_similarity = 1.0 - float(clap(audio_embeds, retain_prompt)[0])

                waveform = waveform_tensor[0].detach().float().cpu().numpy()
                signal = waveform_metrics(waveform, args.silence_threshold, args.clipping_threshold)
                records = [(int(step), float(alpha)) for step, alpha in output.alpha_records]
                alpha_stats = alpha_metrics(records)

                if args.save_audio:
                    audio_relative_path = str(
                        Path("audio") / method_name / f"sample_{sample_id:04d}_seed_{sample_seed}.wav"
                    )
                    save_waveform(output_dir / audio_relative_path, waveform, sampling_rate)

                row = {
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "method": method_name,
                    "prompt": prompt,
                    "target": target,
                    "retain_prompt": retain_prompt,
                    "target_similarity": target_similarity,
                    "prompt_similarity": prompt_similarity,
                    "retain_similarity": retain_similarity,
                    **signal,
                    **alpha_stats,
                }
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()

                alpha_runs.append(
                    {
                        "sample_id": sample_id,
                        "seed": sample_seed,
                        "method": method_name,
                        "records": records,
                    }
                )

                progress["completed"] += 1
                print(
                    f"[{progress['completed']}/{progress['total']}] sample={sample_id} seed={sample_seed} "
                    f"method={method_name} target_cos={target_similarity:+.4f} retain_cos={retain_similarity:+.4f} "
                    f"prompt_cos={prompt_similarity:+.4f}",
                    flush=True,
                )


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_directory(args.output)

    checkpoint = None
    checkpoint_args: dict = {}
    predictor_config: dict = {}
    if args.predictor_checkpoint is not None:
        checkpoint = torch.load(args.predictor_checkpoint, map_location="cpu", weights_only=False)
        required_keys = {"state_dict", "config"}
        missing = required_keys - set(checkpoint)
        if missing:
            raise ValueError(f"checkpoint {args.predictor_checkpoint} is missing keys {sorted(missing)}")
        if "steering_mode" not in checkpoint:
            raise ValueError(
                f"checkpoint {args.predictor_checkpoint} predates target-specific full-to-retain steering. Its"
                " alpha values used the incompatible global-CFG formula, so a new checkpoint must be trained."
            )
        if checkpoint["steering_mode"] != STEERING_MODE:
            raise ValueError(
                f"checkpoint {args.predictor_checkpoint} uses steering mode {checkpoint['steering_mode']!r}, but"
                f" this evaluator requires {STEERING_MODE!r}. Checkpoints trained against MusicLDM cannot be"
                " evaluated on Stable Audio 3: the two backbones do not share a latent space."
            )
        checkpoint_args = checkpoint.get("args", {})
        predictor_config = checkpoint["config"]

    base_generation_config = _finalize_generation_config(dict(GENERATION_DEFAULTS))
    predictor_generation_config = (
        resolve_predictor_generation_config(args, checkpoint_args) if checkpoint is not None else None
    )
    fixed_alpha_generation_config = (
        resolve_generation_config(args, "fixed_alpha") if args.fixed_alpha_values else None
    )
    cfg_diff_generation_config = resolve_generation_config(args, "cfg_diff")

    dataset = PromptTargetDataset(args.dataset, args.max_samples)
    device = resolve_device(args.device)
    model_name = args.model or (checkpoint.get("model") if checkpoint is not None else None) or "small-music-base"
    same_dataset = checkpoint is not None and _same_dataset_as_training(args.dataset, checkpoint_args)
    if same_dataset:
        warnings.warn(
            "The evaluation CSV is the same dataset path stored in the checkpoint. Results measure training-set "
            "behaviour, not held-out generalization.",
            stacklevel=2,
        )

    config = {
        "dataset": str(args.dataset.resolve()),
        "output": str(output_dir),
        "model": model_name,
        "num_prompts": len(dataset),
        "num_seeds": args.num_seeds,
        "first_seed": args.seed,
        "target_counts": dict(Counter(target for _, target, _, _ in dataset.rows)),
        "same_dataset_as_training": same_dataset,
        "device": str(device),
        "model_half": not args.no_half,
        "save_audio": args.save_audio,
        "silence_threshold": args.silence_threshold,
        "clipping_threshold": args.clipping_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "base": {"generation": base_generation_config},
        "predictor": (
            None
            if checkpoint is None
            else {
                "checkpoint": str(args.predictor_checkpoint.resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_training_loss": checkpoint.get("loss"),
                "steering_mode": checkpoint["steering_mode"],
                "generation": predictor_generation_config,
            }
        ),
        "fixed_alpha": (
            None
            if not args.fixed_alpha_values
            else {
                "values": args.fixed_alpha_values,
                "generation": fixed_alpha_generation_config,
            }
        ),
        "cfg_diff": {
            "alpha_min": args.cfg_diff_alpha_min,
            "alpha_max": args.cfg_diff_alpha_max,
            "alpha_magnitude": args.cfg_diff_alpha_magnitude,
            "alpha_quantile_low": args.cfg_diff_alpha_quantile_low,
            "alpha_quantile_high": args.cfg_diff_alpha_quantile_high,
            "generation": cfg_diff_generation_config,
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Loading {model_name} on {device} with {'float32' if args.no_half else 'half'} precision")
    pipe = SteeringStableAudioPipeline.from_pretrained(model_name, device=device, model_half=not args.no_half)
    pipe.diffusion.requires_grad_(False)
    pipe.diffusion.eval()

    predictor = None
    if checkpoint is not None:
        predictor = SteeringPredictor(**predictor_config).to(device)
        predictor.load_state_dict(checkpoint["state_dict"], strict=True)
        predictor.requires_grad_(False)
        predictor.eval()

    clap = ClapLoss.from_pretrained(reduction="none").to(device)
    sampling_rate = pipe.sample_rate

    # the predictor is conditioned on the CLAP embedding of the target, and the dataset holds far
    # fewer distinct targets than rows, so they are embedded once up front
    target_embeds = {target: clap.encode_text([target]) for target in {row[1] for row in dataset.rows}}

    fixed_alpha_methods: dict[str, tuple[nn.Module | None, str]] = {}
    for alpha in args.fixed_alpha_values:
        name = fixed_alpha_name(alpha)
        if name in fixed_alpha_methods:
            raise ValueError(f"duplicate fixed-alpha method {name}; remove repeated --fixed-alpha-values entries")
        fixed_alpha_methods[name] = (FixedAlphaSteering(alpha).to(device), "learned")

    num_methods_per_sample_seed = (
        1  # base
        + (1 if predictor is not None else 0)
        + len(fixed_alpha_methods)
        + 1  # cfg_diff
    )
    total_generations = len(dataset) * args.num_seeds * num_methods_per_sample_seed
    progress = {"completed": 0, "total": total_generations}

    rows: list[dict] = []
    alpha_runs: list[dict] = []
    results_path = output_dir / "results.csv"

    with results_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        # 1. base: the unsteered reference every other method is compared against in `summarize()`.
        run_pipeline(
            pipe, clap, dataset, target_embeds,
            {"base": (FixedAlphaSteering(0.0).to(device), "learned")},
            base_generation_config,
            INERT_ALPHA_BOUNDS,
            args, device, sampling_rate, writer, csv_file, rows, alpha_runs, progress, output_dir,
        )

        # 2. predictor: skipped when no checkpoint was given.
        if predictor is not None:
            run_pipeline(
                pipe, clap, dataset, target_embeds,
                {"learned": (predictor, "learned")},
                predictor_generation_config,
                INERT_ALPHA_BOUNDS,
                args, device, sampling_rate, writer, csv_file, rows, alpha_runs, progress, output_dir,
            )

        # 3. fixed-alpha: skipped when --fixed-alpha-values is empty.
        if fixed_alpha_methods:
            run_pipeline(
                pipe, clap, dataset, target_embeds,
                fixed_alpha_methods,
                fixed_alpha_generation_config,
                INERT_ALPHA_BOUNDS,
                args, device, sampling_rate, writer, csv_file, rows, alpha_runs, progress, output_dir,
            )

        # 4. cfg-diff: always evaluated, zero-cost and training-free.
        run_pipeline(
            pipe, clap, dataset, target_embeds,
            {"cfg_diff": (None, "cfg_diff")},
            cfg_diff_generation_config,
            (
                args.cfg_diff_alpha_min,
                args.cfg_diff_alpha_max,
                args.cfg_diff_alpha_magnitude,
                args.cfg_diff_alpha_quantile_low,
                args.cfg_diff_alpha_quantile_high,
            ),
            args, device, sampling_rate, writer, csv_file, rows, alpha_runs, progress, output_dir,
        )

    summary = summarize(rows, args.bootstrap_samples, args.seed)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_target_similarity(summary, output_dir / "target_similarity.png")
    plot_tradeoff(summary, output_dir / "suppression_fidelity_tradeoff.png")
    plot_alpha_schedules(alpha_runs, output_dir / "alpha_schedules.png")
    print(f"Done. Evaluation artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
