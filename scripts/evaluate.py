r"""
Evaluates a trained ``SteeringPredictor`` against its paired ``alpha=0`` Stable Audio 3 baseline.

Example smoke test (using a checkpoint produced by ``scripts/train.py``):

    uv run python scripts/evaluate.py \
        --dataset datasets/trumpet_simple_splits/validation.csv \
        --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt \
        --output outputs/eval-smoke --max-samples 4 --num-seeds 1 --num-inference-steps 20

Example final run with fixed-alpha controls:

    uv run python scripts/evaluate.py \
        --dataset datasets/trumpet_simple_splits/test.csv \
        --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt \
        --output outputs/eval-final --num-seeds 5 \
        --fixed-alphas 0.25 0.5 0.75 1.0

Preferred paired run from the baseline selector (uses its target-valid pairs, saved baseline audio,
explicit seeds and generation settings):

    uv run python scripts/evaluate.py \
        --baseline-selection outputs/trumpet-baseline-selection \
        --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt \
        --output outputs/eval-selected
"""

import argparse
import csv
import json
import shutil
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from losses import ClapLoss
from pipelines import FixedAlphaSteering, STEERING_MODE, SteeringPredictor, SteeringStableAudioPipeline
from utils import PromptTargetDataset, load_waveform, save_waveform
from utils.instrument_classification import BaselineSelection, load_baseline_selection


MODEL = "medium-base"
FALLBACK_GENERATION_CONFIG = {
    "num_inference_steps": 50,
    "audio_length_in_s": 10.0,
    "cfg_scale": 7.0,
    "apg_scale": 0.0,
    "steering_frac_start": 0.3,
    "steering_frac_end": 0.8,
}
SAMPLE_FIELDS = [
    "pair_id",
    "sample_id",
    "seed_index",
    "seed",
    "method",
    "prompt",
    "target",
    "retain_prompt",
    "audio_path",
    "baseline_source_audio_path",
    "requested_instruments",
    "retain_instruments",
    "baseline_target_instrument_score",
    "baseline_target_valid",
    "baseline_retain_instrument_scores",
    "baseline_retain_instrument_validity",
    "baseline_valid_retain_instruments",
    "baseline_invalid_retain_instruments",
    "all_retain_baseline_valid",
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
SAMPLE_JSON_FIELDS = {
    "requested_instruments",
    "retain_instruments",
    "baseline_retain_instrument_scores",
    "baseline_retain_instrument_validity",
    "baseline_valid_retain_instruments",
    "baseline_invalid_retain_instruments",
}
PLOT_COLORS = ("#2a78d6", "#d56b25", "#39875b", "#845ec2", "#b64c66", "#6b6b6b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SteeringPredictor with paired Stable Audio 3 generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data and checkpoint")
    data.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="legacy evaluation CSV; mutually exclusive with --baseline-selection",
    )
    data.add_argument(
        "--baseline-selection",
        type=Path,
        default=None,
        help=(
            "output directory (or baseline_records.jsonl) from prepare_baselines_stable_audio_3.py; "
            "evaluates only target-valid pairs with their explicit seeds and saved baselines"
        ),
    )
    data.add_argument("--checkpoint", type=Path, required=True, help="checkpoint produced by scripts/train.py")
    data.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    data.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="evaluate the first N prompts in dataset mode or the first N eligible pairs in selection mode",
    )
    data.add_argument(
        "--replace-output",
        action="store_true",
        help="replace an existing evaluation directory (the default is to fail rather than overwrite)",
    )

    methods = parser.add_argument_group("methods")
    methods.add_argument(
        "--fixed-alphas",
        type=float,
        nargs="*",
        default=[],
        metavar="ALPHA",
        help="optional constant-alpha controls evaluated in addition to base and learned",
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument("--model", default=None, help="Stable Audio 3 `-base` checkpoint, defaults to the one trained against")
    generation.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="independent noises per prompt in dataset mode; seeds come from --baseline-selection otherwise",
    )
    generation.add_argument(
        "--seed",
        type=int,
        default=None,
        help="first seed in dataset mode; explicit record seeds are used with --baseline-selection",
    )
    generation.add_argument("--num-inference-steps", type=int, default=None, help="defaults to checkpoint setting")
    generation.add_argument("--audio-length-in-s", type=float, default=None, help="defaults to checkpoint setting")
    generation.add_argument("--cfg-scale", type=float, default=None, help="defaults to checkpoint setting")
    generation.add_argument("--apg-scale", type=float, default=None, help="defaults to checkpoint setting")
    generation.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="must match the baseline-selection setting in selection mode",
    )
    generation.add_argument(
        "--chunked-decode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="must match the baseline-selection setting; defaults to disabled in legacy dataset mode",
    )
    generation.add_argument("--steering-frac-start", type=float, default=None, help="defaults to checkpoint setting")
    generation.add_argument("--steering-frac-end", type=float, default=None, help="defaults to checkpoint setting")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    runtime.add_argument(
        "--no-half",
        action="store_true",
        help="load the diffusion transformer in float32; the autoencoder and the steering algebra always are",
    )
    runtime.add_argument("--save-audio", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--bootstrap-samples", type=int, default=10_000)
    runtime.add_argument("--silence-threshold", type=float, default=1e-4)
    runtime.add_argument("--clipping-threshold", type=float, default=0.999)

    args = parser.parse_args()
    if (args.dataset is None) == (args.baseline_selection is None):
        parser.error("exactly one of --dataset or --baseline-selection must be provided")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.baseline_selection is not None and (args.num_seeds is not None or args.seed is not None):
        parser.error("--num-seeds and --seed cannot be used with --baseline-selection; its explicit seeds are used")
    if args.dataset is not None:
        args.num_seeds = 1 if args.num_seeds is None else args.num_seeds
        args.seed = 1000 if args.seed is None else args.seed
    if args.num_seeds is not None and args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")
    if args.silence_threshold < 0.0:
        parser.error("--silence-threshold cannot be negative")
    if args.clipping_threshold <= 0.0:
        parser.error("--clipping-threshold must be positive")
    return args


def _same_setting(left, right) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-9))
    return left == right


def resolve_generation_config(
    args: argparse.Namespace,
    checkpoint_args: dict,
    baseline_generation: dict | None = None,
) -> dict[str, float | int | str | bool | None]:
    """Keeps saved-baseline settings fixed; legacy mode retains CLI/checkpoint fallback behaviour."""

    resolved = {}
    paired_names = {"num_inference_steps", "audio_length_in_s", "cfg_scale", "apg_scale"}
    for name, fallback in FALLBACK_GENERATION_CONFIG.items():
        cli_value = getattr(args, name)
        if baseline_generation is not None and name in paired_names:
            if name not in baseline_generation:
                raise ValueError(f"baseline-selection config is missing generation setting {name!r}")
            baseline_value = baseline_generation[name]
            if cli_value is not None and not _same_setting(cli_value, baseline_value):
                raise ValueError(
                    f"--{name.replace('_', '-')}={cli_value} does not match saved baseline value {baseline_value}"
                )
            resolved[name] = baseline_value
        else:
            resolved[name] = cli_value if cli_value is not None else checkpoint_args.get(name, fallback)

    if baseline_generation is not None:
        for name, cli_value, fallback in (
            ("negative_prompt", args.negative_prompt, None),
            ("chunked_decode", args.chunked_decode, False),
        ):
            baseline_value = baseline_generation.get(name, fallback)
            if cli_value is not None and not _same_setting(cli_value, baseline_value):
                raise ValueError(
                    f"--{name.replace('_', '-')}={cli_value!r} does not match saved baseline value"
                    f" {baseline_value!r}"
                )
            resolved[name] = baseline_value
    else:
        resolved["negative_prompt"] = args.negative_prompt
        resolved["chunked_decode"] = False if args.chunked_decode is None else args.chunked_decode

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


def validate_checkpoint_selection_compatibility(checkpoint: dict, selection: BaselineSelection | None) -> None:
    r"""Rejects paired runs outside the generation/target domain recorded by new checkpoints."""
    if selection is None:
        return

    baseline_mode = selection.config.get("baseline_mode")
    if baseline_mode != "paired_alpha0":
        warnings.warn(
            "This selection was not generated with paired-alpha0 baselines. Its saved base uses a different "
            "transformer branch from steered methods, so paired deltas may include a computation-path effect.",
            stacklevel=2,
        )

    checkpoint_generation = checkpoint.get("generation")
    if checkpoint_generation is None:
        return
    baseline_generation = selection.config["generation"]
    for name in (
        "model",
        "model_half",
        "num_inference_steps",
        "audio_length_in_s",
        "cfg_scale",
        "apg_scale",
        "negative_prompt",
        "chunked_decode",
        "steering_frac_start",
        "steering_frac_end",
    ):
        if name not in checkpoint_generation or name not in baseline_generation:
            raise ValueError(f"checkpoint or baseline selection is missing compatibility setting {name!r}")
        if not _same_setting(checkpoint_generation[name], baseline_generation[name]):
            raise ValueError(
                f"baseline selection {name}={baseline_generation[name]!r} does not match checkpoint training "
                f"value {checkpoint_generation[name]!r}"
            )

    trained_targets = set(checkpoint.get("training_targets", checkpoint.get("targets", [])))
    evaluation_targets = {str(pair.record["target"]) for pair in selection.pairs}
    unseen_targets = sorted(evaluation_targets - trained_targets)
    if trained_targets and unseen_targets:
        raise ValueError(f"baseline selection contains targets unseen by the checkpoint: {unseen_targets}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return device


def prepare_output_directory(path: Path, replace: bool) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise FileExistsError(
                f"Evaluation output {path} is not empty. Choose another --output or explicitly pass --replace-output."
            )
        # The user explicitly named this evaluation directory and opted into replacement. Refuse broad targets.
        if path == Path.cwd().resolve() or path == path.parent or len(path.parts) < 3:
            raise ValueError(f"Refusing to replace unsafe output directory {path}")
        shutil.rmtree(path)
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


def sample_row_to_csv(row: dict) -> dict:
    return {
        field: json.dumps(row[field], sort_keys=True) if field in SAMPLE_JSON_FIELDS else row[field]
        for field in SAMPLE_FIELDS
    }


def baseline_metadata(record: dict | None, source_audio_path: Path | None = None) -> dict:
    if record is None:
        return {
            "baseline_source_audio_path": "",
            "requested_instruments": [],
            "retain_instruments": [],
            "baseline_target_instrument_score": None,
            "baseline_target_valid": None,
            "baseline_retain_instrument_scores": {},
            "baseline_retain_instrument_validity": {},
            "baseline_valid_retain_instruments": [],
            "baseline_invalid_retain_instruments": [],
            "all_retain_baseline_valid": None,
        }

    validity = {name: bool(value) for name, value in record["retain_instrument_validity"].items()}
    return {
        "baseline_source_audio_path": "" if source_audio_path is None else str(source_audio_path),
        "requested_instruments": list(record["requested_instruments"]),
        "retain_instruments": list(record["retain_instruments"]),
        "baseline_target_instrument_score": float(record["target_score"]),
        "baseline_target_valid": bool(record["target_valid"]),
        "baseline_retain_instrument_scores": {
            name: float(value) for name, value in record["retain_instrument_scores"].items()
        },
        "baseline_retain_instrument_validity": validity,
        "baseline_valid_retain_instruments": [name for name, valid in validity.items() if valid],
        "baseline_invalid_retain_instruments": [name for name, valid in validity.items() if not valid],
        "all_retain_baseline_valid": record["all_retain_valid"],
    }


def build_evaluation_pairs(
    dataset: PromptTargetDataset | None,
    selection: BaselineSelection | None,
    first_seed: int | None,
    num_seeds: int | None,
) -> list[dict]:
    if selection is not None:
        return [
            {
                "pair_id": pair.pair_id,
                "sample_id": int(pair.record["sample_id"]),
                "seed_index": int(pair.record["seed_index"]),
                "seed": int(pair.record["seed"]),
                "prompt": str(pair.record["prompt"]),
                "target": str(pair.record["target"]),
                "retain_prompt": str(pair.record["retain_prompt"]),
                "baseline_record": pair.record,
                "baseline_audio_path": pair.audio_path,
            }
            for pair in selection.pairs
        ]

    if dataset is None or first_seed is None or num_seeds is None:
        raise ValueError("dataset mode requires a dataset, first_seed and num_seeds")
    pairs = []
    for sample_id, (prompt, target, retain_prompt) in enumerate(dataset.rows):
        for seed_index in range(num_seeds):
            seed = first_seed + seed_index * len(dataset) + sample_id
            pairs.append(
                {
                    "pair_id": f"sample_{sample_id:04d}_seed_{seed}",
                    "sample_id": sample_id,
                    "seed_index": seed_index,
                    "seed": seed,
                    "prompt": prompt,
                    "target": target,
                    "retain_prompt": retain_prompt,
                    "baseline_record": None,
                    "baseline_audio_path": None,
                }
            )
    return pairs


def score_waveform_with_clap(
    waveform: np.ndarray,
    sampling_rate: int,
    clap: ClapLoss,
    device: torch.device,
    prompt: str,
    target: str,
    retain_prompt: str,
) -> tuple[float, float, float]:
    waveform_tensor = torch.from_numpy(np.asarray(waveform)).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        audio_embeds = clap.encode_audio(waveform_tensor.mean(dim=1), sampling_rate)
        target_similarity = 1.0 - float(clap(audio_embeds, target)[0])
        prompt_similarity = 1.0 - float(clap(audio_embeds, prompt)[0])
        retain_similarity = 1.0 - float(clap(audio_embeds, retain_prompt)[0])
    return target_similarity, prompt_similarity, retain_similarity


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
            all_retain_valid = [row for row in paired if row.get("all_retain_baseline_valid") is True]
            if all_retain_valid:
                method_summary["retain_similarity_change_all_retain_baseline_valid"] = _mean_ci_by_prompt(
                    all_retain_valid, lambda row: row["retain_change"], bootstrap_samples, rng
                )
                method_summary["num_pairs_all_retain_baseline_valid"] = len(all_retain_valid)

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


def write_report(path: Path, config: dict, summary: dict) -> None:
    seeds_text = "record-specific" if config["num_seeds"] is None else str(config["num_seeds"])
    lines = [
        "# Steering evaluation report",
        "",
        f"- Evaluation prompts: {config['num_prompts']}",
        f"- Prompt/seed pairs: {config['num_pairs']}",
        f"- Seeds per prompt: {seeds_text}",
        f"- Seed source: `{config['seed_source']}`",
        f"- Target counts: `{json.dumps(config['target_counts'], sort_keys=True)}`",
        f"- Denoising steps: {config['generation']['num_inference_steps']}",
        f"- Steering window: [{config['generation']['steering_frac_start']}, "
        f"{config['generation']['steering_frac_end']})",
        f"- Checkpoint selection metric: `{config['checkpoint_selection_metric'] or 'legacy/unknown'}`",
        f"- Checkpoint selection value: `{config['checkpoint_selection_value']}`",
        "",
    ]
    if config["input_mode"] == "baseline_selection":
        lines.extend(
            [
                f"> Loaded only target-valid pairs from `{config['baseline_records']}`. The `base` rows reuse the"
                " exact saved WAV files; steered methods recreate each record's explicit seed and generation"
                " settings.",
                "",
                f"- Baseline computation mode: `{config['baseline_mode']}`",
                f"- Pairs whose complete retain set was present in the baseline: "
                f"{config['num_pairs_all_retain_baseline_valid']}/{config['num_pairs']}",
                f"- Pairs with at least one absent retain instrument: "
                f"{config['num_pairs_with_invalid_retain_baseline']}",
                "",
            ]
        )
    if config["dataset_overlap_role"] is not None:
        lines.extend(
            [
                f"> **Data leakage warning:** the evaluation dataset is the checkpoint's "
                f"`{config['dataset_overlap_role']}` split. These results are diagnostic and must not be "
                "reported as held-out test performance.",
                "",
            ]
        )

    lines.extend(
        [
            "## Summary",
            "",
            "| Method | Target similarity | Suppression gain | Retain similarity | Retain-similarity change |"
            " Silence ratio | Clipping ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, values in summary["methods"].items():
        target = values["target_similarity"]["mean"]
        gain = values.get("target_suppression_gain", {}).get("mean")
        retain = values["retain_similarity"]["mean"]
        retain_change = values.get("retain_similarity_change", {}).get("mean")
        lines.append(
            f"| {method} | {target:.4f} | "
            f"{'—' if gain is None else f'{gain:+.4f}'} | "
            f"{retain:.4f} | "
            f"{'—' if retain_change is None else f'{retain_change:+.4f}'} | "
            f"{values['silence_ratio_mean']:.4f} | {values['clipping_ratio_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower target similarity is better.",
            "- Positive suppression gain means the method is less similar to the target than the paired base audio.",
            "- The retain prompt comes from the dataset when an explicit `retain_prompt` column is present; legacy "
            "two-column datasets fall back to `utils.strip_target`. It is the fidelity measure optimized by "
            "`scripts/train.py` when `--retain-weight` is non-zero, and each row's text is in `sample_metrics.csv`.",
            "- Positive retain-similarity change means the method kept more of the rest of the prompt than the paired "
            "base audio; a large suppression gain paired with a negative retain change usually means degraded audio "
            "rather than a removed concept.",
            "- In baseline-selection mode, `baseline_retain_instrument_validity` in `sample_metrics.csv` says which "
            "retain instruments were actually detected before steering. An absent retain instrument must not be "
            "used to claim collateral preservation or damage; the global CLAP retain score remains a prompt-level "
            "diagnostic. `summary.json` also reports `retain_similarity_change_all_retain_baseline_valid`, restricted "
            "to pairs whose complete retain set was detected in the baseline.",
            "- Only the legacy fallback removal is lexical: modifiers of the target can survive it "
            "(\"muted trumpet with a plunger mute\" becomes \"muted with a plunger mute\"). Explicit retain "
            "prompts avoid this wording artifact.",
            "- `prompt_similarity` in `sample_metrics.csv` and `summary.json` scores the full prompt, which still "
            "contains the target, so it is only a coarse fidelity proxy and partially conflicts with suppression.",
            "- Silence and clipping ratios are sanity checks, not complete perceptual-quality measures.",
            "- Confidence intervals in `summary.json` are clustered by prompt: seeds are averaged first, then prompts "
            "are bootstrapped.",
            "",
            "Listen to paired files with the same sample ID and seed before drawing a final conclusion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_overlap_role(eval_path: Path, checkpoint_args: dict, checkpoint_data: dict | None) -> str | None:
    r"""Returns the checkpoint split whose source dataset is being reused, if any."""
    candidates: list[tuple[str, str | None]] = [("training", checkpoint_args.get("dataset"))]
    if checkpoint_data and checkpoint_data.get("mode") == "target_valid_baseline_selections":
        candidates.extend(
            (role, checkpoint_data.get(role, {}).get("dataset"))
            for role in ("training", "validation")
        )

    try:
        resolved_eval = eval_path.resolve()
        for role, candidate in candidates:
            if candidate is not None and resolved_eval == Path(candidate).resolve():
                return role
    except OSError:
        return None
    return None


def main() -> None:
    args = parse_args()

    selection = (
        None
        if args.baseline_selection is None
        else load_baseline_selection(args.baseline_selection, max_pairs=args.max_samples)
    )
    dataset = None if args.dataset is None else PromptTargetDataset(args.dataset, args.max_samples)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required_keys = {"state_dict", "config"}
    missing = required_keys - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint {args.checkpoint} is missing keys {sorted(missing)}")
    if "steering_mode" not in checkpoint:
        raise ValueError(
            f"checkpoint {args.checkpoint} predates target-specific full-to-retain steering. Its alpha values used"
            " the incompatible global-CFG formula, so a new checkpoint must be trained."
        )
    if checkpoint["steering_mode"] != STEERING_MODE:
        raise ValueError(
            f"checkpoint {args.checkpoint} uses steering mode {checkpoint['steering_mode']!r}, but this evaluator"
            f" requires {STEERING_MODE!r}. Checkpoints trained against MusicLDM cannot be evaluated on Stable Audio 3:"
            " the two backbones do not share a latent space."
        )
    validate_checkpoint_selection_compatibility(checkpoint, selection)
    checkpoint_args = checkpoint.get("args", {})
    baseline_generation = None if selection is None else selection.config["generation"]
    generation_config = resolve_generation_config(args, checkpoint_args, baseline_generation)
    if selection is not None and checkpoint.get("generation") is not None:
        for name in ("steering_frac_start", "steering_frac_end"):
            trained_value = checkpoint["generation"][name]
            if not _same_setting(generation_config[name], trained_value):
                raise ValueError(
                    f"evaluation {name}={generation_config[name]!r} does not match checkpoint training value "
                    f"{trained_value!r}"
                )
    predictor_config = checkpoint["config"]

    if selection is not None:
        if "model" not in baseline_generation:
            raise ValueError("baseline-selection config is missing its Stable Audio 3 model")
        baseline_model = str(baseline_generation["model"])
        if args.model is not None and args.model != baseline_model:
            raise ValueError(
                f"--model={args.model!r} does not match the saved baseline model {baseline_model!r}"
            )
        model_name = baseline_model
        model_half = bool(baseline_generation.get("model_half", True))
        if args.no_half and model_half:
            raise ValueError("--no-half does not match the half-precision transformer used for the saved baselines")
    else:
        model_name = args.model or checkpoint.get("model") or MODEL
        model_half = not args.no_half

    alpha_min = float(predictor_config.get("alpha_min", 0.0))
    alpha_max = float(predictor_config.get("alpha_max", 1.0))
    for alpha in args.fixed_alphas:
        if not alpha_min <= alpha <= alpha_max:
            raise ValueError(
                f"fixed alpha {alpha} lies outside the predictor's configured range [{alpha_min}, {alpha_max}]"
            )

    evaluation_pairs = build_evaluation_pairs(dataset, selection, args.seed, args.num_seeds)
    device = resolve_device(args.device)
    dataset_path = Path(selection.config["dataset"]) if selection is not None else args.dataset
    dataset_overlap_role = _dataset_overlap_role(dataset_path, checkpoint_args, checkpoint.get("data"))
    if dataset_overlap_role is not None:
        warnings.warn(
            f"The evaluation CSV is the checkpoint's {dataset_overlap_role} split. Results are diagnostic, not "
            "held-out test performance.",
            stacklevel=2,
        )

    output_dir = prepare_output_directory(args.output, args.replace_output)
    prompt_targets = {}
    seed_counts: Counter[int] = Counter()
    for pair in evaluation_pairs:
        prompt_targets.setdefault((int(pair["sample_id"]), pair["prompt"]), pair["target"])
        seed_counts[int(pair["sample_id"])] += 1
    seed_count_values = set(seed_counts.values())
    uniform_num_seeds = next(iter(seed_count_values)) if len(seed_count_values) == 1 else None
    summary_seed = int(
        args.seed if args.seed is not None else selection.config.get("first_seed", evaluation_pairs[0]["seed"])
    )
    selection_records = [pair["baseline_record"] for pair in evaluation_pairs if pair["baseline_record"] is not None]

    config = {
        "input_mode": "baseline_selection" if selection is not None else "dataset",
        "dataset": str(dataset_path.resolve()),
        "baseline_selection": None if selection is None else str(selection.root),
        "baseline_records": None if selection is None else str(selection.records_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(output_dir),
        "model": model_name,
        "num_prompts": len(prompt_targets),
        "num_pairs": len(evaluation_pairs),
        "num_seeds": uniform_num_seeds,
        "first_seed": None if selection is not None else args.seed,
        "seed_source": "baseline_selection_records" if selection is not None else "evaluator_formula",
        "target_counts": dict(Counter(prompt_targets.values())),
        "same_dataset_as_training": dataset_overlap_role == "training",
        "same_dataset_as_validation": dataset_overlap_role == "validation",
        "dataset_overlap_role": dataset_overlap_role,
        "device": str(device),
        "model_half": model_half,
        "fixed_alphas": args.fixed_alphas,
        "save_audio": args.save_audio,
        "silence_threshold": args.silence_threshold,
        "clipping_threshold": args.clipping_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "generation": generation_config,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_training_loss": checkpoint.get("loss"),
        "checkpoint_validation_loss": (
            None if checkpoint.get("validation_metrics") is None else checkpoint["validation_metrics"].get("loss")
        ),
        "checkpoint_selection_metric": checkpoint.get("checkpoint_selection_metric"),
        "checkpoint_selection_value": checkpoint.get("checkpoint_selection_value"),
        "checkpoint_data": checkpoint.get("data"),
        "steering_mode": checkpoint["steering_mode"],
        "baseline_mode": None if selection is None else selection.config.get("baseline_mode", "stock_legacy"),
        "baseline_classifier": None if selection is None else selection.config["classifier"],
        "num_pairs_all_retain_baseline_valid": sum(
            record["all_retain_valid"] is True for record in selection_records
        ),
        "num_pairs_with_invalid_retain_baseline": sum(
            any(not bool(value) for value in record["retain_instrument_validity"].values())
            for record in selection_records
        ),
        "bootstrap_seed": summary_seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if selection is not None:
        print(f"Loaded {len(evaluation_pairs)} target-valid prompt/seed pairs from {selection.records_path}")
    print(f"Loading {model_name} on {device} with {'half' if model_half else 'float32'} precision")
    pipe = SteeringStableAudioPipeline.from_pretrained(model_name, device=device, model_half=model_half)
    pipe.diffusion.requires_grad_(False)
    pipe.diffusion.eval()

    predictor = SteeringPredictor(**predictor_config).to(device)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.requires_grad_(False)
    predictor.eval()

    methods: dict[str, nn.Module] = {"learned": predictor}
    if selection is None:
        methods = {"base": FixedAlphaSteering(0.0).to(device), **methods}
    for alpha in args.fixed_alphas:
        name = fixed_alpha_name(alpha)
        if name in methods:
            raise ValueError(f"duplicate evaluation method {name}; remove repeated fixed alphas")
        methods[name] = FixedAlphaSteering(alpha).to(device)

    clap = ClapLoss.from_pretrained(reduction="none").to(device)
    sampling_rate = pipe.sample_rate

    # the predictor is conditioned on the CLAP embedding of the target, and the dataset holds far
    # fewer distinct targets than rows, so they are embedded once up front
    target_embeds = {target: clap.encode_text([target]) for target in {pair["target"] for pair in evaluation_pairs}}

    rows: list[dict] = []
    alpha_runs: list[dict] = []
    samples_path = output_dir / "sample_metrics.csv"
    alpha_path = output_dir / "alpha_records.jsonl"
    total_evaluations = len(evaluation_pairs) * (len(methods) + (1 if selection is not None else 0))
    completed = 0

    with samples_path.open("w", newline="", encoding="utf-8") as csv_file, alpha_path.open(
        "w", encoding="utf-8"
    ) as alpha_file:
        writer = csv.DictWriter(csv_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()

        # Batch size is intentionally one. In selection mode the base row is scored from the exact
        # saved WAV, while every steered method recreates a CPU generator with that record's seed.
        for pair in evaluation_pairs:
            pair_id = pair["pair_id"]
            sample_id = int(pair["sample_id"])
            seed_index = int(pair["seed_index"])
            sample_seed = int(pair["seed"])
            prompt = pair["prompt"]
            target = pair["target"]
            retain_prompt = pair["retain_prompt"]
            metadata = baseline_metadata(pair["baseline_record"], pair["baseline_audio_path"])

            if selection is not None:
                waveform, baseline_rate = load_waveform(pair["baseline_audio_path"])
                if baseline_rate != sampling_rate:
                    raise ValueError(
                        f"saved baseline {pair['baseline_audio_path']} has sample rate {baseline_rate},"
                        f" expected {sampling_rate}"
                    )
                target_similarity, prompt_similarity, retain_similarity = score_waveform_with_clap(
                    waveform, sampling_rate, clap, device, prompt, target, retain_prompt
                )
                signal = waveform_metrics(waveform, args.silence_threshold, args.clipping_threshold)
                records = []
                audio_relative_path = ""
                if args.save_audio:
                    audio_relative_path = str(
                        Path("audio") / "base" / f"sample_{sample_id:04d}_seed_{sample_seed}.wav"
                    )
                    copied_path = output_dir / audio_relative_path
                    copied_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pair["baseline_audio_path"], copied_path)

                row = {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "seed_index": seed_index,
                    "seed": sample_seed,
                    "method": "base",
                    "prompt": prompt,
                    "target": target,
                    "retain_prompt": retain_prompt,
                    "audio_path": audio_relative_path,
                    **metadata,
                    "target_similarity": target_similarity,
                    "prompt_similarity": prompt_similarity,
                    "retain_similarity": retain_similarity,
                    **signal,
                    **alpha_metrics(records),
                }
                rows.append(row)
                writer.writerow(sample_row_to_csv(row))
                alpha_run = {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "seed_index": seed_index,
                    "seed": sample_seed,
                    "method": "base",
                    "records": records,
                }
                alpha_runs.append(alpha_run)
                alpha_file.write(json.dumps(alpha_run) + "\n")
                completed += 1
                print(
                    f"[{completed}/{total_evaluations}] pair={pair_id} method=base(saved) "
                    f"target_cos={target_similarity:+.4f} retain_cos={retain_similarity:+.4f} "
                    f"prompt_cos={prompt_similarity:+.4f}",
                    flush=True,
                )

            for method_name, steering_model in methods.items():
                generator = torch.Generator().manual_seed(sample_seed)
                with torch.inference_mode():
                    output = pipe(
                        prompt=prompt,
                        retain_prompt=retain_prompt,
                        target_embed=target_embeds[target],
                        steering_model=steering_model,
                        steering_frac_start=generation_config["steering_frac_start"],
                        steering_frac_end=generation_config["steering_frac_end"],
                        train=False,
                        num_inference_steps=generation_config["num_inference_steps"],
                        audio_length_in_s=generation_config["audio_length_in_s"],
                        cfg_scale=generation_config["cfg_scale"],
                        negative_prompt=generation_config["negative_prompt"],
                        apg_scale=generation_config["apg_scale"],
                        generator=generator,
                        chunked_decode=generation_config["chunked_decode"],
                        output_type="pt",
                    )
                waveform = output.audios[0].detach().float().cpu().numpy()
                target_similarity, prompt_similarity, retain_similarity = score_waveform_with_clap(
                    waveform, sampling_rate, clap, device, prompt, target, retain_prompt
                )
                signal = waveform_metrics(waveform, args.silence_threshold, args.clipping_threshold)
                records = [(int(step), float(alpha)) for step, alpha in output.alpha_records]

                audio_relative_path = ""
                if args.save_audio:
                    audio_relative_path = str(
                        Path("audio") / method_name / f"sample_{sample_id:04d}_seed_{sample_seed}.wav"
                    )
                    save_waveform(output_dir / audio_relative_path, waveform, sampling_rate)

                row = {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "seed_index": seed_index,
                    "seed": sample_seed,
                    "method": method_name,
                    "prompt": prompt,
                    "target": target,
                    "retain_prompt": retain_prompt,
                    "audio_path": audio_relative_path,
                    **metadata,
                    "target_similarity": target_similarity,
                    "prompt_similarity": prompt_similarity,
                    "retain_similarity": retain_similarity,
                    **signal,
                    **alpha_metrics(records),
                }
                rows.append(row)
                writer.writerow(sample_row_to_csv(row))
                csv_file.flush()

                alpha_run = {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "seed_index": seed_index,
                    "seed": sample_seed,
                    "method": method_name,
                    "records": records,
                }
                alpha_runs.append(alpha_run)
                alpha_file.write(json.dumps(alpha_run) + "\n")
                alpha_file.flush()

                completed += 1
                print(
                    f"[{completed}/{total_evaluations}] pair={pair_id} method={method_name} "
                    f"target_cos={target_similarity:+.4f} retain_cos={retain_similarity:+.4f} "
                    f"prompt_cos={prompt_similarity:+.4f}",
                    flush=True,
                )

    summary = summarize(rows, args.bootstrap_samples, summary_seed)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_target_similarity(summary, output_dir / "target_similarity.png")
    plot_tradeoff(summary, output_dir / "suppression_fidelity_tradeoff.png")
    plot_alpha_schedules(alpha_runs, output_dir / "alpha_schedules.png")
    write_report(output_dir / "report.md", config, summary)
    print(f"Done. Evaluation artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
