r"""
Evaluates a trained ``SteeringPredictor`` against the unsteered Stable Audio 3 baseline.

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
from pipelines import STEERING_MODE, SteeringPredictor, SteeringStableAudioPipeline, FixedAlphaSteering
from utils import PromptTargetDataset, save_waveform


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
    "sample_id",
    "seed_index",
    "seed",
    "method",
    "prompt",
    "target",
    "retain_prompt",
    "audio_path",
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SteeringPredictor with paired Stable Audio 3 generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data and checkpoint")
    data.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="evaluation CSV with prompt,target and preferably an explicit retain_prompt column",
    )
    data.add_argument("--checkpoint", type=Path, required=True, help="checkpoint produced by scripts/train.py")
    data.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    data.add_argument("--max-samples", type=int, default=None, help="evaluate only the first N prompts")
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

    cfg_diff = parser.add_argument_group(
        "cfg-diff regime",
        description=(
            "Regime A: a zero-cost, training-free 'cfg_diff' method is always evaluated alongside base and"
            " learned, deriving a deterministic per-frame alpha_t from the CFG-diff norm instead of calling the"
            " predictor. It reuses alpha_min/alpha_max from the checkpoint's predictor config."
        ),
    )
    cfg_diff.add_argument(
        "--alpha-magnitude",
        type=float,
        default=0.6,
        help="global gain in [0, 1] for the cfg_diff method's alpha_t",
    )
    cfg_diff.add_argument(
        "--alpha-shape-quantile-low",
        type=float,
        default=0.10,
        help="low percentile used by the cfg_diff method to normalize its temporal profile per sample",
    )
    cfg_diff.add_argument(
        "--alpha-shape-quantile-high",
        type=float,
        default=0.90,
        help="high percentile used by the cfg_diff method to normalize its temporal profile per sample",
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument("--model", default=None, help="Stable Audio 3 `-base` checkpoint, defaults to the one trained against")
    generation.add_argument("--num-seeds", type=int, default=1, help="independent noises generated per prompt")
    generation.add_argument("--seed", type=int, default=1000, help="first evaluation seed")
    generation.add_argument("--num-inference-steps", type=int, default=None, help="defaults to checkpoint setting")
    generation.add_argument("--audio-length-in-s", type=float, default=None, help="defaults to checkpoint setting")
    generation.add_argument("--cfg-scale", type=float, default=None, help="defaults to checkpoint setting")
    generation.add_argument("--apg-scale", type=float, default=None, help="defaults to checkpoint setting")
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
    if not 0.0 <= args.alpha_magnitude <= 1.0:
        parser.error("--alpha-magnitude must be in [0, 1]")
    if not 0.0 <= args.alpha_shape_quantile_low < args.alpha_shape_quantile_high <= 1.0:
        parser.error(
            "--alpha-shape-quantile-low and --alpha-shape-quantile-high must satisfy"
            " 0.0 <= low < high <= 1.0"
        )
    return args


def resolve_generation_config(args: argparse.Namespace, checkpoint_args: dict) -> dict[str, float | int]:
    """CLI overrides checkpoint training settings; hard-coded defaults are the last fallback."""

    resolved = {}
    for name, fallback in FALLBACK_GENERATION_CONFIG.items():
        cli_value = getattr(args, name)
        resolved[name] = cli_value if cli_value is not None else checkpoint_args.get(name, fallback)

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


def write_report(path: Path, config: dict, summary: dict) -> None:
    lines = [
        "# Steering evaluation report",
        "",
        f"- Evaluation prompts: {config['num_prompts']}",
        f"- Seeds per prompt: {config['num_seeds']}",
        f"- Target counts: `{json.dumps(config['target_counts'], sort_keys=True)}`",
        f"- Denoising steps: {config['generation']['num_inference_steps']}",
        f"- Steering window: [{config['generation']['steering_frac_start']}, "
        f"{config['generation']['steering_frac_end']})",
        "",
    ]
    if config["same_dataset_as_training"]:
        lines.extend(
            [
                "> **Data leakage warning:** the evaluation dataset path is the same path stored in the training "
                "checkpoint. These results are diagnostic and must not be reported as held-out performance.",
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


def _same_dataset_as_training(eval_path: Path, checkpoint_args: dict) -> bool:
    training_path = checkpoint_args.get("dataset")
    if training_path is None:
        return False
    try:
        return eval_path.resolve() == Path(training_path).resolve()
    except OSError:
        return False


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_directory(args.output, args.replace_output)

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
    checkpoint_args = checkpoint.get("args", {})
    generation_config = resolve_generation_config(args, checkpoint_args)
    predictor_config = checkpoint["config"]

    alpha_min = float(predictor_config.get("alpha_min", 0.0))
    alpha_max = float(predictor_config.get("alpha_max", 1.0))
    for alpha in args.fixed_alphas:
        if not alpha_min <= alpha <= alpha_max:
            raise ValueError(
                f"fixed alpha {alpha} lies outside the predictor's configured range [{alpha_min}, {alpha_max}]"
            )

    dataset = PromptTargetDataset(args.dataset, args.max_samples)
    device = resolve_device(args.device)
    model_name = args.model or checkpoint.get("model") or MODEL
    same_dataset = _same_dataset_as_training(args.dataset, checkpoint_args)
    if same_dataset:
        warnings.warn(
            "The evaluation CSV is the same dataset path stored in the checkpoint. Results measure training-set "
            "behaviour, not held-out generalization.",
            stacklevel=2,
        )

    config = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(output_dir),
        "model": model_name,
        "num_prompts": len(dataset),
        "num_seeds": args.num_seeds,
        "first_seed": args.seed,
        "target_counts": dict(Counter(target for _, target, _ in dataset.rows)),
        "same_dataset_as_training": same_dataset,
        "device": str(device),
        "model_half": not args.no_half,
        "fixed_alphas": args.fixed_alphas,
        "cfg_diff_alpha_min": alpha_min,
        "cfg_diff_alpha_max": alpha_max,
        "alpha_magnitude": args.alpha_magnitude,
        "alpha_shape_quantile_low": args.alpha_shape_quantile_low,
        "alpha_shape_quantile_high": args.alpha_shape_quantile_high,
        "save_audio": args.save_audio,
        "silence_threshold": args.silence_threshold,
        "clipping_threshold": args.clipping_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "generation": generation_config,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_training_loss": checkpoint.get("loss"),
        "steering_mode": checkpoint["steering_mode"],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Loading {model_name} on {device} with {'float32' if args.no_half else 'half'} precision")
    pipe = SteeringStableAudioPipeline.from_pretrained(model_name, device=device, model_half=not args.no_half)
    pipe.diffusion.requires_grad_(False)
    pipe.diffusion.eval()

    predictor = SteeringPredictor(**predictor_config).to(device)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.requires_grad_(False)
    predictor.eval()

    # `cfg_diff` (Regime A) has no `nn.Module`: it derives `alpha_t` deterministically from the
    # CFG-diff instead of calling a steering model, so its entry is `None` and the generation loop
    # branches on `steering_mode` rather than on `steering_model` for it.
    methods: dict[str, nn.Module | None] = {
        "base": FixedAlphaSteering(0.0).to(device),
        "learned": predictor,
        "cfg_diff": None,
    }
    for alpha in args.fixed_alphas:
        name = fixed_alpha_name(alpha)
        if name in methods:
            raise ValueError(f"duplicate evaluation method {name}; remove repeated fixed alphas")
        methods[name] = FixedAlphaSteering(alpha).to(device)

    clap = ClapLoss.from_pretrained(reduction="none").to(device)
    sampling_rate = pipe.sample_rate

    # the predictor is conditioned on the CLAP embedding of the target, and the dataset holds far
    # fewer distinct targets than rows, so they are embedded once up front
    target_embeds = {target: clap.encode_text([target]) for target in {row[1] for row in dataset.rows}}

    rows: list[dict] = []
    alpha_runs: list[dict] = []
    samples_path = output_dir / "sample_metrics.csv"
    alpha_path = output_dir / "alpha_records.jsonl"
    total_generations = len(dataset) * args.num_seeds * len(methods)
    completed = 0

    with samples_path.open("w", newline="", encoding="utf-8") as csv_file, alpha_path.open(
        "w", encoding="utf-8"
    ) as alpha_file:
        writer = csv.DictWriter(csv_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()

        # Batch size is intentionally one. The current pipeline logs alpha averaged over the batch;
        # evaluating one prompt at a time keeps every saved schedule attributable to one prompt.
        for sample_id, (prompt, target, retain_prompt) in enumerate(dataset.rows):
            for seed_index in range(args.num_seeds):
                sample_seed = args.seed + seed_index * len(dataset) + sample_id
                for method_name, steering_model in methods.items():
                    # Recreate the generator for every method so paired runs receive identical noise.
                    generator = torch.Generator().manual_seed(sample_seed)
                    steering_mode = "cfg_diff" if method_name == "cfg_diff" else "learned"
                    with torch.inference_mode():
                        output = pipe(
                            prompt=prompt,
                            retain_prompt=retain_prompt,
                            target_embed=target_embeds[target],
                            steering_model=steering_model,
                            steering_mode=steering_mode,
                            alpha_min=alpha_min,
                            alpha_max=alpha_max,
                            alpha_magnitude=args.alpha_magnitude,
                            alpha_shape_quantile_low=args.alpha_shape_quantile_low,
                            alpha_shape_quantile_high=args.alpha_shape_quantile_high,
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

                    audio_relative_path = ""
                    if args.save_audio:
                        audio_relative_path = str(
                            Path("audio") / method_name / f"sample_{sample_id:04d}_seed_{sample_seed}.wav"
                        )
                        save_waveform(output_dir / audio_relative_path, waveform, sampling_rate)

                    row = {
                        "sample_id": sample_id,
                        "seed_index": seed_index,
                        "seed": sample_seed,
                        "method": method_name,
                        "prompt": prompt,
                        "target": target,
                        "retain_prompt": retain_prompt,
                        "audio_path": audio_relative_path,
                        "target_similarity": target_similarity,
                        "prompt_similarity": prompt_similarity,
                        "retain_similarity": retain_similarity,
                        **signal,
                        **alpha_stats,
                    }
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()

                    alpha_run = {
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
                        f"[{completed}/{total_generations}] sample={sample_id} seed={sample_seed} method={method_name} "
                        f"target_cos={target_similarity:+.4f} retain_cos={retain_similarity:+.4f} "
                        f"prompt_cos={prompt_similarity:+.4f}",
                        flush=True,
                    )

    summary = summarize(rows, args.bootstrap_samples, args.seed)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_target_similarity(summary, output_dir / "target_similarity.png")
    plot_tradeoff(summary, output_dir / "suppression_fidelity_tradeoff.png")
    plot_alpha_schedules(alpha_runs, output_dir / "alpha_schedules.png")
    write_report(output_dir / "report.md", config, summary)
    print(f"Done. Evaluation artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
