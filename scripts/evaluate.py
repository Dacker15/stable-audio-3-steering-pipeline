"""Paired multi-seed evaluation for ACE-Step target-specific steering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from losses import DEFAULT_CLAP_MAX_LENGTH_S, DEFAULT_CLAP_MODEL_ID, DEFAULT_CLAP_REVISION, ClapLoss
from pipelines import (
    ACE_STEP_COMPONENTS_ID,
    ACE_STEP_COMPONENTS_REVISION,
    ACE_STEP_MODEL_ID,
    ACE_STEP_MODEL_REVISION,
    ACE_STEP_SAMPLE_RATE,
    GRADIENT_MODE,
    STEERING_MODE,
    AceStepConditioning,
    SteeringAceStepPipeline,
    SteeringPredictor,
)
from utils import PromptTargetDataset, load_checkpoint, save_waveform


class FixedAlpha(nn.Module):
    """Evaluation-only controller returning one alpha for the whole trajectory."""

    def __init__(self, alpha: float):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, latents: torch.Tensor, **_: object) -> torch.Tensor:
        return self.alpha.expand(latents.shape[0], 1, 1)


def _add_runtime_switches(parser: argparse.ArgumentParser) -> None:
    dit = parser.add_mutually_exclusive_group()
    dit.add_argument("--sequential-dit", dest="sequential_dit", action="store_true")
    dit.add_argument("--batched-dit", dest="sequential_dit", action="store_false")
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument(
        "--offload-dit-after-generation",
        dest="offload_dit_after_generation",
        action="store_true",
        help="offload after every trajectory (lower residency, much slower paired evaluation)",
    )
    offload.add_argument(
        "--keep-dit-on-device",
        dest="offload_dit_after_generation",
        action="store_false",
        help="keep DiT for the trajectory group, then offload once before VAE/CLAP scoring",
    )
    parser.set_defaults(sequential_dit=True, offload_dit_after_generation=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an ACE-Step steering checkpoint with paired noise seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-alphas", type=float, nargs="*", default=(0.25, 0.5, 0.75))

    parser.add_argument("--model-id", default=ACE_STEP_MODEL_ID)
    parser.add_argument("--model-revision", default=ACE_STEP_MODEL_REVISION)
    parser.add_argument("--components-id", default=ACE_STEP_COMPONENTS_ID)
    parser.add_argument("--components-revision", default=ACE_STEP_COMPONENTS_REVISION)
    parser.add_argument("--clap-model-id", default=DEFAULT_CLAP_MODEL_ID)
    parser.add_argument("--clap-revision", default=DEFAULT_CLAP_REVISION)

    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--audio-length-in-s",
        type=float,
        default=DEFAULT_CLAP_MAX_LENGTH_S,
        help="clip duration; capped at the frozen CLAP tower's complete input window",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--guidance-mode", choices=("apg",), default="apg")
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--steering-frac-start", type=float, default=0.3)
    parser.add_argument("--steering-frac-end", type=float, default=0.8)
    parser.add_argument("--thinking", action="store_true", help="unsupported invariant guard")
    parser.add_argument("--dcw-enabled", action="store_true", help="unsupported invariant guard")

    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    _add_runtime_switches(parser)
    args = parser.parse_args(argv)

    exact_artifacts = {
        "model_id": ACE_STEP_MODEL_ID,
        "model_revision": ACE_STEP_MODEL_REVISION,
        "components_id": ACE_STEP_COMPONENTS_ID,
        "components_revision": ACE_STEP_COMPONENTS_REVISION,
        "clap_model_id": DEFAULT_CLAP_MODEL_ID,
        "clap_revision": DEFAULT_CLAP_REVISION,
    }
    for name, expected in exact_artifacts.items():
        if getattr(args, name) != expected:
            parser.error(
                f"--{name.replace('_', '-')} is pinned by this experiment; expected {expected!r}"
            )

    if args.thinking or args.dcw_enabled:
        parser.error("evaluation requires thinking=False and dcw_enabled=False")
    if args.num_seeds < 1 or args.num_inference_steps < 1:
        parser.error("--num-seeds and --num-inference-steps must be positive")
    if (
        not math.isfinite(args.audio_length_in_s)
        or not math.isfinite(args.shift)
        or args.audio_length_in_s <= 0
        or args.audio_length_in_s > DEFAULT_CLAP_MAX_LENGTH_S
        or args.shift <= 0
    ):
        parser.error(
            "--audio-length-in-s must be in (0, 10] for the differentiable CLAP path "
            "and --shift must be positive"
        )
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error("--guidance-scale must be finite and at least 1")
    if not 0 <= args.steering_frac_start < args.steering_frac_end <= 1:
        parser.error("steering fractions must satisfy 0 <= start < end <= 1")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in args.fixed_alphas):
        parser.error("--fixed-alphas values must be finite and in [0,1]")
    return args


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(name)


def _safe_name(text: str, limit: int = 40) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (value or "item")[:limit]


def _move_clap(clap: ClapLoss, device: torch.device) -> None:
    if next(clap.parameters()).device != device:
        clap.to(device=device, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite values in {name}; ACE-Step must remain in FP32")


def _generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_mode": args.guidance_mode,
        "shift": args.shift,
        "steering_frac_start": args.steering_frac_start,
        "steering_frac_end": args.steering_frac_end,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    summaries = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "count": len(selected)}
        for metric in (
            "target_similarity",
            "retain_similarity",
            "prompt_similarity",
            "target_suppression_gain",
            "retain_similarity_change",
            "prompt_similarity_change",
            "alpha_mean",
            "alpha_std",
            "alpha_min",
            "alpha_max",
            "rms",
            "peak",
            "near_silence_ratio",
            "clipping_ratio",
        ):
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(item)
    return summaries


def _add_paired_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = {
        (row["sample_index"], row["seed"]): row for row in rows if row["method"] == "base_full"
    }
    for row in rows:
        base = baseline[(row["sample_index"], row["seed"])]
        row["target_suppression_gain"] = float(base["target_similarity"]) - float(
            row["target_similarity"]
        )
        row["retain_similarity_change"] = float(row["retain_similarity"]) - float(
            base["retain_similarity"]
        )
        row["prompt_similarity_change"] = float(row["prompt_similarity"]) - float(
            base["prompt_similarity"]
        )


def _plot_results(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], output_dir: Path) -> None:
    names = [item["method"] for item in summaries]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.2), 5), constrained_layout=True)
    ax.bar(x - width / 2, [item["target_similarity_mean"] for item in summaries], width, label="target")
    ax.bar(x + width / 2, [item["retain_similarity_mean"] for item in summaries], width, label="retain")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set(ylabel="CLAP cosine", title="Target suppression and retain-prompt preservation")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "method_similarity.png", dpi=150)
    plt.close(fig)

    base_by_pair = {
        (row["sample_index"], row["seed"]): row for row in rows if row["method"] == "base_full"
    }
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for method in names:
        if method == "base_full":
            continue
        selected = [row for row in rows if row["method"] == method]
        target_delta = [
            float(row["target_similarity"])
            - float(base_by_pair[(row["sample_index"], row["seed"])]["target_similarity"])
            for row in selected
        ]
        retain_delta = [
            float(row["retain_similarity"])
            - float(base_by_pair[(row["sample_index"], row["seed"])]["retain_similarity"])
            for row in selected
        ]
        ax.scatter(np.mean(target_delta), np.mean(retain_delta), s=60, label=method)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="change in target similarity (lower is better)",
        ylabel="change in retain similarity (higher is better)",
        title="Paired change from the same-seed full-prompt baseline",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "paired_tradeoff.png", dpi=150)
    plt.close(fig)

    learned = [row for row in rows if row["method"] == "learned" and row["alpha_schedule"]]
    if learned:
        by_t: dict[float, list[float]] = {}
        for row in learned:
            for timestep, alpha in json.loads(row["alpha_schedule"]):
                by_t.setdefault(round(float(timestep), 8), []).append(float(alpha))
        timesteps = sorted(by_t, reverse=True)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.plot(timesteps, [np.mean(by_t[timestep]) for timestep in timesteps], marker="o")
        ax.invert_xaxis()
        ax.set(xlabel="flow timestep", ylabel="mean alpha", title="Learned alpha schedule across evaluation runs")
        ax.grid(alpha=0.25)
        fig.savefig(output_dir / "learned_alpha_schedule.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = _device(args.device)
    _seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    checkpoint = load_checkpoint(
        args.checkpoint,
        expected_model_id=args.model_id,
        expected_model_revision=args.model_revision,
        expected_components_id=args.components_id,
        expected_components_revision=args.components_revision,
        expected_clap_model_id=args.clap_model_id,
        expected_clap_revision=args.clap_revision,
        expected_steering_mode=STEERING_MODE,
        expected_gradient_mode=GRADIENT_MODE,
    )
    saved_train_csv = checkpoint.get("args", {}).get("train_csv")
    if saved_train_csv is not None and Path(saved_train_csv).resolve() == args.eval_csv.resolve():
        warnings.warn(
            "evaluation CSV is the same path recorded for training; use a held-out group-aware split for claims",
            stacklevel=2,
        )
    predictor = SteeringPredictor(**checkpoint["config"]).to(device=device, dtype=torch.float32)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.eval()

    pipeline = SteeringAceStepPipeline.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        components_id=args.components_id,
        components_revision=args.components_revision,
        device=device,
        torch_dtype=torch.float32,
        attention_implementation="eager",
        sequential_dit=args.sequential_dit,
        offload_text_encoder=True,
        offload_dit_after_generation=args.offload_dit_after_generation,
        cache_conditioning=False,
    )
    clap = ClapLoss.from_pretrained(
        args.clap_model_id,
        revision=args.clap_revision,
        torch_dtype=torch.float32,
    ).eval()
    clap_dim = int(getattr(clap.text_encoder.config, "projection_dim", 512))
    if checkpoint["config"].get("target_embed_dim") != clap_dim:
        raise ValueError("checkpoint predictor dimension is incompatible with the selected external CLAP")

    dataset = PromptTargetDataset(args.eval_csv, args.max_samples)
    # Endpoints are always generated under the dedicated alpha_0/alpha_1 names;
    # do not pay for duplicate trajectories if users also list 0 or 1 here.
    fixed = {
        f"fixed_{alpha:g}": FixedAlpha(alpha).to(device)
        for alpha in dict.fromkeys(args.fixed_alphas)
        if alpha not in (0.0, 1.0)
    }
    endpoint_zero = FixedAlpha(0.0).to(device)
    endpoint_one = FixedAlpha(1.0).to(device)
    rows: list[dict[str, Any]] = []
    endpoint_errors: list[dict[str, float | None]] = []

    print(
        f"evaluating {len(dataset)} rows x {args.num_seeds} seeds; device={device}, "
        "dtype=float32, thinking=False, dcw_enabled=False"
    )
    for sample_index, (prompt, target, retain) in enumerate(dataset):
        _move_clap(clap, device)
        with torch.no_grad():
            text_embeds = clap.encode_text([target, retain, prompt])
        target_embed, retain_embed, prompt_embed = text_embeds
        _move_clap(clap, torch.device("cpu"))

        conditioning = pipeline.prepare_conditioning(
            [prompt], [retain], audio_length_in_s=args.audio_length_in_s
        )
        retain_conditioning = replace(
            conditioning,
            full_hidden_states=conditioning.retain_hidden_states,
            full_attention_mask=conditioning.retain_attention_mask,
        )

        for seed_offset in range(args.num_seeds):
            seed = args.seed + seed_offset
            initial = pipeline._prepare_initial_latents(  # paired noise is the evaluation invariant
                conditioning.to(device, torch.float32), seed, None
            ).detach()
            common = {**_generation_kwargs(args), "initial_latents": initial}
            latent_outputs: dict[str, Any] = {}
            latent_outputs["base_full"] = pipeline.denoise(conditioning, **common)
            latent_outputs["base_retain"] = pipeline.denoise(retain_conditioning, **common)
            latent_outputs["alpha_0"] = pipeline.denoise(
                conditioning,
                steering_target_embeds=target_embed[None].to(device),
                steering_model=endpoint_zero,
                **common,
            )
            latent_outputs["alpha_1"] = pipeline.denoise(
                conditioning,
                steering_target_embeds=target_embed[None].to(device),
                steering_model=endpoint_one,
                **common,
            )
            for name, controller in fixed.items():
                latent_outputs[name] = pipeline.denoise(
                    conditioning,
                    steering_target_embeds=target_embed[None].to(device),
                    steering_model=controller,
                    **common,
                )
            latent_outputs["learned"] = pipeline.denoise(
                conditioning,
                steering_target_embeds=target_embed[None].to(device),
                steering_model=predictor,
                **common,
            )

            alpha_zero_error = float(
                (latent_outputs["alpha_0"].audios - latent_outputs["base_full"].audios).abs().max()
            )
            full_window = args.steering_frac_start == 0.0 and args.steering_frac_end == 1.0
            alpha_one_error = (
                float((latent_outputs["alpha_1"].audios - latent_outputs["base_retain"].audios).abs().max())
                if full_window
                else None
            )
            endpoint_errors.append(
                {"alpha_zero_max_abs": alpha_zero_error, "alpha_one_max_abs": alpha_one_error}
            )
            if alpha_zero_error > 1e-5 or (alpha_one_error is not None and alpha_one_error > 1e-4):
                raise AssertionError(
                    f"alpha endpoint mismatch: alpha0={alpha_zero_error:.3g}, alpha1={alpha_one_error}"
                )

            pipeline.offload_dit()
            for method, output in latent_outputs.items():
                _assert_finite(f"{method} latents", output.audios)
                with torch.no_grad():
                    waveform = pipeline.decode_latents(
                        output.audios,
                        num_audio_samples=round(args.audio_length_in_s * ACE_STEP_SAMPLE_RATE),
                    )
                    _assert_finite(f"{method} waveform", waveform)
                    _move_clap(clap, device)
                    audio_embed = clap.encode_audio(waveform, ACE_STEP_SAMPLE_RATE)
                    target_similarity = float((audio_embed[0] * target_embed.to(device)).sum())
                    retain_similarity = float((audio_embed[0] * retain_embed.to(device)).sum())
                    prompt_similarity = float((audio_embed[0] * prompt_embed.to(device)).sum())

                audio_float = waveform[0].detach().float()
                rms = float(audio_float.square().mean().sqrt())
                peak = float(audio_float.abs().max())
                near_silence_ratio = float((audio_float.abs() < 1e-4).float().mean())
                clipping_ratio = float((audio_float.abs() >= 0.999).float().mean())

                schedule = output.alpha_records
                alpha_values = np.asarray([alpha for _, alpha in schedule], dtype=np.float64)
                alpha_mean = float(alpha_values.mean()) if alpha_values.size else 0.0
                alpha_std = float(alpha_values.std()) if alpha_values.size else 0.0
                alpha_min = float(alpha_values.min()) if alpha_values.size else 0.0
                alpha_max = float(alpha_values.max()) if alpha_values.size else 0.0
                audio_name = (
                    f"sample_{sample_index:03d}_seed_{seed}_{_safe_name(method)}_{_safe_name(target)}.wav"
                )
                audio_path = save_waveform(audio_dir / audio_name, waveform[0], ACE_STEP_SAMPLE_RATE)
                rows.append(
                    {
                        "sample_index": sample_index,
                        "seed": seed,
                        "method": method,
                        "prompt": prompt,
                        "target": target,
                        "retain_prompt": retain,
                        "target_similarity": target_similarity,
                        "retain_similarity": retain_similarity,
                        "prompt_similarity": prompt_similarity,
                        "alpha_mean": alpha_mean,
                        "alpha_std": alpha_std,
                        "alpha_min": alpha_min,
                        "alpha_max": alpha_max,
                        "alpha_schedule": json.dumps(schedule),
                        "rms": rms,
                        "peak": peak,
                        "near_silence_ratio": near_silence_ratio,
                        "clipping_ratio": clipping_ratio,
                        "audio_path": str(audio_path),
                    }
                )
                del waveform, audio_embed
            pipeline.offload_vae()
            _move_clap(clap, torch.device("cpu"))
            print(f"sample {sample_index + 1}/{len(dataset)}, seed {seed}: complete")

    _add_paired_deltas(rows)
    summaries = _summaries(rows)
    _write_csv(args.output_dir / "per_sample.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summaries)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_global_step": checkpoint["global_step"],
        "steering_mode": checkpoint["steering_mode"],
        "gradient_mode": checkpoint["gradient_mode"],
        "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"],
        "components_id": checkpoint["components_id"],
        "components_revision": checkpoint["components_revision"],
        "clap_model_id": checkpoint["clap_model_id"],
        "clap_revision": checkpoint["clap_revision"],
        "num_rows": len(rows),
        "endpoint_alpha_zero_max_abs": max(item["alpha_zero_max_abs"] for item in endpoint_errors),
        "endpoint_alpha_one_max_abs": (
            max(
                item["alpha_one_max_abs"]
                for item in endpoint_errors
                if item["alpha_one_max_abs"] is not None
            )
            if any(item["alpha_one_max_abs"] is not None for item in endpoint_errors)
            else None
        ),
        "summaries": summaries,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_lines = [
        "# ACE-Step steering evaluation",
        "",
        f"Checkpoint: `{args.checkpoint.resolve()}`",
        "",
        "| method | target cosine | retain cosine | suppression gain | retain change |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        report_lines.append(
            f"| {item['method']} | {item['target_similarity_mean']:.4f} | "
            f"{item['retain_similarity_mean']:.4f} | {item['target_suppression_gain_mean']:.4f} | "
            f"{item['retain_similarity_change_mean']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "Lower target cosine and higher retain cosine are preferred. Inspect paired stereo audio before "
            "interpreting CLAP-only changes as selective removal.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    _plot_results(rows, summaries, args.output_dir)
    print(f"evaluation complete: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
