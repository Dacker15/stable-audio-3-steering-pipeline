"""Train target-specific ACE-Step 1.5 steering in external CLAP space.

The ACE-Step DiT, Qwen encoder, VAE and CLAP towers are frozen.  The default
``steering_only`` gradient evaluates every DiT branch under ``no_grad`` and
backpropagates through the predictor, alpha interpolation, APG/Euler updates,
the differentiable VAE decode and the CLAP audio tower.  It is a documented
surrogate: it omits the DiT Jacobian with respect to the evolving latent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import DEFAULT_CLAP_MAX_LENGTH_S, DEFAULT_CLAP_MODEL_ID, DEFAULT_CLAP_REVISION, ClapLoss
from pipelines import (
    ACE_STEP_COMPONENTS_ID,
    ACE_STEP_COMPONENTS_REVISION,
    ACE_STEP_LATENT_CHANNELS,
    ACE_STEP_MODEL_ID,
    ACE_STEP_MODEL_REVISION,
    ACE_STEP_SAMPLE_RATE,
    GRADIENT_MODE,
    STEERING_MODE,
    SteeringAceStepPipeline,
    SteeringPredictor,
)
from utils import (
    PromptTargetDataset,
    build_checkpoint,
    collate_prompt_target,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    save_waveform,
)


def _add_runtime_switches(parser: argparse.ArgumentParser) -> None:
    dit = parser.add_mutually_exclusive_group()
    dit.add_argument("--sequential-dit", dest="sequential_dit", action="store_true")
    dit.add_argument("--batched-dit", dest="sequential_dit", action="store_false")
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument(
        "--offload-dit-after-generation", dest="offload_dit_after_generation", action="store_true"
    )
    offload.add_argument("--keep-dit-on-device", dest="offload_dit_after_generation", action="store_false")
    parser.set_defaults(sequential_dit=True, offload_dit_after_generation=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ACE-Step target-specific steering predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data/output")
    data.add_argument("--train-csv", type=Path, required=True)
    data.add_argument("--output-dir", type=Path, required=True)
    data.add_argument("--max-samples", type=int)
    data.add_argument("--batch-size", type=int, default=1)
    data.add_argument("--num-workers", type=int, default=0)
    data.add_argument("--resume", type=Path, help="v2 ACE-Step checkpoint to resume")

    model = parser.add_argument_group("models")
    model.add_argument("--model-id", default=ACE_STEP_MODEL_ID)
    model.add_argument("--model-revision", default=ACE_STEP_MODEL_REVISION)
    model.add_argument("--components-id", default=ACE_STEP_COMPONENTS_ID)
    model.add_argument("--components-revision", default=ACE_STEP_COMPONENTS_REVISION)
    model.add_argument("--clap-model-id", default=DEFAULT_CLAP_MODEL_ID)
    model.add_argument("--clap-revision", default=DEFAULT_CLAP_REVISION)

    generation = parser.add_argument_group("ACE-Step generation")
    generation.add_argument("--num-inference-steps", type=int, default=50)
    generation.add_argument(
        "--audio-length-in-s",
        type=float,
        default=DEFAULT_CLAP_MAX_LENGTH_S,
        help="clip duration; capped at the frozen CLAP tower's complete input window",
    )
    generation.add_argument("--guidance-scale", type=float, default=7.0)
    generation.add_argument("--guidance-mode", choices=("apg",), default="apg")
    generation.add_argument("--shift", type=float, default=1.0)
    generation.add_argument("--steering-frac-start", type=float, default=0.3)
    generation.add_argument("--steering-frac-end", type=float, default=0.8)
    generation.add_argument("--thinking", action="store_true", help="unsupported invariant guard")
    generation.add_argument("--dcw-enabled", action="store_true", help="unsupported invariant guard")

    predictor = parser.add_argument_group("predictor")
    predictor.add_argument("--alpha-min", type=float, default=0.0)
    predictor.add_argument("--alpha-max", type=float, default=1.0)
    predictor.add_argument("--alpha-init", type=float, default=0.15)
    predictor.add_argument("--predictor-channels", type=int, nargs="+", default=(64, 128, 256))
    predictor.add_argument("--predictor-layers-per-block", type=int, default=2)
    predictor.add_argument("--predictor-cond-dim", type=int, default=256)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=5)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-2)
    optim.add_argument("--retain-weight", type=float, default=1.0)
    optim.add_argument("--grad-accum-steps", type=int, default=1)
    optim.add_argument("--max-grad-norm", type=float, default=1.0)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    runtime.add_argument("--dtype", default="float32", choices=("float32",))
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument(
        "--smoke-test",
        action="store_true",
        help="force one sample/epoch, two flow steps, and at most two seconds",
    )
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
        parser.error("this training path requires thinking=False and dcw_enabled=False")
    for name in ("batch_size", "num_workers", "epochs", "grad_accum_steps", "num_inference_steps"):
        value = getattr(args, name)
        minimum = 0 if name == "num_workers" else 1
        if value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}")
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
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and non-negative")
    if (
        not math.isfinite(args.retain_weight)
        or not math.isfinite(args.max_grad_norm)
        or args.retain_weight < 0
        or args.max_grad_norm <= 0
    ):
        parser.error("--retain-weight must be finite/non-negative and --max-grad-norm finite/positive")
    if not 0.0 <= args.alpha_min < args.alpha_init < args.alpha_max <= 1.0:
        parser.error("alpha bounds must satisfy 0 <= min < init < max <= 1")
    if not 0 <= args.steering_frac_start < args.steering_frac_end <= 1:
        parser.error("steering fractions must satisfy 0 <= start < end <= 1")

    if args.smoke_test:
        args.max_samples = 1
        args.batch_size = 1
        args.epochs = 1
        args.grad_accum_steps = 1
        args.num_inference_steps = min(args.num_inference_steps, 2)
        args.audio_length_in_s = min(args.audio_length_in_s, 2.0)

    steered_steps = sum(
        args.steering_frac_start <= index / args.num_inference_steps < args.steering_frac_end
        for index in range(args.num_inference_steps)
    )
    if steered_steps == 0:
        parser.error("the steering window contains no flow step")
    args.num_steered_steps = steered_steps
    return args


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(json.dumps(vars(args), default=str))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(requested)


def _mean_alpha(records: list[tuple[float, float]]) -> list[tuple[float, float]]:
    values: dict[float, list[float]] = defaultdict(list)
    for timestep, alpha in records:
        values[round(float(timestep), 8)].append(float(alpha))
    return [(timestep, float(np.mean(values[timestep]))) for timestep in sorted(values, reverse=True)]


def _plot_history(history: dict[str, Any], output_dir: Path) -> None:
    epochs = history.get("epochs", [])
    if not epochs:
        return
    x = [item["epoch"] for item in epochs]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(x, [item["loss"] for item in epochs], marker="o", label="objective")
    axes[0].set(xlabel="epoch", ylabel="loss", title="ACE-Step steering training loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, [item["target_similarity"] for item in epochs], marker="o", label="target")
    axes[1].plot(x, [item["retain_similarity"] for item in epochs], marker="o", label="retain")
    axes[1].set(xlabel="epoch", ylabel="CLAP cosine", title="Suppression / preservation trade-off")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for item in epochs:
        schedule = item["alpha_by_timestep"]
        ax.plot([point[0] for point in schedule], [point[1] for point in schedule], label=f"epoch {item['epoch']}")
    ax.invert_xaxis()
    ax.set(xlabel="flow timestep (noise to audio)", ylabel="mean alpha", title="Learned alpha schedule")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "alpha_schedule.png", dpi=150)
    plt.close(fig)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"non-finite values detected in {name}; FP32 is required")


def _move_clap(clap: ClapLoss, device: torch.device) -> None:
    if next(clap.parameters()).device != device:
        clap.to(device=device, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()


_RESUME_INVARIANTS = (
    "train_csv",
    "max_samples",
    "batch_size",
    "num_inference_steps",
    "audio_length_in_s",
    "guidance_scale",
    "guidance_mode",
    "shift",
    "steering_frac_start",
    "steering_frac_end",
    "retain_weight",
    "lr",
    "weight_decay",
    "grad_accum_steps",
    "max_grad_norm",
    "alpha_min",
    "alpha_max",
    "alpha_init",
    "predictor_channels",
    "predictor_layers_per_block",
    "predictor_cond_dim",
    "seed",
)


def _validate_resume_configuration(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    saved = checkpoint["args"]
    missing = [name for name in _RESUME_INVARIANTS if name not in saved]
    if missing:
        raise ValueError(f"resume checkpoint is missing exact-training arguments: {missing}")
    mismatches = []
    for name in _RESUME_INVARIANTS:
        previous = saved[name]
        current = getattr(args, name)
        if name == "train_csv":
            previous = str(Path(previous).resolve())
            current = str(Path(current).resolve())
        elif name == "predictor_channels":
            previous = tuple(previous)
            current = tuple(current)
        if previous != current:
            mismatches.append(f"{name}: checkpoint={previous!r}, CLI={current!r}")
    if mismatches:
        raise ValueError(
            "resume would change training semantics; repeat the original flags (only --epochs/runtime/output may "
            "change): " + "; ".join(mismatches)
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    _seed_everything(args.seed)

    checkpoint = None
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            expected_model_id=args.model_id,
            expected_model_revision=args.model_revision,
            expected_components_id=args.components_id,
            expected_components_revision=args.components_revision,
            expected_clap_model_id=args.clap_model_id,
            expected_clap_revision=args.clap_revision,
            expected_steering_mode=STEERING_MODE,
            expected_gradient_mode=GRADIENT_MODE,
        )
        _validate_resume_configuration(checkpoint, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    sample_dir = args.output_dir / "audio"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps(_jsonable_args(args), indent=2), encoding="utf-8"
    )

    print(f"device={device}, dtype=float32, gradient_mode={GRADIENT_MODE}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(
            f"GPU={properties.name}, capability={properties.major}.{properties.minor}, "
            f"VRAM={properties.total_memory / 2**30:.1f} GiB"
        )
    print("thinking=False, dcw_enabled=False, autocast=disabled")

    dataset = PromptTargetDataset(args.train_csv, args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_prompt_target,
        pin_memory=device.type == "cuda",
    )
    print(f"loaded {len(dataset)} rows from {args.train_csv}")

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

    predictor_config: dict[str, Any]
    if checkpoint is not None:
        predictor_config = dict(checkpoint["config"])
        if predictor_config.get("target_embed_dim") != clap_dim:
            raise ValueError(
                "resume checkpoint CLAP dimension does not match the selected external CLAP model"
            )
    else:
        predictor_config = {
            "latent_channels": ACE_STEP_LATENT_CHANNELS,
            "target_embed_dim": clap_dim,
            "block_out_channels": tuple(args.predictor_channels),
            "layers_per_block": args.predictor_layers_per_block,
            "cond_embed_dim": args.predictor_cond_dim,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_init": args.alpha_init,
        }
    predictor = SteeringPredictor(**predictor_config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: dict[str, Any] = {"args": _jsonable_args(args), "iterations": [], "epochs": []}
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    if checkpoint is not None:
        predictor.load_state_dict(checkpoint["state_dict"], strict=True)
        resume = restore_training_state(checkpoint, optimizer=optimizer, restore_rng=True)
        history = dict(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
        global_step = int(resume["global_step"])
        best_loss = float(resume["best_loss"] if resume["best_loss"] is not None else math.inf)
        print(f"resumed {args.resume} at epoch {start_epoch}, global_step {global_step}")

    if start_epoch > args.epochs:
        raise ValueError(f"checkpoint already completed epoch {start_epoch - 1}, but --epochs={args.epochs}")

    print(
        f"predictor={sum(parameter.numel() for parameter in predictor.parameters()) / 1e6:.2f}M params; "
        f"steering {args.num_steered_steps}/{args.num_inference_steps} steps"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        target_scores: list[float] = []
        retain_scores: list[float] = []
        alpha_records: list[tuple[float, float]] = []
        sample_waveform: torch.Tensor | None = None

        for batch_index, (prompts, targets, retains) in enumerate(dataloader):
            global_step += 1

            # CLAP text embeddings are independent from ACE-Step's Qwen
            # embeddings.  Offload CLAP before loading Qwen/DiT on a T4.
            _move_clap(clap, device)
            with torch.no_grad():
                all_text = clap.encode_text([*targets, *retains])
                target_embeds, retain_embeds = all_text.chunk(2)
            _move_clap(clap, torch.device("cpu"))

            latent_output = pipeline(
                prompts,
                retain_prompt=retains,
                steering_target_embeds=target_embeds.to(device),
                steering_model=predictor,
                audio_length_in_s=args.audio_length_in_s,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                guidance_mode=args.guidance_mode,
                shift=args.shift,
                steering_frac_start=args.steering_frac_start,
                steering_frac_end=args.steering_frac_end,
                seed=[args.seed + global_step * args.batch_size + index for index in range(len(prompts))],
                output_type="latent",
                train=True,
                thinking=False,
                dcw_enabled=False,
            )
            latents = latent_output.audios
            _assert_finite("ACE-Step latents", latents)
            alpha_records.extend(latent_output.alpha_records)

            waveform = pipeline.decode_latents(
                latents,
                num_audio_samples=round(args.audio_length_in_s * ACE_STEP_SAMPLE_RATE),
            )
            _assert_finite("decoded waveform", waveform)
            _move_clap(clap, device)
            audio_embeds = clap.encode_audio(waveform, ACE_STEP_SAMPLE_RATE)
            _assert_finite("CLAP audio embeddings", audio_embeds)

            target_embeds = target_embeds.to(device=device, dtype=audio_embeds.dtype)
            retain_embeds = retain_embeds.to(device=device, dtype=audio_embeds.dtype)
            target_similarity = (audio_embeds * target_embeds).sum(dim=-1)
            retain_similarity = (audio_embeds * retain_embeds).sum(dim=-1)
            objective = target_similarity.mean() + args.retain_weight * (
                1.0 - retain_similarity
            ).mean()
            _assert_finite("training objective", objective)
            group_start = (batch_index // args.grad_accum_steps) * args.grad_accum_steps
            accumulation_group_size = min(args.grad_accum_steps, len(dataloader) - group_start)
            (objective / accumulation_group_size).backward()

            should_step = (batch_index + 1) % args.grad_accum_steps == 0 or batch_index + 1 == len(dataloader)
            if should_step:
                gradient_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("non-finite predictor gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            sample_waveform = waveform[0].detach().cpu()
            loss_value = float(objective.detach())
            target_value = float(target_similarity.detach().mean())
            retain_value = float(retain_similarity.detach().mean())
            losses.append(loss_value)
            target_scores.append(target_value)
            retain_scores.append(retain_value)
            history.setdefault("iterations", []).append(
                {
                    "epoch": epoch,
                    "batch": batch_index + 1,
                    "global_step": global_step,
                    "loss": loss_value,
                    "target_similarity": target_value,
                    "retain_similarity": retain_value,
                }
            )
            print(
                f"epoch {epoch}/{args.epochs} batch {batch_index + 1}/{len(dataloader)} "
                f"loss={loss_value:.4f} target={target_value:.4f} retain={retain_value:.4f}"
            )

            # The backward pass no longer needs the frozen VAE/CLAP weights.
            del waveform, audio_embeds, objective, latents, latent_output
            pipeline.offload_vae()
            _move_clap(clap, torch.device("cpu"))

        epoch_loss = float(np.mean(losses))
        epoch_record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "target_similarity": float(np.mean(target_scores)),
            "retain_similarity": float(np.mean(retain_scores)),
            "alpha_by_timestep": _mean_alpha(alpha_records),
        }
        history.setdefault("epochs", []).append(epoch_record)
        is_best = epoch_loss <= best_loss
        best_loss = min(best_loss, epoch_loss)

        payload = build_checkpoint(
            state_dict=predictor.state_dict(),
            config=predictor_config,
            optimizer_state_dict=optimizer.state_dict(),
            history=history,
            args=_jsonable_args(args),
            epoch=epoch,
            global_step=global_step,
            batch_in_epoch=0,
            loss=epoch_loss,
            best_loss=best_loss,
            model_id=args.model_id,
            model_revision=args.model_revision,
            components_id=args.components_id,
            components_revision=args.components_revision,
            clap_model_id=args.clap_model_id,
            clap_revision=args.clap_revision,
            gradient_mode=GRADIENT_MODE,
        )
        save_checkpoint(payload, checkpoint_dir / "latest.pt")
        if is_best:
            save_checkpoint(payload, checkpoint_dir / "best.pt")
        if sample_waveform is not None:
            save_waveform(sample_dir / f"epoch_{epoch:03d}.wav", sample_waveform, ACE_STEP_SAMPLE_RATE)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _plot_history(history, args.output_dir)

    print(f"training complete; latest checkpoint: {checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
