"""Generate plain, unsteered audio with the official ACE-Step 1.5 SFT checkpoint."""

import argparse
import csv
import json
from pathlib import Path

import torch

from pipelines import ACE_STEP_MODEL_ID, SteeringAceStepPipeline
from utils import save_waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio with ACE-Step 1.5 SFT for a list of prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prompts = parser.add_argument_group("prompts")
    prompts.add_argument("--prompt", type=str, nargs="+", default=None, help="one or more prompts")
    prompts.add_argument("--dataset", type=Path, default=None, help="CSV file with a `prompt` column")

    output = parser.add_argument_group("output")
    output.add_argument("--output", type=Path, default=Path("outputs/generated_audio_ace_step"))
    output.add_argument("--batch-size", type=int, default=1)
    output.add_argument("--max-samples", type=int, default=None)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--model", choices=(ACE_STEP_MODEL_ID,), default=ACE_STEP_MODEL_ID)
    generation.add_argument("--steps", type=int, default=50)
    generation.add_argument("--cfg-scale", type=float, default=7.0)
    generation.add_argument("--shift", type=float, default=1.0)
    generation.add_argument("--audio-length-in-s", type=float, default=10.0)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    runtime.add_argument("--no-half", action="store_true", help="use float32 instead of CUDA BF16")
    args = parser.parse_args()

    if (args.prompt is None) == (args.dataset is None):
        parser.error("exactly one of `--prompt` or `--dataset` must be supplied")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.cfg_scale < 1.0:
        parser.error("--cfg-scale must be at least 1.0")
    if args.shift <= 0.0:
        parser.error("--shift must be positive")
    if not 10.0 <= args.audio_length_in_s <= 600.0:
        parser.error("--audio-length-in-s must be in [10, 600]")
    return args


def load_prompts_from_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a `prompt` column; found {reader.fieldnames}")
        prompts = [row["prompt"].strip() for row in reader if (row.get("prompt") or "").strip()]
    if not prompts:
        raise ValueError(f"{path} contains no non-empty prompts")
    return prompts


def main() -> None:
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    prompts = list(args.prompt) if args.prompt is not None else load_prompts_from_csv(args.dataset)
    prompts = prompts if args.max_samples is None else prompts[: args.max_samples]
    print(f"Loaded {len(prompts)} prompts")

    print(f"Loading {args.model} on {args.device} with {'float32' if args.no_half else 'CUDA BF16/float32 fallback'}")
    pipeline = SteeringAceStepPipeline.from_pretrained(
        args.model,
        device=args.device,
        model_half=not args.no_half,
    )

    manifest_path = output_dir / "manifest.csv"
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["index", "prompt", "seed", "audio_path"])
        writer.writeheader()
        for batch_index, start in enumerate(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[start : start + args.batch_size]
            seeds = [args.seed + index for index in range(start, start + len(batch_prompts))]
            generators = [torch.Generator().manual_seed(seed) for seed in seeds]
            with torch.inference_mode():
                output = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=args.steps,
                    audio_length_in_s=args.audio_length_in_s,
                    cfg_scale=args.cfg_scale,
                    shift=args.shift,
                    generator=generators,
                    output_type="pt",
                )

            waveforms = output.audios.detach().cpu().float().numpy()
            for offset, (prompt, seed, waveform) in enumerate(
                zip(batch_prompts, seeds, waveforms, strict=True)
            ):
                sample_index = start + offset
                audio_relative_path = str(Path("audio") / f"{sample_index:04d}.wav")
                save_waveform(output_dir / audio_relative_path, waveform, pipeline.sample_rate)
                writer.writerow(
                    {
                        "index": sample_index,
                        "prompt": prompt,
                        "seed": seed,
                        "audio_path": audio_relative_path,
                    }
                )
            csv_file.flush()
            print(f"batch {batch_index + 1}/{num_batches}: generated {len(batch_prompts)} prompts", flush=True)

    print(f"Done. {len(prompts)} clips and {manifest_path.resolve()} written under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
