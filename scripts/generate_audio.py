r"""
Generates audio with the stock `MusicLDMPipeline` for a list of prompts.

MusicLDM is no longer the backbone of this project, which has moved to Stable Audio 3; this script is
kept as the historical baseline the earlier results were produced with. It never touches any steering
component: there is no steering target, predictor or steering window, just plain text-to-audio
generation.

Example, prompts given directly on the command line:
    uv run python scripts/generate_audio.py --prompt "A laid-back jazz trumpet solo over walking bass" \
        --output outputs/generated --num-inference-steps 200

Example, prompts read from a CSV file with a `prompt` column:
    uv run python scripts/generate_audio.py --dataset datasets/trumpet_prompts_dataset.csv \
        --output outputs/generated --batch-size 4
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from diffusers import MusicLDMPipeline

from utils import save_waveform, tensor_text_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio with MusicLDM for a list of prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    prompts = parser.add_argument_group("prompts")
    prompts.add_argument("--prompt", type=str, nargs="+", default=None, help="one or more prompts to generate")
    prompts.add_argument("--dataset", type=Path, default=None, help="CSV file with a `prompt` column")

    output = parser.add_argument_group("output")
    output.add_argument("--output", type=Path, default=Path("outputs/generated_audio"), help="folder for audio and manifest")
    output.add_argument("--batch-size", type=int, default=1, help="prompts generated per forward pass")
    output.add_argument("--max-samples", type=int, default=None, help="use only the first N prompts")

    generation = parser.add_argument_group("generation")
    generation.add_argument("--num-inference-steps", type=int, default=200, help="denoising steps per generation")
    generation.add_argument("--audio-length-in-s", type=float, default=10.0, help="length of the generated clips, in seconds")
    generation.add_argument("--guidance-scale", type=float, default=2.0, help="classifier free guidance scale")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if (args.prompt is None) == (args.dataset is None):
        parser.error("exactly one of `--prompt` or `--dataset` has to be given")
    if args.batch_size < 1:
        parser.error(f"`--batch-size` has to be at least 1 but is {args.batch_size}")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error(f"`--max-samples` has to be at least 1 but is {args.max_samples}")
    if args.num_inference_steps < 1:
        parser.error(f"`--num-inference-steps` has to be at least 1 but is {args.num_inference_steps}")
    if args.audio_length_in_s <= 0.0:
        parser.error(f"`--audio-length-in-s` has to be positive but is {args.audio_length_in_s}")

    return args


def load_prompts_from_csv(path: Path) -> list[str]:
    r"""
    Reads the `prompt` column of a CSV file, in the same convention as `utils.PromptTargetDataset` but
    without requiring a `target` column, which plain generation has no use for.
    """
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError(f"`dataset` has to contain a `prompt` column but {path} has columns {reader.fieldnames}")
        prompts = [row["prompt"].strip() for row in reader if (row.get("prompt") or "").strip()]

    if not prompts:
        raise ValueError(f"`dataset` has to contain at least one non-empty prompt but {path} contains none")
    return prompts


def resolve_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    return device, dtype


def main() -> None:
    args = parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    prompts = list(args.prompt) if args.prompt is not None else load_prompts_from_csv(args.dataset)
    prompts = prompts if args.max_samples is None else prompts[: args.max_samples]
    print(f"Loaded {len(prompts)} prompts")

    device, dtype = resolve_device_and_dtype()
    print(f"Loading ucsd-reach/musicldm on {device} with {dtype}")
    pipe = MusicLDMPipeline.from_pretrained("ucsd-reach/musicldm", torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    sampling_rate = int(pipe.vocoder.config.sampling_rate)
    audio_dir = output_dir / "audio"
    manifest_path = output_dir / "manifest.csv"
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["index", "prompt", "seed", "audio_path"])
        writer.writeheader()

        for batch_index, start in enumerate(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[start : start + args.batch_size]
            seeds = [args.seed + start + offset for offset in range(len(batch_prompts))]
            generators = [torch.Generator().manual_seed(seed) for seed in seeds]

            with torch.inference_mode(), tensor_text_features(pipe.text_encoder):
                output = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=args.num_inference_steps,
                    audio_length_in_s=args.audio_length_in_s,
                    guidance_scale=args.guidance_scale,
                    generator=generators,
                    output_type="np",
                )
            waveforms = output.audios

            for offset, (prompt, seed, waveform) in enumerate(zip(batch_prompts, seeds, waveforms, strict=True)):
                sample_index = start + offset
                audio_relative_path = str(Path("audio") / f"{sample_index:04d}.wav")
                save_waveform(output_dir / audio_relative_path, waveform, sampling_rate)
                writer.writerow(
                    {"index": sample_index, "prompt": prompt, "seed": seed, "audio_path": audio_relative_path}
                )
            csv_file.flush()

            print(f"batch {batch_index + 1}/{num_batches} generated {len(batch_prompts)} prompts", flush=True)

    print(f"\nDone. {len(prompts)} clips written to {audio_dir.resolve()}, manifest at {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
