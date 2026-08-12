r"""
Generates audio with Stable Audio 3 for a list of prompts.

Same job as `scripts/generate_audio_stable_audio.py`, but Stable Audio 3 is not supported by diffusers:
it ships its own library, `stable_audio_3`, whose `StableAudioModel` replaces `StableAudioPipeline`. Like
the other generation scripts this one never touches MusicLDM or any steering component, it is plain
text-to-audio generation.

The `stabilityai/stable-audio-3-*` weights are gated: accept the Stability AI Community License and the
Gemma Terms of Use on the model page, then authenticate with `hf auth login` or an `HF_TOKEN` variable.

Note on seeding: `StableAudioModel.generate` takes a single `seed` per call rather than one generator per
sample, so with `--batch-size 1` (the default) every clip gets its own reproducible seed, while with a
larger batch the whole batch shares the seed written on each of its manifest rows.

Example, prompts given directly on the command line:
    uv run python scripts/generate_audio_stable_audio_3.py \
        --prompt "A laid-back jazz trumpet solo over walking bass" \
        --output outputs/generated --steps 50

Example, prompts read from a CSV file with a `prompt` column:
    uv run python scripts/generate_audio_stable_audio_3.py --dataset datasets/stable_audio_prompts.csv \
        --output outputs/generated --model small-music
"""

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np
from stable_audio_3 import StableAudioModel

# Post-trained checkpoints are distilled: 8 steps are enough and they ignore guidance, while the `-base`
# ones keep classifier free guidance and want roughly 50 steps. The values are the model's maximum duration.
MODEL_MAX_DURATION_IN_S = {
    "small-music-base": 120.0,
    "small-music": 120.0,
    "small-sfx-base": 120.0,
    "small-sfx": 120.0,
    "medium-base": 380.0,
    "medium": 380.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio with Stable Audio 3 for a list of prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    prompts = parser.add_argument_group("prompts")
    prompts.add_argument("--prompt", type=str, nargs="+", default=None, help="one or more prompts to generate")
    prompts.add_argument("--dataset", type=Path, default=None, help="CSV file with a `prompt` column")

    output = parser.add_argument_group("output")
    output.add_argument("--output", type=Path, default=Path("outputs/generated_audio_sa3"), help="folder for audio and manifest")
    output.add_argument("--batch-size", type=int, default=1, help="prompts generated per forward pass, sharing one seed")
    output.add_argument("--max-samples", type=int, default=None, help="use only the first N prompts")

    generation = parser.add_argument_group("generation")
    generation.add_argument("--model", type=str, default="small-music-base", choices=sorted(MODEL_MAX_DURATION_IN_S), help="Stable Audio 3 checkpoint")
    generation.add_argument("--steps", type=int, default=50, help="sampling steps per generation, 8 is enough for the post-trained checkpoints")
    generation.add_argument("--cfg-scale", type=float, default=7.0, help="classifier free guidance scale, base checkpoints only")
    generation.add_argument("--negative-prompt", type=str, default=None, help="qualities to avoid, base checkpoints only")
    generation.add_argument("--audio-length-in-s", type=float, default=10.0, help="length of the generated clips, in seconds")
    generation.add_argument("--no-chunked-decode", action="store_true", help="decode in one pass, more VRAM but no stitching artifacts")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", type=str, default=None, help="torch device, by default cuda then mps then cpu")
    runtime.add_argument("--no-half", action="store_true", help="run in float32 instead of half precision")

    args = parser.parse_args()

    if (args.prompt is None) == (args.dataset is None):
        parser.error("exactly one of `--prompt` or `--dataset` has to be given")
    if args.batch_size < 1:
        parser.error(f"`--batch-size` has to be at least 1 but is {args.batch_size}")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error(f"`--max-samples` has to be at least 1 but is {args.max_samples}")
    if args.steps < 1:
        parser.error(f"`--steps` has to be at least 1 but is {args.steps}")

    max_duration = MODEL_MAX_DURATION_IN_S[args.model]
    if not 0.0 < args.audio_length_in_s <= max_duration:
        parser.error(
            f"`--audio-length-in-s` has to be in (0, {max_duration}] for `{args.model}` but is {args.audio_length_in_s}"
        )

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


def save_waveform(path: Path, waveform: np.ndarray, sampling_rate: int) -> None:
    """Writes float audio of shape `(channels, samples)` as clipped signed 16-bit PCM, stdlib only."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(np.asarray(waveform), -1.0, 1.0) * 32767.0).astype("<i2")
    frames = np.ascontiguousarray(pcm.T)  # (samples, channels), interleaved for wave's frame layout
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(frames.tobytes())


def main() -> None:
    args = parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    prompts = list(args.prompt) if args.prompt is not None else load_prompts_from_csv(args.dataset)
    prompts = prompts if args.max_samples is None else prompts[: args.max_samples]
    print(f"Loaded {len(prompts)} prompts")

    if not args.model.endswith("-base") and (args.cfg_scale != 1.0 or args.negative_prompt is not None):
        print(f"warning: `{args.model}` is post-trained, `--cfg-scale` and `--negative-prompt` have no effect on it")

    print(f"Loading {args.model} on {args.device or 'the default device'} with {'float32' if args.no_half else 'half'} precision")
    model = StableAudioModel.from_pretrained(args.model, device=args.device, model_half=not args.no_half)

    sampling_rate = int(model.model.sample_rate)
    # `generate` shrinks the latent to the requested duration, but its own default caps at 120s, which would
    # silently truncate the checkpoints that can go longer.
    sample_size = int(model.model_config["sample_size"])
    audio_dir = output_dir / "audio"
    manifest_path = output_dir / "manifest.csv"
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["index", "prompt", "seed", "audio_path"])
        writer.writeheader()

        for batch_index, start in enumerate(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[start : start + args.batch_size]
            seed = args.seed + start  # one seed per call, so with `--batch-size 1` it is the prompt's own seed
            negative_prompts = None if args.negative_prompt is None else [args.negative_prompt] * len(batch_prompts)

            # `generate` is already decorated with `torch.inference_mode`.
            audio = model.generate(
                prompt=batch_prompts,
                negative_prompt=negative_prompts,
                duration=args.audio_length_in_s,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                batch_size=len(batch_prompts),  # has to match the number of prompts to keep them aligned
                sample_size=sample_size,
                seed=seed,
                chunked_decode=not args.no_chunked_decode,
            )
            waveforms = audio.detach().cpu().float().numpy()  # (batch, channels, samples)

            for offset, (prompt, waveform) in enumerate(zip(batch_prompts, waveforms, strict=True)):
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
