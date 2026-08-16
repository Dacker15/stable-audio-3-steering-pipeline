r"""
Generates paired ``alpha=0`` ACE-Step 1.5 SFT baselines and independently tags their instruments.

The target is always read from the input CSV before generation/classification. A prompt/seed pair is
eligible for that target only when the target's baseline score reaches the configured threshold.
Retain instruments are evaluated independently and never change target eligibility.

Preferred CSV columns:
    prompt,target,retain_prompt,requested_instruments,retain_instruments

The two instrument columns are optional and contain either JSON arrays or semicolon-separated names.
When absent, the configured vocabulary extracts instrument mentions from prompt and retain_prompt.
Legacy prompt,target datasets remain accepted and use utils.strip_target for retain_prompt.

Example:
    uv run python scripts/prepare_baselines_ace_step.py \
        --dataset datasets/trumpet_simple_splits/test.csv \
        --output outputs/trumpet-baseline-selection \
        --num-seeds 5 --seed 1000
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from pipelines import ACE_STEP_MODEL_ID, FixedAlphaSteering, SteeringAceStepPipeline
from utils import load_waveform, save_waveform
from utils.instrument_classification import (
    BASELINE_CSV_FIELDS,
    BASELINE_SCHEMA_VERSION,
    DEFAULT_CLASSIFIER_MODEL,
    AudioSetInstrumentClassifier,
    InstrumentVocabulary,
    baseline_record_to_csv,
    build_baseline_record,
    load_baseline_prompts,
    paired_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and classify paired ACE-Step 1.5 SFT baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=Path, required=True, help="CSV with prompt and a preassigned target")
    data.add_argument("--max-samples", type=int, default=None, help="use only the first N dataset rows")
    data.add_argument(
        "--instrument-config",
        type=Path,
        default=None,
        help="optional JSON vocabulary replacing the built-in AudioSet instrument mapping",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output", type=Path, default=Path("outputs/baseline_selection_ace_step"))
    output.add_argument(
        "--replace-output",
        action="store_true",
        help="replace a non-empty output directory instead of failing",
    )

    generation = parser.add_argument_group("ACE-Step 1.5 SFT generation")
    generation.add_argument("--model", choices=(ACE_STEP_MODEL_ID,), default=ACE_STEP_MODEL_ID)
    generation.add_argument(
        "--steps",
        "--num-inference-steps",
        dest="num_inference_steps",
        type=int,
        default=50,
    )
    generation.add_argument("--audio-length-in-s", type=float, default=10.0)
    generation.add_argument("--cfg-scale", type=float, default=7.0)
    generation.add_argument("--shift", type=float, default=1.0)
    generation.add_argument(
        "--baseline-mode",
        choices=("paired-alpha0", "stock"),
        default="paired-alpha0",
        help=(
            "paired-alpha0 runs the same full/unconditional/retain branch used by steering with alpha fixed to 0; "
            "stock keeps the ordinary unsteered two-branch APG path"
        ),
    )
    generation.add_argument("--steering-frac-start", type=float, default=0.3)
    generation.add_argument("--steering-frac-end", type=float, default=0.8)
    generation.add_argument("--num-seeds", type=int, default=1, help="independent baseline seeds per prompt")
    generation.add_argument("--seed", type=int, default=1000, help="first paired seed")

    classifier = parser.add_argument_group("instrument classifier")
    classifier.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_MODEL)
    classifier.add_argument("--classifier-revision", default=None)
    classifier.add_argument("--classifier-device", choices=("cpu", "cuda"), default="cpu")
    classifier.add_argument("--classifier-window-seconds", type=float, default=10.0)
    classifier.add_argument("--classifier-batch-size", type=int, default=8)
    classifier.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="sigmoid score threshold used only for presence/eligibility flags",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    runtime.add_argument("--no-half", action="store_true")

    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.num_inference_steps < 1:
        parser.error("--steps must be at least 1")
    if not 10.0 <= args.audio_length_in_s <= 600.0:
        parser.error("--audio-length-in-s must be in [10, 600]")
    if args.cfg_scale <= 1.0:
        parser.error("--cfg-scale must exceed 1.0 to match the steering evaluation baseline")
    if args.shift <= 0.0:
        parser.error("--shift must be positive")
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if not 0.0 <= args.detection_threshold <= 1.0:
        parser.error("--detection-threshold must be in [0, 1]")
    if args.classifier_window_seconds <= 0.0:
        parser.error("--classifier-window-seconds must be positive")
    if args.classifier_batch_size < 1:
        parser.error("--classifier-batch-size must be at least 1")
    if not 0.0 <= args.steering_frac_start < args.steering_frac_end <= 1.0:
        parser.error("steering fractions must satisfy 0.0 <= start < end <= 1.0")
    active_steps = sum(
        args.steering_frac_start <= step / args.num_inference_steps < args.steering_frac_end
        for step in range(args.num_inference_steps)
    )
    if args.baseline_mode == "paired-alpha0" and active_steps == 0:
        parser.error("the selected steering window contains no denoising step")
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA generation device requested but CUDA is unavailable")
    return device


def prepare_output_directory(path: Path, replace: bool) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise FileExistsError(
                f"Baseline output {path} is not empty. Choose another --output or pass --replace-output."
            )
        if path == Path.cwd().resolve() or path == path.parent or len(path.parts) < 3:
            raise ValueError(f"Refusing to replace unsafe output directory {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vocabulary = (
        InstrumentVocabulary.default()
        if args.instrument_config is None
        else InstrumentVocabulary.from_json(args.instrument_config)
    )
    samples = load_baseline_prompts(args.dataset, vocabulary, args.max_samples)
    generation_device = resolve_device(args.device)
    output_dir = prepare_output_directory(args.output, args.replace_output)

    config = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "dataset": str(args.dataset.resolve()),
        "output": str(output_dir),
        "num_prompts": len(samples),
        "num_seeds": args.num_seeds,
        "num_pairs": len(samples) * args.num_seeds,
        "first_seed": args.seed,
        "target_assignment": "dataset_before_generation_and_classification",
        "eligibility_rule": "target_score >= detection_threshold",
        "baseline_mode": args.baseline_mode.replace("-", "_"),
        "audio_serialization": {
            "format": "wav_pcm_s16le",
            "classifier_input": "saved_waveform",
        },
        "generation": {
            "model": args.model,
            "device": str(generation_device),
            "model_dtype": "bfloat16" if not args.no_half and generation_device.type == "cuda" else "float32",
            "num_inference_steps": args.num_inference_steps,
            "audio_length_in_s": args.audio_length_in_s,
            "cfg_scale": args.cfg_scale,
            "shift": args.shift,
            "steering_frac_start": args.steering_frac_start,
            "steering_frac_end": args.steering_frac_end,
        },
        "classifier": {
            "model": args.classifier_model,
            "requested_revision": args.classifier_revision,
            "resolved_revision": None,
            "device": args.classifier_device,
            "window_seconds": args.classifier_window_seconds,
            "window_aggregation": "maximum_sigmoid_score",
            "batch_size": args.classifier_batch_size,
            "detection_threshold": args.detection_threshold,
            "score_note": "independent sigmoid scores, not calibrated probabilities",
        },
        "instrument_vocabulary": vocabulary.as_json(),
    }
    write_config(output_dir / "config.json", config)

    print(
        f"Loading ACE-Step 1.5 SFT {args.model} on {generation_device} with "
        f"{config['generation']['model_dtype']} DiT precision"
    )
    pipeline = SteeringAceStepPipeline.from_pretrained(
        args.model,
        device=generation_device,
        model_half=not args.no_half,
    )
    pipeline.freeze_backbone()
    alpha_zero = FixedAlphaSteering(0.0).to(generation_device)

    print(f"Loading independent AudioSet classifier {args.classifier_model} on {args.classifier_device}")
    classifier = AudioSetInstrumentClassifier(
        vocabulary,
        model_name=args.classifier_model,
        revision=args.classifier_revision,
        device=args.classifier_device,
        window_seconds=args.classifier_window_seconds,
        batch_size=args.classifier_batch_size,
    )
    config["classifier"]["resolved_revision"] = classifier.revision
    config["classifier"]["sampling_rate"] = classifier.sampling_rate
    write_config(output_dir / "config.json", config)

    records_path = output_dir / "baseline_records.jsonl"
    csv_path = output_dir / "baseline_records.csv"
    eligible_path = output_dir / "eligible_pairs.csv"
    audio_dir = output_dir / "audio" / "baseline"
    total_pairs = len(samples) * args.num_seeds
    records: list[dict] = []

    with records_path.open("w", encoding="utf-8") as jsonl_file, csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file, eligible_path.open("w", newline="", encoding="utf-8") as eligible_file:
        writer = csv.DictWriter(csv_file, fieldnames=BASELINE_CSV_FIELDS)
        eligible_writer = csv.DictWriter(eligible_file, fieldnames=BASELINE_CSV_FIELDS)
        writer.writeheader()
        eligible_writer.writeheader()

        completed = 0
        for sample_id, sample in enumerate(samples):
            for seed_index in range(args.num_seeds):
                sample_seed = paired_seed(args.seed, len(samples), sample_id, seed_index)
                # Keep a CPU generator, exactly as scripts/evaluate.py does. Recreating it with this
                # seed in a future steering call produces the same initial latent noise.
                generator = torch.Generator().manual_seed(sample_seed)
                steering_kwargs = {}
                if args.baseline_mode == "paired-alpha0":
                    # A fixed-alpha controller ignores the embedding contents; only the batch axis
                    # is required. Enabling it deliberately exercises the same three-conditioning
                    # transformer branch as learned steering while remaining mathematically alpha=0.
                    steering_kwargs = {
                        "retain_prompt": sample.retain_prompt,
                        "target_embed": torch.zeros((1, 1), device=generation_device),
                        "steering_model": alpha_zero,
                        "steering_frac_start": args.steering_frac_start,
                        "steering_frac_end": args.steering_frac_end,
                    }
                with torch.inference_mode():
                    output = pipeline(
                        prompt=sample.prompt,
                        num_inference_steps=args.num_inference_steps,
                        audio_length_in_s=args.audio_length_in_s,
                        cfg_scale=args.cfg_scale,
                        shift=args.shift,
                        generator=generator,
                        output_type="pt",
                        **steering_kwargs,
                    )

                waveform = output.audios[0].detach().float().cpu().numpy()
                relative_audio_path = str(
                    Path("audio") / "baseline" / f"sample_{sample_id:04d}_seed_{sample_seed}.wav"
                )
                saved_audio_path = output_dir / relative_audio_path
                save_waveform(saved_audio_path, waveform, pipeline.sample_rate)
                # Selection scores must describe the artifact that later phases actually reuse.
                # Reloading also includes the small, deterministic PCM16 quantization performed by
                # save_waveform, avoiding a threshold decision on an unsaved in-memory variant.
                saved_waveform, saved_sampling_rate = load_waveform(saved_audio_path)
                scores = classifier.score(saved_waveform, saved_sampling_rate)
                record = build_baseline_record(
                    sample=sample,
                    sample_id=sample_id,
                    seed_index=seed_index,
                    seed=sample_seed,
                    audio_path=relative_audio_path,
                    instrument_scores=scores,
                    threshold=args.detection_threshold,
                    instrument_score_proxies={
                        name: spec.is_proxy for name, spec in vocabulary.specs.items()
                    },
                )
                records.append(record)
                csv_record = baseline_record_to_csv(record)
                jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
                writer.writerow(csv_record)
                if record["eligible_for_target_steering"]:
                    eligible_writer.writerow(csv_record)
                jsonl_file.flush()
                csv_file.flush()
                eligible_file.flush()

                completed += 1
                retain_status = (
                    "n/a" if record["all_retain_valid"] is None else str(record["all_retain_valid"]).lower()
                )
                print(
                    f"[{completed}/{total_pairs}] sample={sample_id} seed={sample_seed} "
                    f"target={sample.target} score={record['target_score']:.4f} "
                    f"eligible={str(record['target_valid']).lower()} retain_all={retain_status}",
                    flush=True,
                )

    target_valid = sum(bool(record["target_valid"]) for record in records)
    retain_applicable = [record for record in records if record["all_retain_valid"] is not None]
    by_target = {}
    for target in sorted({record["target"] for record in records}):
        target_records = [record for record in records if record["target"] == target]
        valid_count = sum(bool(record["target_valid"]) for record in target_records)
        by_target[target] = {
            "num_pairs": len(target_records),
            "num_valid": valid_count,
            "valid_rate": valid_count / len(target_records),
        }
    summary = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_mode": config["baseline_mode"],
        "num_pairs": len(records),
        "num_target_valid": target_valid,
        "target_valid_rate": target_valid / len(records),
        "num_with_retain_instruments": len(retain_applicable),
        "num_all_retain_valid": sum(bool(record["all_retain_valid"]) for record in retain_applicable),
        "by_target": by_target,
        "outputs": {
            "records_jsonl": records_path.name,
            "records_csv": csv_path.name,
            "eligible_pairs_csv": eligible_path.name,
            "audio_directory": str(audio_dir.relative_to(output_dir)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. {target_valid}/{len(records)} prompt-seed pairs are eligible; outputs written to {output_dir}")


if __name__ == "__main__":
    main()
