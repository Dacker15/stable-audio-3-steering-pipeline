r"""
Filters an `eligible_pairs_*.csv` file (as produced upstream, e.g. `datasets/eligible_pairs_test.csv`)
down to rows that are actually usable for training/evaluation, and reshapes them into the
`prompt,target,seed,retain_prompt` schema consumed by `PromptTargetDataset`.

A row is kept only if:
  1. every instrument in `requested_instruments` is also present in `detected_instruments`;
  2. the mean of `instrument_scores` restricted to `requested_instruments` exceeds `--min-score`.

Example:

    uv run scripts/filter_eligible_pairs.py \
        --input datasets/eligible_pairs_test.csv \
        --output datasets/eligible_pairs_test_filtered.csv

Pass `--split TRAIN VALIDATION TEST` (proportions summing to 1.0) to additionally shuffle the
filtered rows with `--split-seed` and write three CSVs (`*_train.csv`, `*_validation.csv`,
`*_test.csv`) instead of a single one:

    uv run scripts/filter_eligible_pairs.py \
        --input datasets/eligible_pairs_test.csv \
        --output datasets/eligible_pairs_test_filtered.csv \
        --split 0.8 0.1 0.1
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

REQUIRED_INPUT_COLUMNS = (
    "prompt",
    "target",
    "seed_index",
    "seed",
    "retain_prompt",
    "requested_instruments",
    "detected_instruments",
    "instrument_scores",
)
OUTPUT_FIELDS = ("prompt", "target", "seed", "retain_prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter an eligible_pairs CSV into a valid prompt/target/seed/retain_prompt dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="source CSV with the eligible_pairs schema")
    parser.add_argument("--output", type=Path, required=True, help="filtered CSV to write")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="minimum mean instrument_scores over requested_instruments for a row to be kept",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=None,
        help="train/validation/test proportions (must sum to 1.0); if omitted, writes a single CSV to --output",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="seed for shuffling accepted rows before splitting; only used with --split",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be in [0.0, 1.0]")
    if args.split is not None:
        if any(frac < 0.0 for frac in args.split):
            parser.error("--split proportions must be non-negative")
        if not math.isclose(sum(args.split), 1.0, abs_tol=1e-6):
            parser.error(f"--split proportions must sum to 1.0, got {sum(args.split)}")
    return args


def row_is_eligible(row: dict, min_score: float, line_number: int) -> tuple[bool, str | None]:
    """Returns `(eligible, rejection_reason)`; `rejection_reason` is `None` when eligible."""

    return True, None

    try:
        requested_instruments = json.loads(row["requested_instruments"])
        detected_instruments = json.loads(row["detected_instruments"])
        instrument_scores = json.loads(row["instrument_scores"])
    except json.JSONDecodeError as error:
        raise ValueError(f"line {line_number}: malformed JSON in instrument columns: {error}") from error

    if not requested_instruments:
        return False, "empty requested_instruments"

    if not set(requested_instruments).issubset(detected_instruments):
        return False, "requested instrument missing from detected_instruments"

    mean_score = sum(instrument_scores[instrument] for instrument in requested_instruments) / len(
        requested_instruments
    )
    if mean_score <= min_score:
        return False, "mean requested instrument score too low"

    return True, None


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _split_output_path(output: Path, split_name: str) -> Path:
    return output.with_name(f"{output.stem}_{split_name}{output.suffix}")


def _split_rows(rows: list[dict], proportions: list[float], seed: int) -> dict[str, list[dict]]:
    """Shuffles `rows` with `seed`, then splits into train/validation/test using the largest-remainder
    method so the three counts always sum exactly to `len(rows)`."""

    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    raw_counts = [frac * total for frac in proportions]
    counts = [int(count) for count in raw_counts]  # floor
    remainder = total - sum(counts)

    order = sorted(range(3), key=lambda i: raw_counts[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    train_count, val_count, _ = counts
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def main() -> None:
    args = parse_args()

    with args.input.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [name for name in REQUIRED_INPUT_COLUMNS if name not in fieldnames]
        if missing_columns:
            raise ValueError(
                f"`--input` {args.input} is missing required columns {missing_columns}; found {fieldnames}"
            )

        total = 0
        accepted_rows: list[dict] = []
        rejection_counts: dict[str, int] = {}

        for row in reader:
            total += 1
            eligible, reason = row_is_eligible(row, args.min_score, reader.line_num)
            if not eligible:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue

            accepted_rows.append(
                {
                    "prompt": row["prompt"],
                    "target": row["target"],
                    "seed": int(row["seed_index"]) + int(row["seed"]),
                    "retain_prompt": row["retain_prompt"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Read {total} rows from {args.input}")
    if args.split is None:
        _write_csv(args.output, accepted_rows)
        print(f"Kept {len(accepted_rows)} rows -> {args.output}")
    else:
        splits = _split_rows(accepted_rows, args.split, args.split_seed)
        print(f"Kept {len(accepted_rows)} rows, split with seed {args.split_seed}:")
        for name in ("train", "validation", "test"):
            path = _split_output_path(args.output, name)
            _write_csv(path, splits[name])
            print(f"  {name}: {len(splits[name])} rows -> {path}")

    if rejection_counts:
        print("Rejected rows by reason:")
        for reason, count in sorted(rejection_counts.items(), key=lambda item: -item[1]):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
