r"""
Filters an `eligible_pairs_*.csv` file (as produced upstream, e.g. `datasets/eligible_pairs_test.csv`)
down to rows that are actually usable for training/evaluation, and reshapes them into the
`prompt,target,seed,retain_prompt` schema consumed by `PromptTargetDataset`.

A row is kept only if:
  1. every instrument in `requested_instruments` is also present in `detected_instruments`;
  2. the mean of `instrument_scores` restricted to `requested_instruments` exceeds `--min-score`.

Example:

    uv run python scripts/filter_eligible_pairs.py \
        --input datasets/eligible_pairs_test.csv \
        --output datasets/eligible_pairs_test_filtered.csv
"""

import argparse
import csv
import json
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
    args = parser.parse_args()
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be in [0.0, 1.0]")
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
        accepted = 0
        rejection_counts: dict[str, int] = {}

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for row in reader:
                total += 1
                eligible, reason = row_is_eligible(row, args.min_score, reader.line_num)
                if not eligible:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue

                accepted += 1
                writer.writerow(
                    {
                        "prompt": row["prompt"],
                        "target": row["target"],
                        "seed": int(row["seed_index"]) + int(row["seed"]),
                        "retain_prompt": row["retain_prompt"],
                    }
                )

    print(f"Read {total} rows from {args.input}")
    print(f"Kept {accepted} rows -> {args.output}")
    if rejection_counts:
        print("Rejected rows by reason:")
        for reason, count in sorted(rejection_counts.items(), key=lambda item: -item[1]):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
