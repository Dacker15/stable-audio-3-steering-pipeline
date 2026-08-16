r"""Generate a reproducible 440-row trumpet dataset containing two instruments per prompt.

Every prompt describes a duet: the preassigned target ``trumpet`` plus exactly one explicitly
declared retain instrument. Rows are emitted as adjacent two-template semantic pairs, five pairs per
style, so ``scripts/create_simple_splits.py`` can keep near-duplicates in the same split.
"""

import argparse
import csv
from pathlib import Path


STYLES = (
    "jazz",
    "swing",
    "bebop",
    "cool jazz",
    "hard bop",
    "modal jazz",
    "free jazz",
    "blues",
    "rhythm and blues",
    "funk",
    "soul",
    "gospel",
    "pop",
    "indie pop",
    "dream pop",
    "art pop",
    "rock",
    "indie rock",
    "alternative rock",
    "garage rock",
    "psychedelic rock",
    "progressive rock",
    "post-rock",
    "surf rock",
    "folk",
    "indie folk",
    "country",
    "bluegrass",
    "americana",
    "reggae",
    "ska",
    "dub",
    "bossa nova",
    "samba",
    "salsa",
    "tango",
    "disco",
    "house",
    "techno",
    "trance",
    "ambient",
    "downtempo",
    "chillout",
    "cinematic",
)

# All names are exact, non-proxy canonical entries in the selector's AudioSet vocabulary.
RETAIN_INSTRUMENTS = (
    "drums",
    "piano",
    "acoustic guitar",
    "electric guitar",
    "double bass",
    "violin",
    "cello",
    "clarinet",
    "saxophone",
    "flute",
    "organ",
    "banjo",
    "harmonica",
    "accordion",
    "synthesizer",
)

MODIFIERS = ("slow", "relaxed", "mid-tempo", "upbeat", "fast")
FIELDNAMES = (
    "prompt",
    "target",
    "retain_prompt",
    "requested_instruments",
    "retain_instruments",
)


def with_indefinite_article(phrase: str) -> str:
    article = "An" if phrase[0].casefold() in "aeiou" else "A"
    return f"{article} {phrase}"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for style_index, style in enumerate(STYLES):
        # This rotation keeps five distinct groups inside every block and, with split seed 42,
        # leaves every retain instrument represented in train, validation and test.
        instrument_offset = (style_index + 4) % len(RETAIN_INSTRUMENTS)
        for pair_index, modifier in enumerate(MODIFIERS):
            instrument = RETAIN_INSTRUMENTS[(instrument_offset + pair_index) % len(RETAIN_INSTRUMENTS)]
            requested = f"trumpet; {instrument}"
            rows.extend(
                (
                    {
                        "prompt": (
                            f"{with_indefinite_article(style)} duet featuring a prominent trumpet and {instrument}."
                        ),
                        "target": "trumpet",
                        "retain_prompt": f"{with_indefinite_article(style)} piece with {instrument}.",
                        "requested_instruments": requested,
                        "retain_instruments": instrument,
                    },
                    {
                        "prompt": (
                            f"{with_indefinite_article(f'{modifier} {style}')} duet led by trumpet alongside "
                            f"{instrument}."
                        ),
                        "target": "trumpet",
                        "retain_prompt": (
                            f"{with_indefinite_article(f'{modifier} {style}')} track featuring {instrument}."
                        ),
                        "requested_instruments": requested,
                        "retain_instruments": instrument,
                    },
                )
            )

    expected_rows = len(STYLES) * len(MODIFIERS) * 2
    if len(rows) != expected_rows or len({row["prompt"] for row in rows}) != expected_rows:
        raise RuntimeError("internal big-dataset construction error")
    return rows


def write_dataset(path: Path, overwrite: bool = False) -> int:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 440-row, two-instrument trumpet prompt dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/trumpet_prompts_simple_dataset_big.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_dataset(args.output, args.overwrite)
    print(f"Created {count} two-instrument prompts in {args.output.resolve()}")


if __name__ == "__main__":
    main()
