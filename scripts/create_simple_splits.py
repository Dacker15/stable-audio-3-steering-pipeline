r"""Create reproducible group-aware splits for the simplified trumpet prompt dataset.

The simplified dataset is ordered as 22 genre blocks of 10 rows. Within each block, adjacent rows
form five prompt pairs that share genre and accompaniment but use two different templates. This
script keeps each pair in one split and assigns, per genre block, three pairs to train, one to
validation and one to test. The resulting row counts are 132/44/44.

Example:
    python scripts/create_simple_splits.py \
        --input datasets/trumpet_prompts_simple_dataset.csv \
        --output-dir datasets/trumpet_simple_splits --seed 42
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


EXPECTED_ROWS = 220
BLOCK_SIZE = 10
PAIR_SIZE = 2
GROUPS_PER_BLOCK = BLOCK_SIZE // PAIR_SIZE
SPLIT_GROUP_COUNTS = {"train": 3, "validation": 1, "test": 1}
OUTPUT_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
ANNOTATION_COLUMNS = ("group_id", "split")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = {"prompt", "target", "retain_prompt"} - set(fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns {sorted(missing)}")

        rows: list[dict[str, str]] = []
        for row in reader:
            line_number = reader.line_num
            if None in row:
                raise ValueError(f"{path} has values without a matching header at CSV line {line_number}")

            normalized = {name: (value or "").strip() for name, value in row.items()}
            empty = [name for name in ("prompt", "target", "retain_prompt") if not normalized[name]]
            if empty:
                raise ValueError(f"{path} has empty required value(s) {empty} at CSV line {line_number}")
            rows.append(normalized)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"the simplified dataset must contain exactly {EXPECTED_ROWS} rows but {path} contains {len(rows)}"
        )
    if len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate prompts; pair grouping would be ambiguous")

    return fieldnames, rows


def _normalize_template_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.casefold()).strip()
    return text.rstrip(" .")


def _without_article(text: str) -> str:
    return re.sub(r"^(?:a|an)\s+", "", text, count=1)


def _pair_signature(pair: list[dict[str, str]], block_index: int, pair_index: int) -> tuple[str, str]:
    """Validates the two simplified templates and returns `(genre, accompaniment)`."""

    if len(pair) != PAIR_SIZE:
        raise ValueError(f"pair {pair_index} of genre block {block_index} has {len(pair)} rows")

    piece = _without_article(_normalize_template_text(pair[0]["retain_prompt"]))
    track = _without_article(_normalize_template_text(pair[1]["retain_prompt"]))
    piece_marker = " piece with "
    track_marker = " track featuring "
    if piece_marker not in piece or track_marker not in track:
        raise ValueError(
            f"pair {pair_index} of genre block {block_index} does not follow the expected "
            "`<genre> piece with ...` / `<modifier> <genre> track featuring ...` templates"
        )

    genre, piece_accompaniment = piece.split(piece_marker, maxsplit=1)
    track_prefix, track_accompaniment = track.split(track_marker, maxsplit=1)
    if not genre or not piece_accompaniment:
        raise ValueError(f"pair {pair_index} of genre block {block_index} has an empty template component")
    if track_prefix != genre and not track_prefix.endswith(f" {genre}"):
        raise ValueError(
            f"pair {pair_index} of genre block {block_index} has inconsistent genres: "
            f"{genre!r} and {track_prefix!r}"
        )
    if piece_accompaniment != track_accompaniment:
        raise ValueError(
            f"pair {pair_index} of genre block {block_index} does not share the same accompaniment: "
            f"{piece_accompaniment!r} != {track_accompaniment!r}"
        )

    return genre, piece_accompaniment


def _group_id(genre: str, accompaniment: str) -> str:
    genre_slug = re.sub(r"[^a-z0-9]+", "_", genre).strip("_")
    digest = hashlib.sha256(f"{genre}\0{accompaniment}".encode("utf-8")).hexdigest()[:10]
    return f"{genre_slug}_{digest}"


def _ranked_pair_indices(seed: int, group_ids: list[str]) -> list[int]:
    """Ranks groups with a stable SHA-256 key instead of version-dependent random shuffling."""

    def rank(pair_index: int) -> bytes:
        return hashlib.sha256(f"{seed}:{group_ids[pair_index]}".encode("utf-8")).digest()

    return sorted(range(len(group_ids)), key=rank)


def assign_splits(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows but received {len(rows)}")

    splits: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_GROUP_COUNTS}
    num_blocks = EXPECTED_ROWS // BLOCK_SIZE

    for block_index in range(num_blocks):
        block_start = block_index * BLOCK_SIZE
        block = rows[block_start : block_start + BLOCK_SIZE]
        block_targets = {row["target"] for row in block}
        if len(block_targets) != 1:
            raise ValueError(
                f"genre block {block_index} contains multiple targets {sorted(block_targets)}; its grouping is invalid"
            )

        pairs = [block[index : index + PAIR_SIZE] for index in range(0, BLOCK_SIZE, PAIR_SIZE)]
        signatures = [_pair_signature(pair, block_index, pair_index) for pair_index, pair in enumerate(pairs)]
        block_genres = {genre for genre, _ in signatures}
        if len(block_genres) != 1:
            raise ValueError(
                f"genre block {block_index} contains multiple parsed genres {sorted(block_genres)}"
            )
        group_ids = [_group_id(*signature) for signature in signatures]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(f"genre block {block_index} contains duplicate semantic groups")

        ranked_pairs = _ranked_pair_indices(seed, group_ids)
        split_by_pair: dict[int, str] = {}
        cursor = 0
        for split_name, group_count in SPLIT_GROUP_COUNTS.items():
            for pair_index in ranked_pairs[cursor : cursor + group_count]:
                split_by_pair[pair_index] = split_name
            cursor += group_count

        for pair_index in range(GROUPS_PER_BLOCK):
            pair = pairs[pair_index]
            if len({row["target"] for row in pair}) != 1:
                raise ValueError(f"pair {pair_index} of genre block {block_index} contains inconsistent targets")

            split_name = split_by_pair[pair_index]
            group_id = group_ids[pair_index]
            for row in pair:
                annotated = dict(row)
                annotated["group_id"] = group_id
                annotated["split"] = split_name
                splits[split_name].append(annotated)

    expected_counts = {"train": 132, "validation": 44, "test": 44}
    actual_counts = {name: len(split_rows) for name, split_rows in splits.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(f"internal split-count error: expected {expected_counts}, produced {actual_counts}")

    return splits


def write_splits(
    input_path: Path,
    output_dir: Path,
    fieldnames: list[str],
    splits: dict[str, list[dict[str, str]]],
    seed: int,
    overwrite: bool,
) -> dict:
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    manifest_path = output_dir / "split_manifest.json"
    existing = [path for path in (*output_paths.values(), manifest_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to replace existing split output(s): {joined}; pass --overwrite to replace them"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_fieldnames = [name for name in fieldnames if name not in ANNOTATION_COLUMNS] + list(ANNOTATION_COLUMNS)
    for split_name, output_path in output_paths.items():
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(splits[split_name])

    input_hash = sha256_file(input_path)
    manifest = {
        "input": str(input_path.resolve()),
        "input_sha256": input_hash,
        "seed": seed,
        "strategy": "22 genre blocks; validated semantic pairs kept together; 3/1/1 pairs per block",
        "block_size": BLOCK_SIZE,
        "pair_size": PAIR_SIZE,
        "row_counts": {name: len(rows) for name, rows in splits.items()},
        "group_counts": {
            name: len({row["group_id"] for row in rows}) for name, rows in splits.items()
        },
        "files": {name: path.name for name, path in output_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def create_splits(input_path: Path, output_dir: Path, seed: int, overwrite: bool = False) -> dict:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    fieldnames, rows = read_dataset(input_path)
    splits = assign_splits(rows, seed)
    return write_splits(input_path, output_dir, fieldnames, splits, seed, overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic group-aware train/validation/test splits for the simplified dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/trumpet_prompts_simple_dataset.csv"),
        help="simplified 220-row CSV with prompt, target and retain_prompt columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/trumpet_simple_splits"),
        help="directory that will receive train.csv, validation.csv, test.csv and the manifest",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed used for stable pair assignment")
    parser.add_argument("--overwrite", action="store_true", help="replace existing split outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_splits(args.input, args.output_dir, args.seed, args.overwrite)
    counts = manifest["row_counts"]
    print(
        f"Created train/validation/test splits with {counts['train']}/{counts['validation']}/{counts['test']} rows "
        f"in {args.output_dir.resolve()} (input SHA-256: {manifest['input_sha256']})"
    )


if __name__ == "__main__":
    main()
