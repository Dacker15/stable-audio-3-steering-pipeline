import csv
import re
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_simple_splits import assign_splits, create_splits, read_dataset
from scripts.create_big_trumpet_dataset import RETAIN_INSTRUMENTS, STYLES, build_rows


class CreateSimpleSplitsTests(unittest.TestCase):
    def write_source_dataset(self, directory: str) -> tuple[Path, dict[str, str]]:
        path = Path(directory) / "simple.csv"
        retain_by_prompt: dict[str, str] = {}
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["prompt", "target", "retain_prompt"])
            writer.writeheader()
            for block_index in range(22):
                for pair_index in range(5):
                    genre = f"genre {block_index}"
                    accompaniment = f"instrument {pair_index} and drums"
                    variants = (
                        (
                            f"A {genre} piece with trumpet, {accompaniment}.",
                            f"A {genre} piece with {accompaniment}.",
                        ),
                        (
                            f"A slow {genre} track featuring trumpet with {accompaniment}.",
                            f"A slow {genre} track featuring {accompaniment}.",
                        ),
                    )
                    for prompt, retain in variants:
                        retain_by_prompt[prompt] = retain
                        writer.writerow({"prompt": prompt, "target": "trumpet", "retain_prompt": retain})
        return path, retain_by_prompt

    @staticmethod
    def read_split(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))

    def test_create_splits_has_expected_counts_and_keeps_pairs_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, retain_by_prompt = self.write_source_dataset(directory)
            output_dir = Path(directory) / "splits"

            manifest = create_splits(source, output_dir, seed=42)
            rows_by_split = {
                name: self.read_split(output_dir / filename)
                for name, filename in {
                    "train": "train.csv",
                    "validation": "validation.csv",
                    "test": "test.csv",
                }.items()
            }

            self.assertEqual(manifest["row_counts"], {"train": 132, "validation": 44, "test": 44})
            self.assertEqual(manifest["group_counts"], {"train": 66, "validation": 22, "test": 22})

            group_splits: dict[str, set[str]] = defaultdict(set)
            group_sizes: Counter[str] = Counter()
            block_split_groups: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
            for split_name, rows in rows_by_split.items():
                for row in rows:
                    self.assertEqual(row["retain_prompt"], retain_by_prompt[row["prompt"]])
                    self.assertEqual(row["split"], split_name)
                    group_splits[row["group_id"]].add(split_name)
                    group_sizes[row["group_id"]] += 1
                    block_index = int(re.search(r"genre (\d+)", row["prompt"]).group(1))
                    block_split_groups[block_index][split_name].add(row["group_id"])

            self.assertTrue(all(splits and len(splits) == 1 for splits in group_splits.values()))
            self.assertTrue(all(size == 2 for size in group_sizes.values()))
            for block_index in range(22):
                self.assertEqual(len(block_split_groups[block_index]["train"]), 3)
                self.assertEqual(len(block_split_groups[block_index]["validation"]), 1)
                self.assertEqual(len(block_split_groups[block_index]["test"]), 1)

    def test_same_seed_is_reproducible_and_different_seed_changes_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.write_source_dataset(directory)
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            third = Path(directory) / "third"

            create_splits(source, first, seed=7)
            create_splits(source, second, seed=7)
            create_splits(source, third, seed=8)

            first_train = (first / "train.csv").read_text(encoding="utf-8")
            self.assertEqual(first_train, (second / "train.csv").read_text(encoding="utf-8"))
            self.assertNotEqual(first_train, (third / "train.csv").read_text(encoding="utf-8"))

    def test_existing_outputs_require_overwrite_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.write_source_dataset(directory)
            output_dir = Path(directory) / "splits"
            create_splits(source, output_dir, seed=42)

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                create_splits(source, output_dir, seed=42)

            # Also exercise the input validator while the generated source still exists.
            fields, rows = read_dataset(source)
            self.assertEqual(fields, ["prompt", "target", "retain_prompt"])
            self.assertEqual(len(rows), 220)

    def test_mispaired_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.write_source_dataset(directory)
            _, rows = read_dataset(source)
            rows[1], rows[2] = rows[2], rows[1]

            with self.assertRaisesRegex(ValueError, "expected|accompaniment|genres"):
                assign_splits(rows, seed=42)

    def test_big_dataset_doubles_rows_and_keeps_exactly_two_declared_instruments(self) -> None:
        rows = build_rows()

        self.assertEqual(len(rows), 440)
        self.assertEqual(len(STYLES), 44)
        self.assertEqual(len({row["prompt"] for row in rows}), 440)
        for row in rows:
            self.assertNotRegex(row["prompt"], r"^A (?:ambient|americana|art|indie|alternative|upbeat)\b")
            self.assertEqual(row["target"], "trumpet")
            requested = row["requested_instruments"].split("; ")
            self.assertEqual(len(requested), 2)
            self.assertEqual(requested[0], "trumpet")
            self.assertIn(requested[1], RETAIN_INSTRUMENTS)
            self.assertEqual(row["retain_instruments"], requested[1])
            self.assertNotIn("trumpet", row["retain_prompt"].casefold())

        splits = assign_splits(rows, seed=42)
        self.assertEqual({name: len(split) for name, split in splits.items()}, {
            "train": 264,
            "validation": 88,
            "test": 88,
        })
        self.assertTrue(
            all({row["retain_instruments"] for row in split_rows} == set(RETAIN_INSTRUMENTS)
                for split_rows in splits.values())
        )
        group_splits: dict[str, set[str]] = defaultdict(set)
        for split_name, split_rows in splits.items():
            for row in split_rows:
                group_splits[row["group_id"]].add(split_name)
        self.assertTrue(all(len(names) == 1 for names in group_splits.values()))


if __name__ == "__main__":
    unittest.main()
