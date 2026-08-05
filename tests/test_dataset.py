import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dataset parsing itself only needs the Dataset base class. Keep these stdlib tests runnable in a
# lightweight checkout where the project's PyTorch dependency has not been installed yet.
if importlib.util.find_spec("torch") is None:
    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_module.__path__ = []
    torch_data_module = types.ModuleType("torch.utils.data")

    class TorchDatasetStub:
        pass

    torch_data_module.Dataset = TorchDatasetStub
    torch_utils_module.data = torch_data_module
    torch_module.utils = torch_utils_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils_module
    sys.modules["torch.utils.data"] = torch_data_module

from utils.dataset import PromptTargetDataset, strip_target


class PromptTargetDatasetTests(unittest.TestCase):
    def write_csv(self, directory: str, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
        path = Path(directory) / "dataset.csv"
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_explicit_retain_prompt_is_used_and_extra_split_columns_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "retain_prompt", "group_id", "split"],
                [
                    {
                        "prompt": "A jazz piece with trumpet, drums, and piano.",
                        "target": "trumpet",
                        "retain_prompt": "  A jazz piece with drums and piano.  ",
                        "group_id": "genre_00_pair_00",
                        "split": "train",
                    }
                ],
            )

            dataset = PromptTargetDataset(path)

            self.assertEqual(
                dataset[0],
                (
                    "A jazz piece with trumpet, drums, and piano.",
                    "trumpet",
                    "A jazz piece with drums and piano.",
                ),
            )
            self.assertNotEqual(dataset[0][2], strip_target(dataset[0][0], dataset[0][1]))

    def test_two_column_csv_falls_back_to_strip_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target"],
                [{"prompt": "Bright trumpet over drums.", "target": "trumpet"}],
            )

            dataset = PromptTargetDataset(path)

            self.assertEqual(dataset[0], ("Bright trumpet over drums.", "trumpet", "Bright over drums."))

    def test_empty_explicit_retain_prompt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "retain_prompt"],
                [{"prompt": "Trumpet and drums.", "target": "trumpet", "retain_prompt": ""}],
            )

            with self.assertRaisesRegex(ValueError, "empty `retain_prompt`"):
                PromptTargetDataset(path)

    def test_explicit_retain_prompt_cannot_still_mention_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "retain_prompt"],
                [
                    {
                        "prompt": "Trumpet and drums.",
                        "target": "trumpet",
                        "retain_prompt": "Quiet trumpets and drums.",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "still mentions target"):
                PromptTargetDataset(path)

    def test_non_positive_max_samples_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target"],
                [{"prompt": "Trumpet and drums.", "target": "trumpet"}],
            )

            with self.assertRaisesRegex(ValueError, "at least 1"):
                PromptTargetDataset(path, max_samples=0)


if __name__ == "__main__":
    unittest.main()
