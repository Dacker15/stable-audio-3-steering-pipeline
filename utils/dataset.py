import csv
from pathlib import Path

from torch.utils.data import Dataset


class PromptTargetDataset(Dataset):
    r"""
    Reads the `prompt,target` CSV consumed by `SteeringMusicLDMPipeline`.

    Args:
        path (`Path`): Path to a CSV file with a `prompt` and a `target` column.
        max_samples (`int` or `None`, *optional*): Keep only the first `max_samples` rows. Useful for
            smoke tests, since one optimizer step costs a full generation.
    """

    def __init__(self, path: Path, max_samples: int | None = None):
        with open(path, newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = {"prompt", "target"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"`dataset` has to contain a `prompt` and a `target` column but {sorted(missing)} are missing from"
                    f" {path}, which has columns {reader.fieldnames}"
                )
            rows = [
                (row["prompt"].strip(), row["target"].strip())
                for row in reader
                if (row.get("prompt") or "").strip() and (row.get("target") or "").strip()
            ]

        if not rows:
            raise ValueError(f"`dataset` has to contain at least one non-empty row but {path} contains none")

        self.rows = rows if max_samples is None else rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.rows[index]


def collate_prompt_target(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    r"""
    Collates into two `list`s of `str`. The default collate would produce tuples, while the pipeline
    validates `prompt` and `steering_target` against `list`.
    """
    return [prompt for prompt, _ in batch], [target for _, target in batch]