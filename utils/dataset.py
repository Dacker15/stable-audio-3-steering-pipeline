import csv
import re
import warnings
from pathlib import Path

from torch.utils.data import Dataset


def strip_target(prompt: str, target: str) -> str:
    r"""
    Removes every mention of `target` from `prompt`, leaving everything else the prompt asks for.

    This is the text the retain loss is scored against: the concept being suppressed has to be gone
    from it, or the term would pull the audio back towards what the suppression term pushes it away
    from.

    The removal is lexical, so the modifiers that were attached to the target survive it and
    `"muted trumpet with a plunger mute"` becomes `"muted with a plunger mute"`. The result is not
    fluent English, it is what is left of the requested attributes, which is what CLAP scores.

    Args:
        prompt (`str`): The full prompt.
        target (`str`): The concept to remove. Matched case-insensitively, on word boundaries and
            with an optional plural or possessive suffix, so `"trumpet"` also removes `"Trumpets"`
            and `"trumpet's"`.

    Returns:
        `str`: `prompt` without the target, or `prompt` unchanged if the target does not occur in it
        or if removing it would leave nothing but punctuation.
    """
    words = target.split()
    if not words:
        return prompt

    pattern = r"\b" + r"\s+".join(re.escape(word) for word in words) + r"(?:'s|es|s)?\b"
    stripped = re.sub(pattern, "", prompt, flags=re.IGNORECASE)

    # clean up the debris the removal leaves: doubled spaces, punctuation that lost the word it
    # followed, and the separators of an enumeration that lost one of its items
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", stripped)
    stripped = stripped.strip(" ,;:-")

    if not re.search(r"\w", stripped):
        return prompt

    return stripped


class PromptTargetDataset(Dataset):
    r"""
    Reads the `prompt,target` CSV consumed by `SteeringMusicLDMPipeline`.

    Every row also carries the retain prompt derived by `strip_target`, i.e. the prompt with the
    target concept removed, so that the suppression and the retain side of the objective are built
    from one deterministic source instead of being re-derived per script.

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

        rows = rows if max_samples is None else rows[:max_samples]

        self.rows = [(prompt, target, strip_target(prompt, target)) for prompt, target in rows]

        # a retain prompt identical to the prompt means the target was never mentioned in it, so the
        # retain term would pull the audio towards the concept the suppression term pushes it away
        # from. worth surfacing once, since it is a property of the data rather than of a run
        unchanged = [index for index, (prompt, _, retain) in enumerate(self.rows) if retain == prompt]
        if unchanged:
            warnings.warn(
                f"{len(unchanged)} of {len(self.rows)} rows of {path} contain no mention of their target, so their"
                f" retain prompt is the full prompt and the retain loss there works against the suppression loss."
                f" First such row: index {unchanged[0]}, target {self.rows[unchanged[0]][1]!r}.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        return self.rows[index]


def collate_prompt_target(batch: list[tuple[str, str, str]]) -> tuple[list[str], list[str], list[str]]:
    r"""
    Collates into three `list`s of `str`. The default collate would produce tuples, while the pipeline
    validates `prompt` and `steering_target` against `list`.
    """
    return (
        [prompt for prompt, _, _ in batch],
        [target for _, target, _ in batch],
        [retain for _, _, retain in batch],
    )
