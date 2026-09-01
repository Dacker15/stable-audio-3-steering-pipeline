import csv
import re
import warnings
from pathlib import Path

from torch.utils.data import Dataset


def _target_pattern(target: str) -> str | None:
    words = target.split()
    if not words:
        return None
    return r"\b" + r"\s+".join(re.escape(word) for word in words) + r"(?:'s|es|s)?\b"


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
    pattern = _target_pattern(target)
    if pattern is None:
        return prompt

    stripped = re.sub(pattern, "", prompt, flags=re.IGNORECASE)

    # normalize what the removal leaves behind: doubled spaces, punctuation that lost the word it
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
    Reads the prompt CSV consumed by `SteeringStableAudioPipeline`.

    A CSV may contain either `prompt,target` or `prompt,target,retain_prompt`. An explicit
    `retain_prompt` is used as written (apart from surrounding whitespace); the two-column format
    falls back to `strip_target` for backwards compatibility. If the column is present, every row
    must provide a non-empty retain prompt that does not still mention its target.

    The CSV may also contain a `seed` column. If present, every row must provide an integer seed,
    which callers should use to generate that row instead of a seed derived from a global base
    seed. If absent, every row's seed is `None`.

    Args:
        path (`Path`): Path to a CSV file with `prompt` and `target` columns and, optionally,
            `retain_prompt` and `seed` columns.
        max_samples (`int` or `None`, *optional*): Keep only the first `max_samples` rows. Useful for
            smoke tests, since one optimizer step costs a full generation.
    """

    def __init__(self, path: Path, max_samples: int | None = None):
        if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int)):
            raise TypeError(f"`max_samples` has to be an integer or `None` but is {type(max_samples).__name__}")
        if max_samples is not None and max_samples < 1:
            raise ValueError(f"`max_samples` has to be at least 1 but is {max_samples}")

        path = Path(path)
        with path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            duplicate_columns = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
            if duplicate_columns:
                raise ValueError(f"`dataset` {path} has duplicate columns {duplicate_columns}")

            missing = {"prompt", "target"} - set(fieldnames)
            if missing:
                raise ValueError(
                    f"`dataset` has to contain a `prompt` and a `target` column but {sorted(missing)} are missing from"
                    f" {path}, which has columns {reader.fieldnames}"
                )

            has_explicit_retain = "retain_prompt" in fieldnames
            has_explicit_seed = "seed" in fieldnames
            rows: list[tuple[str, str, str, int | None]] = []
            for row in reader:
                line_number = reader.line_num
                if None in row:
                    raise ValueError(
                        f"`dataset` {path} has values without a matching header at CSV line {line_number}"
                    )

                prompt = (row.get("prompt") or "").strip()
                target = (row.get("target") or "").strip()
                if not prompt or not target:
                    missing_values = [name for name, value in (("prompt", prompt), ("target", target)) if not value]
                    raise ValueError(
                        f"`dataset` {path} has empty required value(s) {missing_values} at CSV line {line_number}"
                    )

                if has_explicit_retain:
                    retain = (row.get("retain_prompt") or "").strip()
                    if not retain or not re.search(r"\w", retain):
                        raise ValueError(
                            f"`dataset` {path} has an empty `retain_prompt` at CSV line {line_number}"
                        )

                    target_pattern = _target_pattern(target)
                    if target_pattern is not None and re.search(target_pattern, retain, flags=re.IGNORECASE):
                        raise ValueError(
                            f"`dataset` {path} has a `retain_prompt` that still mentions target {target!r} at CSV"
                            f" line {line_number}: {retain!r}"
                        )
                else:
                    retain = strip_target(prompt, target)

                if has_explicit_seed:
                    seed_value = (row.get("seed") or "").strip()
                    try:
                        seed = int(seed_value)
                    except ValueError:
                        raise ValueError(
                            f"`dataset` {path} has a non-integer `seed` {seed_value!r} at CSV line {line_number}"
                        ) from None
                else:
                    seed = None

                rows.append((prompt, target, retain, seed))

        if not rows:
            raise ValueError(f"`dataset` has to contain at least one non-empty row but {path} contains none")

        rows = rows if max_samples is None else rows[:max_samples]
        self.rows = rows

        # An identical retain prompt means that the row supplied no lexical evidence that the target
        # was removed. Keep this a warning: an explicit retain may legitimately remove a synonym
        # that cannot be inferred from the canonical target string.
        unchanged = [index for index, (prompt, _, retain, _) in enumerate(self.rows) if retain == prompt]
        if unchanged:
            warnings.warn(
                f"{len(unchanged)} of {len(self.rows)} selected rows of {path} have a retain prompt identical to the"
                " full prompt, so the retain and suppression objectives may conflict."
                f" First such row: index {unchanged[0]}, target {self.rows[unchanged[0]][1]!r}.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str, str, int | None]:
        return self.rows[index]


def collate_prompt_target(
    batch: list[tuple[str, str, str, int | None]],
) -> tuple[list[str], list[str], list[str], list[int | None]]:
    r"""
    Collates into four `list`s. The default collate would produce tuples, while the pipeline
    validates `prompt` and `steering_target` against `list`.
    """
    return (
        [prompt for prompt, _, _, _ in batch],
        [target for _, target, _, _ in batch],
        [retain for _, _, retain, _ in batch],
        [seed for _, _, _, seed in batch],
    )
