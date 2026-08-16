import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from utils.dataset import strip_target


DEFAULT_CLASSIFIER_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
BASELINE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class InstrumentSpec:
    classifier_labels: tuple[str, ...]
    aliases: tuple[str, ...]
    is_proxy: bool = False


# AudioSet has exact labels for the most common instruments and only family labels for a few
# instruments used by the repository's prompt datasets. The latter remain separate canonical
# concepts, but their classifier_labels make the coarser proxy explicit in config.json.
DEFAULT_INSTRUMENT_SPECS: dict[str, InstrumentSpec] = {
    "accordion": InstrumentSpec(("Accordion",), ("accordion",)),
    "acoustic guitar": InstrumentSpec(
        ("Acoustic guitar",),
        ("acoustic guitar", "nylon-string guitar", "nylon string guitar"),
    ),
    "banjo": InstrumentSpec(("Banjo",), ("banjo",)),
    "bagpipes": InstrumentSpec(("Bagpipes",), ("bagpipe", "bagpipes")),
    "bass": InstrumentSpec(("Bass guitar", "Double bass", "Synthesizer"), ("deep bass", "bassline"), True),
    "bass drum": InstrumentSpec(("Bass drum",), ("bass drum",)),
    "bassoon": InstrumentSpec(("Wind instrument, woodwind instrument",), ("bassoon",), True),
    "bell": InstrumentSpec(("Bell",), ("bell", "bells")),
    "brass": InstrumentSpec(("Brass instrument",), ("brass", "brass instrument")),
    "brass section": InstrumentSpec(("Brass instrument",), ("brass section", "horn section"), True),
    "cello": InstrumentSpec(("Cello",), ("cello", "cellos")),
    "clarinet": InstrumentSpec(("Clarinet",), ("clarinet", "clarinets")),
    "chime": InstrumentSpec(("Chime",), ("chime", "chimes")),
    "church bell": InstrumentSpec(("Church bell",), ("church bell", "church bells")),
    "congas": InstrumentSpec(("Percussion", "Drum"), ("conga", "congas"), True),
    "cymbal": InstrumentSpec(("Cymbal",), ("cymbal", "cymbals")),
    "didgeridoo": InstrumentSpec(("Didgeridoo",), ("didgeridoo", "didgeridoos")),
    "double bass": InstrumentSpec(
        ("Double bass",),
        ("double bass", "upright bass", "acoustic bass", "walking bass"),
    ),
    "drum machine": InstrumentSpec(
        ("Drum machine", "Sampler"),
        ("drum machine", "drum machines", "sampled drums"),
    ),
    "drums": InstrumentSpec(
        ("Drum kit", "Drum"),
        ("drum kit", "drum kits", "drums", "brushed drums"),
    ),
    "electric bass": InstrumentSpec(("Bass guitar",), ("electric bass", "bass guitar")),
    "electric guitar": InstrumentSpec(
        ("Electric guitar",),
        ("electric guitar", "electric guitars", "distorted electric guitar"),
    ),
    "electronic organ": InstrumentSpec(("Electronic organ",), ("electronic organ", "electronic organs")),
    "electric piano": InstrumentSpec(("Electric piano",), ("electric piano", "electric pianos")),
    "flute": InstrumentSpec(("Flute",), ("flute", "flutes")),
    "french horn": InstrumentSpec(("French horn",), ("french horn", "french horns")),
    "guitar": InstrumentSpec(("Guitar",), ("guitar", "guitars", "rhythm guitar", "lead guitar")),
    "glockenspiel": InstrumentSpec(("Glockenspiel",), ("glockenspiel", "glockenspiels")),
    "gong": InstrumentSpec(("Gong",), ("gong", "gongs")),
    "hammond organ": InstrumentSpec(("Hammond organ",), ("hammond organ", "hammond organs")),
    "harmonica": InstrumentSpec(("Harmonica",), ("harmonica", "harmonicas")),
    "harp": InstrumentSpec(("Harp",), ("harp", "harps")),
    "harpsichord": InstrumentSpec(("Harpsichord",), ("harpsichord", "harpsichords")),
    "hi-hat": InstrumentSpec(("Hi-hat",), ("hi-hat", "hi-hats", "hi hat", "hi hats")),
    "keyboard": InstrumentSpec(("Keyboard (musical)",), ("keyboard", "keyboards")),
    "mandolin": InstrumentSpec(("Mandolin",), ("mandolin", "mandolins")),
    "mallet percussion": InstrumentSpec(("Mallet percussion",), ("mallet percussion",)),
    "maraca": InstrumentSpec(("Maraca",), ("maraca", "maracas")),
    "marimba/xylophone": InstrumentSpec(
        ("Marimba, xylophone",),
        ("marimba", "marimbas", "xylophone", "xylophones"),
    ),
    "oboe": InstrumentSpec(("Wind instrument, woodwind instrument",), ("oboe", "oboes"), True),
    "orchestra": InstrumentSpec(("Orchestra",), ("orchestra", "orchestras")),
    "organ": InstrumentSpec(("Organ",), ("organ", "organs")),
    "percussion": InstrumentSpec(
        ("Percussion",),
        ("percussion", "hand percussion", "soft percussion", "atmospheric percussion"),
    ),
    "piano": InstrumentSpec(
        ("Piano",),
        ("piano", "pianos", "acoustic piano", "acoustic grand piano", "grand piano"),
    ),
    "plucked strings": InstrumentSpec(
        ("Plucked string instrument",),
        ("plucked strings", "plucked string instrument"),
    ),
    "rattle": InstrumentSpec(("Rattle (instrument)",), ("rattle", "rattles")),
    "sampler": InstrumentSpec(("Sampler",), ("sampler", "samplers")),
    "saxophone": InstrumentSpec(
        ("Saxophone",),
        ("saxophone", "saxophones", "tenor saxophone", "alto saxophone", "sax"),
    ),
    "slide guitar": InstrumentSpec(
        ("Steel guitar, slide guitar",),
        ("slide guitar", "steel guitar", "steel guitars"),
    ),
    "shofar": InstrumentSpec(("Shofar",), ("shofar", "shofars")),
    "singing bowl": InstrumentSpec(("Singing bowl",), ("singing bowl", "singing bowls")),
    "sitar": InstrumentSpec(("Sitar",), ("sitar", "sitars")),
    "snare drum": InstrumentSpec(("Snare drum",), ("snare drum", "snare drums")),
    "strings": InstrumentSpec(
        ("String section", "Bowed string instrument"),
        (
            "strings",
            "string section",
            "string ensemble",
            "string orchestra",
            "string quartet",
            "bowed strings",
            "sustained strings",
        ),
    ),
    "steelpan": InstrumentSpec(("Steelpan",), ("steelpan", "steelpans", "steel pan", "steel pans")),
    "synthesizer": InstrumentSpec(("Synthesizer",), ("synthesizer", "synthesizers", "synth")),
    "tabla": InstrumentSpec(("Tabla",), ("tabla", "tablas")),
    "tambourine": InstrumentSpec(("Tambourine",), ("tambourine", "tambourines")),
    "taiko drums": InstrumentSpec(("Drum", "Percussion"), ("taiko drum", "taiko drums"), True),
    "timbales": InstrumentSpec(("Drum", "Percussion"), ("timbale", "timbales"), True),
    "timpani": InstrumentSpec(("Timpani",), ("timpani",)),
    "theremin": InstrumentSpec(("Theremin",), ("theremin", "theremins")),
    "trombone": InstrumentSpec(("Trombone",), ("trombone", "trombones")),
    "trumpet": InstrumentSpec(("Trumpet",), ("trumpet", "trumpets")),
    "tuba": InstrumentSpec(("Brass instrument",), ("tuba", "tubas"), True),
    "tubular bells": InstrumentSpec(("Tubular bells",), ("tubular bell", "tubular bells")),
    "tuning fork": InstrumentSpec(("Tuning fork",), ("tuning fork", "tuning forks")),
    "ukulele": InstrumentSpec(("Ukulele",), ("ukulele", "ukuleles")),
    "vibraphone": InstrumentSpec(("Vibraphone",), ("vibraphone", "vibraphones")),
    "violin": InstrumentSpec(("Violin, fiddle",), ("violin", "violins", "fiddle", "fiddles")),
    "wind chime": InstrumentSpec(("Wind chime",), ("wind chime", "wind chimes")),
    "wood block": InstrumentSpec(("Wood block",), ("wood block", "wood blocks")),
    "woodwind": InstrumentSpec(
        ("Wind instrument, woodwind instrument",),
        ("woodwind", "woodwinds", "woodwind instrument", "wind instrument"),
    ),
    "zither": InstrumentSpec(("Zither",), ("zither", "zithers")),
}


@dataclass(frozen=True)
class BaselinePrompt:
    prompt: str
    target: str
    retain_prompt: str
    requested_instruments: tuple[str, ...]
    retain_instruments: tuple[str, ...]
    source_metadata: dict[str, str]


@dataclass(frozen=True)
class SelectedBaselinePair:
    record: dict
    audio_path: Path

    @property
    def pair_id(self) -> str:
        return str(self.record["pair_id"])


@dataclass(frozen=True)
class BaselineSelection:
    root: Path
    records_path: Path
    config: dict
    pairs: tuple[SelectedBaselinePair, ...]


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold().replace("_", " "))


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class InstrumentVocabulary:
    def __init__(self, specs: Mapping[str, InstrumentSpec]):
        if not specs:
            raise ValueError("instrument vocabulary cannot be empty")

        normalized_specs: dict[str, InstrumentSpec] = {}
        alias_to_name: dict[str, str] = {}
        for raw_name, raw_spec in specs.items():
            name = _normalize_name(raw_name)
            labels = _unique(tuple(label.strip() for label in raw_spec.classifier_labels if label.strip()))
            aliases = _unique(
                tuple(_normalize_name(alias) for alias in (name, *raw_spec.aliases) if _normalize_name(alias))
            )
            if not name or not labels or not aliases:
                raise ValueError(f"invalid instrument specification for {raw_name!r}")
            if name in normalized_specs:
                raise ValueError(f"duplicate canonical instrument {name!r}")
            normalized_specs[name] = InstrumentSpec(labels, aliases, bool(raw_spec.is_proxy))
            for alias in aliases:
                previous = alias_to_name.get(alias)
                if previous is not None and previous != name:
                    raise ValueError(f"instrument alias {alias!r} is shared by {previous!r} and {name!r}")
                alias_to_name[alias] = name

        self.specs = dict(sorted(normalized_specs.items()))
        self._alias_to_name = alias_to_name
        self._patterns = sorted(
            (
                (
                    alias,
                    canonical,
                    re.compile(r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"[\s-]+") + r"(?!\w)", re.I),
                )
                for alias, canonical in alias_to_name.items()
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    @classmethod
    def default(cls) -> "InstrumentVocabulary":
        return cls(DEFAULT_INSTRUMENT_SPECS)

    @classmethod
    def from_json(cls, path: Path) -> "InstrumentVocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("instrument config must be a JSON object keyed by canonical instrument name")
        specs: dict[str, InstrumentSpec] = {}
        for name, value in data.items():
            if not isinstance(value, dict):
                raise ValueError(f"instrument config entry {name!r} must be an object")
            labels = value.get("classifier_labels")
            aliases = value.get("aliases", [])
            is_proxy = value.get("is_proxy", False)
            if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
                raise ValueError(f"instrument config entry {name!r} needs a string list `classifier_labels`")
            if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
                raise ValueError(f"instrument config entry {name!r} needs a string list `aliases`")
            if not isinstance(is_proxy, bool):
                raise ValueError(f"instrument config entry {name!r} needs a boolean `is_proxy`")
            specs[name] = InstrumentSpec(tuple(labels), tuple(aliases), is_proxy)
        return cls(specs)

    def resolve(self, value: str) -> str:
        normalized = _normalize_name(value)
        try:
            return self._alias_to_name[normalized]
        except KeyError as error:
            raise ValueError(
                f"unknown instrument {value!r}; use a canonical name/alias from the configured vocabulary"
            ) from error

    def extract(self, text: str) -> tuple[str, ...]:
        # Prefer the longest non-overlapping mention, so "electric guitar" does not also become
        # generic "guitar", and return concepts in prompt order.
        occupied: list[tuple[int, int]] = []
        matches: list[tuple[int, str]] = []
        for _, canonical, pattern in self._patterns:
            for match in pattern.finditer(text):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                occupied.append(span)
                matches.append((span[0], canonical))
        return _unique(tuple(name for _, name in sorted(matches)))

    def as_json(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "classifier_labels": list(spec.classifier_labels),
                "aliases": list(spec.aliases),
                "is_proxy": spec.is_proxy,
            }
            for name, spec in self.specs.items()
        }


def parse_instrument_list(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value:
        return ()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("instrument list JSON must be an array of strings")
        return _unique(tuple(item.strip() for item in parsed if item.strip()))
    return _unique(tuple(item.strip() for item in re.split(r"[;,]", value) if item.strip()))


def _resolve_list(values: Sequence[str], vocabulary: InstrumentVocabulary) -> tuple[str, ...]:
    return _unique(tuple(vocabulary.resolve(value) for value in values))


def load_baseline_prompts(
    path: Path,
    vocabulary: InstrumentVocabulary,
    max_samples: int | None = None,
) -> list[BaselinePrompt]:
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        duplicate_columns = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
        if duplicate_columns:
            raise ValueError(f"dataset {path} has duplicate columns {duplicate_columns}")
        missing = {"prompt", "target"} - set(fieldnames)
        if missing:
            raise ValueError(f"dataset {path} is missing required columns {sorted(missing)}")

        has_retain_prompt = "retain_prompt" in fieldnames
        has_requested = "requested_instruments" in fieldnames
        has_retain_instruments = "retain_instruments" in fieldnames
        reserved = {
            "prompt",
            "target",
            "retain_prompt",
            "requested_instruments",
            "retain_instruments",
        }
        rows: list[BaselinePrompt] = []
        for row in reader:
            line_number = reader.line_num
            if None in row:
                raise ValueError(f"dataset {path} has values without a header at CSV line {line_number}")
            prompt = (row.get("prompt") or "").strip()
            raw_target = (row.get("target") or "").strip()
            if not prompt or not raw_target:
                raise ValueError(f"dataset {path} has an empty prompt or target at CSV line {line_number}")

            # This resolution happens while loading the dataset, before any audio is generated or
            # classified. Classification is therefore incapable of choosing or changing the target.
            try:
                target = vocabulary.resolve(raw_target)
            except ValueError as error:
                raise ValueError(f"dataset {path}, CSV line {line_number}: {error}") from error
            if vocabulary.specs[target].is_proxy:
                raise ValueError(
                    f"dataset {path}, CSV line {line_number}: target {target!r} has only a coarse AudioSet proxy;"
                    " configure an exact classifier label before using it for target eligibility"
                )

            if has_retain_prompt:
                retain_prompt = (row.get("retain_prompt") or "").strip()
                if not retain_prompt:
                    raise ValueError(f"dataset {path} has an empty retain_prompt at CSV line {line_number}")
            else:
                retain_prompt = strip_target(prompt, raw_target)
            if strip_target(retain_prompt, raw_target) != retain_prompt:
                raise ValueError(
                    f"dataset {path} has a retain_prompt that still mentions target {raw_target!r} at CSV line"
                    f" {line_number}"
                )

            try:
                if has_retain_instruments:
                    retain = _resolve_list(parse_instrument_list(row.get("retain_instruments") or ""), vocabulary)
                else:
                    retain = vocabulary.extract(retain_prompt)

                if has_requested:
                    requested = _resolve_list(
                        parse_instrument_list(row.get("requested_instruments") or ""), vocabulary
                    )
                    if target not in requested:
                        raise ValueError(
                            f"preassigned target {target!r} is missing from explicit requested_instruments"
                        )
                else:
                    requested = vocabulary.extract(prompt)
                    requested = _unique((*requested, target, *retain))
            except ValueError as error:
                raise ValueError(f"dataset {path}, CSV line {line_number}: {error}") from error

            if target in retain:
                raise ValueError(
                    f"dataset {path} has target {target!r} among retain_instruments at CSV line {line_number}"
                )
            not_requested = [instrument for instrument in retain if instrument not in requested]
            if not_requested:
                raise ValueError(
                    f"dataset {path} has retain instruments not present in requested_instruments at CSV line"
                    f" {line_number}: {not_requested}"
                )

            metadata = {name: (row.get(name) or "").strip() for name in fieldnames if name not in reserved}
            rows.append(BaselinePrompt(prompt, target, retain_prompt, requested, retain, metadata))

    if not rows:
        raise ValueError(f"dataset {path} contains no samples")
    return rows if max_samples is None else rows[:max_samples]


def paired_seed(first_seed: int, num_samples: int, sample_id: int, seed_index: int) -> int:
    if num_samples < 1 or not 0 <= sample_id < num_samples or seed_index < 0:
        raise ValueError("invalid sample or seed index")
    return int(first_seed + seed_index * num_samples + sample_id)


def build_baseline_record(
    *,
    sample: BaselinePrompt,
    sample_id: int,
    seed_index: int,
    seed: int,
    audio_path: str,
    instrument_scores: Mapping[str, float],
    threshold: float,
    instrument_score_proxies: Mapping[str, bool] | None = None,
) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("detection threshold must be in [0, 1]")
    required = {sample.target, *sample.requested_instruments, *sample.retain_instruments}
    absent = sorted(required - set(instrument_scores))
    if absent:
        raise ValueError(f"classifier did not return scores for required instruments {absent}")

    scores = {name: float(score) for name, score in sorted(instrument_scores.items())}
    invalid_scores = {name: score for name, score in scores.items() if not math.isfinite(score) or not 0 <= score <= 1}
    if invalid_scores:
        raise ValueError(f"classifier returned invalid sigmoid scores {invalid_scores}")
    proxies = {
        name: bool((instrument_score_proxies or {}).get(name, False))
        for name in scores
    }
    requested_scores = {name: scores[name] for name in sample.requested_instruments}
    requested_validity = {name: score >= threshold for name, score in requested_scores.items()}
    retain_scores = {name: scores[name] for name in sample.retain_instruments}
    retain_validity = {name: score >= threshold for name, score in retain_scores.items()}
    target_score = scores[sample.target]
    target_valid = target_score >= threshold

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "pair_id": f"sample_{sample_id:04d}_seed_{seed}",
        "sample_id": sample_id,
        "seed_index": seed_index,
        "seed": seed,
        "prompt": sample.prompt,
        "target": sample.target,
        "retain_prompt": sample.retain_prompt,
        "requested_instruments": list(sample.requested_instruments),
        "retain_instruments": list(sample.retain_instruments),
        "audio_path": audio_path,
        "detection_threshold": threshold,
        "instrument_scores": scores,
        "instrument_score_proxies": proxies,
        "detected_instruments": [
            name for name, score in scores.items() if score >= threshold and not proxies[name]
        ],
        "detected_proxy_instruments": [
            name for name, score in scores.items() if score >= threshold and proxies[name]
        ],
        "requested_instrument_scores": requested_scores,
        "requested_instrument_validity": requested_validity,
        "target_score": target_score,
        "target_valid": target_valid,
        "eligible_for_target_steering": target_valid,
        "retain_instrument_scores": retain_scores,
        "retain_instrument_validity": retain_validity,
        "all_retain_valid": all(retain_validity.values()) if retain_validity else None,
        "source_metadata": dict(sample.source_metadata),
    }


BASELINE_CSV_FIELDS = [
    "schema_version",
    "pair_id",
    "sample_id",
    "seed_index",
    "seed",
    "prompt",
    "target",
    "retain_prompt",
    "requested_instruments",
    "retain_instruments",
    "audio_path",
    "detection_threshold",
    "instrument_scores",
    "instrument_score_proxies",
    "detected_instruments",
    "detected_proxy_instruments",
    "requested_instrument_scores",
    "requested_instrument_validity",
    "target_score",
    "target_valid",
    "eligible_for_target_steering",
    "retain_instrument_scores",
    "retain_instrument_validity",
    "all_retain_valid",
    "source_metadata",
]


def baseline_record_to_csv(record: Mapping) -> dict:
    nested = {
        "requested_instruments",
        "retain_instruments",
        "instrument_scores",
        "instrument_score_proxies",
        "detected_instruments",
        "detected_proxy_instruments",
        "requested_instrument_scores",
        "requested_instrument_validity",
        "retain_instrument_scores",
        "retain_instrument_validity",
        "source_metadata",
    }
    return {
        field: json.dumps(record[field], sort_keys=True) if field in nested else record[field]
        for field in BASELINE_CSV_FIELDS
    }


def load_baseline_selection(path: Path, max_pairs: int | None = None) -> BaselineSelection:
    r"""Loads target-valid records and their exact seeds/audio from a baseline-selection run."""
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be at least 1")

    path = Path(path).resolve()
    if path.is_dir():
        root = path
        records_path = root / "baseline_records.jsonl"
    elif path.suffix.casefold() == ".jsonl":
        root = path.parent
        records_path = path
    else:
        raise ValueError("baseline selection must be its output directory or baseline_records.jsonl")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"baseline selection config not found at {config_path}")
    if not records_path.is_file():
        raise FileNotFoundError(f"baseline selection records not found at {records_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported baseline selection schema {config.get('schema_version')!r};"
            f" expected {BASELINE_SCHEMA_VERSION}"
        )
    required_config = {"generation", "classifier", "instrument_vocabulary", "dataset"}
    missing_config = sorted(required_config - set(config))
    if missing_config:
        raise ValueError(f"baseline selection config is missing keys {missing_config}")

    required_record = {
        "schema_version",
        "pair_id",
        "sample_id",
        "seed_index",
        "seed",
        "prompt",
        "target",
        "retain_prompt",
        "requested_instruments",
        "retain_instruments",
        "audio_path",
        "detection_threshold",
        "instrument_scores",
        "target_score",
        "target_valid",
        "eligible_for_target_steering",
        "retain_instrument_scores",
        "retain_instrument_validity",
        "all_retain_valid",
    }
    selected: list[SelectedBaselinePair] = []
    pair_ids: set[str] = set()
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing_record = sorted(required_record - set(record))
        if missing_record:
            raise ValueError(f"baseline record line {line_number} is missing keys {missing_record}")
        if int(record["schema_version"]) != BASELINE_SCHEMA_VERSION:
            raise ValueError(f"baseline record line {line_number} uses an unsupported schema")
        if not bool(record["eligible_for_target_steering"]):
            continue
        if not bool(record["target_valid"]):
            raise ValueError(f"eligible baseline record {record['pair_id']!r} has target_valid=false")

        pair_id = str(record["pair_id"])
        if pair_id in pair_ids:
            raise ValueError(f"duplicate eligible baseline pair_id {pair_id!r}")
        pair_ids.add(pair_id)

        audio_path = Path(str(record["audio_path"]))
        audio_path = audio_path if audio_path.is_absolute() else root / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"baseline audio for pair {pair_id!r} not found at {audio_path}")
        selected.append(SelectedBaselinePair(record=record, audio_path=audio_path))

    if not selected:
        raise ValueError(f"baseline selection {records_path} contains no target-valid pair")
    if max_pairs is not None:
        selected = selected[:max_pairs]
    return BaselineSelection(root=root, records_path=records_path, config=config, pairs=tuple(selected))


class AudioSetInstrumentClassifier:
    """Multi-label AST/AudioSet inference with sigmoid scores and max aggregation over 10 s windows."""

    def __init__(
        self,
        vocabulary: InstrumentVocabulary,
        model_name: str = DEFAULT_CLASSIFIER_MODEL,
        revision: str | None = None,
        device: str = "cpu",
        window_seconds: float = 10.0,
        batch_size: int = 8,
    ):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        if window_seconds <= 0.0 or batch_size < 1:
            raise ValueError("classifier window_seconds and batch_size must be positive")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA classifier device requested but CUDA is unavailable")

        kwargs = {} if revision is None else {"revision": revision}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name, **kwargs).to(resolved_device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.vocabulary = vocabulary
        self.model_name = model_name
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.device = resolved_device
        self.sampling_rate = int(self.feature_extractor.sampling_rate)
        self.window_samples = max(1, int(round(window_seconds * self.sampling_rate)))
        self.batch_size = batch_size

        label_to_id = {label.casefold(): int(index) for index, label in self.model.config.id2label.items()}
        missing_labels = sorted(
            {
                label
                for spec in vocabulary.specs.values()
                for label in spec.classifier_labels
                if label.casefold() not in label_to_id
            }
        )
        if missing_labels:
            raise ValueError(f"classifier {model_name!r} is missing configured AudioSet labels {missing_labels}")
        self._label_ids = {
            name: tuple(label_to_id[label.casefold()] for label in spec.classifier_labels)
            for name, spec in vocabulary.specs.items()
        }

    def score(self, waveform: np.ndarray, sampling_rate: int) -> dict[str, float]:
        import torch
        import torchaudio.functional as audio_functional

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("waveform must contain mono samples or have shape (channels, samples)")

        samples = torch.from_numpy(np.ascontiguousarray(waveform))
        if sampling_rate != self.sampling_rate:
            samples = audio_functional.resample(samples, sampling_rate, self.sampling_rate)
        samples = samples.numpy()
        chunks = [samples[start : start + self.window_samples] for start in range(0, len(samples), self.window_samples)]

        max_probabilities = None
        with torch.inference_mode():
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                inputs = self.feature_extractor(batch, sampling_rate=self.sampling_rate, return_tensors="pt")
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                # AudioSet is multi-label. The checkpoint exposes raw logits and does not declare a
                # problem_type in config, so apply sigmoid explicitly instead of pipeline softmax.
                probabilities = torch.sigmoid(self.model(**inputs).logits).amax(dim=0).cpu()
                max_probabilities = (
                    probabilities if max_probabilities is None else torch.maximum(max_probabilities, probabilities)
                )

        return {
            name: float(max_probabilities[list(label_ids)].amax())
            for name, label_ids in self._label_ids.items()
        }
