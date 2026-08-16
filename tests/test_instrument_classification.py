import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Core schema/vocabulary tests do not instantiate the runtime classifier. Keep them runnable in a
# lightweight checkout just like tests/test_dataset.py.
if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
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

from utils.instrument_classification import (
    BaselinePrompt,
    InstrumentVocabulary,
    baseline_record_to_csv,
    build_baseline_record,
    load_baseline_selection,
    load_baseline_prompts,
    paired_seed,
)
from utils.audio import load_waveform, save_waveform


class InstrumentClassificationTests(unittest.TestCase):
    @staticmethod
    def write_csv(directory: str, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
        path = Path(directory) / "dataset.csv"
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_default_vocabulary_prefers_specific_mentions(self) -> None:
        vocabulary = InstrumentVocabulary.default()

        instruments = vocabulary.extract(
            "Trumpet with brushed drums, acoustic grand piano, electric guitar, and upright bass."
        )

        self.assertEqual(
            instruments,
            ("trumpet", "drums", "piano", "electric guitar", "double bass"),
        )
        self.assertNotIn("guitar", instruments)

    def test_existing_dataset_contract_infers_requested_and_retain_instruments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "retain_prompt", "group_id"],
                [
                    {
                        "prompt": "A jazz piece with trumpet, drums, and piano.",
                        "target": "trumpet",
                        "retain_prompt": "A jazz piece with drums and piano.",
                        "group_id": "jazz_01",
                    }
                ],
            )

            sample = load_baseline_prompts(path, InstrumentVocabulary.default())[0]

            self.assertEqual(sample.target, "trumpet")
            self.assertEqual(sample.requested_instruments, ("trumpet", "drums", "piano"))
            self.assertEqual(sample.retain_instruments, ("drums", "piano"))
            self.assertEqual(sample.source_metadata, {"group_id": "jazz_01"})

    def test_all_repository_simple_splits_are_accepted(self) -> None:
        vocabulary = InstrumentVocabulary.default()
        paths = sorted((ROOT / "datasets").glob("*_simple_splits/*.csv"))

        self.assertTrue(paths)
        for path in paths:
            samples = load_baseline_prompts(path, vocabulary)
            self.assertTrue(samples, path)
            for sample in samples:
                self.assertIn(sample.target, sample.requested_instruments)
                self.assertTrue(set(sample.retain_instruments) <= set(sample.requested_instruments))

    def test_explicit_instrument_columns_are_authoritative_and_target_is_preassigned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "retain_prompt", "requested_instruments", "retain_instruments"],
                [
                    {
                        "prompt": "A deliberately abstract prompt.",
                        "target": "Trumpet",
                        "retain_prompt": "Keep the rhythm section.",
                        "requested_instruments": '["trumpet", "electric bass", "drums"]',
                        "retain_instruments": "electric bass; drums",
                    }
                ],
            )

            sample = load_baseline_prompts(path, InstrumentVocabulary.default())[0]

            self.assertEqual(sample.target, "trumpet")
            self.assertEqual(sample.requested_instruments, ("trumpet", "electric bass", "drums"))
            self.assertEqual(sample.retain_instruments, ("electric bass", "drums"))

    def test_explicit_requested_instruments_must_include_preassigned_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "requested_instruments", "retain_instruments"],
                [
                    {
                        "prompt": "Trumpet and drums.",
                        "target": "trumpet",
                        "requested_instruments": "drums",
                        "retain_instruments": "drums",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "preassigned target.*missing"):
                load_baseline_prompts(path, InstrumentVocabulary.default())

    def test_retain_prompt_cannot_still_mention_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "retain_prompt", "requested_instruments", "retain_instruments"],
                [
                    {
                        "prompt": "Trumpet and drums.",
                        "target": "trumpet",
                        "retain_prompt": "Quiet trumpet and drums.",
                        "requested_instruments": "trumpet; drums",
                        "retain_instruments": "drums",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "retain_prompt.*still mentions target"):
                load_baseline_prompts(path, InstrumentVocabulary.default())

    def test_coarse_proxy_cannot_be_used_as_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                ["prompt", "target", "retain_prompt"],
                [
                    {
                        "prompt": "Tuba and drums.",
                        "target": "tuba",
                        "retain_prompt": "Drums.",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "target 'tuba'.*coarse AudioSet proxy"):
                load_baseline_prompts(path, InstrumentVocabulary.default())

    def test_record_eligibility_depends_only_on_preassigned_target(self) -> None:
        sample = BaselinePrompt(
            prompt="Trumpet, drums, and piano.",
            target="trumpet",
            retain_prompt="Drums and piano.",
            requested_instruments=("trumpet", "drums", "piano"),
            retain_instruments=("drums", "piano"),
            source_metadata={},
        )

        record = build_baseline_record(
            sample=sample,
            sample_id=3,
            seed_index=1,
            seed=105,
            audio_path="audio/baseline/sample_0003_seed_105.wav",
            instrument_scores={"trumpet": 0.49, "drums": 0.91, "piano": 0.12, "violin": 0.8},
            threshold=0.5,
            instrument_score_proxies={"trumpet": False, "drums": False, "piano": False, "violin": False},
        )

        self.assertFalse(record["target_valid"])
        self.assertFalse(record["eligible_for_target_steering"])
        self.assertEqual(record["target_score"], 0.49)
        self.assertEqual(record["retain_instrument_validity"], {"drums": True, "piano": False})
        self.assertFalse(record["all_retain_valid"])
        self.assertIn("violin", record["detected_instruments"])
        csv_record = baseline_record_to_csv(record)
        self.assertEqual(json.loads(csv_record["instrument_scores"])["trumpet"], 0.49)

    def test_pair_seed_formula_matches_evaluator_layout(self) -> None:
        seeds = [paired_seed(1000, 3, sample_id, seed_index) for seed_index in range(2) for sample_id in range(3)]
        self.assertEqual(seeds, [1000, 1001, 1002, 1003, 1004, 1005])

    def test_selection_loader_keeps_only_eligible_pairs_and_resolves_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "audio" / "baseline" / "sample.wav"
            save_waveform(audio_path, np.zeros((2, 16), dtype=np.float32), 44_100)
            config = {
                "schema_version": 2,
                "dataset": str(root / "dataset.csv"),
                "generation": {"model": "ACE-Step/acestep-v15-sft"},
                "classifier": {"model": "fake"},
                "instrument_vocabulary": {"trumpet": {}},
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

            base = {
                "schema_version": 2,
                "sample_id": 0,
                "seed_index": 0,
                "seed": 1000,
                "prompt": "Trumpet and drums.",
                "target": "trumpet",
                "retain_prompt": "Drums.",
                "requested_instruments": ["trumpet", "drums"],
                "retain_instruments": ["drums"],
                "audio_path": "audio/baseline/sample.wav",
                "detection_threshold": 0.5,
                "instrument_scores": {"trumpet": 0.8, "drums": 0.7},
                "target_score": 0.8,
                "target_valid": True,
                "retain_instrument_scores": {"drums": 0.7},
                "retain_instrument_validity": {"drums": True},
                "all_retain_valid": True,
            }
            invalid = {
                **base,
                "pair_id": "invalid",
                "target_score": 0.1,
                "target_valid": False,
                "eligible_for_target_steering": False,
            }
            eligible = {
                **base,
                "pair_id": "eligible",
                "eligible_for_target_steering": True,
            }
            (root / "baseline_records.jsonl").write_text(
                json.dumps(invalid) + "\n" + json.dumps(eligible) + "\n",
                encoding="utf-8",
            )

            selection = load_baseline_selection(root)
            waveform, sampling_rate = load_waveform(selection.pairs[0].audio_path)

            self.assertEqual([pair.pair_id for pair in selection.pairs], ["eligible"])
            self.assertEqual(selection.pairs[0].record["seed"], 1000)
            self.assertEqual(waveform.shape, (2, 16))
            self.assertEqual(sampling_rate, 44_100)


if __name__ == "__main__":
    unittest.main()
