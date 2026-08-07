import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pipelines import GRADIENT_MODE as PIPELINE_GRADIENT_MODE
from pipelines import STEERING_MODE as PIPELINE_STEERING_MODE
from utils.checkpoint import (
    CLAP_MODEL_ID,
    CLAP_REVISION,
    COMPONENTS_ID,
    COMPONENTS_REVISION,
    FORMAT_VERSION,
    GRADIENT_MODE,
    MODEL_FAMILY,
    MODEL_ID,
    MODEL_REVISION,
    STEERING_MODE,
    CheckpointCompatibilityError,
    build_checkpoint,
    build_resume_fields,
    capture_rng_state,
    extract_resume_fields,
    load_checkpoint,
    restore_optimizer_state,
    restore_rng_state,
    restore_training_state,
    save_checkpoint,
    validate_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self, **overrides):
        values = {
            "state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "config": {"latent_channels": 8, "alpha_init": 0.15},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "history": {"epochs": [{"epoch": 1, "loss": 0.25}]},
            "args": {"seed": 42},
            "epoch": 1,
            "global_step": 7,
            "loss": 0.25,
            "best_loss": 0.25,
        }
        values.update(overrides)
        return build_checkpoint(**values)

    def test_build_and_round_trip_preserve_v2_metadata_and_training_state(self) -> None:
        checkpoint = self.make_checkpoint()

        self.assertEqual(checkpoint["format_version"], FORMAT_VERSION)
        self.assertEqual(checkpoint["model_family"], MODEL_FAMILY)
        self.assertEqual(checkpoint["model_id"], MODEL_ID)
        self.assertEqual(checkpoint["model_revision"], MODEL_REVISION)
        self.assertEqual(checkpoint["components_id"], COMPONENTS_ID)
        self.assertEqual(checkpoint["components_revision"], COMPONENTS_REVISION)
        self.assertEqual(checkpoint["clap_model_id"], CLAP_MODEL_ID)
        self.assertEqual(checkpoint["clap_revision"], CLAP_REVISION)
        self.assertEqual(checkpoint["steering_mode"], STEERING_MODE)
        self.assertEqual(checkpoint["gradient_mode"], GRADIENT_MODE)

        self.assertEqual(STEERING_MODE, PIPELINE_STEERING_MODE)
        self.assertEqual(GRADIENT_MODE, PIPELINE_GRADIENT_MODE)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.pt"
            saved_path = save_checkpoint(checkpoint, path)
            loaded = load_checkpoint(path)

        self.assertEqual(saved_path, path.resolve())
        self.assertTrue(torch.equal(loaded["state_dict"]["weight"], torch.tensor([1.0, 2.0])))
        self.assertEqual(loaded["history"], checkpoint["history"])
        self.assertEqual(
            extract_resume_fields(loaded),
            {
                "epoch": 1,
                "global_step": 7,
                "batch_in_epoch": 0,
                "loss": 0.25,
                "best_loss": 0.25,
            },
        )

    def test_unversioned_musicldm_checkpoint_is_rejected_explicitly(self) -> None:
        legacy = {"state_dict": {}, "config": {}, "steering_mode": "full_to_retain_v1"}

        with self.assertRaisesRegex(CheckpointCompatibilityError, "legacy MusicLDM"):
            validate_checkpoint(legacy)

    def test_old_and_future_versions_are_rejected_with_distinct_messages(self) -> None:
        checkpoint = self.make_checkpoint()

        old = copy.deepcopy(checkpoint)
        old["format_version"] = 1
        with self.assertRaisesRegex(CheckpointCompatibilityError, "legacy MusicLDM-era"):
            validate_checkpoint(old)

        future = copy.deepcopy(checkpoint)
        future["format_version"] = FORMAT_VERSION + 1
        with self.assertRaisesRegex(CheckpointCompatibilityError, "newer than the supported"):
            validate_checkpoint(future)

    def test_model_family_model_id_and_steering_mode_mismatches_are_rejected(self) -> None:
        checkpoint = self.make_checkpoint()

        wrong_family = copy.deepcopy(checkpoint)
        wrong_family["model_family"] = "musicldm"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "model family.*MusicLDM"):
            validate_checkpoint(wrong_family)

        wrong_model = copy.deepcopy(checkpoint)
        wrong_model["model_id"] = "ACE-Step/acestep-v15-turbo"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "base model"):
            validate_checkpoint(wrong_model)

        wrong_mode = copy.deepcopy(checkpoint)
        wrong_mode["steering_mode"] = "full_to_retain_v1"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "steering mode"):
            validate_checkpoint(wrong_mode)

        wrong_revision = copy.deepcopy(checkpoint)
        wrong_revision["model_revision"] = "moving-main"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "model revision"):
            validate_checkpoint(wrong_revision)

        wrong_clap = copy.deepcopy(checkpoint)
        wrong_clap["clap_model_id"] = "some/other-clap"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "CLAP model"):
            validate_checkpoint(wrong_clap)

        wrong_gradient = copy.deepcopy(checkpoint)
        wrong_gradient["gradient_mode"] = "full_bptt"
        with self.assertRaisesRegex(CheckpointCompatibilityError, "gradient mode"):
            validate_checkpoint(wrong_gradient)

    def test_rng_state_round_trip_reproduces_python_numpy_and_torch_draws(self) -> None:
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        state = capture_rng_state(include_cuda=False)

        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.random()
        np.random.rand()
        torch.rand(3)
        restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_resume_helpers_restore_optimizer_rng_and_detached_history(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        checkpoint = self.make_checkpoint(
            state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            batch_in_epoch=3,
        )
        resumed_model = torch.nn.Linear(2, 1)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.9)

        self.assertTrue(restore_optimizer_state(resumed_optimizer, checkpoint))
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], 0.01)
        state = restore_training_state(
            checkpoint,
            optimizer=resumed_optimizer,
            restore_rng=False,
        )
        state["history"]["epochs"].append({"epoch": 99})

        self.assertEqual(state["batch_in_epoch"], 3)
        self.assertEqual(len(checkpoint["history"]["epochs"]), 1)

    def test_resume_fields_validate_non_negative_integer_cursors(self) -> None:
        self.assertEqual(build_resume_fields(epoch=2, global_step=9)["global_step"], 9)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "non-negative integer"):
            build_resume_fields(epoch=-1)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "non-negative integer"):
            build_resume_fields(global_step=True)

    def test_eval_only_checkpoint_reports_missing_optimizer_on_resume(self) -> None:
        checkpoint = self.make_checkpoint(optimizer_state_dict=None)
        optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters())

        with self.assertRaisesRegex(CheckpointCompatibilityError, "cannot be resumed exactly"):
            restore_optimizer_state(optimizer, checkpoint)
        self.assertFalse(restore_optimizer_state(optimizer, checkpoint, required=False))


if __name__ == "__main__":
    unittest.main()
