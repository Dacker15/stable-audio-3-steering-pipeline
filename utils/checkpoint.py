"""Versioned checkpoints for ACE-Step steering training.

The checkpoint format is deliberately owned by this project rather than by the
underlying ACE-Step runtime.  A steering predictor trained against MusicLDM has
the same broad shape of payload (``state_dict`` plus ``config``), but its inputs
and denoising semantics are incompatible with ACE-Step.  Requiring the metadata
below prevents such a checkpoint from being accepted accidentally.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 2
MODEL_FAMILY = "ace-step-1.5"
MODEL_ID = "ACE-Step/acestep-v15-sft"
MODEL_REVISION = "c410d249e71ea9385a7b586865e65b1473e1098d"
COMPONENTS_ID = "ACE-Step/Ace-Step1.5"
COMPONENTS_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
STEERING_MODE = "ace_step_apg_cond_velocity_lerp_canonical_null_v2"
GRADIENT_MODE = "steering_only_surrogate_v1"

# Longer aliases make imports at call sites self-documenting while the short
# names remain the canonical serialized values.
CHECKPOINT_FORMAT_VERSION = FORMAT_VERSION
CHECKPOINT_MODEL_FAMILY = MODEL_FAMILY
CHECKPOINT_MODEL_ID = MODEL_ID
CHECKPOINT_MODEL_REVISION = MODEL_REVISION
CHECKPOINT_COMPONENTS_ID = COMPONENTS_ID
CHECKPOINT_COMPONENTS_REVISION = COMPONENTS_REVISION
CHECKPOINT_CLAP_MODEL_ID = CLAP_MODEL_ID
CHECKPOINT_CLAP_REVISION = CLAP_REVISION
CHECKPOINT_STEERING_MODE = STEERING_MODE
CHECKPOINT_GRADIENT_MODE = GRADIENT_MODE


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be used by the ACE-Step v2 runtime."""


_REQUIRED_FIELDS = {
    "format_version",
    "model_family",
    "model_id",
    "model_revision",
    "components_id",
    "components_revision",
    "clap_model_id",
    "clap_revision",
    "steering_mode",
    "gradient_mode",
    "state_dict",
    "config",
    "optimizer_state_dict",
    "rng_state",
    "history",
    "args",
    "epoch",
    "global_step",
    "batch_in_epoch",
    "loss",
    "best_loss",
}


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    """Capture all RNGs used by the training loop.

    CUDA state is recorded only when CUDA is available.  CPU-only checkpoints
    therefore remain portable, and loading a CUDA checkpoint on CPU is allowed.
    """

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if include_cuda and torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(rng_state: Mapping[str, Any], *, strict_cuda: bool = False) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`.

    Args:
        rng_state: Serialized Python, NumPy and Torch RNG states.
        strict_cuda: Require CUDA state to be applicable.  The default permits
            inspecting or evaluating a GPU-created checkpoint on a CPU host.
    """

    if not isinstance(rng_state, Mapping):
        raise TypeError("`rng_state` must be a mapping")

    missing = {"python", "numpy", "torch_cpu"} - set(rng_state)
    if missing:
        raise CheckpointCompatibilityError(
            f"checkpoint RNG state is missing required entries {sorted(missing)}"
        )

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"].cpu())

    cuda_state = rng_state.get("torch_cuda")
    if cuda_state is None:
        if strict_cuda and torch.cuda.is_available():
            raise CheckpointCompatibilityError("checkpoint has no CUDA RNG state to restore")
        return
    if not torch.cuda.is_available():
        if strict_cuda:
            raise CheckpointCompatibilityError(
                "checkpoint contains CUDA RNG state, but CUDA is unavailable on this host"
            )
        return
    torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


def build_resume_fields(
    *,
    epoch: int = 0,
    global_step: int = 0,
    batch_in_epoch: int = 0,
    loss: float | None = None,
    best_loss: float | None = None,
) -> dict[str, int | float | None]:
    """Build and validate the progress fields required for exact resume.

    ``epoch`` is the last fully entered epoch and ``batch_in_epoch`` is the
    number of batches already consumed in it.  A checkpoint written at an epoch
    boundary therefore normally stores ``batch_in_epoch=0`` and the training
    loop resumes from ``epoch + 1``.
    """

    fields: dict[str, int | float | None] = {
        "epoch": epoch,
        "global_step": global_step,
        "batch_in_epoch": batch_in_epoch,
        "loss": loss,
        "best_loss": best_loss,
    }
    _validate_resume_fields(fields)
    return fields


def extract_resume_fields(checkpoint: Mapping[str, Any]) -> dict[str, int | float | None]:
    """Return a detached copy of the progress fields from a v2 checkpoint."""

    fields = {key: checkpoint.get(key) for key in ("epoch", "global_step", "batch_in_epoch", "loss", "best_loss")}
    _validate_resume_fields(fields)
    return fields


def _validate_resume_fields(fields: Mapping[str, Any]) -> None:
    for name in ("epoch", "global_step", "batch_in_epoch"):
        value = fields.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckpointCompatibilityError(f"checkpoint `{name}` must be a non-negative integer")
    for name in ("loss", "best_loss"):
        value = fields.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise CheckpointCompatibilityError(f"checkpoint `{name}` must be numeric or None")


def _mapping_copy(value: Mapping[str, Any] | argparse.Namespace | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping or argparse.Namespace")
    return copy.deepcopy(dict(value))


def build_checkpoint(
    *,
    state_dict: Mapping[str, Any],
    config: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
    args: Mapping[str, Any] | argparse.Namespace | None = None,
    epoch: int = 0,
    global_step: int = 0,
    batch_in_epoch: int = 0,
    loss: float | None = None,
    best_loss: float | None = None,
    rng_state: Mapping[str, Any] | None = None,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    components_id: str = COMPONENTS_ID,
    components_revision: str = COMPONENTS_REVISION,
    clap_model_id: str = CLAP_MODEL_ID,
    clap_revision: str = CLAP_REVISION,
    gradient_mode: str = GRADIENT_MODE,
) -> dict[str, Any]:
    """Construct a complete ACE-Step v2 training checkpoint.

    The model and optimizer state dictionaries are kept tensor-preserving; the
    smaller metadata containers are copied so later history/config mutations do
    not silently alter the in-memory checkpoint.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("`state_dict` must be a mapping")
    if not isinstance(config, Mapping):
        raise TypeError("`config` must be a mapping")
    if optimizer_state_dict is not None and not isinstance(optimizer_state_dict, Mapping):
        raise TypeError("`optimizer_state_dict` must be a mapping or None")
    if rng_state is not None and not isinstance(rng_state, Mapping):
        raise TypeError("`rng_state` must be a mapping or None")

    checkpoint: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_family": MODEL_FAMILY,
        "model_id": model_id,
        "model_revision": model_revision,
        "components_id": components_id,
        "components_revision": components_revision,
        "clap_model_id": clap_model_id,
        "clap_revision": clap_revision,
        "steering_mode": STEERING_MODE,
        "gradient_mode": gradient_mode,
        "state_dict": dict(state_dict),
        "config": _mapping_copy(config, "config"),
        "optimizer_state_dict": (
            # Keep optimizer tensors shallow just like model tensors.  Deep-copying AdamW's
            # moment buffers would briefly duplicate them on the already memory-constrained T4.
            None if optimizer_state_dict is None else dict(optimizer_state_dict)
        ),
        "rng_state": copy.deepcopy(dict(rng_state)) if rng_state is not None else capture_rng_state(),
        "history": _mapping_copy(history, "history"),
        "args": _mapping_copy(args, "args"),
        **build_resume_fields(
            epoch=epoch,
            global_step=global_step,
            batch_in_epoch=batch_in_epoch,
            loss=loss,
            best_loss=best_loss,
        ),
    }
    validate_checkpoint(checkpoint)
    return checkpoint


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_model_family: str = MODEL_FAMILY,
    expected_model_id: str | None = MODEL_ID,
    expected_model_revision: str | None = MODEL_REVISION,
    expected_components_id: str | None = COMPONENTS_ID,
    expected_components_revision: str | None = COMPONENTS_REVISION,
    expected_clap_model_id: str | None = CLAP_MODEL_ID,
    expected_clap_revision: str | None = CLAP_REVISION,
    expected_steering_mode: str = STEERING_MODE,
    expected_gradient_mode: str = GRADIENT_MODE,
) -> None:
    """Validate format and semantic compatibility of a checkpoint payload."""

    if not isinstance(checkpoint, Mapping):
        raise CheckpointCompatibilityError(
            f"checkpoint payload must be a mapping, got {type(checkpoint).__name__}"
        )

    if "format_version" not in checkpoint:
        raise CheckpointCompatibilityError(
            "checkpoint is missing `format_version`; unversioned/legacy MusicLDM checkpoints are "
            "not compatible with the ACE-Step steering runtime and must be retrained"
        )

    version = checkpoint["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise CheckpointCompatibilityError("checkpoint `format_version` must be an integer")
    if version < FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint format v{version} is a legacy MusicLDM-era format and is incompatible with "
            f"ACE-Step checkpoint format v{FORMAT_VERSION}; retrain the steering predictor"
        )
    if version > FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint format v{version} is newer than the supported v{FORMAT_VERSION}; upgrade this "
            "project before loading it"
        )

    missing = _REQUIRED_FIELDS - set(checkpoint)
    if missing:
        raise CheckpointCompatibilityError(
            f"ACE-Step checkpoint format v{FORMAT_VERSION} is missing required fields {sorted(missing)}"
        )

    model_family = checkpoint["model_family"]
    if model_family != expected_model_family:
        legacy_hint = (
            " MusicLDM checkpoints cannot be migrated because UNet noise predictions are incompatible "
            "with ACE-Step DiT flow predictions."
            if "musicldm" in str(model_family).lower()
            else ""
        )
        raise CheckpointCompatibilityError(
            f"checkpoint model family {model_family!r} does not match required ACE-Step family "
            f"{expected_model_family!r}.{legacy_hint}"
        )

    model_id = checkpoint["model_id"]
    if expected_model_id is not None and model_id != expected_model_id:
        raise CheckpointCompatibilityError(
            f"checkpoint base model {model_id!r} does not match required model {expected_model_id!r}"
        )

    identities = (
        ("model revision", "model_revision", expected_model_revision),
        ("component bundle", "components_id", expected_components_id),
        ("component revision", "components_revision", expected_components_revision),
        ("CLAP model", "clap_model_id", expected_clap_model_id),
        ("CLAP revision", "clap_revision", expected_clap_revision),
    )
    for label, key, expected in identities:
        actual = checkpoint[key]
        if expected is not None and actual != expected:
            raise CheckpointCompatibilityError(
                f"checkpoint {label} {actual!r} does not match required {expected!r}"
            )

    steering_mode = checkpoint["steering_mode"]
    if steering_mode != expected_steering_mode:
        raise CheckpointCompatibilityError(
            f"checkpoint steering mode {steering_mode!r} does not match required mode "
            f"{expected_steering_mode!r}"
        )

    gradient_mode = checkpoint["gradient_mode"]
    if gradient_mode != expected_gradient_mode:
        raise CheckpointCompatibilityError(
            f"checkpoint gradient mode {gradient_mode!r} does not match required mode "
            f"{expected_gradient_mode!r}"
        )

    for name in ("state_dict", "config", "history", "args", "rng_state"):
        if not isinstance(checkpoint[name], Mapping):
            raise CheckpointCompatibilityError(f"checkpoint `{name}` must be a mapping")
    optimizer_state = checkpoint["optimizer_state_dict"]
    if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
        raise CheckpointCompatibilityError("checkpoint `optimizer_state_dict` must be a mapping or None")

    rng_missing = {"python", "numpy", "torch_cpu"} - set(checkpoint["rng_state"])
    if rng_missing:
        raise CheckpointCompatibilityError(
            f"checkpoint RNG state is missing required entries {sorted(rng_missing)}"
        )
    if not torch.is_tensor(checkpoint["rng_state"]["torch_cpu"]):
        raise CheckpointCompatibilityError("checkpoint RNG entry `torch_cpu` must be a tensor")

    extract_resume_fields(checkpoint)


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Validate and atomically save a checkpoint, returning its resolved path."""

    validate_checkpoint(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(checkpoint), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
    expected_model_family: str = MODEL_FAMILY,
    expected_model_id: str | None = MODEL_ID,
    expected_model_revision: str | None = MODEL_REVISION,
    expected_components_id: str | None = COMPONENTS_ID,
    expected_components_revision: str | None = COMPONENTS_REVISION,
    expected_clap_model_id: str | None = CLAP_MODEL_ID,
    expected_clap_revision: str | None = CLAP_REVISION,
    expected_steering_mode: str = STEERING_MODE,
    expected_gradient_mode: str = GRADIENT_MODE,
) -> dict[str, Any]:
    """Load an ACE-Step checkpoint only after strict compatibility checks."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    validate_checkpoint(
        checkpoint,
        expected_model_family=expected_model_family,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        expected_components_id=expected_components_id,
        expected_components_revision=expected_components_revision,
        expected_clap_model_id=expected_clap_model_id,
        expected_clap_revision=expected_clap_revision,
        expected_steering_mode=expected_steering_mode,
        expected_gradient_mode=expected_gradient_mode,
    )
    return dict(checkpoint)


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: Mapping[str, Any],
    *,
    required: bool = True,
) -> bool:
    """Restore optimizer slots from a checkpoint and report whether they existed."""

    state = checkpoint.get("optimizer_state_dict")
    if state is None:
        if required:
            raise CheckpointCompatibilityError(
                "checkpoint has no optimizer state; it can be evaluated but cannot be resumed exactly"
            )
        return False
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint `optimizer_state_dict` must be a mapping")
    optimizer.load_state_dict(state)
    return True


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
    require_optimizer: bool = True,
) -> dict[str, Any]:
    """Restore mutable training state and return history plus resume cursors."""

    validate_checkpoint(checkpoint)
    if optimizer is not None:
        restore_optimizer_state(optimizer, checkpoint, required=require_optimizer)
    elif require_optimizer and checkpoint["optimizer_state_dict"] is None:
        raise CheckpointCompatibilityError(
            "checkpoint has no optimizer state; it can be evaluated but cannot be resumed exactly"
        )
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return {
        **extract_resume_fields(checkpoint),
        "history": copy.deepcopy(dict(checkpoint["history"])),
    }


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CHECKPOINT_MODEL_FAMILY",
    "CHECKPOINT_MODEL_ID",
    "CHECKPOINT_MODEL_REVISION",
    "CHECKPOINT_COMPONENTS_ID",
    "CHECKPOINT_COMPONENTS_REVISION",
    "CHECKPOINT_CLAP_MODEL_ID",
    "CHECKPOINT_CLAP_REVISION",
    "CHECKPOINT_STEERING_MODE",
    "CHECKPOINT_GRADIENT_MODE",
    "CLAP_MODEL_ID",
    "CLAP_REVISION",
    "COMPONENTS_ID",
    "COMPONENTS_REVISION",
    "FORMAT_VERSION",
    "MODEL_FAMILY",
    "MODEL_ID",
    "MODEL_REVISION",
    "GRADIENT_MODE",
    "STEERING_MODE",
    "CheckpointCompatibilityError",
    "build_checkpoint",
    "build_resume_fields",
    "capture_rng_state",
    "extract_resume_fields",
    "load_checkpoint",
    "restore_optimizer_state",
    "restore_rng_state",
    "restore_training_state",
    "save_checkpoint",
    "validate_checkpoint",
]
