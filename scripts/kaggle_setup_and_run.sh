#!/usr/bin/env bash
# Reproducible ACE-Step 1.5 SFT setup, diagnostics, smoke training, and optional
# first real experiment for a Kaggle Tesla T4.
#
# Kaggle setup:
#   1. Enable a GPU accelerator.
#   2. Add a private Kaggle secret named HF_TOKEN (never paste it into this file).
#   3. Upload/clone this repository below /kaggle/working.
#   4. Point TRAIN_CSV at an attached three-column CSV if it is not in the repo.
#
# Smoke only (default):
#   bash scripts/kaggle_setup_and_run.sh
#
# Smoke followed by the small first experiment:
#   RUN_FULL_TRAINING=1 TRAIN_CSV=/kaggle/input/my-dataset/train.csv \
#     bash scripts/kaggle_setup_and_run.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
KAGGLE_WORKING="${KAGGLE_WORKING:-/kaggle/working}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${KAGGLE_WORKING}/ace-step-steering/${RUN_ID}}"
HF_HOME="${HF_HOME:-${KAGGLE_WORKING}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${KAGGLE_WORKING}/pip-cache}"

ACE_MODEL_ID="ACE-Step/acestep-v15-sft"
ACE_MODEL_REVISION="c410d249e71ea9385a7b586865e65b1473e1098d"
ACE_COMPONENTS_ID="ACE-Step/Ace-Step1.5"
ACE_COMPONENTS_REVISION="19671f406d603126926c1b7e2adc169acbcade22"
CLAP_MODEL_ID="laion/clap-htsat-unfused"
CLAP_REVISION="8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"

TRAIN_CSV="${TRAIN_CSV:-${PROJECT_DIR}/datasets/trumpet_simple_splits/train.csv}"
SIMPLE_DATASET_CSV="${SIMPLE_DATASET_CSV:-}"
SMOKE_OUTPUT="${OUTPUT_ROOT}/smoke"
TRAIN_OUTPUT="${OUTPUT_ROOT}/first-experiment"
LOG_DIR="${OUTPUT_ROOT}/logs"

case "${PROJECT_DIR}" in
  "${KAGGLE_WORKING}"|"${KAGGLE_WORKING}"/*) ;;
  *)
    echo "PROJECT_DIR must be inside ${KAGGLE_WORKING}; got ${PROJECT_DIR}" >&2
    exit 2
    ;;
esac
case "${OUTPUT_ROOT}" in
  "${KAGGLE_WORKING}"|"${KAGGLE_WORKING}"/*) ;;
  *)
    echo "OUTPUT_ROOT must be inside ${KAGGLE_WORKING}; got ${OUTPUT_ROOT}" >&2
    exit 2
    ;;
esac
if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  echo "No pyproject.toml found at ${PROJECT_DIR}. Upload or clone the repository below ${KAGGLE_WORKING}, then set PROJECT_DIR." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${HF_HUB_CACHE}" "${PIP_CACHE_DIR}"
export HF_HOME HF_HUB_CACHE PIP_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT_DIR}"

echo "Run ID: ${RUN_ID}"
echo "Project: ${PROJECT_DIR}"
echo "Outputs: ${OUTPUT_ROOT}"
python --version | tee "${LOG_DIR}/python-version.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "${LOG_DIR}/nvidia-smi.log"
else
  echo "nvidia-smi is unavailable. Enable a Kaggle GPU accelerator before continuing." >&2
  exit 2
fi

python -m pip install --upgrade pip setuptools wheel 'uv>=0.8,<1' 2>&1 | tee "${LOG_DIR}/pip-bootstrap.log"
uv sync --frozen --group dev 2>&1 | tee "${LOG_DIR}/uv-sync.log"
# All remaining commands use the exact environment resolved in uv.lock,
# including ACE-Step's official Linux CUDA 12.8 PyTorch build.
source "${PROJECT_DIR}/.venv/bin/activate"
uv pip check 2>&1 | tee "${LOG_DIR}/pip-check.log"
uv pip freeze > "${LOG_DIR}/pip-freeze.txt"

python - <<'PY' | tee "${LOG_DIR}/torch-diagnostics.log"
import platform
import torch

print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. In Kaggle, enable a GPU accelerator and restart the session.")
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("supported ACE-Step dtype: torch.float32")
PY

# Prefer an already exported token. Otherwise read the private Kaggle secret. Command
# substitution captures it without printing it to the notebook output.
if [[ -z "${HF_TOKEN:-}" ]]; then
  if ! HF_TOKEN="$(python - <<'PY'
from kaggle_secrets import UserSecretsClient

try:
    print(UserSecretsClient().get_secret("HF_TOKEN"), end="")
except Exception as exc:
    raise SystemExit(
        "Cannot read the Kaggle secret HF_TOKEN. Add it under Add-ons > Secrets and enable it for this notebook."
    ) from exc
PY
)"; then
    exit 2
  fi
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN}" ]]; then
  echo "HF_TOKEN is empty. Add a private Kaggle secret named HF_TOKEN." >&2
  exit 2
fi

python - <<'PY' | tee "${LOG_DIR}/huggingface-download.log"
import os
from huggingface_hub import snapshot_download

token = os.environ["HF_TOKEN"]
# Use the same default Hub cache path that transformers/diffusers will consult
# during training, so the explicit prefetch is not downloaded a second time.
cache_dir = os.environ["HF_HUB_CACHE"]
artifacts = (
    ("ACE-Step/acestep-v15-sft", "c410d249e71ea9385a7b586865e65b1473e1098d", None),
    (
        "ACE-Step/Ace-Step1.5",
        "19671f406d603126926c1b7e2adc169acbcade22",
        ["vae/**", "Qwen3-Embedding-0.6B/**"],
    ),
    (
        "laion/clap-htsat-unfused",
        "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        None,
    ),
)
for repo_id, revision, allow_patterns in artifacts:
    print(f"Downloading {repo_id} at revision {revision or 'repository default'}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        token=token,
    )
print("Pinned ACE-Step SFT, VAE, Qwen, and external CLAP assets are cached.")
PY

# Run cheap unit tests and all opt-in diagnostics whose prerequisites were supplied.
# Real-audio CLAP tests should skip themselves unless both fixture paths are set.
if [[ -n "${TRUMPET_AUDIO_PATH:-}" || -n "${NON_TRUMPET_AUDIO_PATH:-}" ]]; then
  if [[ ! -f "${TRUMPET_AUDIO_PATH:-}" || ! -f "${NON_TRUMPET_AUDIO_PATH:-}" ]]; then
    echo "Set both TRUMPET_AUDIO_PATH and NON_TRUMPET_AUDIO_PATH to readable licensed clips, or neither." >&2
    exit 2
  fi
  export RUN_CLAP_DIAGNOSTIC=1
  export CLAP_TRUMPET_AUDIO="${TRUMPET_AUDIO_PATH}"
  export CLAP_NON_TRUMPET_AUDIO="${NON_TRUMPET_AUDIO_PATH}"
fi
python -m pytest -q 2>&1 | tee "${LOG_DIR}/pytest.log"

# An optional real-model GPU diagnostic may be expensive. Unit tests and the smoke
# training below run regardless; this block is enabled explicitly.
if [[ "${RUN_REAL_ACE_DIAGNOSTICS:-0}" == "1" ]]; then
  RUN_ACE_GPU_TESTS=1 python -m pytest -q -m gpu 2>&1 | tee "${LOG_DIR}/pytest-real-ace.log"
fi

# If a raw simple dataset is provided, create group-aware splits inside
# /kaggle/working and use its training partition.
if [[ ! -f "${TRAIN_CSV}" && -n "${SIMPLE_DATASET_CSV}" ]]; then
  if [[ ! -f "${SIMPLE_DATASET_CSV}" ]]; then
    echo "SIMPLE_DATASET_CSV does not exist: ${SIMPLE_DATASET_CSV}" >&2
    exit 2
  fi
  SPLIT_DIR="${OUTPUT_ROOT}/dataset-splits"
  python scripts/create_simple_splits.py \
    --input "${SIMPLE_DATASET_CSV}" \
    --output-dir "${SPLIT_DIR}" \
    --seed 42 2>&1 | tee "${LOG_DIR}/dataset-split.log"
  TRAIN_CSV="${SPLIT_DIR}/train.csv"
fi
if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "Training CSV not found: ${TRAIN_CSV}" >&2
  echo "Attach a CSV and set TRAIN_CSV=/kaggle/input/<dataset>/train.csv, or set SIMPLE_DATASET_CSV to create splits." >&2
  exit 2
fi

echo "Starting the mandatory one-sample, two-step, two-second smoke training."
python scripts/train.py \
  --train-csv "${TRAIN_CSV}" \
  --output-dir "${SMOKE_OUTPUT}" \
  --model-id "${ACE_MODEL_ID}" \
  --model-revision "${ACE_MODEL_REVISION}" \
  --components-id "${ACE_COMPONENTS_ID}" \
  --components-revision "${ACE_COMPONENTS_REVISION}" \
  --clap-model-id "${CLAP_MODEL_ID}" \
  --clap-revision "${CLAP_REVISION}" \
  --device cuda \
  --dtype float32 \
  --guidance-mode apg \
  --guidance-scale 7.0 \
  --shift 1.0 \
  --sequential-dit \
  --offload-dit-after-generation \
  --smoke-test 2>&1 | tee "${LOG_DIR}/train-smoke.log"

echo "Smoke artifacts: ${SMOKE_OUTPUT}"
if [[ "${RUN_FULL_TRAINING:-0}" != "1" ]]; then
  echo "Smoke test complete. Set RUN_FULL_TRAINING=1 to launch the first real experiment."
  exit 0
fi

REAL_STEPS="${REAL_STEPS:-4}"
REAL_DURATION="${REAL_DURATION:-10}"
REAL_EPOCHS="${REAL_EPOCHS:-1}"
REAL_MAX_SAMPLES="${REAL_MAX_SAMPLES:-8}"
REAL_GRAD_ACCUM="${REAL_GRAD_ACCUM:-4}"

if ! python -c 'import math, sys; value = float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) and 0 < value <= 10 else 1)' "${REAL_DURATION}"; then
  echo "REAL_DURATION must be a finite number in (0, 10]; CLAP scores one complete 10-second window." >&2
  exit 2
fi

TRAIN_ARGS=(
  --train-csv "${TRAIN_CSV}"
  --output-dir "${TRAIN_OUTPUT}"
  --model-id "${ACE_MODEL_ID}"
  --model-revision "${ACE_MODEL_REVISION}"
  --components-id "${ACE_COMPONENTS_ID}"
  --components-revision "${ACE_COMPONENTS_REVISION}"
  --clap-model-id "${CLAP_MODEL_ID}"
  --clap-revision "${CLAP_REVISION}"
  --device cuda
  --dtype float32
  --batch-size 1
  --grad-accum-steps "${REAL_GRAD_ACCUM}"
  --epochs "${REAL_EPOCHS}"
  --max-samples "${REAL_MAX_SAMPLES}"
  --audio-length-in-s "${REAL_DURATION}"
  --num-inference-steps "${REAL_STEPS}"
  --guidance-mode apg
  --guidance-scale 7.0
  --shift 1.0
  --sequential-dit
  --offload-dit-after-generation
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ "${REAL_EPOCHS}" -le 1 ]]; then
    echo "When RESUME_CHECKPOINT is set, choose REAL_EPOCHS greater than the epoch stored in it (for example 2)." >&2
    exit 2
  fi
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Starting the first experiment: ${REAL_MAX_SAMPLES} samples, ${REAL_STEPS} flow steps, ${REAL_DURATION}s audio."
python scripts/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train-first-experiment.log"

echo "Training complete."
echo "Latest checkpoint: ${TRAIN_OUTPUT}/checkpoints/latest.pt"
echo "Best checkpoint: ${TRAIN_OUTPUT}/checkpoints/best.pt"
echo "All artifacts and logs: ${OUTPUT_ROOT}"
