# ACE-Step 1.5 SFT Steering Pipeline

This project learns target-specific instrument steering (currently `trumpet`) by interpolating at
every active denoising step between:

- ACE-Step APG for the complete prompt (`alpha=0`), and
- ACE-Step APG for the retain prompt with the target removed (`alpha=1`).

The backbone is pinned to the official 2B checkpoint `ACE-Step/acestep-v15-sft`. The loader rejects
all other variants. In particular it does **not** use `acestep-v15-base`, which is the checkpoint
that supports Extract/Lego/Complete, nor Turbo, whose distilled path has no CFG branch. Generation
uses text-to-music only: no extractor, source-audio editing, or 5 Hz language-model planner.

The implementation is in `pipelines/steering_ace_step_pipeline.py`. It loads the SFT DiT and its
official shared Qwen3 embedding encoder, silence latent, and Oobleck VAE. The standard defaults are
50 Euler steps, CFG/APG scale 7, timestep shift 1, 48 kHz stereo, and 10-second clips.

## Install

```powershell
uv sync
```

Use Python 3.11 or 3.12. On 64-bit Windows the lockfile installs the ACE-Step-compatible
PyTorch/torchaudio 2.7.1 CUDA 12.8 pair. The first model-backed command downloads the public
ACE-Step weights from Hugging Face. CUDA uses the checkpoint's native BF16; `--no-half` selects
float32. The differentiable latent trajectory and VAE decoder remain float32 during training.

## Prepare the simplified dataset

Place `trumpet_prompts_simple_dataset.csv` in `datasets/`, then create deterministic group-aware
splits. Equivalent prompt templates remain in the same split.

```powershell
uv run python scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset.csv `
  --output-dir datasets/trumpet_simple_splits `
  --seed 42
```

This creates 132 training rows, 44 validation rows, and 44 test rows.

### Larger two-instrument dataset

`datasets/trumpet_simple_splits_big` contains a 440-prompt alternative. Every prompt is an explicit
duet: trumpet plus exactly one retain instrument. Its group-aware split contains 264/88/88
train/validation/test rows while paired templates remain confined to one split.

The committed files can be regenerated with:

```powershell
uv run python scripts/create_big_trumpet_dataset.py --overwrite

uv run python scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset_big.csv `
  --output-dir datasets/trumpet_simple_splits_big `
  --seed 42 --overwrite
```

### Multi-target two-instrument dataset

`datasets/multi_instrument_splits` is the balanced multi-target alternative. It covers 15 target
instruments and all 105 unordered pairs between them. Every prompt requests exactly two instruments,
exposes them separately, and then asks for call-and-response or a joint ending so both timbres have
a clear chance to be heard. All four variants of an instrument pair remain in the same split; every
instrument appears as a target and as a retain instrument in train, validation, and test. The split
sizes are 300/60/60.

Prompt wording raises the probability that both instruments are rendered, but no text-to-music
model can guarantee that for every seed. During baseline preparation, use the recorded
`requested_instrument_validity` and `all_retain_valid` fields to retain only generations in which
the independent AudioSet classifier detects both requested instruments.

Regenerate the raw 420-row CSV and its deterministic splits with:

```powershell
uv run python scripts/create_multi_instrument_dataset.py --seed 42 --overwrite
```

Use the new dataset by pointing baseline preparation to, for example:

```text
--dataset datasets/multi_instrument_splits/train.csv
```

## Train

First generate paired `alpha=0` baselines for training and validation. Only records where the
preassigned target is detected by the independent AudioSet classifier are consumed by training.

```powershell
uv run python scripts/prepare_baselines_ace_step.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-train-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 1000

uv run python scripts/prepare_baselines_ace_step.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --output outputs/trumpet-validation-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 100000

uv run python scripts/train.py `
  --train-selection outputs/trumpet-train-baselines `
  --validation-selection outputs/trumpet-validation-baselines `
  --output outputs/trumpet-target-specific
```

Both selections must have identical generation/classifier settings and disjoint dataset groups and
seeds. `alpha` initializes at `0.15`. CLAP supplies both the suppression/retention loss and the target
embedding that conditions the predictor; ACE-Step itself uses Qwen3 for prompt conditioning. The
best checkpoint is selected by fixed held-out validation loss.

The historical `--dataset` training mode remains available, but it has no target-valid validation
selection and therefore selects by training loss.

## Evaluate

Prepare target-valid test baselines, then evaluate those exact prompt/seed pairs:

```powershell
uv run python scripts/prepare_baselines_ace_step.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --output outputs/trumpet-baseline-selection `
  --num-seeds 5 --seed 200000

uv run python scripts/evaluate.py `
  --baseline-selection outputs/trumpet-baseline-selection `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/trumpet-eval-selected `
  --fixed-alphas 0.25 0.5 0.75 1.0
```

See [BASELINE_SELECTION.md](BASELINE_SELECTION.md) for selection semantics and
[EVALUATION.md](EVALUATION.md) for metrics, controls, and outputs. Old Stable Audio 3 and MusicLDM
predictor checkpoints are intentionally incompatible and must be retrained.

## Plain generation

```powershell
uv run python scripts/generate_audio_ace_step.py `
  --prompt "A laid-back jazz trumpet solo over walking bass" `
  --output outputs/generated
```

`scripts/generate_audio_stable_audio.py` (Stable Audio Open 1.0) and `scripts/generate_audio.py`
(MusicLDM) remain only as historical baselines.
