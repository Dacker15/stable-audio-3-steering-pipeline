# Stable Audio 3 Steering Pipeline

The current experiment learns target-specific steering for one instrument (`trumpet`) by
interpolating between full-prompt and retain-prompt classifier-free guidance.

Stable Audio 3 is not supported by diffusers, so the steering is not a pipeline subclass: it
overrides `DiffusionTransformer.forward`, the component of the `stable_audio_3` library where
classifier-free guidance is actually computed. See `pipelines/steering_stable_audio_pipeline.py`.

Only the `-base` checkpoints work. The post-trained ones (`small-music`, `medium`, `small-sfx`) are
distilled and ignore `cfg_scale`, and steering lives inside the guidance branch. The default is
`medium-base`; `small-music-base` is the lighter fallback. Both are gated on HuggingFace, so accept
the licence on the model page and authenticate with `hf auth login` or an `HF_TOKEN` variable first.

## Prepare the simplified dataset

Place `trumpet_prompts_simple_dataset.csv` in `datasets/`, then create deterministic group-aware
splits. Each pair of equivalent prompt templates stays in the same split.

```powershell
uv run python scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset.csv `
  --output-dir datasets/trumpet_simple_splits `
  --seed 42
```

This creates 132 training rows, 44 validation rows and 44 test rows.

## Train

First generate paired `alpha=0` baselines on the training and validation splits. These calls use
different seed ranges, and only records where the preassigned target is detected will be consumed by
training:

```powershell
uv run python scripts/prepare_baselines_stable_audio_3.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-train-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 1000

uv run python scripts/prepare_baselines_stable_audio_3.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --output outputs/trumpet-validation-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 100000

uv run python scripts/train.py `
  --train-selection outputs/trumpet-train-baselines `
  --validation-selection outputs/trumpet-validation-baselines `
  --output outputs/trumpet-target-specific
```

`alpha` starts at `0.15`: `0` follows full-prompt guidance and `1` follows retain-prompt guidance.
Both selections must use identical model/generation/classifier settings, disjoint dataset groups and
disjoint seed values. The training loader contains only target-valid prompt/seed pairs. Validation
runs on its fixed held-out pairs after every epoch and `steering_predictor_best.pt` is selected by
validation loss. CLAP is loaded on its own — Stable Audio 3 conditions on T5Gemma — and provides
both the loss and the target embedding the predictor is conditioned on.

The historical `--dataset` training mode remains available for compatibility, but has no target-valid
validation selection and therefore falls back to choosing the best checkpoint by training loss.

## Evaluate

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/trumpet-target-specific-validation
```

See [EVALUATION.md](EVALUATION.md) for the full paired evaluation workflow.

## Select valid prompt/seed baselines

Before a suppression evaluation, generate paired `alpha=0` baselines and verify that the target
assigned by the dataset is actually audible according to an independent multi-label AudioSet
classifier:

```powershell
uv run python scripts/prepare_baselines_stable_audio_3.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --output outputs/trumpet-baseline-selection `
  --num-seeds 5 --seed 200000
```

The target is fixed from the CSV before classification. Only prompt/seed pairs where it is detected
are written to `eligible_pairs.csv`; complete scores, invalid pairs, requested instruments, retain
instrument validity and saved baseline audio remain in the full manifests. See
[BASELINE_SELECTION.md](BASELINE_SELECTION.md) for the schema, thresholding, exact/proxy AudioSet
labels, and the same-seed contract used by the integrated paired baseline/steering evaluation.

Evaluate those exact target-valid pairs with:

```powershell
uv run python scripts/evaluate.py `
  --baseline-selection outputs/trumpet-baseline-selection `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/trumpet-eval-selected
```

## Plain generation

`scripts/generate_audio_stable_audio_3.py` generates with the stock Stable Audio 3, which is the
unsteered reference the evaluation can be sanity-checked against.
`scripts/generate_audio_stable_audio.py` (Stable Audio Open 1.0) and `scripts/generate_audio.py`
(MusicLDM, the backbone this project used before the migration) are kept as historical baselines.
