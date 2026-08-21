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

`scripts/train_regime_b.py` trains a `MagnitudePredictor`: a learned per-sample gain on Regime A's
deterministic CFG-diff shape ("Regime B", `steering_mode="cfg_diff_magnitude"`).

```powershell
uv run python scripts/train_regime_b.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-regime-b `
  --epochs 5 --margin-target 0.30 --margin-retain 0.40 `
  --lambda-reg 0.01 --lambda-reg-warmup-steps 200 --lambda-fid 0.1
```

CLAP is loaded on its own — Stable Audio 3 conditions on T5Gemma — and provides both the loss and
the target embedding the predictor is conditioned on. `scripts/evaluate.py` also has a zero-cost,
training-free `cfg-diff` method that needs no checkpoint at all.

## Evaluate

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --magnituder-checkpoint outputs/trumpet-regime-b/magnitude_predictor_best.pt `
  --output outputs/trumpet-regime-b-validation
```

See [EVALUATION.md](EVALUATION.md) for the full paired evaluation workflow.

## Plain generation

`scripts/generate_audio_stable_audio_3.py` generates with the stock Stable Audio 3, which is the
unsteered reference the evaluation can be sanity-checked against.