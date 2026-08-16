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

```powershell
uv run python scripts/train.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-target-specific `
  --model medium-base --cfg-scale 7.0 --num-inference-steps 50
```

`alpha` starts at `0.15`: `0` follows full-prompt guidance and `1` follows retain-prompt guidance.
The existing weighted target/retain loss is unchanged. CLAP is loaded on its own — Stable Audio 3
conditions on T5Gemma — and provides both the loss and the target embedding the predictor is
conditioned on.

## Evaluate

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/trumpet-target-specific-validation
```

See [EVALUATION.md](EVALUATION.md) for the full paired evaluation workflow.

## Plain generation

`scripts/generate_audio_stable_audio_3.py` generates with the stock Stable Audio 3, which is the
unsteered reference the evaluation can be sanity-checked against.