# Stable Audio 3 Steering Pipeline

This project focuses on applying a target-specific steering on Stable Audio 3 by
interpolating between full-prompt and retain-prompt classifier-free guidance. 

We used `stable_audio_3` library to override `DiffusionTransformer.forward`, where
classifier-free guidance is actually computed. See `pipelines/steering_stable_audio_pipeline.py`.

Only the `-base` checkpoints work. The post-trained ones (`small-music`, `medium`, `small-sfx`) are
distilled and ignore `cfg_scale`, and steering lives inside the guidance branch.
The default is `small-music-base` since `medium-base` is too heavy to train. The models are available on HuggingFace.

## Train

`scripts/train.py` trains a `MagnitudePredictor`: a learned per-sample gain on the
deterministic CFG-diff shape (`steering_mode="cfg_diff_magnitude"`), in contrast to the fixed
`magnitude` hyperparameter `steering_mode="cfg_diff"` uses.

```powershell
uv run scripts/train.py `
  --dataset datasets/train_set.csv `
  --validation-dataset datasets/validation_set `
  --output outputs/small-music-base-training `
  --experiments-csv experiments/base.csv
```

## Evaluate

`scripts/evaluate.py` is used to evaluate the different output models provided by `scripts/train.py`.
It also has a zero-cost, training-free `cfg-diff` method that needs no checkpoint at all.

```powershell
uv run scripts/evaluate.py `
  --dataset datasets/test_csv.csv `
  --experiments-csv outputs/small-music-base-training/experiments.csv`
  --output  outputs/small-music-base-evaluate
```

## Plain generation

`scripts/generate_audio_stable_audio_3.py` generates with the stock Stable Audio 3, which is the unsteered reference the evaluation can be sanity-checked against.
