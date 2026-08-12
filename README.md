# MusicLDM Steering Pipeline

The current experiment learns target-specific steering for one instrument (`trumpet`) by
interpolating between full-prompt and retain-prompt classifier-free guidance.

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
  --output outputs/trumpet-target-specific
```

`alpha` starts at `0.15`: `0` follows full-prompt guidance and `1` follows retain-prompt guidance.
The existing weighted target/retain loss is unchanged.

## Evaluate

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/trumpet-target-specific-validation
```

See [EVALUATION.md](EVALUATION.md) for the full paired evaluation workflow.
