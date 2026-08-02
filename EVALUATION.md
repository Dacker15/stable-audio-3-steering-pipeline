# Evaluation pipeline

`scripts/evaluate.py` evaluates checkpoints

## What it compares

Every prompt/seed pair is generated with identical initial noise under:

- `base`: constant `alpha=0`, exactly preserving the ordinary CFG formula;
- `learned`: the `SteeringPredictor` loaded from the checkpoint;
- optional fixed-alpha controls supplied through `--fixed-alphas`.

Evaluation runs with batch size one because the current pipeline records alpha averaged across the batch. This makes each recorded schedule belong unambiguously to one prompt.

## Before running

First produce a checkpoint with the existing training entry point, for example:

```powershell
uv run python scripts/train.py `
  --dataset datasets/trumpet_prompts_dataset.csv `
  --output outputs/trumpet
```

For reportable results, evaluate a held-out CSV that was not passed to `train.py`. The evaluator
checks the evaluation path against the dataset path stored in the checkpoint and writes a leakage
warning when they match. Evaluating the training CSV remains useful as a diagnostic or smoke test,
but does not measure generalization.

The current CSV contract remains unchanged:

```csv
prompt,target
"A jazz piece featuring bright trumpet and swung drums","trumpet"
```

## Smoke test

This checks the end-to-end evaluator with a small workload. Twenty denoising steps are useful only
for checking that the machinery works; they are not comparable to a full 200-step evaluation.

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_prompts_dataset.csv `
  --checkpoint outputs/trumpet/steering_predictor_best.pt `
  --output outputs/eval-smoke `
  --max-samples 4 `
  --num-seeds 1 `
  --num-inference-steps 20
```

## Final paired evaluation

Use multiple seeds and include fixed-alpha controls to test whether the learned schedule improves
over a constant intervention:

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_test.csv `
  --checkpoint outputs/trumpet/steering_predictor_best.pt `
  --output outputs/eval-final `
  --num-seeds 5 `
  --fixed-alphas 0.25 0.5 0.75 1.0
```

Generation settings default to those stored in the checkpoint and can be overridden with:

- `--num-inference-steps`;
- `--audio-length-in-s`;
- `--guidance-scale`;
- `--steering-frac-start` and `--steering-frac-end`.

On CUDA, dtype `auto` uses float16. On CPU it uses float32. Audio saving can be disabled with
`--no-save-audio`, although paired listening is strongly recommended.

An existing non-empty output directory is never replaced accidentally. Pass `--replace-output`
only when the named evaluation directory may be removed and recreated.

## Metrics

For each generated waveform the evaluator records:

- `target_similarity`: CLAP cosine similarity between audio and target; lower is better;
- `prompt_similarity`: CLAP cosine similarity between audio and the full prompt;
- RMS, peak, near-silence ratio and clipping ratio;
- alpha mean, standard deviation, minimum and maximum;
- fraction of steered steps with `alpha > 0.5`, where guidance is inverted.

Paired metrics compare a method with `base` for the same prompt and seed:

```text
target suppression gain = target_similarity(base) - target_similarity(method)
prompt similarity change = prompt_similarity(method) - prompt_similarity(base)
```

A positive suppression gain is desirable. Prompt-similarity change is only a coarse fidelity proxy:
in the current dataset the full prompt includes the target word `trumpet`, so suppressing the target
and matching the full prompt are partially conflicting objectives.

Confidence intervals use prompts as the independent units. Results from multiple seeds are averaged
within each prompt before bootstrap resampling, avoiding artificially narrow intervals from treating
seeds of the same prompt as independent observations.

CLAP is also used by training, so these metrics must be complemented by paired listening and,
eventually, an independent instrument classifier. RMS, silence and clipping are sanity checks rather
than perceptual-quality metrics.

## Outputs

The evaluation directory contains:

```text
config.json
sample_metrics.csv
alpha_records.jsonl
summary.json
report.md
target_similarity.png
suppression_fidelity_tradeoff.png
alpha_schedules.png
audio/
  base/
  learned/
  fixed_alpha_*/
```

`sample_metrics.csv` is the detailed table with one row per prompt, seed and method. `summary.json`
contains prompt-clustered bootstrap confidence intervals, while `report.md` provides the compact
human-readable summary.
