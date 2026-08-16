# Evaluation pipeline

`scripts/evaluate.py` compares the learned ACE-Step 1.5 SFT controller with a paired `alpha=0`
baseline and optional fixed-alpha controls. Every method starts from identical latent noise for each
prompt/seed pair.

## Recommended workflow

Generate target-valid selections for train and validation with disjoint seed ranges, then train:

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

For final evaluation, first select target-valid baselines from the held-out test split:

```powershell
uv run python scripts/prepare_baselines_ace_step.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --output outputs/trumpet-test-baselines `
  --baseline-mode paired-alpha0 --num-seeds 5 --seed 200000

uv run python scripts/evaluate.py `
  --baseline-selection outputs/trumpet-test-baselines `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-final `
  --fixed-alphas 0.25 0.5 0.75 1.0
```

Selection mode evaluates only target-valid records, uses their explicit seeds, and scores the exact
saved baseline WAV as `base`. Model, precision, step count, duration, CFG scale, timestep shift, and
steering window are checked against the training checkpoint. Conflicting overrides are rejected.

The legacy `--dataset`, `--num-seeds`, and `--seed` path remains useful for diagnostics. For a small
plumbing check:

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-smoke `
  --max-samples 4 --num-seeds 1 --num-inference-steps 8
```

Eight steps are only a software smoke test; the SFT checkpoint's normal setting is 50 steps.

## Compared methods

- `base`: fixed `alpha=0`, preserving complete-prompt APG;
- `learned`: the trained per-step predictor;
- `fixed_alpha_*`: optional constant controls from `--fixed-alphas`.

Evaluation uses batch size one so each recorded alpha schedule belongs to one prompt. `alpha=1`
fully selects retain-prompt APG within the steering window; outside the window all methods follow the
complete prompt.

## Configuration contract

The only accepted backbone is `ACE-Step/acestep-v15-sft`. The important generation controls are:

- `--num-inference-steps` (50 by default);
- `--audio-length-in-s` (10–600 seconds; experiments default to 10 for CLAP coverage);
- `--cfg-scale` (7 by default and greater than 1 for steering);
- `--shift` (1 by default for SFT);
- `--steering-frac-start` and `--steering-frac-end`.

CUDA loads the DiT/text encoder in BF16 unless `--no-half` is used. The Euler trajectory and VAE
decoder stay float32. Output is 48 kHz stereo; CLAP scores the mono channel average.

Checkpoints carry a versioned `steering_mode`. Predictors trained for Stable Audio 3, MusicLDM, an
older guidance formula, or a different latent layout are deliberately rejected.

## Metrics

For each waveform the evaluator records:

- `target_similarity`: CLAP audio/target cosine similarity; lower is better;
- `retain_similarity`: CLAP audio/retain-prompt similarity; higher is better;
- `prompt_similarity`: CLAP audio/full-prompt similarity;
- RMS, peak, near-silence ratio, and clipping ratio;
- alpha mean, standard deviation, minimum, maximum, and retain-dominant ratio.

Paired changes are calculated within the same prompt and seed:

```text
target suppression gain  = target_similarity(base) - target_similarity(method)
retain similarity change = retain_similarity(method) - retain_similarity(base)
prompt similarity change = prompt_similarity(method) - prompt_similarity(base)
```

A positive suppression gain is desirable. A large gain combined with a strongly negative retain
change usually indicates degraded audio rather than selective instrument removal. Selection mode
also propagates the independent classifier's target and retain-instrument scores. Confidence
intervals bootstrap prompts, after averaging multiple seeds within a prompt.

CLAP is also used during training, so report these numbers together with the independent instrument
classifier and paired listening.

## Outputs

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

In selection mode `audio/base/` contains copies of the saved selector baselines. A non-empty output
directory is never overwritten unless `--replace-output` is explicitly passed.
