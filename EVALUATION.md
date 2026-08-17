# Evaluation pipeline

`scripts/evaluate.py` evaluates checkpoints

## What it compares

Every prompt/seed pair is generated with identical initial noise under:

- `base`: constant `alpha=0`, exactly preserving full-prompt CFG. Always evaluated, and independent
  of the three pipelines below — it's the reference point every other method is compared against;
- `learned`: the `SteeringPredictor` loaded from `--predictor-checkpoint`, interpolating from
  full-prompt CFG towards retain-prompt CFG. Skipped when `--predictor-checkpoint` is omitted;
- optional fixed-alpha controls supplied through `--fixed-alpha-values`;
- `cfg_diff`: a zero-cost, training-free method, always evaluated.

Each of the three pipelines (`predictor`, `fixed-alpha`, `cfg-diff`) has its own prefixed CLI
parameters (e.g. `--predictor-cfg-scale`, `--fixed-alpha-cfg-scale`, `--cfg-diff-cfg-scale`), so they
can be tuned and tested independently.

Evaluation runs with batch size one because the current pipeline records alpha averaged across the batch. This makes each recorded schedule belong unambiguously to one prompt.

## Before running

First produce a checkpoint with the existing training entry point, for example:

```powershell
uv run python scripts/train.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-target-specific
```

Create the group-aware splits first with:

```powershell
uv run python scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset.csv `
  --output-dir datasets/trumpet_simple_splits `
  --seed 42
```

The split contains 132/44/44 train/validation/test rows. The two templates that share a genre and
accompaniment are assigned together, preventing their near-duplicate pair from leaking across splits.

For reportable results, evaluate a held-out CSV that was not passed to `train.py`. The evaluator
checks the evaluation path against the dataset path stored in the checkpoint and writes a leakage
warning when they match. Evaluating the training CSV remains useful as a diagnostic or smoke test,
but does not measure generalization.

The preferred CSV contract contains an explicit retain prompt:

```csv
prompt,target,retain_prompt
"A jazz piece featuring bright trumpet and swung drums","trumpet","A jazz piece featuring swung drums"
```

Legacy `prompt,target` CSV files remain supported; in that case the retain prompt is derived by
lexically removing the target. An explicit `retain_prompt` is always preferred.

## Smoke test

This checks the end-to-end evaluator with a small workload. Eight denoising steps are useful only
for checking that the machinery works; they are not comparable to a full 50-step evaluation.

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --predictor-checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-smoke `
  --max-samples 4 `
  --num-seeds 1 `
  --predictor-num-inference-steps 8
```

## Final paired evaluation

Use multiple seeds and include fixed-alpha controls to test whether the learned schedule improves
over a constant intervention:

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --predictor-checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-final `
  --num-seeds 5 `
  --fixed-alpha-values 0.25 0.5 0.75 1.0
```

Checkpoints whose `steering_mode` does not match the evaluator's are intentionally rejected. That
covers the previous global-CFG formula, whose `alpha` values have a different meaning, and every
checkpoint trained against MusicLDM, which does not share a latent space with Stable Audio 3.

`--model` names the Stable Audio 3 `-base` checkpoint loaded for every pipeline (it has to be a
`-base` checkpoint); it defaults to the predictor checkpoint's own model when
`--predictor-checkpoint` is given, else to `small-music-base`.

Each of the three pipelines resolves its own generation settings independently, through its own
prefixed flags:

- `--predictor-num-inference-steps`, `--predictor-audio-length-in-s`, `--predictor-cfg-scale`,
  `--predictor-apg-scale`, `--predictor-steering-frac-start`, `--predictor-steering-frac-end` —
  default to the checkpoint's stored training settings, then to hard-coded defaults;
- `--fixed-alpha-num-inference-steps`, `--fixed-alpha-audio-length-in-s`, `--fixed-alpha-cfg-scale`,
  `--fixed-alpha-apg-scale`, `--fixed-alpha-steering-frac-start`, `--fixed-alpha-steering-frac-end` —
  default straight to hard-coded defaults, since this pipeline never needs a checkpoint;
- `--cfg-diff-num-inference-steps`, `--cfg-diff-audio-length-in-s`, `--cfg-diff-cfg-scale`,
  `--cfg-diff-apg-scale`, `--cfg-diff-steering-frac-start`, `--cfg-diff-steering-frac-end` — same
  fallback as fixed-alpha.

The `base` reference always uses the hard-coded defaults and has no flags of its own, since its
output doesn't depend on the steering window.

The diffusion transformer is loaded in half precision unless `--no-half` is passed; the latent
trajectory, the guidance algebra and the autoencoder are always float32. Audio saving can be disabled
with `--no-save-audio`, although paired listening is strongly recommended. Saved clips are stereo at
44.1 kHz; CLAP scores a mono downmix of them, since its audio tower is mono.

An existing non-empty output directory is never replaced accidentally; choose another `--output`.

## Metrics

For each generated waveform the evaluator records:

- `target_similarity`: CLAP cosine similarity between audio and target; lower is better;
- `retain_similarity`: CLAP cosine similarity between audio and the explicit retain prompt, or the
  fallback produced by `utils.strip_target` for legacy datasets; higher is better, and this is the
  term `scripts/train.py` optimizes alongside suppression;
- `prompt_similarity`: CLAP cosine similarity between audio and the full prompt;
- RMS, peak, near-silence ratio and clipping ratio;
- alpha mean, standard deviation, minimum and maximum;
- fraction of steered steps with `alpha > 0.5`, where the interpolation weight assigned to
  retain-prompt guidance is larger than the weight assigned to full-prompt guidance.

Paired metrics compare a method with `base` for the same prompt and seed:

```text
target suppression gain  = target_similarity(base) - target_similarity(method)
retain similarity change = retain_similarity(method) - retain_similarity(base)
prompt similarity change = prompt_similarity(method) - prompt_similarity(base)
```

A positive suppression gain is desirable, and the trade-off plot reads it against the retain
similarity change: a large gain paired with a negative retain change usually means the audio was
degraded rather than the concept removed.

Prompt-similarity change is only a coarse fidelity proxy: in the current dataset the full prompt
includes the target word `trumpet`, so suppressing the target and matching the full prompt are
partially conflicting objectives. Explicit retain prompts remove that conflict more cleanly. For
legacy two-column datasets the fallback derivation is lexical, so modifiers of the target survive it
(`"muted trumpet with a plunger mute"` becomes `"muted with a plunger mute"`).

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
results.csv
summary.json
target_similarity.png
suppression_fidelity_tradeoff.png
alpha_schedules.png
audio/
  base/
  learned/
  fixed_alpha_*/
  cfg_diff/
```

`results.csv` is the detailed table with one row per prompt, seed and method. `summary.json`
contains prompt-clustered bootstrap confidence intervals.
