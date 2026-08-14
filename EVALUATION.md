# Evaluation pipeline

`scripts/evaluate.py` evaluates checkpoints

## What it compares

Every prompt/seed pair is generated with identical initial noise under:

- `base`: constant `alpha=0`, exactly preserving full-prompt CFG;
- `learned`: the `SteeringPredictor` loaded from the checkpoint, interpolating from full-prompt CFG
  towards retain-prompt CFG;
- optional fixed-alpha controls supplied through `--fixed-alphas`.

Evaluation runs with batch size one because the current pipeline records alpha averaged across the batch. This makes each recorded schedule belong unambiguously to one prompt.

## Before running

For a reportable suppression experiment, first use
[`scripts/prepare_baselines_stable_audio_3.py`](BASELINE_SELECTION.md) to verify that each preassigned
target is present in its paired `alpha=0` prompt/seed baseline. This prevents an absent baseline
target from being counted as successful suppression. The selector also records which retain
instruments were present.

The preferred evaluator input is now the selector output directory:

```powershell
uv run python scripts/evaluate.py `
  --baseline-selection outputs/trumpet-baseline-selection `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-selected
```

This mode evaluates only target-valid records, uses their explicit seeds, takes paired generation
settings from the selector config, and scores the exact saved WAV as `base`. Conflicting settings are
rejected rather than silently creating an unpaired comparison. The legacy `--dataset`, `--num-seeds`
and `--seed` mode remains supported for diagnostics.

First produce target-valid paired-alpha0 selections for the training and validation splits, using
different numeric seed ranges, then train from those manifests:

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

The best checkpoint is selected by validation loss. It records hashes and metadata for both
selection manifests and datasets, their target/seed counts, the complete generation configuration,
optimizer settings, target vocabulary, training metrics and validation metrics. Legacy `--dataset`
training remains supported, but selects by training loss because it has no held-out selection.

Create the group-aware splits first with:

```powershell
uv run python scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset.csv `
  --output-dir datasets/trumpet_simple_splits `
  --seed 42
```

The split contains 132/44/44 train/validation/test rows. The two templates that share a genre and
accompaniment are assigned together, preventing their near-duplicate pair from leaking across splits.

For reportable results, evaluate a held-out CSV that was not used for either training or checkpoint
selection. The evaluator checks the evaluation path against both dataset paths stored in new
checkpoints and writes a leakage warning when it matches the training or validation split. Reusing
either remains useful as a diagnostic, but does not measure held-out test generalization.

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
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-smoke `
  --max-samples 4 `
  --num-seeds 1 `
  --num-inference-steps 8
```

## Final paired evaluation

Use multiple seeds and include fixed-alpha controls to test whether the learned schedule improves
over a constant intervention:

```powershell
uv run python scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-final `
  --num-seeds 5 `
  --fixed-alphas 0.25 0.5 0.75 1.0
```

For a target-valid final evaluation, prefer the selection-driven equivalent:

```powershell
uv run python scripts/evaluate.py `
  --baseline-selection outputs/trumpet-baseline-selection `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/eval-final-selected `
  --fixed-alphas 0.25 0.5 0.75 1.0
```

Checkpoints whose `steering_mode` does not match the evaluator's are intentionally rejected. That
covers the previous global-CFG formula, whose `alpha` values have a different meaning, and every
checkpoint trained against MusicLDM, which does not share a latent space with Stable Audio 3.

Generation settings default to those stored in the checkpoint and can be overridden with:

- `--model`, which has to name a `-base` checkpoint;
- `--num-inference-steps`;
- `--audio-length-in-s`;
- `--cfg-scale` and `--apg-scale`;
- `--steering-frac-start` and `--steering-frac-end`.

That override rule applies to legacy `--dataset` mode. With `--baseline-selection`, model, step count,
duration, CFG/APG, negative prompt, precision and decode mode are inherited from the saved baselines;
the steering window must also match both the selection and the checkpoint that was trained from it.
A conflicting paired-setting override is rejected.

The diffusion transformer is loaded in half precision unless `--no-half` is passed; the latent
trajectory, the guidance algebra and the autoencoder are always float32. Audio saving can be disabled
with `--no-save-audio`, although paired listening is strongly recommended. Saved clips are stereo at
44.1 kHz; CLAP scores a mono downmix of them, since its audio tower is mono.

An existing non-empty output directory is never replaced accidentally. Pass `--replace-output`
only when the named evaluation directory may be removed and recreated.

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

With `--baseline-selection`, every method row additionally records the baseline classifier's target
score, requested and retain instruments, per-retain score/validity, and the lists
`baseline_valid_retain_instruments` / `baseline_invalid_retain_instruments`. A retain instrument that
was absent before steering is not evidence of collateral damage; the global CLAP retain score remains
available as a prompt-level diagnostic. `summary.json` additionally reports
`retain_similarity_change_all_retain_baseline_valid` for the clean subset where every requested retain
instrument was detected in the baseline.

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
human-readable summary. In selection mode `audio/base/` contains exact copies of the saved selector
baselines rather than regenerated clips.
