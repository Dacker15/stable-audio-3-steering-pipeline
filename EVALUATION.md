# Evaluation pipeline

`scripts/evaluate.py` evaluates checkpoints

## What it compares

Every prompt/seed pair is generated with identical initial noise under:

- `base`: constant `alpha=0`, exactly preserving full-prompt CFG. Always evaluated, and independent
  of the pipelines below — it's the reference point every other method is compared against;
- `magnituder`: the `MagnitudePredictor` loaded from `--magnituder-checkpoint`, a learned per-sample
  gain on the deterministic CFG-diff shape (in contrast to `cfg_diff`'s fixed `magnitude`
  hyperparameter). Skipped when `--magnituder-checkpoint` is omitted;
- optional fixed-alpha controls supplied through `--fixed-alpha-values`;
- `cfg_diff`: a zero-cost, training-free method, always evaluated.

Each of the three pipelines (`magnituder`, `fixed-alpha`, `cfg-diff`) has its own prefixed CLI
parameters (e.g. `--magnituder-cfg-scale`, `--fixed-alpha-cfg-scale`, `--cfg-diff-cfg-scale`), so they
can be tuned and tested independently.

Evaluation runs with batch size one because the current pipeline records alpha averaged across the batch. This makes each recorded schedule belong unambiguously to one prompt.

## Before running

To evaluate the `magnituder` method, first produce a checkpoint with `scripts/train.py`,
for example:

```powershell
uv run scripts/train_regime_b.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-regime-b `
  --epochs 5 --margin-target 0.30 --margin-retain 0.40 `
  --lambda-reg 0.01 --lambda-reg-warmup-steps 200 --lambda-fid 0.1
```

`base`, fixed-alpha and `cfg_diff` need no checkpoint at all.

Create the group-aware splits first with:

```powershell
uv run scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset.csv `
  --output-dir datasets/trumpet_simple_splits `
  --seed 42
```

The split contains 132/44/44 train/validation/test rows. The two templates that share a genre and
accompaniment are assigned together, preventing their near-duplicate pair from leaking across splits.

For reportable results, evaluate a held-out CSV that was not passed to `train.py`. The
evaluator checks the evaluation path against the dataset path stored in the checkpoint and writes a
leakage warning when they match. Evaluating the training CSV remains useful as a diagnostic or smoke
test, but does not measure generalization.

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
uv run scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --magnituder-checkpoint outputs/trumpet-regime-b/magnitude_predictor_best.pt `
  --output outputs/eval-smoke `
  --max-samples 4 `
  --num-seeds 1 `
  --magnituder-num-inference-steps 8
```

## Final paired evaluation

Use multiple seeds and include fixed-alpha controls to test whether the learned schedule improves
over a constant intervention:

```powershell
uv run scripts/evaluate.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --magnituder-checkpoint outputs/trumpet-regime-b/magnitude_predictor_best.pt `
  --output outputs/eval-final `
  --num-seeds 5 `
  --fixed-alpha-values 0.25 0.5 0.75 1.0
```

Checkpoints whose `steering_mode` does not match the evaluator's are intentionally rejected. That
covers every checkpoint trained against MusicLDM, which does not share a latent space with Stable
Audio 3.

`--model` names the Stable Audio 3 `-base` checkpoint loaded for every pipeline (it has to be a
`-base` checkpoint); it defaults to the magnituder checkpoint's own model when
`--magnituder-checkpoint` is given, else to `small-music-base`.

Each of the three pipelines resolves its own generation settings independently, through its own
prefixed flags:

- `--magnituder-num-inference-steps`, `--magnituder-audio-length-in-s`, `--magnituder-cfg-scale`,
  `--magnituder-apg-scale`, `--magnituder-steering-frac-start`, `--magnituder-steering-frac-end` —
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
Alignment Gain (CLAP)    = target_similarity(base) - target_similarity(method)
retain similarity change = retain_similarity(method) - retain_similarity(base)
prompt similarity change = prompt_similarity(method) - prompt_similarity(base)
```

A positive suppression gain is desirable, and the trade-off plot reads it against the retain
similarity change: a large gain paired with a negative retain change usually means the audio was
degraded rather than the concept removed. This field is still named `target_suppression_gain` in
`results.csv`/`summary.json` for backward compatibility; "Alignment Gain (CLAP)" is only the display
label (see `docs/metriche-valutazione-steering.md`).

Prompt-similarity change is only a coarse fidelity proxy: in the current dataset the full prompt
includes the target word `trumpet`, so suppressing the target and matching the full prompt are
partially conflicting objectives. Explicit retain prompts remove that conflict more cleanly. For
legacy two-column datasets the fallback derivation is lexical, so modifiers of the target survive it
(`"muted trumpet with a plunger mute"` becomes `"muted with a plunger mute"`).

Confidence intervals use prompts as the independent units. Results from multiple seeds are averaged
within each prompt before bootstrap resampling, avoiding artificially narrow intervals from treating
seeds of the same prompt as independent observations.

CLAP is also used by training, so these metrics must be complemented by paired listening and an
independent instrument classifier. RMS, silence and clipping are sanity checks rather than
perceptual-quality metrics.

### Additional metrics for `base` and `magnituder`

The following four metrics (adapted from TADA, Staniszewski et al., 2026, at the `magnituder`'s
native operating point rather than via a steering-strength sweep — see
`docs/metriche-valutazione-steering.md` for the full rationale) are computed only for the `base`
reference and each `magnituder` experiment; `fixed-alpha` and `cfg_diff` rows always leave these
columns blank/`null`, a deliberate scope decision, not an oversight.

- **Alignment Gain (AST)** (`target_suppression_gain_ast` in `results.csv`/`summary.json`): the
  same paired-gain formula as Alignment Gain (CLAP) above, but computed from
  `target_instrument_score` (the `AudioSetInstrumentClassifier` / AST score) instead of CLAP. A
  secondary, independent check: AST is a closed-vocabulary classifier, so it can only stand in for
  the target-suppression comparison, never for `retain_similarity_change`/`prompt_similarity_change`
  (those require scoring arbitrary free text, which a fixed-class classifier cannot do). Never
  average or merge the CLAP and AST versions numerically — the underlying scales aren't comparable
  (cosine similarity vs. sigmoid probability); read them side by side instead.
- **Preservation (LPAPS)** (`lpaps_preservation`): perceptual distance between a method's audio and
  the paired `base` audio (same prompt and seed), computed in the feature space of "VGGish-ish", a
  VGG16-style classifier trained from scratch on VGGSound (Iashin & Rahtu, 2021,
  `github.com/v-iashin/SpecVQGAN`). Lower is more preserved. **Domain-mismatch caveat**: VGGish-ish
  is trained on VGGSound (environmental/video sound events), not music, so this is a reasonable
  approximation of preserved perceptual structure, not a metric calibrated for music. The checkpoint
  (~140M parameters) is downloaded once from `a3s.fi` and cached under
  `~/.cache/musicldm-steering-pipeline/lpaps/`; environments without network access to `a3s.fi` (e.g.
  some CI/offline setups) cannot compute this metric on first run.
- **Audio Quality (Audiobox Aesthetics)** (`audiobox_ce`/`audiobox_cu`/`audiobox_pc`/`audiobox_pq`/
  `audiobox_mean`): a no-reference quality score (Content Enjoyment, Content Usefulness, Production
  Complexity, Production Quality; Tjandra et al., Meta, 2025) computed for every in-scope row,
  including `base`, since it needs no paired reference. Its checkpoint downloads automatically from
  the `facebook/audiobox-aesthetics` HuggingFace repo on first use.
- **Smoothness**: not computed, and not approximated. TADA's Smoothness is defined on the variance
  between consecutive points of a steering-strength sweep; the `magnituder` predicts a single
  per-sample `magnitude` scalar with no sweep to speak of (see §4 of this doc and
  `docs/metriche-valutazione-steering.md`). `summary.json` reports `"smoothness": null` explicitly
  for `magnituder` methods, so the omission is visible rather than silent.

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
  magnituder/
  fixed_alpha_*/
  cfg_diff/
```

`results.csv` is the detailed table with one row per prompt, seed and method. Besides the metrics
listed above it also carries `target_suppression_gain_ast`, `lpaps_preservation`, `audiobox_ce`,
`audiobox_cu`, `audiobox_pc`, `audiobox_pq` and `audiobox_mean` — populated only for `base`/
`magnituder` rows, blank for `fixed-alpha`/`cfg_diff`. `summary.json` contains prompt-clustered
bootstrap confidence intervals, plus the plain per-method means for Audiobox Aesthetics and the
explicit `"smoothness": null` marker for `magnituder` methods.
