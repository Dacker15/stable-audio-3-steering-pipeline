# Baseline instrument selection

`scripts/prepare_baselines_stable_audio_3.py` generates Stable Audio 3 clips with steering fixed to
`alpha=0`, runs an
independent multi-label instrument classifier only on the final decoded waveforms, and records which
prompt/seed pairs are eligible for target steering.

`paired-alpha0` is the default computation mode. It is mathematically unsteered, but deliberately
executes the same full/unconditional/retain transformer branch used by learned steering. This avoids
confounding a paired comparison with the numerical difference between the stock two-conditioning
path and the steering three-conditioning path. `--baseline-mode stock` remains available only as a
legacy diagnostic.

This stage intentionally does not inspect diffusion states and does not run a probe. Its purpose is
to prevent a later suppression experiment from receiving credit when the assigned target was already
absent from the baseline.

## Selection rule

The target comes from the dataset's `target` column and is resolved before any audio is generated or
classified. It is never selected from the classifier output.

For a fixed detection threshold:

```text
target_valid = target_score >= detection_threshold
eligible_for_target_steering = target_valid
```

Retain instruments are scored independently. Their per-instrument flags and the aggregate
`all_retain_valid` value are recorded, but they do not change target eligibility. This keeps target
selection separate from the later collateral-damage analysis.

Scores are sigmoid-transformed AudioSet logits from
[`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593).
They are useful ranking/detection scores, not calibrated probabilities. The default threshold is
`0.5`; report it with results and tune it on separately annotated calibration data when possible.
Changing the threshold after generation only requires recomputing flags from the stored scores.

## Input CSV

The existing dataset contract remains valid:

```csv
prompt,target,retain_prompt
"A jazz piece with trumpet, drums, and piano.",trumpet,"A jazz piece with drums and piano."
```

The preferred contract adds explicit instrument lists:

```csv
prompt,target,retain_prompt,requested_instruments,retain_instruments
"A jazz piece with trumpet, drums, and piano.",trumpet,"A jazz piece with drums and piano.","trumpet; drums; piano","drums; piano"
```

Instrument-list cells accept semicolon/comma-separated names or a JSON string array. Explicit lists
are authoritative, and `requested_instruments` must contain the preassigned target. When the columns
are absent, the built-in vocabulary extracts instrument mentions from `prompt` and `retain_prompt`.
Legacy `prompt,target` CSVs also work and derive `retain_prompt` with `utils.strip_target`.

Explicit instrument lists are preferable for new datasets: lexical extraction is deterministic and
keeps old files usable, but it cannot infer an instrument that the prompt describes without a known
name or alias.

Some instruments in the current datasets do not have an exact AudioSet class. Their scores use a
documented family proxy, for example `tuba -> Brass instrument` and `oboe -> Wind instrument,
woodwind instrument`. They are marked by `instrument_score_proxies` and listed separately in
`detected_proxy_instruments`. A target is rejected at input time if it has only a coarse proxy; target
eligibility requires an exact classifier class. Retain proxies remain available as explicitly marked
diagnostics.

## Generate and classify baselines

The generation settings below match the default `base` settings in `scripts/evaluate.py`:

```powershell
uv run python scripts/prepare_baselines_stable_audio_3.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --output outputs/trumpet-baseline-selection `
  --model medium-base `
  --num-seeds 5 --seed 200000 `
  --steps 50 --cfg-scale 7.0 --apg-scale 0.0 `
  --baseline-mode paired-alpha0 `
  --steering-frac-start 0.3 --steering-frac-end 0.8 `
  --detection-threshold 0.5
```

Stable Audio 3 normally runs on the selected generation device, while AST defaults to CPU to avoid
competing for diffusion-model VRAM. Use `--classifier-device cuda` when enough VRAM is available.
Both model repositories may be downloaded on the first run; Stable Audio 3 remains gated as described
in the project README.

For a quick plumbing check, reduce `--max-samples`, `--num-seeds`, `--steps`, and audio length. Scores
from a short/low-step smoke test are not reportable selection results.

## Outputs

```text
config.json
summary.json
baseline_records.jsonl
baseline_records.csv
eligible_pairs.csv
audio/
  baseline/
    sample_0000_seed_1000.wav
```

Each record contains:

- prompt, retain prompt, sample ID, seed index, exact seed, and stable `pair_id`;
- preassigned target, requested instruments, and retain instruments;
- all canonical instrument scores and detected exact/proxy instruments;
- target score, `target_valid`, and `eligible_for_target_steering`;
- per-instrument requested/retain scores and validity flags, plus `all_retain_valid`;
- relative baseline audio path and passthrough metadata such as split/group columns.

JSONL preserves the nested dictionaries directly. CSV stores list/dictionary fields as JSON strings.
`eligible_pairs.csv` contains only rows where the preassigned target was detected; the complete CSV
and JSONL retain invalid rows so absence is auditable. Classification is run on the reloaded PCM16
WAV, so every stored score describes the exact baseline artifact reused by training/evaluation.

## Reusing the same noise later

Seeds follow the evaluator's layout:

```text
seed = first_seed + seed_index * number_of_selected_dataset_rows + sample_id
```

Generation uses the same Stable Audio 3 sampler and a freshly recreated CPU `torch.Generator` for
each pair. A steered generation must consume the exact `seed` stored in the record and keep the
same model, step count, duration, CFG/APG settings, negative prompt, and decode settings.

The evaluator now consumes the selector output directly:

```powershell
uv run python scripts/evaluate.py `
  --baseline-selection outputs/trumpet-baseline-selection `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/trumpet-eval-selected
```

It reads target-valid records from `baseline_records.jsonl`, reuses each saved baseline WAV, and
recreates the explicit record seed for every steered method. Generation settings that determine the
paired sample are taken from the selector config; conflicting CLI overrides are rejected. Retain
validity is propagated to `sample_metrics.csv`, including separate lists of baseline-valid and
baseline-invalid retain instruments.

## Target-valid training and validation pools

Generate the two splits independently and use non-overlapping numeric seed ranges:

```powershell
uv run python scripts/prepare_baselines_stable_audio_3.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-train-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 1000

uv run python scripts/prepare_baselines_stable_audio_3.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --output outputs/trumpet-validation-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 100000
```

Then train directly from the selected records:

```powershell
uv run python scripts/train.py `
  --train-selection outputs/trumpet-train-baselines `
  --validation-selection outputs/trumpet-validation-baselines `
  --output outputs/trumpet-target-specific
```

`train.py` rejects stock baselines, mismatched generation/classifier settings, overlapping seed
values, the same dataset file, semantic `group_id` leakage, and validation targets unseen in
training. Both loaders contain only records with `target_valid=true`. Validation uses its saved seeds
after every epoch and selects the best checkpoint by validation loss.

## Custom instrument vocabulary

Pass `--instrument-config path/to/instruments.json` to replace the built-in vocabulary. The JSON
object maps canonical names to exact AudioSet labels and aliases:

```json
{
  "trumpet": {
    "classifier_labels": ["Trumpet"],
    "aliases": ["trumpets"],
    "is_proxy": false
  },
  "tuba": {
    "classifier_labels": ["Brass instrument"],
    "aliases": ["tubas"],
    "is_proxy": true
  }
}
```

Every classifier label is validated against the loaded checkpoint before generation begins.
