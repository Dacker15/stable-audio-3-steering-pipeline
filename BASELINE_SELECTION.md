# Baseline instrument selection

`scripts/prepare_baselines_ace_step.py` generates ACE-Step 1.5 SFT clips, runs an independent
multi-label AudioSet classifier on the decoded waveforms, and records which prompt/seed pairs are
eligible for target steering.

The loader is pinned to `ACE-Step/acestep-v15-sft`. It does not use the Base checkpoint or any of its
Extract/Lego/Complete modes: baseline generation is ordinary text-to-music with empty lyric
conditioning, silence context, 50-step CFG/APG, and no 5 Hz LM planner.

`paired-alpha0` is the default. It is mathematically unsteered but executes the same
full/null/retain DiT batching used by learned steering, with `alpha` fixed to zero. `stock` uses the
ordinary full/null path and is retained only as a diagnostic.

## Selection rule

The target is always read from the dataset's `target` column before generation and classification.
It is never inferred from classifier output.

```text
target_valid = target_score >= detection_threshold
eligible_for_target_steering = target_valid
```

Retain instruments are scored independently. Their flags and `all_retain_valid` are recorded but do
not change target eligibility.

Scores are sigmoid-transformed AudioSet logits from
[`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593).
They are detection/ranking scores, not calibrated probabilities. The default threshold is `0.5`.

## Input CSV

The preferred schema explicitly names the retain prompt and instruments:

```csv
prompt,target,retain_prompt,requested_instruments,retain_instruments
"A jazz piece with trumpet, drums, and piano.",trumpet,"A jazz piece with drums and piano.","trumpet; drums; piano","drums; piano"
```

Instrument cells accept semicolon/comma-separated names or JSON arrays. Legacy `prompt,target` CSVs
are accepted and derive the retain prompt lexically. Explicit fields are preferred.

Some retain instruments use documented AudioSet family proxies (for example `tuba -> Brass
instrument`). Proxy status is saved. Target eligibility requires an exact classifier class; coarse
proxies are not accepted for the target.

## Generate and classify

```powershell
uv run python scripts/prepare_baselines_ace_step.py `
  --dataset datasets/trumpet_simple_splits/test.csv `
  --output outputs/trumpet-baseline-selection `
  --model ACE-Step/acestep-v15-sft `
  --num-seeds 5 --seed 200000 `
  --steps 50 --cfg-scale 7.0 --shift 1.0 `
  --baseline-mode paired-alpha0 `
  --steering-frac-start 0.3 --steering-frac-end 0.8 `
  --detection-threshold 0.5
```

ACE-Step runs on the selected generation device; AST defaults to CPU to avoid competing for VRAM.
Use `--classifier-device cuda` when memory permits. First use downloads the public model weights.

## Outputs

```text
config.json
summary.json
baseline_records.jsonl
baseline_records.csv
eligible_pairs.csv
audio/
  baseline/
```

Each record contains prompt metadata, target, retain prompt, stable `pair_id`, exact seed, requested
and retain instruments, all classifier scores and proxy flags, target eligibility, retain validity,
and the relative baseline WAV path. Classification runs on the reloaded PCM16 WAV, so stored scores
describe the exact artifact reused by training and evaluation.

## Same-noise contract

Seeds follow:

```text
seed = first_seed + seed_index * number_of_dataset_rows + sample_id
```

Every pair recreates a CPU `torch.Generator` from its stored seed. Paired steering must keep the same
checkpoint, model dtype, step count, duration, CFG scale, shift, and steering window.

```powershell
uv run python scripts/evaluate.py `
  --baseline-selection outputs/trumpet-baseline-selection `
  --checkpoint outputs/trumpet-target-specific/steering_predictor_best.pt `
  --output outputs/trumpet-eval-selected
```

The evaluator reuses each saved base WAV and explicit seed. Conflicting settings are rejected.

## Training and validation pools

Create them independently with disjoint seeds:

```powershell
uv run python scripts/prepare_baselines_ace_step.py `
  --dataset datasets/trumpet_simple_splits/train.csv `
  --output outputs/trumpet-train-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 1000

uv run python scripts/prepare_baselines_ace_step.py `
  --dataset datasets/trumpet_simple_splits/validation.csv `
  --output outputs/trumpet-validation-baselines `
  --baseline-mode paired-alpha0 --num-seeds 3 --seed 100000
```

`train.py` rejects stock baselines, mismatched generation/classifier settings, overlapping seeds,
identical dataset files, semantic `group_id` leakage, and validation targets unseen during training.

## Custom instrument vocabulary

Pass `--instrument-config path/to/instruments.json` to replace the built-in mapping:

```json
{
  "trumpet": {
    "classifier_labels": ["Trumpet"],
    "aliases": ["trumpets"],
    "is_proxy": false
  }
}
```

Every classifier label is validated against the loaded AudioSet checkpoint before generation.
