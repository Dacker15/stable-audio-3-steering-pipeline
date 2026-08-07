# ACE-Step steering evaluation

`scripts/evaluate.py` performs paired evaluation of a versioned ACE-Step 1.5 steering checkpoint.
Every comparison uses the same prompt, retain prompt, target, seed, initial latent, flow schedule,
APG settings, and SFT model revision. The method is the only variable.

## Methods

- `base_full`: steering is disabled; the DiT follows the full prompt with native APG.
- `base_retain`: an independently prepared, unsteered retain-prompt trajectory.
- `alpha_0` / `alpha_1`: explicit endpoint controls used to enforce the steering invariants.
- `learned`: `alpha_t` comes from the target-conditioned 1D steering predictor.
- `fixed_*`: optional constant controls supplied with `--fixed-alphas`.

`alpha=0` is expected to reproduce `base_full` exactly within floating-point tolerance. `alpha=1`
selects the retain condition at every active steering step, still combined with ACE-Step's learned
null condition through one APG operation. It reproduces `base_retain` only when the steering window
covers the full interval (`--steering-frac-start 0 --steering-frac-end 1`); outside a partial window,
the sampler intentionally follows the full prompt. These endpoint checks are part of the test suite;
they are not implemented by mixing finished waveforms.

The two captions are encoded independently. A canonical one-token learned-null branch avoids
making either endpoint depend on the other caption's padded sequence width; the opt-in real-model
test also compares it against ACE-Step's upstream repeated-null construction.

APG has trajectory state (its momentum buffer), so each method starts a fresh sampler and APG buffer.
Reusing identical initial latents makes methods comparable without incorrectly sharing mutable
sampler state.

## Required inputs

Use a held-out CSV with an explicit retain prompt:

```csv
prompt,target,retain_prompt
"jazz with trumpet, piano and drums","trumpet","jazz with piano and drums"
```

Legacy `prompt,target` input is accepted by deriving a retain prompt lexically, but that is less
reliable. For example, removing only the noun may leave target-specific modifiers behind. The
group-aware split script keeps equivalent templates in one partition and prevents near-duplicate
train/test leakage:

```powershell
uv run python scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset.csv `
  --output-dir datasets/trumpet_simple_splits `
  --seed 42
```

The evaluator warns if its CSV path matches the training CSV recorded in the checkpoint. A training
split is useful for debugging but cannot support a generalization claim.

The checkpoint must declare the current checkpoint format, `ACE-Step 1.5` model family, and
`ace_step_apg_cond_velocity_lerp_canonical_null_v2` steering semantics. Old MusicLDM checkpoints, unversioned
checkpoints, and checkpoints whose alpha interpolates a different quantity are rejected before
weights are applied.

## Quick evaluator smoke run

First create a two-step training checkpoint with `scripts/train.py --smoke-test`. Then check the
evaluation machinery on one item and one seed:

```powershell
uv run python scripts/evaluate.py `
  --eval-csv datasets/trumpet_simple_splits/validation.csv `
  --checkpoint outputs/ace-step-smoke/checkpoints/best.pt `
  --output-dir outputs/ace-step-eval-smoke `
  --device cuda `
  --dtype float32 `
  --max-samples 1 `
  --num-seeds 1 `
  --audio-length-in-s 2 `
  --num-inference-steps 2 `
  --guidance-mode apg `
  --guidance-scale 7.0 `
  --shift 1.0 `
  --steering-frac-start 0.0 `
  --steering-frac-end 1.0 `
  --fixed-alphas `
  --sequential-dit `
  --keep-dit-on-device
```

This validates plumbing only. Two flow steps and two seconds of audio do not provide reportable
music quality or suppression results.

## Final paired run

After choosing a step count that fits the GPU, evaluate multiple seeds on the untouched test split:

```powershell
uv run python scripts/evaluate.py `
  --eval-csv datasets/trumpet_simple_splits/test.csv `
  --checkpoint outputs/trumpet-ace-step-first/checkpoints/best.pt `
  --output-dir outputs/trumpet-ace-step-eval `
  --device cuda `
  --dtype float32 `
  --num-seeds 5 `
  --audio-length-in-s 10 `
  --num-inference-steps 32 `
  --guidance-mode apg `
  --guidance-scale 7.0 `
  --shift 1.0 `
  --fixed-alphas 0.25 0.5 0.75 `
  --sequential-dit `
  --keep-dit-on-device
```

Use 32-50 steps for a stronger final sampler if runtime permits. Do not compare results obtained
with different step counts, guidance modes, guidance scales, shifts, durations, model revisions, or
CLAP revisions as though they were paired.

Generation values are explicit CLI settings; the checkpoint records the training settings for
audit and strict resume validation. The supported project configuration remains `thinking=False`,
`dcw_enabled=False`, FP32, eager attention, and the pinned SFT/component/CLAP revisions.
`guidance_scale` is configurable but defaults to `7.0`, and the base/SFT schedule defaults to
`shift=1.0`. APG is the supported train/evaluation path; classical CFG remains available only as a
low-level pipeline diagnostic and cannot be mislabeled in a training checkpoint.

## Metrics

For each waveform, the same frozen external CLAP model encodes the downmixed/resampled audio and all
three text roles:

- `target_similarity = cosine(CLAP(audio), CLAP(target))`: lower is better;
- `retain_similarity = cosine(CLAP(audio), CLAP(retain_prompt))`: higher is better;
- `prompt_similarity = cosine(CLAP(audio), CLAP(full_prompt))`: a coarse full-caption fidelity
  measure.

For a method `m`, paired deltas against `base_full` are:

```text
target suppression gain  = target_similarity(base_full) - target_similarity(m)
retain similarity change = retain_similarity(m) - retain_similarity(base_full)
prompt similarity change = prompt_similarity(m) - prompt_similarity(base_full)
```

A useful result has positive target suppression without a large negative retain change. A target
gain accompanied by silence, clipping, or a severe retain loss is likely degradation rather than
selective removal.

Prompt similarity has an intentional tension: the full prompt still names the concept being
suppressed. Retain similarity therefore carries more direct evidence of preservation. CLAP is also
the training objective, so report paired listening and, when available, an independent instrument
classifier alongside these scores. The opt-in real-audio diagnostic checks that CLAP ranks a
licensed trumpet clip above a non-trumpet clip for the word "trumpet"; if that test is skipped, do
not assume the ranking was validated locally.

Audio-health fields include RMS, peak, near-silence ratio, and clipping ratio. Alpha records include
mean, standard deviation, minimum, maximum, and the per-timestep schedule. ACE-Step WAV output is
kept stereo at 48 kHz; only CLAP scoring uses a differentiable mono downmix and resampling.

Confidence intervals should use prompts, not individual seeds, as independent units. Seed results
are averaged within a prompt before prompt-level bootstrap resampling, avoiding falsely narrow
intervals caused by repeated noises for one caption.

## Interpreting endpoint and determinism checks

Before trusting learned steering, verify these invariants with the same initial latent:

1. disabled steering and fixed `alpha=0` agree;
2. with a full steering window, fixed `alpha=1` agrees with a retain-conditioned trajectory under
   the same APG rule;
3. repeated base generation with the same prompt and seed agrees;
4. changing only the seed changes the initial latent;
5. learned alpha is finite and stays within its configured bounds.

A discrepancy at an endpoint is an implementation error, not a model-quality trade-off. Tiny
device-dependent numerical differences can occur, so use tolerances rather than byte equality for
floating-point tensors.

## Runtime and memory

Evaluation does not need predictor gradients, but the FP32 DiT is still large. On a 16 GB T4 use
batch size one, sequential full/retain/null DiT passes, and DiT offloading before VAE decode and
CLAP scoring. The optional 5 Hz LM is absent. Increasing flow steps primarily costs time; increasing
duration also increases latent and audio-tower activation memory. The train/evaluation CLIs cap
clips at 10 seconds: the pinned CLAP tower scores a fixed 10-second window, so generating a longer
differentiable clip would otherwise consume extra T4 memory only for the reference processor to
crop it before the loss. If the smoke evaluator OOMs, reduce duration first, then retain sequential
execution and verify no other process occupies the GPU.

The training checkpoint was produced with a `steering_only` surrogate gradient: predictor,
interpolation, APG, Euler, frozen VAE operations, differentiable mono/resampling, and frozen CLAP
audio operations remain on the graph, while DiT evaluations, Qwen, and CLAP text embeddings do not.
Evaluation measures the resulting policy directly, but it does not remove that training
approximation.

## Outputs

The evaluation directory contains the resolved configuration, detailed per-sample metrics, alpha
records, aggregate summaries, plots, a Markdown report, and paired stereo audio. The expected layout
is:

```text
args.json
per_sample.csv
summary.csv
report.json
report.md
method_similarity.png
paired_tradeoff.png
learned_alpha_schedule.png
audio/
  sample_000_seed_42_base_full_<target>.wav
  sample_000_seed_42_base_retain_<target>.wav
  sample_000_seed_42_alpha_0_<target>.wav
  sample_000_seed_42_alpha_1_<target>.wav
  sample_000_seed_42_fixed_0.5_<target>.wav
  sample_000_seed_42_learned_<target>.wav
```

Keep this directory together with the exact checkpoint and dataset split. They are all needed to
reproduce the paired analysis.
