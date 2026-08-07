# ACE-Step 1.5 target-specific steering

This project learns a timestep-dependent, target-specific `alpha_t` that suppresses one concept
while preserving the rest of a music prompt. The generative backbone is
[`ACE-Step/acestep-v15-sft`](https://huggingface.co/ACE-Step/acestep-v15-sft); a separate frozen
[`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused) model supplies the
audio-text training loss.

The supported generation configuration is deliberately narrow and reproducible:

- ACE-Step 1.5 SFT, pinned to revision
  `c410d249e71ea9385a7b586865e65b1473e1098d`;
- VAE and Qwen components from `ACE-Step/Ace-Step1.5`, revision
  `19671f406d603126926c1b7e2adc169acbcade22`;
- external CLAP pinned to revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`;
- `thinking=False`: the optional 5 Hz language model is not downloaded;
- `dcw_enabled=False`;
- native APG guidance, with `guidance_scale=7.0` by default;
- the native base/SFT flow schedule, `shift=1.0` by default (`shift=3.0` is for turbo);
- true FP32 and eager attention. FP16 is intentionally rejected because it produced NaNs on a
  Tesla T4.

`thinking` and DCW are not silent compatibility switches. Passing `--thinking` or
`--dcw-enabled` is an error in the training and evaluation commands.

## How steering works

ACE-Step uses a flow-matching DiT over temporal latents shaped `[batch, frames, 64]`, not the 2D
UNet noise prediction used by MusicLDM. At every active steering step, the frozen DiT evaluates the
same `x_t` and timestep with three compatible conditions:

```text
v_cond = v_full + alpha_t * (v_retain - v_full)
v_t    = APG(v_cond, v_null, guidance_scale)
x_next = x_t - (t_current - t_next) * v_t
```

The interpolation happens between the two **conditional flow velocities before one native APG
operation**. Consequently, `alpha=0` follows the full-prompt trajectory and `alpha=1` follows the
retain condition at every active steering step. Exact equality with the unsteered retain-prompt
trajectory requires steering over the full interval (`--steering-frac-start 0` and
`--steering-frac-end 1`); with the default partial window, steps outside it still follow the full
prompt. ACE-Step's learned null condition is used for APG. The default APG setup matches the SFT
implementation (`momentum=-0.75`, `eta=0`, norm threshold `2.5`).

Full and retain captions are Qwen-encoded independently, so their native sequence lengths cannot
change each other through padding. APG uses the learned one-token null embedding directly. The
pinned eager decoder has no encoder-position encoding and ignores its encoder mask, making this
canonical token equivalent to upstream's repetition of that same token to the conditional width.
The opt-in real-model diagnostic checks that equivalence before relying on the endpoint identities.

The steering predictor is a temporal 1D network. It sees the current `[B,T,64]` latent, the flow
timestep, and an external CLAP embedding of the target. Its bounded sigmoid output starts near
`0.15`; `--alpha-min`, `--alpha-max`, and `--alpha-init` are configurable. Qwen conditions
ACE-Step only. Qwen and CLAP embeddings are never compared or mixed.

The loss minimizes target similarity while retaining the explicit retain prompt:

```text
loss = cosine(CLAP(audio), CLAP(target))
     + retain_weight * (1 - cosine(CLAP(audio), CLAP(retain_prompt)))
```

ACE-Step decodes stereo at 48 kHz. Audio remains stereo when saved and is differentiably downmixed
to mono, then resampled to the CLAP input rate, for the loss. The pinned unfused CLAP model scores
one 10-second / 480,000-sample window. Shorter clips use repeat-padding; longer clips are rejected
so the tail is never silently omitted from the objective or evaluation metrics.

## Installation

Use Python 3.11 or 3.12. A CUDA GPU is required for meaningful training; CPU is suitable for unit
tests only.

```powershell
python -m pip install --upgrade pip
python -m pip install "uv>=0.8,<1"
uv sync --frozen --group dev
```

Authenticate with Hugging Face before the first model download. Do not put a token in source
control:

```powershell
$env:HF_TOKEN = "hf_..."
huggingface-cli login --token $env:HF_TOKEN
```

The loader uses remote model code from the pinned ACE-Step revision. Review that revision before
running it in a sensitive environment.

## Dataset and group-aware splits

The preferred CSV contract is:

```csv
prompt,target,retain_prompt
"jazz with trumpet, piano and drums","trumpet","jazz with piano and drums"
```

Legacy `prompt,target` files still load by deriving a retain prompt lexically, but an explicit
`retain_prompt` is strongly preferred. Create deterministic group-aware splits so equivalent prompt
templates never leak across partitions:

```powershell
uv run python scripts/create_simple_splits.py `
  --input datasets/trumpet_prompts_simple_dataset.csv `
  --output-dir datasets/trumpet_simple_splits `
  --seed 42
```

The supplied simple dataset produces 132/44/44 train/validation/test rows.

## Tests and diagnostics

Run the locally affordable suite first:

```powershell
uv run python -m pytest -q
```

The unit tests use small stand-ins to check base/`alpha=0`/`alpha=1` trajectory equivalence,
deterministic initial noise, finite and non-zero alpha gradients, a finite-difference check, stereo
handling, checkpoint rejection, and a one-sample training step. Real ACE-Step and CLAP diagnostics
are opt-in because they download several gigabytes and require a CUDA GPU or two licensed audio
clips. The real-model diagnostic also compares the complete disabled-steering trajectory with the
pinned checkpoint's upstream `generate_audio` APG/Euler loop. See the skip message emitted by
`pytest` for the environment variables and fixture paths it expects. A skipped diagnostic is not
evidence that the real model passed it.

## T4 smoke test

This is the first command to run after installation and authentication. `--smoke-test` caps the run
at one sample, one epoch, two denoising steps, and two seconds of audio.

```powershell
uv run python scripts/train.py `
  --train-csv datasets/trumpet_simple_splits/train.csv `
  --output-dir outputs/ace-step-smoke `
  --model-id ACE-Step/acestep-v15-sft `
  --model-revision c410d249e71ea9385a7b586865e65b1473e1098d `
  --components-id ACE-Step/Ace-Step1.5 `
  --components-revision 19671f406d603126926c1b7e2adc169acbcade22 `
  --clap-model-id laion/clap-htsat-unfused `
  --clap-revision 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a `
  --device cuda `
  --dtype float32 `
  --guidance-mode apg `
  --guidance-scale 7.0 `
  --shift 1.0 `
  --sequential-dit `
  --offload-dit-after-generation `
  --smoke-test
```

## First real experiment on a Tesla T4

Start small. Four flow steps are not a final-quality sampler; they establish whether the full
training path fits and makes progress before spending hours on a larger run.

```powershell
uv run python scripts/train.py `
  --train-csv datasets/trumpet_simple_splits/train.csv `
  --output-dir outputs/trumpet-ace-step-first `
  --model-id ACE-Step/acestep-v15-sft `
  --model-revision c410d249e71ea9385a7b586865e65b1473e1098d `
  --components-id ACE-Step/Ace-Step1.5 `
  --components-revision 19671f406d603126926c1b7e2adc169acbcade22 `
  --clap-model-id laion/clap-htsat-unfused `
  --clap-revision 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a `
  --device cuda `
  --dtype float32 `
  --batch-size 1 `
  --grad-accum-steps 4 `
  --epochs 1 `
  --max-samples 8 `
  --audio-length-in-s 10 `
  --num-inference-steps 4 `
  --guidance-mode apg `
  --guidance-scale 7.0 `
  --shift 1.0 `
  --sequential-dit `
  --offload-dit-after-generation
```

Increase `--num-inference-steps` (8 first, then 16 or more) and the dataset size only after this
run succeeds. A 32-50 step sampler is more appropriate for final evaluation than for iterative T4
training.

Resume without resetting the optimizer, epoch, history, or RNG state:

```powershell
uv run python scripts/train.py `
  --train-csv datasets/trumpet_simple_splits/train.csv `
  --output-dir outputs/trumpet-ace-step-first `
  --resume outputs/trumpet-ace-step-first/checkpoints/latest.pt `
  --device cuda `
  --dtype float32 `
  --batch-size 1 `
  --grad-accum-steps 4 `
  --epochs 2 `
  --max-samples 8 `
  --audio-length-in-s 10 `
  --num-inference-steps 4 `
  --guidance-mode apg `
  --guidance-scale 7.0 `
  --shift 1.0 `
  --sequential-dit `
  --offload-dit-after-generation
```

Resume validates the original dataset path, sampling schedule, objective, accumulation, clipping,
and seed. Repeat the original semantic flags exactly; only the total epoch count, runtime/offload
choices, and output location may change.

Checkpoints carry a format version, the ACE-Step model family, and the steering formulation. Legacy
MusicLDM checkpoints and checkpoints produced by a different alpha meaning are rejected instead of
being loaded partially.

## Memory strategy and gradient limitation

Full backpropagation through a roughly 2B-parameter FP32 DiT over many denoising steps is not
realistic on a 16 GB T4. The default `steering_only` surrogate uses the following boundary:

| Component | Parameters | Autograd through operations | Device schedule |
|---|---:|---:|---|
| Qwen text encoder | frozen | no | GPU only while preparing conditioning, then CPU |
| ACE-Step DiT | frozen | no | sequential full/retain/null passes, then CPU |
| steering predictor and alpha/APG/Euler path | trainable | yes | GPU |
| ACE-Step VAE decoder | frozen | **yes, with respect to latents** | GPU after DiT offload |
| CLAP text tower | frozen | no | target/retain embeddings are detached |
| CLAP audio tower, mono/resampling, loss | frozen | **yes, with respect to waveform** | GPU after DiT offload |

This preserves a useful gradient from CLAP through audio, VAE, Euler integration, APG and
`alpha_t`, but omits the Jacobian of later DiT predictions with respect to the evolving latent. It
is a documented first-order surrogate, not mathematically identical to full end-to-end BPTT.
Activation checkpointing is applied to the predictor. Conditioning can be cached on CPU; CUDA cache
is cleared at phase boundaries. The custom three-branch sampler deliberately disables persistent
cross-attention KV caches: retaining separate full/retain/null caches costs substantial extra VRAM,
so the T4-safe default recomputes their K/V projections each step and trades speed for memory.
Even with these measures, long clips or many steps can be slow and may still exceed T4 memory. Keep
batch size one, prefer sequential DiT branches, and reduce audio duration or flow steps if the smoke
run OOMs. The train/evaluation CLIs intentionally reject durations above 10 seconds: the pinned
unfused CLAP tower consumes a fixed 10-second window, so a longer differentiable VAE decode would
otherwise increase T4 memory only for the reference processor to crop it before the loss or metrics.

## Evaluation

Evaluate the best versioned checkpoint on the held-out test split with paired seeds:

```powershell
uv run python scripts/evaluate.py `
  --eval-csv datasets/trumpet_simple_splits/test.csv `
  --checkpoint outputs/trumpet-ace-step-first/checkpoints/best.pt `
  --output-dir outputs/trumpet-ace-step-eval `
  --device cuda `
  --dtype float32 `
  --num-seeds 5 `
  --num-inference-steps 32 `
  --guidance-mode apg `
  --guidance-scale 7.0 `
  --fixed-alphas 0.25 0.5 0.75 `
  --sequential-dit `
  --keep-dit-on-device
```

The evaluator records target, retain, and full-prompt CLAP similarity; paired suppression and
retention changes; alpha schedules; audio-health statistics; plots; summaries; and stereo WAV
files. See [EVALUATION.md](EVALUATION.md) for interpretation and the full protocol.

## Kaggle

Upload or clone this repository into `/kaggle/working/musicldm-steering-pipeline`, create a Kaggle
secret named `HF_TOKEN`, attach the CSV dataset, and run:

```bash
cd /kaggle/working/musicldm-steering-pipeline
bash scripts/kaggle_setup_and_run.sh
```

That command installs the project, reads the token without printing or persisting it, downloads only the pinned
SFT/Qwen/VAE/CLAP assets, runs tests and diagnostics, and launches the two-step smoke test. To run
the small first experiment after the smoke test:

```bash
cd /kaggle/working/musicldm-steering-pipeline
RUN_FULL_TRAINING=1 TRAIN_CSV=/kaggle/working/musicldm-steering-pipeline/datasets/trumpet_simple_splits/train.csv \
  REAL_STEPS=4 REAL_DURATION=10 REAL_EPOCHS=1 REAL_MAX_SAMPLES=8 \
  bash scripts/kaggle_setup_and_run.sh
```

Every checkpoint, log, plot, and audio artifact is written below `/kaggle/working`; the Hugging Face
cache is placed there as well so it can be inspected or preserved as a Kaggle output.

## Upstream references

- [ACE-Step 1.5 repository](https://github.com/ace-step/ACE-Step-1.5)
- [Official inference guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)
- [ACE-Step GPU compatibility notes](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md)
- [Pinned SFT checkpoint](https://huggingface.co/ACE-Step/acestep-v15-sft)
