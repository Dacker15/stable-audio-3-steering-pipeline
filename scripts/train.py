r"""
Trains a `SteeringPredictor` to suppress a concept in `SteeringMusicLDMPipeline` generations.

Each training step generates a full batch of waveforms with steering enabled, embeds them with the
CLAP audio tower and scores them against two texts:

* the steering target, whose cosine distance is *maximized*, so the predictor learns a per-timestep
  `alpha_t` that moves the denoising trajectory from full-prompt guidance towards retain-prompt
  guidance;
* the retain prompt supplied by the dataset (or derived with `utils.strip_target` for legacy
  two-column CSVs), whose cosine distance is *minimized*, so everything else the prompt asks for
  survives the suppression.

The retain term is what separates removing the concept from degrading the audio: silence and noise
are both far from the target, and only the retain term tells them apart from a faithful rendition of
the rest of the prompt.

Gradients reach the predictor because the pipeline is called with `output_type="latent"`: the
`torch.no_grad()` in its post-processing block only stops new grad-tracking ops, it does not detach
the latents, which are returned before any decoding happens. The UNet itself runs under `no_grad`,
so the gradient flows to each `alpha_t` through the scheduler's linear recurrence only. That is a
first-order approximation of the true gradient and is what keeps the unrolled trajectory affordable.

Example:
    uv run python scripts/train.py --dataset datasets/trumpet_simple_splits/train.csv --batch-size 2 \
        --output outputs/trumpet-target-specific --epochs 5 --steering-frac-start 0.1 --steering-frac-end 0.8
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

# non-interactive backend: this script only saves figures to disk and never calls `plt.show()`, and
# the default TkAgg backend on Windows raises a spurious "main thread is not in main loop" from
# tkinter's Image.__del__ when the garbage collector runs between epochs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from losses import ClapLoss
from pipelines import STEERING_MODE, SteeringMusicLDMPipeline, SteeringPredictor
from utils import PromptTargetDataset, collate_prompt_target

# static light-surface artifacts: chrome and ink from the reference palette, the first two
# categorical slots for the target/retain pair and a validated 5-step ordinal blue ramp for the
# per-epoch alpha lines
PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": "#2a78d6",
    "series_2": "#eb6834",
}
ORDINAL_BLUE = ("#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")


def mean_alpha_by_timestep(records: list[tuple[float, float]]) -> list[tuple[float, float]]:
    r"""
    Returns:
        `list[tuple[float, float]]`: `(timestep, mean_alpha)` pairs, ordered from the noisiest to
        the cleanest timestep, i.e. in denoising order.
    """
    totals: dict[float, float] = defaultdict(float)
    counts: dict[float, int] = defaultdict(int)
    for timestep, alpha in records:
        totals[timestep] += alpha
        counts[timestep] += 1
    return [(timestep, totals[timestep] / counts[timestep]) for timestep in sorted(totals, reverse=True)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SteeringPredictor to suppress a concept in MusicLDM generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data and output")
    data.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CSV with prompt,target and preferably an explicit retain_prompt column",
    )
    data.add_argument("--output", type=Path, default=Path("outputs"), help="folder for weights and plots")
    data.add_argument("--batch-size", type=int, default=1, help="prompts generated per forward pass")
    data.add_argument("--max-samples", type=int, default=None, help="use only the first N rows of the dataset")

    steering = parser.add_argument_group("steering")
    steering.add_argument("--steering-frac-start", type=float, default=0.3, help="fraction of the loop steering starts")
    steering.add_argument("--steering-frac-end", type=float, default=0.8, help="fraction of the loop steering ends")
    steering.add_argument("--alpha-min", type=float, default=0.0, help="lower bound of the predicted alpha_t")
    steering.add_argument("--alpha-max", type=float, default=1.0, help="upper bound of the predicted alpha_t")
    steering.add_argument(
        "--alpha-init",
        type=float,
        default=0.15,
        help="initial alpha_t: 0 follows the full prompt and 1 follows the retain prompt",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=5)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-2)
    optim.add_argument("--grad-accum-steps", type=int, default=4, help="batches accumulated per optimizer step")
    optim.add_argument("--max-grad-norm", type=float, default=1.0)
    optim.add_argument(
        "--retain-weight",
        type=float,
        default=1.0,
        help=(
            "weight of the retain term, which pulls the audio towards the prompt with the target removed. pure"
            " suppression can be solved by degrading the audio rather than by removing the target concept from it,"
            " and this term is what rules that solution out. 0.0 disables it and optimizes suppression alone"
        ),
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument("--num-inference-steps", type=int, default=200, help="denoising steps per generation")
    generation.add_argument(
        "--audio-length-in-s",
        type=float,
        default=10.0,
        help=(
            "the default maps exactly onto the 10s window of CLAP's audio tower, so the whole clip is"
            " scored and nothing is generated that the loss cannot see"
        ),
    )
    generation.add_argument("--guidance-scale", type=float, default=2.0, help="must exceed 1.0 to enable steering")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"`--batch-size` has to be at least 1 but is {args.batch_size}")
    if args.grad_accum_steps < 1:
        raise ValueError(f"`--grad-accum-steps` has to be at least 1 but is {args.grad_accum_steps}")
    if args.retain_weight < 0.0:
        # a negative weight would push the audio away from the rest of the prompt as well, i.e. ask
        # for degraded audio outright
        raise ValueError(f"`--retain-weight` has to be non-negative but is {args.retain_weight}")
    if args.alpha_min >= args.alpha_max:
        raise ValueError(
            f"`--alpha-min` has to be smaller than `--alpha-max` but got {args.alpha_min} >= {args.alpha_max}"
        )
    if not args.alpha_min < args.alpha_init < args.alpha_max:
        raise ValueError(
            f"`--alpha-init` has to lie strictly inside ({args.alpha_min}, {args.alpha_max}) but is"
            f" {args.alpha_init}"
        )
    if not 0.0 <= args.steering_frac_start < args.steering_frac_end <= 1.0:
        raise ValueError(
            "`--steering-frac-start` and `--steering-frac-end` have to satisfy"
            f" `0.0 <= start < end <= 1.0` but are {args.steering_frac_start} and {args.steering_frac_end}"
        )
    if args.guidance_scale <= 1.0:
        # the pipeline only calls the steering model inside its classifier free guidance branch
        raise ValueError(
            f"`--guidance-scale` has to be greater than 1.0 for steering to be applied but is {args.guidance_scale}."
            " With a lower value the pipeline skips classifier free guidance and never calls the steering model, so"
            " the predictor would receive no gradient."
        )

    num_steered_steps = sum(
        1
        for step in range(args.num_inference_steps)
        if args.steering_frac_start <= step / args.num_inference_steps < args.steering_frac_end
    )
    if num_steered_steps == 0:
        raise ValueError(
            f"The steering window [{args.steering_frac_start}, {args.steering_frac_end}) contains no step of the"
            f" {args.num_inference_steps} denoising steps, so the predictor would receive no gradient. Widen the"
            " window or raise `--num-inference-steps`."
        )
    args.num_steered_steps = num_steered_steps

    return args


def style_axes(ax: plt.Axes) -> None:
    r"""Applies the recessive chrome shared by every plot: hairline solid grid and muted axes."""
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(PALETTE["ink_secondary"])
    ax.yaxis.label.set_color(PALETTE["ink_secondary"])
    ax.title.set_color(PALETTE["ink"])


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, facecolor=PALETTE["surface"], bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history: dict, path: Path) -> None:
    r"""
    Writes the optimized loss and both CLAP cosine similarities, one point per epoch.

    Two stacked panels rather than one plot with two y-scales: the loss and the similarities have
    unrelated ranges, and overlaying them on separate scales would invent a relationship. The two
    similarities do share a panel, because they share the cosine scale and reading one against the
    other *is* the suppression/retain trade-off. Per-batch values (`history["iterations"]`) only
    support the epoch means plotted here; they are not drawn themselves, but stay in `history.json`.
    """
    epochs = history["epochs"]
    epoch_numbers = [record["epoch"] for record in epochs]

    fig, (ax_loss, ax_cos) = plt.subplots(2, 1, figsize=(9, 7.5), facecolor=PALETTE["surface"])

    def draw_series(ax: plt.Axes, key: str, color: str, label: str | None = None) -> None:
        values = [record[key] for record in epochs]
        ax.plot(
            epoch_numbers,
            values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=8,
            markeredgecolor=PALETTE["surface"],
            markeredgewidth=2.0,
            label=label,
        )
        # direct-label only the endpoint, so the value is readable without a tooltip
        ax.annotate(
            f"{values[-1]:.3f}",
            (epoch_numbers[-1], values[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=9,
            color=PALETTE["ink"],
        )

    draw_series(ax_loss, "loss", PALETTE["series"])
    draw_series(ax_cos, "target_similarity", PALETTE["series"], "target — suppressed")
    draw_series(ax_cos, "retain_similarity", PALETTE["series_2"], "retain prompt — preserved")

    for ax, title, ylabel in (
        (ax_loss, "Training loss — the suppression and retain terms combined", "loss"),
        (ax_cos, "CLAP cosine similarity, target against retain prompt", "cosine similarity"),
    ):
        ax.set_xticks(epoch_numbers)
        ax.set_title(title, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        # the default margin is thinner than the markers, which leaves the extreme points clipped
        # by the axes
        ax.margins(y=0.15)
        style_axes(ax)

    legend = ax_cos.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    fig.tight_layout()
    save_figure(fig, path)


def plot_alpha_schedule(history: dict, path: Path, alpha_min: float, alpha_max: float) -> None:
    r"""
    Writes the mean predicted `alpha_t` against the denoising timestep, one line per epoch.

    This is the load-bearing diagnostic: it shows whether the predictor learned a schedule that
    varies along the trajectory or collapsed to a constant.

    Epochs are an ordered quantity, so the lines use a single-hue ordinal ramp rather than
    categorical hues. At most `len(ORDINAL_BLUE)` epochs are drawn, evenly spaced and always
    including the first and the last; `history.json` carries every epoch.
    """
    epochs = history["epochs"]

    if len(epochs) <= len(ORDINAL_BLUE):
        selected = epochs
    else:
        stride = (len(epochs) - 1) / (len(ORDINAL_BLUE) - 1)
        selected = [epochs[round(index * stride)] for index in range(len(ORDINAL_BLUE))]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["surface"])

    # `alpha=0.5` is the midpoint between full-prompt and retain-prompt classifier-free guidance.
    if alpha_min < 0.5 < alpha_max:
        ax.axhline(0.5, color=PALETTE["muted"], linewidth=1.0, linestyle="--", zorder=1)
        ax.annotate(
            "α = 0.5 — full/retain midpoint",
            (0.995, 0.5),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 5),
            ha="right",
            fontsize=8,
            color=PALETTE["muted"],
        )

    # spread the drawn epochs across the whole ramp rather than crowding one end, so the lightness
    # gaps stay visible for any epoch count. a lone line is not an ordinal encoding at all
    if len(selected) == 1:
        colors = [PALETTE["series"]]
    else:
        step_size = (len(ORDINAL_BLUE) - 1) / (len(selected) - 1)
        colors = [ORDINAL_BLUE[round(index * step_size)] for index in range(len(selected))]

    for record, color in zip(selected, colors, strict=True):
        timesteps = [timestep for timestep, _ in record["alpha_by_timestep"]]
        alphas = [alpha for _, alpha in record["alpha_by_timestep"]]
        ax.plot(timesteps, alphas, color=color, linewidth=2.0, label=f"epoch {record['epoch']}", zorder=2)

    ax.set_title("Predicted steering strength across the denoising trajectory", fontsize=11, loc="left", pad=10)
    ax.set_xlabel("denoising timestep (noisy → clean)")
    ax.set_ylabel("mean α")
    # denoising runs from high to low timesteps, so read the trajectory left to right
    ax.invert_xaxis()
    margin = 0.04 * (alpha_max - alpha_min)
    ax.set_ylim(alpha_min - margin, alpha_max + margin)
    style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    args = parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    dataset = PromptTargetDataset(args.dataset, args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_prompt_target,
    )
    print(f"Loaded {len(dataset)} prompts from {args.dataset} -> {len(dataloader)} batches per epoch")
    # Show the effective retain prompt, whether explicit or derived, before an expensive first step.
    _, example_target, example_retain = dataset.rows[0]
    print(f"Retain weight {args.retain_weight}, target {example_target!r}, example retain prompt: {example_retain!r}")

    # float32 throughout: the gradient reaches `alpha_t` through the scheduler recurrence, the VAE,
    # the vocoder and the CLAP tower, and in half precision that chain returns a gradient whose sign
    # disagrees with a finite-difference check of the same loss more often than not
    pipe = SteeringMusicLDMPipeline.from_pretrained("ucsd-reach/musicldm", torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    for module in (pipe.unet, pipe.vae, pipe.vocoder, pipe.text_encoder):
        module.requires_grad_(False)
        module.eval()

    # the predictor casts its inputs and its output to the latents' dtype itself, so it keeps
    # working unchanged if the pipeline above is ever loaded in half precision again
    predictor_config = {
        "latent_channels": pipe.unet.config.in_channels,
        "target_embed_dim": pipe.text_encoder.config.projection_dim,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_init": args.alpha_init,
    }
    predictor = SteeringPredictor(**predictor_config).to(device)
    num_parameters = sum(parameter.numel() for parameter in predictor.parameters())
    print(f"SteeringPredictor: {num_parameters / 1e6:.2f}M parameters, config {predictor_config}")

    clap_loss = ClapLoss(pipe.text_encoder, pipe.tokenizer, pipe.feature_extractor).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_samples = int(args.audio_length_in_s * pipe.vocoder.config.sampling_rate)
    print(
        f"Steering {args.num_steered_steps}/{args.num_inference_steps} steps in"
        f" [{args.steering_frac_start}, {args.steering_frac_end}), {args.audio_length_in_s}s clips"
    )

    history: dict = {"args": json.loads(json.dumps(vars(args), default=str)), "iterations": [], "epochs": []}
    best_loss = float("inf")
    step = 0

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_losses: list[float] = []
        epoch_target_similarities: list[float] = []
        epoch_retain_similarities: list[float] = []
        epoch_alpha_records: list[tuple[float, float]] = []

        for batch_index, (prompts, targets, retains) in enumerate(dataloader):
            step += 1
            # fresh noise every batch, but reproducible across runs
            generator = torch.Generator().manual_seed(args.seed + step)

            output = pipe(
                prompt=prompts,
                retain_prompt=retains,
                steering_target=targets,
                steering_model=predictor,
                steering_frac_start=args.steering_frac_start,
                steering_frac_end=args.steering_frac_end,
                train=True,
                num_inference_steps=args.num_inference_steps,
                audio_length_in_s=args.audio_length_in_s,
                guidance_scale=args.guidance_scale,
                num_waveforms_per_prompt=1,
                generator=generator,
                output_type="latent",
            )
            latents = output.audios
            epoch_alpha_records.extend(output.alpha_records)

            # mirrors the pipeline's post-processing, but outside `no_grad` and without the move to
            # the CPU that `mel_spectrogram_to_waveform` performs
            mel_spectrogram = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
            if mel_spectrogram.dim() == 4:
                mel_spectrogram = mel_spectrogram.squeeze(1)
            waveform = pipe.vocoder(mel_spectrogram)[:, :num_samples]

            audio_embeds = clap_loss.encode_audio(waveform, pipe.vocoder.config.sampling_rate)

            # `ClapLoss` returns `1 - cosine_similarity`, so negating it maximizes the distance to
            # the target, i.e. suppresses the concept the prompt asks for
            target_distance = clap_loss(audio_embeds, targets)
            loss = -target_distance

            # the same quantity against the prompt without the target, this time minimized, so the
            # audio keeps everything the prompt asks for besides the concept being suppressed
            retain_distance = clap_loss(audio_embeds, retains) if args.retain_weight > 0.0 else None
            if retain_distance is not None:
                loss = loss + args.retain_weight * retain_distance

            (loss / args.grad_accum_steps).backward()

            grad_norm = None
            is_last_batch = batch_index == len(dataloader) - 1
            if (batch_index + 1) % args.grad_accum_steps == 0 or is_last_batch:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.detach())
            target_similarity = 1.0 - float(target_distance.detach())
            # measured even when the term is switched off, since it is the diagnostic that tells
            # concept removal apart from plain audio degradation
            with torch.no_grad():
                retain_similarity = (
                    1.0 - float(retain_distance.detach())
                    if retain_distance is not None
                    else 1.0 - float(clap_loss(audio_embeds.detach(), retains))
                )
            epoch_losses.append(loss_value)
            epoch_target_similarities.append(target_similarity)
            epoch_retain_similarities.append(retain_similarity)
            history["iterations"].append(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_value,
                    "target_similarity": target_similarity,
                    "retain_similarity": retain_similarity,
                    "grad_norm": grad_norm,
                }
            )
            print(
                f"epoch {epoch}/{args.epochs} batch {batch_index + 1}/{len(dataloader)}"
                f" loss {loss_value:+.4f} target_cos {target_similarity:+.4f} retain_cos {retain_similarity:+.4f}"
                f" grad_norm {'-' if grad_norm is None else f'{grad_norm:.3e}'}",
                flush=True,
            )

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        history["epochs"].append(
            {
                "epoch": epoch,
                "last_step": step,
                "loss": epoch_loss,
                "target_similarity": sum(epoch_target_similarities) / len(epoch_target_similarities),
                "retain_similarity": sum(epoch_retain_similarities) / len(epoch_retain_similarities),
                "alpha_by_timestep": mean_alpha_by_timestep(epoch_alpha_records),
            }
        )
        print(f"epoch {epoch}/{args.epochs} mean loss {epoch_loss:+.4f}")

        checkpoint = {
            "state_dict": predictor.state_dict(),
            "config": predictor_config,
            "steering_mode": STEERING_MODE,
            "args": history["args"],
            "epoch": epoch,
            "loss": epoch_loss,
        }
        torch.save(checkpoint, output_dir / "steering_predictor.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, output_dir / "steering_predictor_best.pt")

        # written every epoch so an interrupted run keeps its numbers, and doubles as the
        # machine-readable twin of the two plots
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_loss_curve(history, output_dir / "loss_curve.png")
        plot_alpha_schedule(history, output_dir / "alpha_schedule.png", args.alpha_min, args.alpha_max)

    print(f"\nDone. Best epoch mean loss {best_loss:+.4f}. Artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
