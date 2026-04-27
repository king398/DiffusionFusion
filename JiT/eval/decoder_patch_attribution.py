#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from JiT.decoder.dataset import RamLoadedShardDataset, inspect_feature_shards
from JiT.eval.eval_decoder import (
    build_decoder_model_from_args,
    extract_image_normalization,
    load_checkpoint_args,
    resolve_batch_size,
    resolve_decoder_state_dict,
    resolve_feature_dir_name,
    select_checkpoint_key,
)
from JiT.eval.utils import images_to_uint8, load_checkpoint_payload


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_patch(value: str) -> tuple[int, int]:
    try:
        row_text, col_text = value.split(",", maxsplit=1)
        row = int(row_text)
        col = int(col_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected patch as row,col, got {value!r}."
        ) from exc
    if row < 0 or col < 0:
        raise argparse.ArgumentTypeError("Patch row and col must be non-negative.")
    return row, col


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Grad-CAM-style patch attribution on JiT.decoder.model.Decoder context "
            "tokens, split into DINO and latent token heatmaps."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature-root", type=str, required=True)
    parser.add_argument("--image-data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--latent-dir-name", type=str, default=None)
    parser.add_argument("--dino-dir-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of samples to process. Defaults to 100.",
    )
    parser.add_argument(
        "--patch",
        type=parse_patch,
        action="append",
        default=None,
        help=(
            "Output patch as row,col. Repeat for multiple patches per sample. "
            "If omitted, one high-variance decoded patch is selected per sample."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default="mean",
        choices=["mean", "square", "red", "green", "blue"],
        help="Scalar target computed from the selected output patch.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        dest="pin_mem",
        help="Pin dataloader memory when using CUDA.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=None)
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save raw signed and absolute attribution maps beside each figure.",
    )
    parser.add_argument(
        "--heatmap-percentile",
        type=float,
        default=100.0,
        help=(
            "Percentile used to scale attribution heatmaps for display. Defaults to "
            "100, which is plain min/max scaling per saved figure."
        ),
    )
    parser.add_argument(
        "--heatmap-gamma",
        type=float,
        default=1.0,
        help=(
            "Gamma applied after heatmap normalization. Values below 1 brighten "
            "low attribution values. Defaults to 1 for linear scaling."
        ),
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload the output directory to Hugging Face after generation.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="Mithilss/decoder_patch_attribution",
        help=(
            "Hugging Face repo id used with --upload-to-hf. A bare namespace "
            "like `Mithilss` is expanded to `Mithilss/decoder_patch_attribution`."
        ),
    )
    parser.add_argument(
        "--hf-repo-type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
    )
    parser.add_argument(
        "--hf-path-in-repo",
        type=str,
        default="decoder_patch_attribution",
    )
    return parser.parse_args()


def tensor_to_uint8_image(
    image: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Image.Image:
    array = images_to_uint8(image.unsqueeze(0), mean, std)[0]
    return Image.fromarray(array, mode="RGB")


def robust_vmax(values: np.ndarray, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    positive = arr[arr > 0.0]
    if positive.size == 0:
        return 0.0
    pct = float(np.clip(percentile, 0.0, 100.0))
    return float(max(np.percentile(positive, pct), positive.max() * 1.0e-6))


def normalize_map(
    values: np.ndarray,
    vmax: float | None = None,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if vmax is None:
        vmax = float(arr.max()) if arr.size else 0.0
    if vmax <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    normed = np.clip(arr / vmax, 0.0, 1.0)
    if gamma > 0.0 and gamma != 1.0:
        normed = np.power(normed, gamma)
    return normed


def _interpolate_colormap(x: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    out = np.zeros((*x.shape, 3), dtype=np.float32)
    for idx in range(len(stops) - 1):
        left_x, left_rgb = stops[idx]
        right_x, right_rgb = stops[idx + 1]
        if idx == len(stops) - 2:
            mask = (x >= left_x) & (x <= right_x)
        else:
            mask = (x >= left_x) & (x < right_x)
        if not np.any(mask):
            continue
        denom = max(right_x - left_x, 1.0e-12)
        t = ((x[mask] - left_x) / denom).reshape(-1, 1)
        left = np.asarray(left_rgb, dtype=np.float32)
        right = np.asarray(right_rgb, dtype=np.float32)
        out[mask] = left * (1.0 - t) + right * t
    return np.clip(out, 0, 255).astype(np.uint8)


ATTRIBUTION_COLORMAP = [
    (0.00, (72, 0, 118)),
    (0.16, (45, 23, 180)),
    (0.32, (24, 103, 220)),
    (0.48, (28, 185, 205)),
    (0.64, (62, 210, 84)),
    (0.78, (238, 220, 58)),
    (0.90, (246, 130, 42)),
    (1.00, (210, 32, 32)),
]

DINO_COLOR = (0, 190, 220)
LATENT_COLOR = (236, 120, 36)
NEUTRAL_COLOR = (246, 246, 246)


def heatmap_image(
    values: np.ndarray,
    size: int,
    *,
    vmax: float | None = None,
    gamma: float = 1.0,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    normed = normalize_map(values, vmax=vmax, gamma=gamma)
    colors = _interpolate_colormap(
        normed,
        ATTRIBUTION_COLORMAP,
    )
    return Image.fromarray(colors, mode="RGB").resize((size, size), resample)


def attribution_colorbar(width: int, vmax: float, *, height: int = 46) -> Image.Image:
    bar_h = 16
    margin_x = 8
    gradient_w = max(1, width - margin_x * 2)
    gradient = np.linspace(0.0, 1.0, gradient_w, dtype=np.float32).reshape(1, gradient_w)
    colors = _interpolate_colormap(gradient, ATTRIBUTION_COLORMAP)
    bar = Image.fromarray(colors, mode="RGB").resize(
        (gradient_w, bar_h),
        Image.Resampling.BILINEAR,
    )

    panel = Image.new("RGB", (width, height), "white")
    panel.paste(bar, (margin_x, 4))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((margin_x, 24), "0", fill=(24, 24, 24), font=font)
    max_text = f"{vmax:.2e}"
    text_bbox = draw.textbbox((0, 0), max_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    draw.text((width - margin_x - text_w, 24), max_text, fill=(24, 24, 24), font=font)
    title = "shared DINO/latent abs attribution scale"
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_w) // 2, 24), title, fill=(24, 24, 24), font=font)
    return panel


def _source_rgb(source_share: np.ndarray) -> np.ndarray:
    """Map 0=latent, 0.5=neutral, 1=DINO to a diverging RGB ramp."""
    share = np.clip(source_share, 0.0, 1.0)
    out = np.zeros((*share.shape, 3), dtype=np.float32)
    latent = np.asarray(LATENT_COLOR, dtype=np.float32)
    dino = np.asarray(DINO_COLOR, dtype=np.float32)
    neutral = np.asarray(NEUTRAL_COLOR, dtype=np.float32)

    lower = share <= 0.5
    if np.any(lower):
        t = (share[lower] / 0.5).reshape(-1, 1)
        out[lower] = latent * (1.0 - t) + neutral * t

    upper = ~lower
    if np.any(upper):
        t = ((share[upper] - 0.5) / 0.5).reshape(-1, 1)
        out[upper] = neutral * (1.0 - t) + dino * t

    return np.clip(out, 0, 255).astype(np.uint8)


def source_mix_image(
    dino_abs: np.ndarray,
    latent_abs: np.ndarray,
    size: int,
    *,
    percentile: float = 98.0,
    gamma: float = 0.65,
) -> Image.Image:
    dino = np.nan_to_num(np.asarray(dino_abs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    latent = np.nan_to_num(np.asarray(latent_abs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    total = np.maximum(dino + latent, 0.0)
    share = np.divide(dino, total, out=np.full_like(total, 0.5), where=total > 0.0)
    source_colors = _source_rgb(share).astype(np.float32)

    strength = normalize_map(total, robust_vmax(total, percentile), gamma=gamma)
    strength = np.clip(0.12 + 0.88 * strength, 0.0, 1.0)[..., None]
    white = np.full_like(source_colors, 255.0)
    colors = white * (1.0 - strength) + source_colors * strength
    return Image.fromarray(np.clip(colors, 0, 255).astype(np.uint8), mode="RGB").resize(
        (size, size),
        Image.Resampling.NEAREST,
    )


def difference_image(values: np.ndarray, size: int, *, percentile: float = 98.0) -> Image.Image:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_values = np.abs(arr)
    positive = abs_values[abs_values > 0.0]
    scale = float(np.percentile(positive, np.clip(percentile, 0.0, 100.0))) if positive.size else 0.0
    if positive.size:
        scale = max(scale, float(positive.max()) * 1.0e-6)
    if scale <= 0.0:
        normed = np.full_like(arr, 0.5, dtype=np.float32)
    else:
        normed = np.clip((arr / scale + 1.0) * 0.5, 0.0, 1.0)
    colors = _interpolate_colormap(
        normed,
        [
            (0.00, (49, 76, 160)),
            (0.50, (245, 245, 245)),
            (1.00, (180, 42, 42)),
        ],
    )
    return Image.fromarray(colors, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def _share(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_source_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    dino_share: float,
) -> None:
    x0, y0, x1, y1 = box
    dino_share = float(np.clip(dino_share, 0.0, 1.0))
    split = int(round(x0 + (x1 - x0) * dino_share))
    draw.rectangle([x0, y0, x1, y1], fill=LATENT_COLOR)
    if split > x0:
        draw.rectangle([x0, y0, split, y1], fill=DINO_COLOR)
    draw.rectangle([x0, y0, x1, y1], outline=(210, 210, 210), width=1)


def attribution_summary_panel(
    width: int,
    *,
    dino_abs: np.ndarray,
    latent_abs: np.ndarray,
    row: int,
    col: int,
) -> Image.Image:
    height = 92
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    title_font = _load_font(14)
    font = _load_font(12)

    dino_total = float(np.asarray(dino_abs, dtype=np.float32).sum())
    latent_total = float(np.asarray(latent_abs, dtype=np.float32).sum())
    total_share = _share(dino_total, dino_total + latent_total)

    dino_aligned = float(dino_abs[row, col])
    latent_aligned = float(latent_abs[row, col])
    aligned_share = _share(dino_aligned, dino_aligned + latent_aligned)

    dino_nonlocal = max(0.0, dino_total - dino_aligned)
    latent_nonlocal = max(0.0, latent_total - latent_aligned)
    nonlocal_share = _share(dino_nonlocal, dino_nonlocal + latent_nonlocal)

    draw.text((8, 6), "source summary", fill=(24, 24, 24), font=title_font)
    legend_x = width - 242
    draw.rectangle([legend_x, 10, legend_x + 12, 22], fill=DINO_COLOR)
    draw.text((legend_x + 18, 8), "DINO", fill=(24, 24, 24), font=font)
    draw.rectangle([legend_x + 78, 10, legend_x + 90, 22], fill=LATENT_COLOR)
    draw.text((legend_x + 96, 8), "latent", fill=(24, 24, 24), font=font)

    rows = [
        ("total abs", total_share, dino_total, latent_total),
        ("target patch", aligned_share, dino_aligned, latent_aligned),
        ("nonlocal", nonlocal_share, dino_nonlocal, latent_nonlocal),
    ]
    label_x = 8
    bar_x = 116
    value_space = 250 if width >= 700 else 170
    bar_w = max(120, width - bar_x - value_space)
    value_x = bar_x + bar_w + 10
    for idx, (label, share, dino_value, latent_value) in enumerate(rows):
        y = 34 + idx * 18
        draw.text((label_x, y - 2), label, fill=(40, 40, 40), font=font)
        _draw_source_bar(draw, (bar_x, y, bar_x + bar_w, y + 10), share)
        value = f"DINO {_format_pct(share)} | {dino_value:.2e} / {latent_value:.2e}"
        draw.text((value_x, y - 3), value, fill=(40, 40, 40), font=font)

    return panel


def draw_patch_box(
    image: Image.Image,
    row: int,
    col: int,
    patch_size: int,
    *,
    color: tuple[int, int, int] = (255, 48, 48),
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    x0 = col * patch_size
    y0 = row * patch_size
    x1 = (col + 1) * patch_size - 1
    y1 = (row + 1) * patch_size - 1
    max_width = max(1, min(width, (x1 - x0 + 1) // 2, (y1 - y0 + 1) // 2))
    for offset in range(max_width):
        draw.rectangle(
            [x0 + offset, y0 + offset, x1 - offset, y1 - offset],
            outline=color,
        )


def label_panel(image: Image.Image, title: str) -> Image.Image:
    title_h = 28
    panel = Image.new("RGB", (image.width, image.height + title_h), "white")
    panel.paste(image, (0, title_h))
    draw = ImageDraw.Draw(panel)
    font = _load_font(15)
    for size in (15, 14, 13, 12, 11):
        candidate = _load_font(size)
        bbox = draw.textbbox((0, 0), title, font=candidate)
        if bbox[2] - bbox[0] <= image.width - 14:
            font = candidate
            break
    draw.text((7, 6), title, fill=(24, 24, 24), font=font)
    return panel


def save_figure(
    path: Path,
    *,
    recon: Image.Image,
    dino_abs: np.ndarray,
    latent_abs: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
    grid_size: int,
    heatmap_percentile: float = 98.0,
    heatmap_gamma: float = 0.65,
) -> None:
    image_size = recon.width
    recon_panel = recon.copy()
    draw_patch_box(recon_panel, row, col, patch_size)

    dino_vmax = robust_vmax(dino_abs, heatmap_percentile)
    latent_vmax = robust_vmax(latent_abs, heatmap_percentile)
    total_abs = dino_abs + latent_abs
    total_vmax = robust_vmax(total_abs, heatmap_percentile)
    dino_panel = heatmap_image(
        dino_abs,
        image_size,
        vmax=dino_vmax,
        gamma=heatmap_gamma,
        resample=Image.Resampling.NEAREST,
    )
    latent_panel = heatmap_image(
        latent_abs,
        image_size,
        vmax=latent_vmax,
        gamma=heatmap_gamma,
        resample=Image.Resampling.NEAREST,
    )
    total_panel = heatmap_image(
        total_abs,
        image_size,
        vmax=total_vmax,
        gamma=heatmap_gamma,
        resample=Image.Resampling.NEAREST,
    )
    mix_panel = source_mix_image(
        dino_abs,
        latent_abs,
        image_size,
        percentile=heatmap_percentile,
        gamma=heatmap_gamma,
    )
    diff_panel = difference_image(
        latent_abs - dino_abs,
        image_size,
        percentile=heatmap_percentile,
    )

    heat_patch = max(image_size // grid_size, 1)
    for panel in (dino_panel, latent_panel, total_panel, mix_panel, diff_panel):
        draw_patch_box(panel, row, col, heat_patch, color=(0, 255, 255), width=2)

    panels = [
        label_panel(recon_panel, "decoded image"),
        label_panel(dino_panel, f"DINO abs pattern | pmax {dino_vmax:.2e}"),
        label_panel(latent_panel, f"latent abs pattern | pmax {latent_vmax:.2e}"),
        label_panel(total_panel, f"total abs | pmax {total_vmax:.2e}"),
        label_panel(mix_panel, "source mix | cyan=DINO"),
        label_panel(diff_panel, "excess: latent red, DINO blue"),
    ]
    gap = 8
    cols = 3
    rows = 2
    summary = attribution_summary_panel(
        panels[0].width * cols + gap * (cols - 1),
        dino_abs=dino_abs,
        latent_abs=latent_abs,
        row=row,
        col=col,
    )
    canvas = Image.new(
        "RGB",
        (
            panels[0].width * cols + gap * (cols - 1),
            panels[0].height * rows + gap * (rows - 1) + summary.height,
        ),
        "white",
    )
    for idx, panel in enumerate(panels):
        x = (idx % cols) * (panel.width + gap)
        y = (idx // cols) * (panel.height + gap)
        canvas.paste(panel, (x, y))
    canvas.paste(summary, (0, panels[0].height * rows + gap * (rows - 1)))
    canvas.save(path, format="PNG", compress_level=3)


def select_visible_patch(
    image: torch.Tensor,
    patch_size: int,
    *,
    border: int = 1,
) -> tuple[int, int]:
    _, height, width = image.shape
    rows = height // patch_size
    cols = width // patch_size
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid image shape for patch selection: {tuple(image.shape)}")

    patches = image[:, : rows * patch_size, : cols * patch_size]
    patches = patches.reshape(3, rows, patch_size, cols, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(rows, cols, -1)
    scores = patches.float().std(dim=-1)

    if rows > border * 2 and cols > border * 2:
        masked = torch.full_like(scores, -1.0)
        masked[border : rows - border, border : cols - border] = scores[
            border : rows - border, border : cols - border
        ]
        scores = masked

    flat_idx = int(torch.argmax(scores).item())
    return flat_idx // cols, flat_idx % cols


def patch_target(
    image: torch.Tensor,
    row: int,
    col: int,
    patch_size: int,
    mode: str,
) -> torch.Tensor:
    y0 = row * patch_size
    y1 = (row + 1) * patch_size
    x0 = col * patch_size
    x1 = (col + 1) * patch_size
    patch = image[0, :, y0:y1, x0:x1]
    if patch.numel() == 0:
        raise ValueError(
            f"Patch ({row}, {col}) is outside decoded image shape {tuple(image.shape)}."
        )
    if mode == "mean":
        return patch.mean()
    if mode == "square":
        return patch.square().mean()
    channel = {"red": 0, "green": 1, "blue": 2}[mode]
    return patch[channel].mean()


def decoder_forward_with_context(
    decoder: torch.nn.Module,
    dino: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_tokens = decoder.latent_tokenizer(latent)
    latent_tokens = latent_tokens + decoder.pos_embed
    dino_tokens = decoder._prepare_dino_tokens(dino)

    x = decoder.query_tokens.expand(latent.shape[0], -1, -1)
    x = x + decoder.query_pos_embed
    ctx_tokens = torch.cat([dino_tokens, latent_tokens], dim=1)
    ctx_tokens.retain_grad()
    for block in decoder.blocks:
        x = block(x, ctx_tokens)
    return decoder.tokens_to_image(x), ctx_tokens


def compute_attribution(
    decoder: torch.nn.Module,
    dino: torch.Tensor,
    latent: torch.Tensor,
    *,
    row: int,
    col: int,
    target_mode: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decoder.zero_grad(set_to_none=True)
    image, ctx_tokens = decoder_forward_with_context(decoder, dino, latent)
    target = patch_target(image, row, col, decoder.patch_size, target_mode)
    target.backward()

    if ctx_tokens.grad is None:
        raise RuntimeError("Context token gradients were not retained.")
    token_attr = (ctx_tokens.grad * ctx_tokens).sum(dim=-1)[0].detach().float().cpu()
    num_patches = int(decoder.num_patches)
    grid_size = int(num_patches**0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Decoder num_patches={num_patches} is not a square grid.")

    dino_signed = token_attr[:num_patches].reshape(grid_size, grid_size).numpy()
    latent_signed = token_attr[num_patches:].reshape(grid_size, grid_size).numpy()
    dino_abs = np.abs(dino_signed)
    latent_abs = np.abs(latent_signed)
    return image.detach(), dino_signed, latent_signed, dino_abs, latent_abs


def attribution_metrics(
    *,
    sample_id: int,
    row: int,
    col: int,
    dino_signed: np.ndarray,
    latent_signed: np.ndarray,
    dino_abs: np.ndarray,
    latent_abs: np.ndarray,
) -> dict[str, float | int]:
    dino_abs_sum = float(dino_abs.sum())
    latent_abs_sum = float(latent_abs.sum())
    total = dino_abs_sum + latent_abs_sum
    dino_aligned_abs = float(dino_abs[row, col])
    latent_aligned_abs = float(latent_abs[row, col])
    return {
        "sample_id": sample_id,
        "patch_row": row,
        "patch_col": col,
        "dino_abs_sum": dino_abs_sum,
        "latent_abs_sum": latent_abs_sum,
        "dino_share": float(dino_abs_sum / total) if total > 0.0 else 0.0,
        "dino_aligned_abs": dino_aligned_abs,
        "latent_aligned_abs": latent_aligned_abs,
        "dino_nonlocal_abs": float(dino_abs_sum - dino_aligned_abs),
        "latent_nonlocal_abs": float(latent_abs_sum - latent_aligned_abs),
        "dino_aligned_signed": float(dino_signed[row, col]),
        "latent_aligned_signed": float(latent_signed[row, col]),
    }


def iter_requested_patches(
    explicit_patches: Iterable[tuple[int, int]] | None,
    image: torch.Tensor,
    patch_size: int,
) -> list[tuple[int, int]]:
    if explicit_patches:
        return list(explicit_patches)
    return [select_visible_patch(image[0].detach().cpu(), patch_size)]


def upload_to_huggingface(args: argparse.Namespace, output_dir: Path) -> tuple[str, str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --upload-to-hf. Install it or rerun without upload."
        ) from exc

    repo_id = args.hf_repo_id.strip("/")
    if "/" not in repo_id:
        repo_id = f"{repo_id}/decoder_patch_attribution"
    path_in_repo = args.hf_path_in_repo.strip("/") or "."

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type=args.hf_repo_type,
        exist_ok=True,
    )
    upload_large_folder = getattr(api, "upload_large_folder", None)
    if upload_large_folder is not None and path_in_repo == ".":
        upload_large_folder(
            repo_id=repo_id,
            repo_type=args.hf_repo_type,
            folder_path=str(output_dir),
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            repo_type=args.hf_repo_type,
            folder_path=str(output_dir),
            path_in_repo=path_in_repo,
        )
    return repo_id, path_in_repo


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint_path}")

    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    checkpoint_args = load_checkpoint_args(checkpoint_payload)
    checkpoint_key = select_checkpoint_key(args, checkpoint_payload)
    image_size = int(getattr(checkpoint_args, "decoder_output_image_size"))
    latent_dir_name = resolve_feature_dir_name(
        args.latent_dir_name,
        getattr(checkpoint_args, "latent_dir_name", None),
        "imagenet256_latents",
        args.split,
    )
    dino_dir_name = resolve_feature_dir_name(
        args.dino_dir_name,
        getattr(checkpoint_args, "dino_dir_name", None),
        "imagenet256_dinov3_features",
        args.split,
    )
    batch_size = resolve_batch_size(args.batch_size, checkpoint_args, world_size=1)
    pin_mem = bool(getattr(checkpoint_args, "pin_mem", True)) if args.pin_mem is None else args.pin_mem
    decoder_batch_prefetch = bool(getattr(checkpoint_args, "decoder_batch_prefetch", True))
    image_model_name = getattr(
        checkpoint_args,
        "image_model_name",
        "vit_base_patch16_dinov3.lvd1689m",
    )

    latent_store = inspect_feature_shards(args.feature_root, latent_dir_name)
    dino_store = inspect_feature_shards(args.feature_root, dino_dir_name)
    dataset = RamLoadedShardDataset(
        latent_store=latent_store,
        dino_store=dino_store,
        batch_size=batch_size,
        num_replicas=1,
        rank=0,
        shuffle_shards=False,
        seed=0,
        preload_next_shard=False,
        preload_next_batch=decoder_batch_prefetch,
        image_data_path=args.image_data_path,
        image_data_split=args.split,
        image_model_name=image_model_name,
        image_size=image_size,
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=pin_mem and device.type == "cuda",
    )
    image_mean, image_std = extract_image_normalization(data_loader)

    decoder = build_decoder_model_from_args(checkpoint_args).to(device)
    checkpoint_state, _stripped_prefix = resolve_decoder_state_dict(
        checkpoint_payload[checkpoint_key],
        decoder,
    )
    decoder.load_state_dict(checkpoint_state, strict=True)
    decoder.eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    arrays_dir = output_dir / "arrays"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npz:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.jsonl"
    all_metrics: list[dict[str, float | int | str]] = []
    processed_samples = 0
    progress = tqdm(data_loader, desc="decoder attribution")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        for batch in progress:
            latent_batch = batch["latent"].to(device, non_blocking=True)
            dino_batch = batch["dino"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"].cpu().numpy().astype(np.int64, copy=False)

            for item_idx, sample_id in enumerate(sample_ids.tolist()):
                if processed_samples >= args.num_images:
                    break

                latent = latent_batch[item_idx : item_idx + 1].float()
                dino = dino_batch[item_idx : item_idx + 1].float()

                with torch.no_grad():
                    preview = decoder.generate(latent, dino)
                patches = iter_requested_patches(args.patch, preview, decoder.patch_size)

                for row, col in patches:
                    if row >= int(decoder.num_patches**0.5) or col >= int(decoder.num_patches**0.5):
                        raise ValueError(
                            f"Patch ({row}, {col}) exceeds decoder grid for num_patches={decoder.num_patches}."
                        )

                    image, dino_signed, latent_signed, dino_abs, latent_abs = compute_attribution(
                        decoder,
                        dino,
                        latent,
                        row=row,
                        col=col,
                        target_mode=args.target,
                    )
                    recon = tensor_to_uint8_image(image[0].cpu(), image_mean, image_std)
                    stem = f"decoder_attr_sample{sample_id:08d}_patch{row:02d}_{col:02d}"
                    figure_path = figures_dir / f"{stem}.png"
                    save_figure(
                        figure_path,
                        recon=recon,
                        dino_abs=dino_abs,
                        latent_abs=latent_abs,
                        row=row,
                        col=col,
                        patch_size=decoder.patch_size,
                        grid_size=int(decoder.num_patches**0.5),
                        heatmap_percentile=args.heatmap_percentile,
                        heatmap_gamma=args.heatmap_gamma,
                    )

                    metrics = attribution_metrics(
                        sample_id=int(sample_id),
                        row=row,
                        col=col,
                        dino_signed=dino_signed,
                        latent_signed=latent_signed,
                        dino_abs=dino_abs,
                        latent_abs=latent_abs,
                    )
                    metrics.update(
                        {
                            "figure": str(figure_path.relative_to(output_dir)),
                            "target": args.target,
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_key": checkpoint_key,
                            "split": args.split,
                            "latent_dir_name": latent_dir_name,
                            "dino_dir_name": dino_dir_name,
                        }
                    )
                    metrics_path = metrics_dir / f"{stem}.json"
                    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                    summary_file.write(json.dumps(metrics) + "\n")
                    summary_file.flush()
                    all_metrics.append(metrics)

                    if args.save_npz:
                        np.savez_compressed(
                            arrays_dir / f"{stem}.npz",
                            dino_signed=dino_signed,
                            latent_signed=latent_signed,
                            dino_abs=dino_abs,
                            latent_abs=latent_abs,
                        )

                processed_samples += 1
                progress.set_postfix(samples=processed_samples)

            if processed_samples >= args.num_images:
                break

    manifest = {
        "num_samples": processed_samples,
        "num_figures": len(all_metrics),
        "target": args.target,
        "checkpoint": str(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "split": args.split,
        "feature_root": args.feature_root,
        "latent_dir_name": latent_dir_name,
        "dino_dir_name": dino_dir_name,
        "image_data_path": args.image_data_path,
        "figures_dir": str(figures_dir),
        "summary_jsonl": str(summary_path),
    }
    if all_metrics:
        dino_shares = np.asarray([float(item["dino_share"]) for item in all_metrics])
        manifest["mean_dino_share"] = float(dino_shares.mean())
        manifest["median_dino_share"] = float(np.median(dino_shares))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if args.upload_to_hf:
        uploaded_repo_id, uploaded_path = upload_to_huggingface(args, output_dir)
        print(
            json.dumps(
                {
                    "uploaded_to_hf": uploaded_repo_id,
                    "repo_type": args.hf_repo_type,
                    "path_in_repo": uploaded_path,
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
