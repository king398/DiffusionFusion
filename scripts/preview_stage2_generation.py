#!/usr/bin/env python3
"""Generate retained decoder previews from a legacy checkpoint through Stage 2 JiT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from JiT.denoiser import Denoiser  # noqa: E402
from JiT.eval.diffusion_decoder import (  # noqa: E402
    decode_with_decoder,
    load_decoder_for_eval,
)
from JiT.eval.utils import (  # noqa: E402
    autocast_context,
    load_checkpoint_args,
    load_checkpoint_payload,
    save_uint8_pngs,
)
from JiT.main_jit import (  # noqa: E402
    _maybe_remap_dual_checkpoint,
    get_args_parser,
    load_stream_specs,
)


LEGACY_MARKERS = (
    "latent_blocks.",
    "dino_blocks.",
    "cross_fusion_blocks.",
    "x_embedder.",
    "dino_embedder.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an old dual-stream EMA checkpoint through the Stage 2 remap path, "
            "sample a small batch, decode it, and retain PNGs for visual inspection."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--state-key",
        default="model_ema1",
        choices=("model", "model_ema1", "model_ema2"),
        help="JiT state dict to load. Baseline evaluation uses model_ema1.",
    )
    parser.add_argument(
        "--decoder-checkpoint-key",
        default="model_ema",
        choices=("auto", "model", "model_ema"),
    )
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--start-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-method", default="heun", choices=("euler", "heun"))
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=2.9)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=1.0)
    return parser.parse_args()


def is_legacy_dual_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    for key in state_dict:
        bare = key.removeprefix("net.")
        if any(bare.startswith(marker) for marker in LEGACY_MARKERS):
            return True
    return False


def build_eval_args(payload: dict, cli_args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(get_args_parser().parse_args([]))
    defaults.update(vars(load_checkpoint_args(payload, label="JiT")))
    defaults.update(
        {
            "sampling_method": cli_args.sampling_method,
            "num_sampling_steps": cli_args.num_sampling_steps,
            "cfg": cli_args.cfg,
            "cfg_scale": cli_args.cfg,
            "interval_min": cli_args.interval_min,
            "interval_max": cli_args.interval_max,
            "seed": cli_args.seed,
            "streams_config": None,
        }
    )
    args = argparse.Namespace(**defaults)
    args.streams, _ = load_stream_specs(args)
    return args


@torch.inference_mode()
def main() -> None:
    cli_args = parse_args()
    if cli_args.num_images <= 0:
        raise ValueError("--num-images must be positive.")

    device = torch.device(cli_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    checkpoint_path = cli_args.checkpoint.expanduser().resolve()
    decoder_checkpoint = cli_args.decoder_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"JiT checkpoint not found: {checkpoint_path}")
    if not decoder_checkpoint.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_checkpoint}")

    payload = load_checkpoint_payload(checkpoint_path)
    if cli_args.state_key not in payload:
        raise KeyError(f"Checkpoint does not contain `{cli_args.state_key}`.")
    source_state = payload[cli_args.state_key]
    if not is_legacy_dual_state_dict(source_state):
        raise RuntimeError(
            f"`{cli_args.state_key}` is not a legacy dual-stream state dict; "
            "this preview is intended to verify the Stage 2 remap path."
        )

    eval_args = build_eval_args(payload, cli_args)
    model = Denoiser(eval_args).to(device).eval()
    remapped_state = _maybe_remap_dual_checkpoint(source_state, label=cli_args.state_key)
    model.load_state_dict(remapped_state, strict=True)
    print(
        f"Strictly loaded remapped `{cli_args.state_key}` into Stage 2 Denoiser: "
        "zero missing or unexpected keys.",
        flush=True,
    )

    decoder = load_decoder_for_eval(
        str(decoder_checkpoint),
        device,
        cli_args.decoder_checkpoint_key,
    )
    output_dir = cli_args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cli_args.seed)
    labels = (
        torch.arange(cli_args.start_class, cli_args.start_class + cli_args.num_images)
        .remainder(eval_args.class_num)
        .to(device=device, dtype=torch.long)
    )
    with autocast_context(device):
        sampled = model.generate(labels)
    if set(sampled) != {"latent", "dino"}:
        raise RuntimeError(f"Stage 2 generation returned unexpected stream keys: {set(sampled)}")
    images = decode_with_decoder(decoder, sampled["latent"], sampled["dino"])
    save_uint8_pngs(
        images,
        np.arange(cli_args.num_images, dtype=np.int64),
        output_dir,
        width=5,
    )
    print(
        "PASS: generated and decoded "
        f"{cli_args.num_images} retained previews to {output_dir} "
        f"using {cli_args.sampling_method}-{cli_args.num_sampling_steps}, "
        f"CFG {cli_args.cfg}, interval [{cli_args.interval_min}, {cli_args.interval_max}].",
        flush=True,
    )


if __name__ == "__main__":
    main()
