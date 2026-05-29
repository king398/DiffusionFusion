#!/usr/bin/env python3
"""Validate Stage 2 JiT multistream parity against a legacy checkpoint."""

from __future__ import annotations

import argparse
import gc
import inspect
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from JiT.model_jit import (  # noqa: E402
    JiT_Dual_B_2_4C_896,
    JiT_models,
    JiTMultiStream,
    remap_dual_to_multistream,
)
from JiT.denoiser import Denoiser  # noqa: E402


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
            "Run full-scale fp32 forward parity between the legacy dual-stream JiT "
            "and the Stage 2 multistream JiT after remapping an old checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Legacy checkpoint-last.pth containing Denoiser state dicts.",
    )
    parser.add_argument(
        "--state-key",
        default="model",
        choices=("model", "model_ema1", "model_ema2"),
        help="Checkpoint state dict to compare. The raw model is the default gate.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def strip_denoiser_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        raise ValueError("Checkpoint state dict is empty.")

    prefixed = [key.startswith("net.") for key in state_dict]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Checkpoint mixes `net.`-prefixed and bare model keys.")

    bare = {
        key[len("net.") :] if key.startswith("net.") else key: value
        for key, value in state_dict.items()
    }
    if not any(
        key.startswith(marker) for key in bare for marker in LEGACY_MARKERS
    ):
        raise ValueError(
            "Selected state dict does not contain legacy dual-stream parameter names; "
            "this script requires a pre-refactor checkpoint."
        )
    return bare


def verify_stage2_code() -> None:
    model_module = Path(sys.modules[JiTMultiStream.__module__].__file__).resolve()
    denoiser_module = Path(sys.modules[Denoiser.__module__].__file__).resolve()
    forward_parameters = list(inspect.signature(Denoiser.forward).parameters)
    if forward_parameters[:3] != ["self", "streams", "labels"]:
        raise RuntimeError(
            "Imported Denoiser does not expose the Stage 2 dict-keyed "
            "`forward(self, streams, labels)` API."
        )
    print(f"Stage 2 model module: {model_module}", flush=True)
    print(f"Stage 2 denoiser module: {denoiser_module}", flush=True)
    print("Stage 2 Denoiser dict-keyed forward signature confirmed.", flush=True)


@torch.inference_mode()
def compare_forward(
    label: str,
    legacy: torch.nn.Module,
    multistream: torch.nn.Module,
    latent: torch.Tensor,
    dino: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    *,
    mask_dino_to_latent: bool,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    latent_ref, dino_ref = legacy(
        latent,
        dino,
        t,
        y,
        mask_dino_to_latent=mask_dino_to_latent,
    )
    mask = {"dino": {"latent": True}} if mask_dino_to_latent else None
    outputs = multistream({"latent": latent, "dino": dino}, t, y, mask=mask)

    latent_diff = (outputs["latent"] - latent_ref).abs().max().item()
    dino_diff = (outputs["dino"] - dino_ref).abs().max().item()
    print(
        f"{label}: latent max_abs_diff={latent_diff:.8g}, "
        f"dino max_abs_diff={dino_diff:.8g}",
        flush=True,
    )
    torch.testing.assert_close(outputs["latent"], latent_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(outputs["dino"], dino_ref, atol=atol, rtol=rtol)
    return latent_diff, dino_diff


def main() -> None:
    args = parse_args()
    verify_stage2_code()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    print(f"Loading `{args.state_key}` from {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.state_key not in payload:
        raise KeyError(f"Checkpoint does not contain `{args.state_key}`.")
    bare_state = strip_denoiser_prefix(payload[args.state_key])

    legacy = JiT_Dual_B_2_4C_896(num_classes=1000).eval()
    multistream = JiT_models["JiT-Dual-B/2-4C-896"](num_classes=1000).eval()
    if not isinstance(multistream, JiTMultiStream):
        raise RuntimeError("Production registry did not construct JiTMultiStream.")
    legacy.load_state_dict(bare_state, strict=True)
    multistream.load_state_dict(
        remap_dual_to_multistream(bare_state),
        strict=True,
    )
    print("Strict checkpoint loads succeeded for legacy and multistream models.", flush=True)

    del payload, bare_state
    gc.collect()

    legacy.to(device)
    multistream.to(device)
    torch.manual_seed(args.seed)
    latent = torch.randn(args.batch_size, 4, 32, 32, device=device)
    dino = torch.randn(args.batch_size, 768, 16, 16, device=device)
    t = torch.rand(args.batch_size, device=device)
    y = torch.randint(0, 1000, (args.batch_size,), device=device)

    print(f"Running fp32 parity on {device} with no autocast.", flush=True)
    diffs = []
    diffs.extend(
        compare_forward(
            "unmasked",
            legacy,
            multistream,
            latent,
            dino,
            t,
            y,
            mask_dino_to_latent=False,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    diffs.extend(
        compare_forward(
            "masked",
            legacy,
            multistream,
            latent,
            dino,
            t,
            y,
            mask_dino_to_latent=True,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    print(f"PASS: maximum absolute difference across both paths is {max(diffs):.8g}.")


if __name__ == "__main__":
    main()
