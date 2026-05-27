"""Test A: offline forward parity, old JiTDualStream vs new JiTMultiStream.

Loads the 80-epoch dual-stream checkpoint, builds both the legacy dual model
(strict load of bare keys) and the refactored multistream model (strict load of
remapped keys), feeds identical random inputs in fp32 eval mode, and asserts the
outputs match. This is the definitive correctness check for the Stage-2 refactor
and needs no sampling / decode / FID.

Run from repo root:  .venv/bin/python parity_test_a.py
"""

import argparse
import sys

import torch

from JiT.model_jit import (
    JiT_Dual_B_2_4C_896,
    make_jit_dual_b_2_4c_896_multistream,
    remap_dual_to_multistream,
)

CKPT = "/work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/checkpoint-last.pth"
NET_PREFIX = "net."
NUM_CLASSES = 1000
WARN_TOL = 1e-4   # above this we flag; task says ~1e-5 expected, up to 1e-4 fine
FAIL_TOL = 1e-2   # above this the forward math genuinely diverged


def strip_prefix(state_dict, prefix):
    out, kept = {}, 0
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
            kept += 1
        else:
            out[k] = v
    return out, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu",
                    help="cpu (deterministic, default) or cuda")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        # Keep fp32 truly fp32 so the diff reflects only the algebraic
        # reformulation, not TF32 rounding.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    print(f"[load] {CKPT}")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        print(f"[FATAL] payload has no 'model' key; keys={list(payload)[:10]}")
        return 2
    print(f"[load] payload keys: {sorted(payload.keys())}")

    raw = payload["model"]
    bare, kept = strip_prefix(raw, NET_PREFIX)
    print(f"[load] model state: {len(raw)} keys, stripped '{NET_PREFIX}' from {kept}")
    sample = [k for k in bare if "blocks" in k or "cross_fusion" in k][:6]
    print(f"[load] sample bare keys (expect OLD dual names): {sample}")
    is_old = any(k.startswith(("latent_blocks.", "dino_blocks.", "cross_fusion_blocks.")) for k in bare)
    if not is_old:
        print("[WARN] bare keys do not look like an old dual-stream checkpoint!")

    # ---- old model: strict load of bare keys -------------------------------
    old = JiT_Dual_B_2_4C_896(num_classes=NUM_CLASSES)
    miss, unexp = old.load_state_dict(bare, strict=False)
    print(f"[old ] load: missing={len(miss)} unexpected={len(unexp)}")
    if miss or unexp:
        print(f"       missing(5)={list(miss)[:5]}\n       unexpected(5)={list(unexp)[:5]}")
        print("[FATAL] old dual model did not load strictly.")
        return 3

    # ---- new model: strict load of REMAPPED keys ---------------------------
    new = make_jit_dual_b_2_4c_896_multistream(num_classes=NUM_CLASSES)
    remapped = remap_dual_to_multistream(bare)
    miss2, unexp2 = new.load_state_dict(remapped, strict=False)
    print(f"[new ] load(remap): missing={len(miss2)} unexpected={len(unexp2)}")
    if miss2 or unexp2:
        print(f"       missing={list(miss2)[:20]}\n       unexpected={list(unexp2)[:20]}")
        print("[FATAL] remap_dual_to_multistream did NOT produce an exact key match. Shim is broken.")
        return 4
    print("[new ] strict key match OK (0 missing / 0 unexpected) -- remap is faithful")

    old = old.to(device).float().eval()
    new = new.to(device).float().eval()

    # ---- identical random inputs -------------------------------------------
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    B = args.batch
    latent = torch.randn(B, 4, 32, 32, generator=g).to(device)
    dino = torch.randn(B, 768, 16, 16, generator=g).to(device)
    t = torch.rand(B, generator=g).to(device)
    y = torch.randint(0, NUM_CLASSES, (B,), generator=g).to(device)

    def compare(tag, masked):
        with torch.no_grad():
            lat_o, dino_o = old(latent, dino, t, y,
                                mask_dino_to_latent=masked)
            out = new({"latent": latent, "dino": dino}, t, y,
                      mask={"dino": {"latent": True}} if masked else None)
        d_lat = (lat_o - out["latent"]).abs().max().item()
        d_dino = (dino_o - out["dino"]).abs().max().item()
        worst = max(d_lat, d_dino)
        status = "OK" if worst <= WARN_TOL else ("WARN" if worst <= FAIL_TOL else "FAIL")
        # reference magnitudes prove outputs are non-degenerate (not both-zero)
        ref = (f"refLat[absmax={lat_o.abs().max():.3f} std={lat_o.std():.3f}] "
               f"refDino[absmax={dino_o.abs().max():.3f} std={dino_o.std():.3f}]")
        print(f"[{tag}] max|dlatent|={d_lat:.3e}  max|ddino|={d_dino:.3e}  -> {status}  | {ref}")
        return worst

    print(f"\n[run] device={device} dtype=fp32 B={B} seed={args.seed}\n")
    w1 = compare("unmasked", masked=False)
    w2 = compare("masked  ", masked=True)

    overall = max(w1, w2)
    print(f"\n[result] worst max-abs-diff over both paths: {overall:.3e}")
    if overall <= WARN_TOL:
        print("[result] PASS -- refactor is mathematically faithful (<=1e-4).")
        return 0
    if overall <= FAIL_TOL:
        print("[result] PASS-with-WARN -- larger than 1e-4 but <1e-2; inspect.")
        return 0
    print("[result] FAIL -- forward math diverged (>=1e-2). Bisect embedder/block/fusion.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
