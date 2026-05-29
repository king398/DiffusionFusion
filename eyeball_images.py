"""Standalone: generate + decode a handful of images with EMA1 so we can eyeball
quality (the eval deletes its image folder after FID). Mirrors the engine eval
path: remap dual checkpoint -> load -> swap to ema_params1 -> generate (bf16
autocast) -> decode_with_decoder -> save PNGs.

Run from repo root:  .venv/bin/python eyeball_images.py
"""
import os
import numpy as np
import torch

import JiT.main_jit as M
from JiT.denoiser import Denoiser
from JiT.engine_jit import _swap_ema_parameters, _save_png
from JiT.eval.diffusion_decoder import decode_with_decoder

CKPT = "/work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/checkpoint-last.pth"
DECODER = "/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth"
OUTDIR = "/u/galoiz/DiffusionFusion-independent-backup/artifacts/eyeball_testb"
# recognizable ImageNet classes for a visual sanity check
CLASSES = [207, 281, 388, 130, 1, 933, 949, 980, 817, 248, 90, 309, 555, 963, 417, 11]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda")

    parser = M.get_args_parser()
    args = parser.parse_args([
        "--model", "JiT-Dual-B/2-4C-896",
        "--cfg", "2.9", "--interval_min", "0.1", "--interval_max", "1.0",
        "--P_mean", "-0.8", "--P_std", "0.8", "--noise_scale", "1.0",
        "--class_num", "1000", "--latent_size", "32",
        "--decoder_checkpoint", DECODER, "--decoder_checkpoint_key", "model_ema",
        "--output_dir", OUTDIR, "--resume", "",
    ])
    # sampling_method/num_sampling_steps default to heun/50 (the baseline)
    args.streams = M.load_stream_specs(args)[0]
    print(f"streams: {[s.name for s in args.streams]}  "
          f"method={args.sampling_method} steps={args.num_sampling_steps} "
          f"cfg={args.cfg} interval=[{args.interval_min},{args.interval_max}]")

    model = Denoiser(args).to(device).eval()

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd_model = M._maybe_remap_dual_checkpoint(payload["model"], label="model")
    sd_ema1 = M._maybe_remap_dual_checkpoint(payload["model_ema1"], label="model_ema1")
    model.load_state_dict(sd_model)  # strict
    model.ema_params1 = [sd_ema1[name].to(device)
                         for name, _ in model.named_parameters()]
    print("loaded model + ema_params1")

    decoder = M.load_eval_decoder(args, device, global_rank=0)

    labels = torch.tensor(CLASSES, dtype=torch.long, device=device)
    with _swap_ema_parameters(model, model.ema_params1):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sampled = model.generate(labels)
    print("generated:", {k: tuple(v.shape) for k, v in sampled.items()})

    images = decode_with_decoder(decoder, sampled["latent"], sampled["dino"])
    images = np.asarray(images)
    print("decoded images:", images.shape, images.dtype,
          "min", float(images.min()), "max", float(images.max()))

    for i, (img, cls) in enumerate(zip(images, CLASSES)):
        _save_png(os.path.join(OUTDIR, f"eye_{i:02d}_cls{cls}.png"), img)
    print(f"saved {len(images)} PNGs to {OUTDIR}")


if __name__ == "__main__":
    main()
