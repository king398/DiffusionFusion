# Environment setup: DiffusionFusion on NCSA DeltaAI (GH200)

Verified 2026-05-26 / 2026-05-27 on `gh*` compute nodes (NVIDIA GH200 Grace-Hopper, aarch64, sm_90), under user `galoiz`.

The canonical build is `scripts/setup_env.sh` (idempotent). The canonical eval launchers are `eval_galoiz.sh` (interactive, 1 GPU) and `sbatch/jit_eval_galoiz.sbatch` (batch, 4 GPUs by default). This document explains *why* each step is needed.

## Cluster / account facts

- **Login / batch host:** `gh-login01.delta.ncsa.illinois.edu`.
- **Partition:** `ghx4` (batch) / `ghx4-interactive` (interactive). GH200 nodes have 4 GPUs each (`gpu:nvidia_gh200_120gb:4`).
- **SLURM account:** `bgnp-dtai-gh` for `galoiz`. The original `sbatch/jit_eval.sbatch` hardcodes `betw-dtai-gh` (msalunkhe's account); submitting it as `galoiz` fails with `Invalid account or account/partition combination specified`. Always pass `-A bgnp-dtai-gh` (already set in `sbatch/jit_eval_galoiz.sbatch`).
- **Home quota:** 102 GiB soft / 103 GiB hard for `/u/galoiz` (the venv is ~12 GiB).
- **Shared data (read-only to `galoiz`):**
  - JiT baseline checkpoint (the 4.03 reference): `/work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/checkpoint-last.pth`
  - Decoder checkpoint: `/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth` (`model_ema` key)
  - FID stats: `JiT/fid_stats/jit_in256_stats.npz`
  - ImageNet (FID ref only): `/work/hdd/bgnp/msalunkhe/data/imagenet/`

## Building the venv

```bash
bash scripts/setup_env.sh
```

This runs `module load python/3.11.9`, creates `.venv/` in-repo, and `pip install -r requirements.txt`, then applies the four gotchas below.

### Gotcha 1 — torch defaults to a CPU wheel on aarch64

`requirements.txt` pins `torch==2.10.0` with no index URL. On aarch64 plain PyPI serves the CPU wheel (`2.10.0+cpu`, `torch.cuda.is_available() == False`); a `JiTMultiStream(...).cuda()` call eventually crashes with an opaque error far from the root cause. Fix:

```bash
pip install --no-deps --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"
```

The pinned `nvidia-*-cu12==12.8.*` libs in `requirements.txt` are already the right deps for cu128; `--no-deps` keeps them.

### Gotcha 2 — editable install for `from JiT...`

The eval/sbatch scripts `cd` into `JiT/` and run `main_jit.py`, which contains `from JiT...` imports. With CWD inside the package, `import JiT` fails with `ModuleNotFoundError: No module named 'JiT'`. Fix:

```bash
pip install -e . --no-deps
```

(Running scripts from the repo root also works without the editable install, but the launchers don't.)

### Gotcha 3 — torch-fidelity must be the LTH14 fork

`engine_jit.py` calls `calculate_metrics(..., fid_statistics_file=<npz>)`. Stock PyPI `torch-fidelity==0.3.0` **silently ignores** the `fid_statistics_file` kwarg and instead demands `input2` (a real-image dataset). The eval runs through all of generation + decode and then crashes at the very end with:

```
ValueError: Second input is required for "fid"
```

This is the crash that bit the first Test B run on 2026-05-26. Fix (URL is from `JiT/README.md`):

```bash
pip install --no-deps --force-reinstall \
  "git+https://github.com/LTH14/torch-fidelity.git"
```

(Installs `0.4.0-beta`.) The fork reads `mu`/`sigma` from `JiT/fid_stats/jit_in256_stats.npz`. Verify:

```python
import inspect, torch_fidelity
from torch_fidelity import metrics
assert "fid_statistics_file" in inspect.getsource(metrics)
```

### Gotcha 4 — DeltaAI's Cray PE / HPC-SDK cuBLAS shadows torch's

This is the most painful one and *cannot* be fixed at install time — it's a runtime env issue.

The DeltaAI login env preloads Cray PE (`cudatoolkit/12.9`) + NVIDIA HPC-SDK 25.5 CUDA libs on `LD_LIBRARY_PATH`. These shadow torch's bundled cuBLAS 12.8 at runtime and throw `CUBLAS_STATUS_INVALID_VALUE` on non-contiguous bf16 batched GEMMs — which is exactly what the dino `nn.Linear` embedder produces during generation (`flatten(2).transpose(1, 2)` makes the input non-contiguous). The crash sometimes hits even on trivial GEMMs. Notably, `module load cuda/13.1.1` (what the original sbatch does) does **not** fix it; it just adds more shadowing libs.

Fix: prepend the venv's bundled NVIDIA libs so torch's cuBLAS wins.

```bash
NVLIBS=$(echo "$REPO"/.venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"
```

`eval_galoiz.sh` bakes this in. **Not a refactor regression** — old and new code both use the same non-contiguous pattern and both crash under shadowed cuBLAS.

When porting to another cluster: 1-3 likely still apply (1/3 specifically for aarch64); 4 is DeltaAI-specific (Cray PE shadowing).

## Verifying the venv

`scripts/setup_env.sh` ends with a smoke check:

```python
import torch, inspect, torch_fidelity
from torch_fidelity import metrics
print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("torch_fidelity:", torch_fidelity.__version__,
      "| fid_statistics_file supported:", "fid_statistics_file" in inspect.getsource(metrics))
```

Expected output on a working install:

```
torch: 2.10.0+cu128 | cuda_available: True | device: NVIDIA GH200 120GB
torch_fidelity: 0.4.0-beta | fid_statistics_file supported: True
```

## Launchers

- **`eval_galoiz.sh`** — interactive launcher. Activates `.venv`, applies the LD fix (gotcha 4), `cd`s into `JiT/`, and runs `torchrun ... main_jit.py` with the baseline args (Heun/50, CFG 2.9, interval [0.1, 1.0], EMA `model_ema1` hard-coded in `engine_jit.py`, gen_bsz 64). Override via env: `NPROC`, `NUM_IMAGES`, `OUTPUT_DIR`, `GEN_BSZ`.
- **`sbatch/jit_eval_galoiz.sbatch`** — single-eval batch wrapper. Defaults: 4 GPUs, 3 h, `NUM_IMAGES=50000`.
- **`sbatch/jit_eval_galoiz_testcd.sbatch`** — sequential Test C (10K) + Test D (50K) under one 2-GPU / 6 h allocation, for this validation session.

Do **not** use the original `sbatch/jit_eval.sbatch` as `galoiz`: it points at msalunkhe's checkout/venv, hardcodes the wrong account, defaults to the 200ep training dir (not the 80ep baseline), and lacks the LD fix.

## Baseline eval config (must match to hit FID 4.03)

| Setting | Value | Source |
|---|---|---|
| Model | `JiT-Dual-B/2-4C-896` | `eval_galoiz.sh` `MODEL` |
| Latent size | 32 (256-px image) | `main_jit.py` default |
| Sampler | Heun, 50 steps | `main_jit.py` defaults |
| CFG scale | 2.9 | `eval_galoiz.sh` `CFG` |
| CFG interval | [0.1, 1.0] | `INTERVAL_MIN`/`INTERVAL_MAX` |
| EMA | `model_ema1` | `engine_jit.py` hard-coded `ema_params1` |
| Precision | bf16 autocast | `engine_jit.py` |
| `P_mean` / `P_std` / noise_scale | -0.8 / 0.8 / 1.0 | `main_jit.py` defaults |
| `dino_time_shift` | 0.0 | default |
| `class_num` | 1000 | default |
| gen_bsz | 64 | `eval_galoiz.sh` |

Leave them at defaults — that is what produced the recorded FID-50K 4.03 / IS-50K 166.14 on 2026-05-05.
