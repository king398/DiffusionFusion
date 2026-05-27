#!/bin/bash
# Reproducible GPU venv setup for DiffusionFusion on NCSA DeltaAI GH200 (aarch64 / Grace-Hopper).
# Verified 2026-05-26 on gh* compute nodes. Encodes 4 gotchas that requirements.txt alone does NOT
# capture; without them GPU eval/training fails (silently CPU, import error, cuBLAS crash, or FID error).
#
#   bash scripts/setup_env.sh
#
# After setup, launch eval with scripts-adjacent eval_galoiz.sh (it applies the runtime LD fix below).
set -eo pipefail
REPO="${REPO:-/u/galoiz/DiffusionFusion-independent-backup}"
cd "$REPO"

module load python/3.11.9

# --- venv ---
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# --- base deps ---
# NOTE: on aarch64 this pulls the CPU-only torch wheel (torch==2.10.0+cpu); fixed in the next step.
.venv/bin/pip install -r requirements.txt

# --- GOTCHA 1: CUDA torch ---
# requirements.txt pins torch==2.10.0 with no index, so plain PyPI serves the aarch64 CPU wheel
# (torch.cuda.is_available() == False). Reinstall the CUDA 12.8 build from the pytorch index.
# The pinned nvidia-*-cu12==12.8.* libs are already the right deps, so --no-deps keeps them.
.venv/bin/pip install --no-deps --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"

# --- GOTCHA 2: editable install ---
# The eval/sbatch `cd`s into JiT/ and runs main_jit.py, which does `from JiT...`. With CWD inside the
# package, `import JiT` fails (ModuleNotFoundError). Editable-install so it resolves from any CWD.
.venv/bin/pip install -e . --no-deps

# --- GOTCHA 3: customized torch-fidelity ---
# engine_jit.py calls calculate_metrics(..., fid_statistics_file=<npz>), supported only by the LTH14
# fork (stock PyPI torch-fidelity==0.3.0 ignores it and demands input2 -> ValueError). URL is from
# JiT/README.md. The fork reads mu/sigma from JiT/fid_stats/jit_in256_stats.npz.
.venv/bin/pip install --no-deps --force-reinstall \
  "git+https://github.com/LTH14/torch-fidelity.git"

# --- verify ---
.venv/bin/python - <<'PY'
import torch, inspect, torch_fidelity
from torch_fidelity import metrics
print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("torch_fidelity:", torch_fidelity.__version__,
      "| fid_statistics_file supported:", "fid_statistics_file" in inspect.getsource(metrics))
PY

cat <<'EOF'

[setup] Done.
RUNTIME GOTCHA 4 (not an install step -- already baked into eval_galoiz.sh):
  The ambient Cray PE (cudatoolkit/12.9) + NVIDIA HPC-SDK CUDA libs on LD_LIBRARY_PATH shadow torch's
  bundled cuBLAS 12.8 and throw CUBLAS_STATUS_INVALID_VALUE on non-contiguous bf16 batched GEMMs
  (the dino nn.Linear embedder during generation). `module load cuda/13.1.1` does NOT fix it.
  Prepend the venv's bundled nvidia libs before any GPU job:
    NVLIBS=$(echo "$REPO"/.venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
    export LD_LIBRARY_PATH="$NVLIBS:$LD_LIBRARY_PATH"
EOF
