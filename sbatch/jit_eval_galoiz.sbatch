#!/bin/bash
# Galoiz-owned eval sbatch: runs the VERIFIED env (own checkout + .venv + cuBLAS LD fix)
# via eval_galoiz.sh, NOT msalunkhe's checkout/venv. Used for the 50K FID gate (Test D),
# which is too slow for a 1-GPU interactive session (~5h); here it gets its own 4-GPU node.
# Override via env, e.g.: NUM_IMAGES=50000 NPROC=4 OUTPUT_DIR=... sbatch sbatch/jit_eval_galoiz.sbatch
#SBATCH -J jit_eval_galoiz
#SBATCH -A bgnp-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH -o slurm-galoiz-%j.out
#SBATCH -e slurm-galoiz-%j.err

set -eo pipefail
REPO=/u/galoiz/DiffusionFusion-independent-backup

export NPROC="${NPROC:-4}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export GEN_BSZ="${GEN_BSZ:-64}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/artifacts/eval_stage2_testd}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# CFG 2.9, interval 0.1-1.0, Heun/50, ema1, decoder + FID stats all come from eval_galoiz.sh
# defaults + main_jit defaults -> matches the 4.03 baseline.

echo "[jit_eval_galoiz] host=$(hostname) NPROC=${NPROC} NUM_IMAGES=${NUM_IMAGES} OUTPUT_DIR=${OUTPUT_DIR}"
exec bash "${REPO}/eval_galoiz.sh"
