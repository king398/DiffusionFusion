#!/bin/bash
# Combined Test C (10K) + Test D (50K) eval for the Stage-2 refactor validation,
# sized to the session budget: ghx4 / 2 GPUs / 6h. Runs C first (directional ~30min),
# then D (the 4.03 gate, ~160min on 2 GPUs); set -eo pipefail so a broken C aborts D.
#SBATCH -J jit_eval_galoiz_testcd
#SBATCH -A bgnp-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH -o slurm-galoiz-%j.out
#SBATCH -e slurm-galoiz-%j.err

set -eo pipefail
REPO=/u/galoiz/DiffusionFusion-independent-backup

echo "[jit_eval_galoiz_testcd] host=$(hostname) start=$(date -Iseconds)"

# ---- Test C: 10K FID (directional) ----
echo "[jit_eval_galoiz_testcd] ===== Test C (10K) starting ====="
NPROC=2 \
NUM_IMAGES=10000 \
GEN_BSZ=64 \
OUTPUT_DIR="${REPO}/artifacts/eval_stage2_testc" \
WANDB_MODE=offline \
bash "${REPO}/eval_galoiz.sh"
echo "[jit_eval_galoiz_testcd] ===== Test C (10K) completed at $(date -Iseconds) ====="

# ---- Test D: 50K FID (the 4.03 gate) ----
echo "[jit_eval_galoiz_testcd] ===== Test D (50K) starting ====="
NPROC=2 \
NUM_IMAGES=50000 \
GEN_BSZ=64 \
OUTPUT_DIR="${REPO}/artifacts/eval_stage2_testd" \
WANDB_MODE=offline \
bash "${REPO}/eval_galoiz.sh"
echo "[jit_eval_galoiz_testcd] ===== Test D (50K) completed at $(date -Iseconds) ====="
