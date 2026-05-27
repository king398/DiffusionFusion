#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Joint-attention SMOKE test — joint 2-stream (latent + DINO), short budget.
# galoiz-owned defaults (checkout + venv-bundled CUDA + cuBLAS LD_LIBRARY_PATH fix).
#
# Purpose: de-risk the V-Co joint path (`fusion=joint`) on REAL multi-GPU before
# spending an 80-epoch budget. The joint graph (JointStreamBlock, the [L,L]
# additive CFG bias, per-stream qkv/norm/mlp) has only ever run on CPU in tests.
# This run answers three questions and nothing else:
#   1. Does the joint graph TRAIN under ddp_static_graph=True + torch.compile?
#      (the whole reason the CFG mask is an additive bias, never a module-skip)
#   2. Does the loss descend / not NaN over a handful of epochs?
#   3. Does the joint SAMPLING path (generate()->dict -> decode latent+dino ->
#      FID) run end-to-end without error? (an eval fires at epoch 0 -> fail fast)
# It is NOT a quality run — ignore the FID number; it's on barely-trained weights.
#
# If green, promote to the full control: same config, EPOCHS=80, realistic
# NUM_IMAGES, separate output dir. That control vs the 4.03 pairwise baseline is
# the clean fusion-mechanism comparison (Experiment 1 in docs/experiment_plan.md).
#
# Submit:  sbatch sbatch/jit_joint2_smoke.sbatch
# Override checkout/venv at submit time if needed:
#   sbatch --export=ALL,REPO=/u/galoiz/DiffusionFusion sbatch/jit_joint2_smoke.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J jit_joint2_smoke
#SBATCH -A bgnp-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o slurm-galoiz-%j.out
#SBATCH -e slurm-galoiz-%j.err

set -eo pipefail

REPO="${REPO:-/u/galoiz/DiffusionFusion-independent-backup}"
module load python/3.11.9
source "${REPO}/.venv/bin/activate"
cd "${REPO}/JiT"

# Force torch's bundled CUDA 12.8 libs to win over the ambient Cray/HPC-SDK CUDA
# libs on LD_LIBRARY_PATH. Without this, a shadowing system cuBLAS throws
# CUBLAS_STATUS_INVALID_VALUE on non-contiguous bf16 batched GEMMs (dino embedder).
NVLIBS=$(echo "${REPO}"/.venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Single-node job. Avoid the site OFI net plugin (cuda129) clashing with the venv's cu128 torch.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN_OVERRIDE:-none}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# torch.compile needs a real C++ compiler (the Cray 'CC' wrapper breaks inductor).
if [[ -z "${CXX:-}" || "${CXX}" == "CC" ]]; then
  if command -v g++ >/dev/null 2>&1; then export CXX="$(command -v g++)";
  elif command -v c++ >/dev/null 2>&1; then export CXX="$(command -v c++)"; fi
fi
if [[ -n "${CXX:-}" ]]; then export TORCHINDUCTOR_CXX="${TORCHINDUCTOR_CXX:-${CXX}}"; fi


IMAGENET_PATH="${IMAGENET_PATH:-/work/hdd/bgnp/msalunkhe/data/imagenet/}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/outputs/jit_joint2_smoke}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-jit_joint2_smoke}"
FID_STATS_PATH="${FID_STATS_PATH:-${REPO}/JiT/fid_stats/jit_in256_stats.npz}"
# streams_config lives at repo root; main_jit.py runs from ${REPO}/JiT.
STREAMS_CONFIG="${STREAMS_CONFIG:-../configs/streams/latent_dino_joint.yaml}"
DECODER_CKPT="${DECODER_CKPT:-/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth}"

# Smoke knobs: short, with an early eval so a broken joint-sampling path fails fast.
EPOCHS="${EPOCHS:-10}"
EVAL_FREQ="${EVAL_FREQ:-5}"          # fires at epoch 0 and 5 -> exercises generate->decode early
NUM_IMAGES="${NUM_IMAGES:-2000}"     # tiny: FID is meaningless here, we only check it runs
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_ITER="${ACCUM_ITER:-2}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
mkdir -p "${OUTPUT_DIR}"

COMPILE_ARGS=()
if [[ "${JIT_COMPILE:-1}" == "1" || "${JIT_COMPILE:-1}" == "true" ]]; then
  COMPILE_ARGS+=(--compile_model)    # default ON — this is the risky config we want to validate
else
  COMPILE_ARGS+=(--no_compile_model) # JIT_COMPILE=0 to isolate a compile failure from a graph failure
fi

echo "[joint2_smoke] REPO=${REPO} NPROC=${NPROC_PER_NODE} EPOCHS=${EPOCHS} compile=${JIT_COMPILE:-1}"
echo "[joint2_smoke] streams=${STREAMS_CONFIG} fusion=joint OUTPUT_DIR=${OUTPUT_DIR}"

OMP_NUM_THREADS=16 torchrun \
--nproc_per_node="${NPROC_PER_NODE}" \
--nnodes=1 \
--node_rank=0 \
main_jit.py \
--model JiT-Dual-B/2-4C-896 \
--streams_config "${STREAMS_CONFIG}" \
--fusion joint \
--proj_dropout 0.0 \
--P_mean -0.8 \
--P_std 0.8 \
--noise_scale 1.0 \
--batch_size "${BATCH_SIZE}" \
--accum_iter "${ACCUM_ITER}" \
--blr 5e-5 \
--epochs "${EPOCHS}" \
--warmup_epochs 5 \
--lr_schedule "${LR_SCHEDULE}" \
--gen_bsz 128 \
--num_images "${NUM_IMAGES}" \
--cfg 2.9 \
--interval_min 0.1 \
--interval_max 1.0 \
--output_dir "${OUTPUT_DIR}" \
--wandb_run_name "${WANDB_RUN_NAME}" \
--data_path "${IMAGENET_PATH}" \
--online_eval \
--fid_stats_path "${FID_STATS_PATH}" \
--decoder_checkpoint "${DECODER_CKPT}" \
--decoder_checkpoint_key model_ema \
--class_num 1000 \
--eval_freq "${EVAL_FREQ}" \
"${COMPILE_ARGS[@]}"
