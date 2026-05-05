#!/bin/bash
#SBATCH -J JiT_struct_cfg_mask_80ep
#SBATCH -A betw-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-user=mithils3@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail


CUDA_MODULE="${CUDA_MODULE:-cuda/13.1.1}"
module load python/3.11.9
module load "${CUDA_MODULE}"

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/JiT

EXPECTED_GPUS="${EXPECTED_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-${EXPECTED_GPUS}}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV,NET}"
# This is a single-node job. Avoid loading the site OFI net plugin by default;
# the loaded plugin is cuda129, while the current venv may use a cu130 PyTorch.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN_OVERRIDE:-none}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"



IMAGENET_PATH="/work/hdd/bgnp/msalunkhe/data/imagenet/"
OUTPUT_DIR="/work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_200ep"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-jit_dual_struct_cfg_mask_80ep}"
FID_STATS_PATH="${FID_STATS_PATH:-/u/msalunkhe/DiffusionFusion/JiT/fid_stats/jit_in256_stats.npz}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
EVAL_FREQ="${EVAL_FREQ:-40}"
EPOCHS="${EPOCHS:-200}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
DINO_TIME_SHIFT="${DINO_TIME_SHIFT:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_ITER="${ACCUM_ITER:-2}"
mkdir -p "${OUTPUT_DIR}"
OMP_NUM_THREADS=16 torchrun \
--nproc_per_node="${NPROC_PER_NODE}" \
--nnodes=1 \
--node_rank=0 \
main_jit.py \
--model JiT-Dual-B/2-4C-896 \
--proj_dropout 0.0 \
--P_mean -0.8 \
--P_std 0.8 \
--noise_scale 1.0 \
--dino_time_shift "${DINO_TIME_SHIFT}" \
--no_ddp_static_graph \
--ddp_find_unused_parameters \
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
--output_dir ${OUTPUT_DIR} \
--wandb_run_name "${WANDB_RUN_NAME}" \
--data_path ${IMAGENET_PATH} \
--online_eval \
--fid_stats_path "${FID_STATS_PATH}" \
--decoder_checkpoint /work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth \
--decoder_checkpoint_key model_ema \
--class_num 1000 \
--eval_freq "${EVAL_FREQ}"
