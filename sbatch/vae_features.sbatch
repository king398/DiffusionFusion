#!/bin/bash
#SBATCH -J DiT_FID
#SBATCH -A betw-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail


module load python/3.11.9
module load cuda/12.6.1

source /u/msalunkhe/.venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/JiT
IMAGENET_PATH="timm/mini-imagenet"
OUTPUT_DIR="/work/nvme/betw/msalunkhe/data/"
# Avoid thread explosion across DDP ranks + DataLoader workers.
torchrun --nproc_per_node=auto --nnodes=1 --node_rank=0 \
extract_features.py \
--data-path ${IMAGENET_PATH}  \
--features-path ${OUTPUT_DIR} \
--num-workers 2
  IMAGENET_PATH="timm/mini-imagenet"
  OUTPUT_DIR="/work/nvme/betw/msalunkhe/data"

  python prepare_ref.py \
    --use_hf_dataset \
    --data_path "$IMAGENET_PATH" \
    --hf_split train \
    --hf_image_column image \
    --img_size 256 \
    --output_path "${OUTPUT_DIR}/mini-imagenet-train-256"