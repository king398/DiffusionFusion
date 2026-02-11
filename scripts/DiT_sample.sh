module load python/3.11.9
module load cuda/12.6.1

source ./venv/bin/activate
cd /u/msalunkhe/DiffusionFusion/DiT
torchrun --nnodes=1 --nproc_per_node=auto sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000  --sample-dir /projects/betw/msalunkhe/DiT_generations
