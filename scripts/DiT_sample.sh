module load python/3.11.9
module load cuda/12.6.1

source ./venv/bin/activate

torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000