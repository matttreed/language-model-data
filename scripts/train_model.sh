#!/bin/bash                                                                                                                                                                    
#SBATCH --job-name=vscode
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=scripts/logs/train_%j.out
#SBATCH --error=scripts/logs/train_%j.err
#SBATCH --mem=100G       

export WANDB_API_KEY="a881776c91971a761af061cba1423c1d6e38ab60"

eval "$(conda shell.bash hook)"
conda activate cs336_data

torchrun --standalone --nproc_per_node=2 /home/c-mattreed/language-model-data/cs336-basics/scripts/train.py \
--train-path /home/c-mattreed/language-model-data/filtered_data/data_tokenized.bin \
--dev-path /home/shared/paloma_c4_100_domains_val_tokenized.bin \
--output-dir /home/c-mattreed/language-model-data/models \
--vocab-size 50257 \
--context-length 512 \
--d-model 768 \
--num-layers 12 \
--num-heads 12 \
--d-ff 3072 \
--attn-pdrop 0.1 \
--residual-pdrop 0.1 \
--batch-size 128 \
--train-steps 200000 \
--eval-iters 1000 \
--eval-interval 2000 \
--learning-rate 1e-3 \
--lr-scheduler cosine \
--weight-decay 0.1 \
--warmup-ratio 0.01 \
--grad-clip 1.0 \
--dtype bfloat16 \
--wandb-project cs336-data \
	 --compile \
	 --device cuda
