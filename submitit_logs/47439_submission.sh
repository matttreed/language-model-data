#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=/home/c-mattreed/language-model-data/submitit_logs/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --mem=10GB
#SBATCH --nodes=2
#SBATCH --open-mode=append
#SBATCH --output=/home/c-mattreed/language-model-data/submitit_logs/%j_0_log.out
#SBATCH --partition=batch
#SBATCH --signal=USR2@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/c-mattreed/language-model-data/submitit_logs/%j_%t_log.out --error /home/c-mattreed/language-model-data/submitit_logs/%j_%t_log.err /home/c-mattreed/miniconda3/envs/cs336_data/bin/python -u -m submitit.core._submit /home/c-mattreed/language-model-data/submitit_logs
