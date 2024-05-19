#!/bin/bash                                                                                                                                                                    
#SBATCH --job-name=vscode
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --output=scripts/logs/filter_%j.out
#SBATCH --error=scripts/logs/filter_%j.err
#SBATCH --mem=10G       

eval "$(conda shell.bash hook)"
conda activate cs336_data

python cs336-data/cs336_data/common_crawl.py $1 $2

