#!/bin/bash                                                                                                                                                                    
#SBATCH --job-name=create_data                                                                                                                                                    
#SBATCH --partition=batch                                                                                                                                                      
#SBATCH --ntasks=1                                                                                                                                                             
#SBATCH --cpus-per-task=1                                                                                                                                                      
#SBATCH --time=10:00:00                                                                                                                                                        
#SBATCH --output=scripts/logs/c_data_%j.out                                                                                                                                    
#SBATCH --error=scripts/logs/c_data_%j.err                                                                                                                                     
#SBATCH --mem=100G                                                                                                                                                                                                                                                                                                      
#SBATCH --gpus=1 

eval "$(conda shell.bash hook)"
conda activate cs336_data

python cs336-data/cs336_data/classifier.py --create_dataset

