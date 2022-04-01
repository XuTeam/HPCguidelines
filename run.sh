#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --job-name=DLCodeDemo 
#SBATCH --output=DLCodeDemo.out 
#SBATCH --error=DLCodeDemo.err 
#SBATCH --time=8:00:00 
#SBATCH --partition=batch 
#SBATCH --gres=gpu:a100:1

source ~/.bashrc 
conda activate anconda3
python run.py