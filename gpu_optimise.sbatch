#!/bin/sh
#SBATCH --mem=40G
#SBATCH --job-name=GPU_OPTIM_LIGHTDARK
#SBATCH --output=logs/optim/%x_%j.out
#SBATCH --error=logs/optim/%x_%j.err
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --qos=orchid
#SBATCH --exclude=gpuhost015







source /home/users/ucakir/Experiments_POMDP_DRL/.venv/bin/activate


python /home/users/ucakir/Experiments_POMDP_DRL/optimise.py




