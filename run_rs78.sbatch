#!/bin/bash
#SBATCH --job-name=RS-78           
#SBATCH --account=orchid
#SBATCH --partition=orchid             
#SBATCH --qos=orchid                  
#SBATCH --time=24:00:00                
#SBATCH --ntasks=1                    

#SBATCH --output=logs/rs78/rs_%A_%a.out  
#SBATCH --error=logs/rs78/rs_%A_%a.err  
#SBATCH --array=1-5
#SBATCH --mem=16G





export SEED=${SLURM_ARRAY_TASK_ID}
julia /home/users/ucakir/Experiments_POMDP_DRL/Julia_DRL_experiments/PPO-RNN-Experiments/rs78.jl $SEED







#SBATCH --cpus-per-task=20