#!/bin/bash
#SBATCH --job-name=RS-78           
#SBATCH --account=iecdt
#SBATCH --partition=standard             
#SBATCH --qos=high                  
#SBATCH --time=24:00:00                
#SBATCH --ntasks=1   
#SBATCH --cpus-per-task=40                

#SBATCH --output=logs/rs15/rs_%A_%a.out  
#SBATCH --error=logs/rs15/rs_%A_%a.err  
#SBATCH --array=1-5
#SBATCH --mem=24G





export SEED=${SLURM_ARRAY_TASK_ID}
julia -t 40 /home/users/ucakir/Experiments_POMDP_DRL/Julia_DRL_experiments/PPO-RNN-Experiments/rs15.jl $SEED







