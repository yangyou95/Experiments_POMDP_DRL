#!/bin/bash
#SBATCH --job-name=RS-GridSweep            
#SBATCH --account=iecdt
#SBATCH --partition=standard             
#SBATCH --qos=standard                  
#SBATCH --time=02:00:00                
#SBATCH --ntasks=1                    
#SBATCH --output=logs/pomdp/rs_%A_%a.out  
#SBATCH --error=logs/pomdp/rs_%A_%a.err  
#SBATCH --array=1-4
#SBATCH --mem=16G


# 2) Define your (rows,cols) pairs here – adjust or extend as needed:
PARAMS=(
  "7 8"
  "11 11"
  "15 15"
  "20 20"
)

# 3) Select the right pair for this array task
IDX=$((SLURM_ARRAY_TASK_ID - 1))
read ROWS COLS <<< "${PARAMS[$IDX]}"

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: RockSample ${ROWS}×${COLS}"

# 4) Run your Julia script with the parsed arguments
julia /home/users/ucakir/Experiments_POMDP_DRL/Julia_DRL_experiments/PPO-RNN-Experiments/ppo-rnn-rs79.jl --rows $ROWS --cols $COLS