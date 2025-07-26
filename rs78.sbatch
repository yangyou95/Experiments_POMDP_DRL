#!/bin/bash
#SBATCH --job-name=RS-GridSweep            
#SBATCH --account=iecdt
#SBATCH --partition=standard             
#SBATCH --qos=high                  
#SBATCH --time=48:00:00                
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/pomdp/rs_%A_%a.out  
#SBATCH --error=logs/pomdp/rs_%A_%a.err  
#SBATCH --array=1-5
#SBATCH --mem=64G
# export Julia threads to match SLURM thread allocation
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK


julia /home/users/ucakir/Experiments_POMDP_DRL/Julia_DRL_experiments/PPO-RNN-Experiments/ppo-rnn-rs79.jl \
      --device cpu \
      --rows 7 --cols 8 \
      --layer-size 256 \
      --rnn-hidden-size 128 \
      --lr-policy 1e-3 \
      --lr-value 3e-4 \
      --gamma 0.95 \
      --epsilon 0.2 \
      --ent-coef 0.02 \
      --lambda 0.95 \
      --n-epochs 4 \
      --batch-size 4096 \
      --minibatch-size 64 \
      --sequence-length 32 \
      --max-episode-length 200 \
      --num-updates 10000 \
      --runname LSTM_RELU \
      --seed ${SLURM_ARRAY_TASK_ID} \
      --wandb-project "RS78" \