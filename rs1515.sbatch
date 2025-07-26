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
      --rows 15 --cols 15 \
      --layer-size 384 \
      --rnn-hidden-size 384 \
      --sequence-length 96 \
      --max-episode-length 500 \
      --batch-size 16384 \
      --minibatch-size 256 \
      --n-epochs 4 \
      --lr-policy 1.5e-4 \
      --lr-value 7.5e-5 \
      --gamma 0.995 \
      --lambda 0.95 \
      --epsilon 0.18 \
      --ent-coef 0.008 \
      --num-updates 20000 \
      --runname LSTM_RELU \
      --seed ${SLURM_ARRAY_TASK_ID} \
      --wandb-project "RS1515" \