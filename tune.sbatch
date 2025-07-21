#!/bin/bash
#SBATCH --job-name=RS-GridSweep            
#SBATCH --account=iecdt
#SBATCH --partition=standard             
#SBATCH --qos=standard                  
#SBATCH --time=24:00:00                
#SBATCH --ntasks=1                    
#SBATCH --output=logs/pomdp/rs_%A_%a.out  
#SBATCH --error=logs/pomdp/rs_%A_%a.err  
#SBATCH --array=1-10
#SBATCH --mem=16G


# Hyperparameter combinations for rows=7 and cols=8
# Define layer and RNN hidden sizes to tune
LAYER_SIZES=(64 64 64 128 128 128 256 256 512 512)

RNN_HIDDEN_SIZES=(64 128 256 64 128 256 64 128 256 512)

# Learning rate and other hyperparameter sweeps
LR_POLICIES=(3e-4 3e-4 1e-4 1e-4 5e-4 5e-4 1e-3 1e-3 2e-4 2e-4)
LR_VALUES=(1e-4 3e-4 1e-4 3e-4 1e-4 3e-4 1e-4 3e-4 1e-4 3e-4)
GAMMAS=(0.99 0.99 0.95 0.95 0.99 0.95 0.99 0.95 0.99 0.95)
EPSILONS=(0.1  0.2  0.1  0.2  0.1  0.2  0.1  0.2  0.1  0.2)
ENT_COEFS=(0.0  0.01 0.01 0.02 0.02 0.005 0.005 0.02 0.01 0.005)

# Additional hyperparameter sweeps
LAMBDA_VALUES=(0.90 0.95 0.99 0.90 0.95 0.99 0.90 0.95 0.99 0.95)
N_EPOCHS_LIST=(3 4 5 3 4 5 3 4 5 4)
BATCH_SIZES=(2048 4096 8192 2048 4096 8192 2048 4096 8192)
MINIBATCH_SIZES=(32 64 128 32 64 128 32 64 128 64)
SEQ_LENGTHS=(16 32 64 16 32 64 16 32 64 32)
MAX_EPISODE_LENGTHS=(100 200 300 100 200 300 100 200 300 200)

# 3) Select hyperparameters for this array task
IDX=$((SLURM_ARRAY_TASK_ID - 1))
ROWS=7
COLS=8
LAYER_SIZE=${LAYER_SIZES[$IDX]}
RNN_HIDDEN_SIZE=${RNN_HIDDEN_SIZES[$IDX]}

LR_POLICY=${LR_POLICIES[$IDX]}
LR_VALUE=${LR_VALUES[$IDX]}
GAMMA=${GAMMAS[$IDX]}
EPSILON=${EPSILONS[$IDX]}
ENT_COEF=${ENT_COEFS[$IDX]}

# Additional selected hyperparameters
LAMBDA=${LAMBDA_VALUES[$IDX]}
N_EPOCHS=${N_EPOCHS_LIST[$IDX]}
BATCH_SIZE=${BATCH_SIZES[$IDX]}
MINIBATCH_SIZE=${MINIBATCH_SIZES[$IDX]}
SEQUENCE_LENGTH=${SEQ_LENGTHS[$IDX]}
MAX_EPISODE_LENGTH=${MAX_EPISODE_LENGTHS[$IDX]}

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: RockSample ${ROWS}×${COLS} | Layer ${LAYER_SIZE} | RNN ${RNN_HIDDEN_SIZE}"

# 4) Run your Julia script with the parsed arguments
julia /home/users/ucakir/Experiments_POMDP_DRL/Julia_DRL_experiments/PPO-RNN-Experiments/ppo-rnn-rs79.jl \
      --rows $ROWS --cols $COLS \
      --layer-size $LAYER_SIZE --rnn-hidden-size $RNN_HIDDEN_SIZE \
      --lr-policy $LR_POLICY --lr-value $LR_VALUE \
      --gamma $GAMMA --epsilon $EPSILON --ent-coef $ENT_COEF \
      --lambda $LAMBDA --n-epochs $N_EPOCHS \
      --batch-size $BATCH_SIZE --minibatch-size $MINIBATCH_SIZE \
      --sequence-length $SEQUENCE_LENGTH --max-episode-length $MAX_EPISODE_LENGTH