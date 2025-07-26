#!/bin/bash
#SBATCH --job-name=RS-GridSweep            
#SBATCH --account=iecdt
#SBATCH --partition=standard             
#SBATCH --qos=high                  
#SBATCH --time=24:00:00                
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/pomdp/rs_%A_%a.out  
#SBATCH --error=logs/pomdp/rs_%A_%a.err  
#SBATCH --array=1-4
#SBATCH --mem=64G
# export Julia threads to match SLURM thread allocation
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# # 2) Define your (rows,cols) pairs here – adjust or extend as needed:
# PARAMS=(
#   "7 8"
#   "11 11"
#   "15 15"
#   "20 20"
# )
# # Define layer and RNN hidden sizes for each grid
# LAYER_SIZES=(64 64 128 256)
# RNN_HIDDEN_SIZES=(64 64 128 256)

# # 3) Select the right pair for this array task
# IDX=$((SLURM_ARRAY_TASK_ID - 1))
# read ROWS COLS <<< "${PARAMS[$IDX]}"
# LAYER_SIZE=${LAYER_SIZES[$IDX]}
# RNN_HIDDEN_SIZE=${RNN_HIDDEN_SIZES[$IDX]}

# echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: RockSample ${ROWS}×${COLS}"

# # 4) Run your Julia script with the parsed arguments
# julia /home/users/ucakir/Experiments_POMDP_DRL/Julia_DRL_experiments/PPO-RNN-Experiments/ppo-rnn-rs79.jl \
#       --rows $ROWS --cols $COLS \
#       --layer-size $LAYER_SIZE --rnn-hidden-size $RNN_HIDDEN_SIZE



PARAMS=(
  "7 8 128 2.5e-4 0.2"
  "11 11 256 2.5e-4 0.2"
  "15 15 512 2.5e-4 0.2"
  "20 20 1024 2.5e-4 0.2"
)

# Select the right parameter set for this array task
IDX=$((SLURM_ARRAY_TASK_ID - 1))
read ROWS COLS RNN_HIDDEN_SIZE LR ENT_COEF <<< "${PARAMS[$IDX]}"

# The paper trains for 5e6 steps. Your script uses `num_updates`.
# Let's calculate num_updates based on batch_size.
# num_updates = total_steps / batch_size
# We will set a high batch_size for stability, e.g., 4096
BATCH_SIZE=4096
TOTAL_STEPS=5000000
NUM_UPDATES=$((TOTAL_STEPS / BATCH_SIZE))

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID: Reproducing POBAX RockSample ${ROWS}x${COLS}"
echo "Threads: $JULIA_NUM_THREADS"
echo "Parameters: RNN Size=$RNN_HIDDEN_SIZE, LR=$LR, Ent_Coef=$ENT_COEF, Updates=$NUM_UPDATES"

# Run your Julia script with the parsed arguments
# Note: I've added arguments for lr and ent-coef
# In your ArgParse, you should add --ent-coef and use the same value for both lr-policy and lr-value
julia /home/users/ucakir/Experiments_POMDP_DRL/Julia_DRL_experiments/PPO-RNN-Experiments/ppo-rnn-rs79.jl \
      --device cpu \
      --rows $ROWS --cols $COLS \
      --rnn-hidden-size $RNN_HIDDEN_SIZE \
      --hidden-dim $RNN_HIDDEN_SIZE \
      --batch-size $BATCH_SIZE \
      --num-updates $NUM_UPDATES \
      --lr-policy $LR \
      --lr-value $LR \
      --ent-coef $ENT_COEF \
      --lambda 0.95 \
      --epsilon 0.2 \
      --n-epochs 4 # This is a common PPO default, paper doesn't specify it in the tables