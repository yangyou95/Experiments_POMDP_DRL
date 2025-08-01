#!/bin/sh
#SBATCH --mem=40G
#SBATCH --job-name=GPU_OPTIM_ROCKSAMPLE
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=orchid
#SBATCH --exclude=gpuhost015

# --- Job Array Setup ---
# This will create a job array with 4 tasks, indexed 1, 2, 3, 4.
# Each task will have a unique SLURM_ARRAY_TASK_ID.
#SBATCH --array=1-4

# --- Log File Configuration ---
# Use the master job ID (%A) and the array task ID (%a) for unique log files
# %x is the job name, so logs will look like: logs/optim/GPU_OPTIM_ROCKSAMPLE_12345_1.out
#SBATCH --output=logs/optim/%x_%A_%a.out
#SBATCH --error=logs/optim/%x_%A_%a.err


# --- Environment Setup ---
echo "------------------------------------------------"
echo "Starting Slurm Job Array Task"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "------------------------------------------------"

# Define the list of environments in a shell array
# Note: The array is 0-indexed
ENVS=("rocksample_7_8" "rocksample_11_11" "rocksample_15_15" "rocksample_20_20")

# Map the 1-based Slurm Task ID to the 0-based array index
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Get the environment name for the current task
CURRENT_ENV=${ENVS[$INDEX]}

echo "Selected Environment for this task: $CURRENT_ENV"


# --- Activate Python Environment ---
source /home/users/ucakir/Experiments_POMDP_DRL/.venv/bin/activate


# --- Run the Python Script ---
# Pass the selected environment name to the optimisation script
# The --n_trials argument is also added so you can easily change it here
python /home/users/ucakir/Experiments_POMDP_DRL/optimise.py \
    --env "$CURRENT_ENV" \
    --n_trials 50 --gamma 0.95


echo "------------------------------------------------"
echo "Job Array Task Finished"
echo "------------------------------------------------"