#!/bin/bash
#SBATCH --array=0-3               # one task per RockSample size
#SBATCH --job-name=RS_TRAIN_ALL
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00           # double the original wall-time
#SBATCH --output=logs/optim/%x_%A_%a.out
#SBATCH --error=logs/optim/%x_%A_%a.err
#SBATCH --exclude=gpuhost015      # keep if host is unreliable

# ------------------------------------------------------------
# Environment list and corresponding hyper-parameters
# ------------------------------------------------------------
ENVS=("rocksample_7_8"  "rocksample_11_11"  "rocksample_15_15"  "rocksample_20_20")

# Each entry (same order) is a quoted block of CLI args.
HPARAMS=(
  "--lr 6.923e-04  --clip_eps 0.3148801  --entropy_coeff 0.01787555  --vf_coeff 0.3363666  \
   --lambda1 0.9124030 --max_grad_norm 1.5795778 --num_envs 128 --num_steps 128 \
   --hidden_size 256  --update_epochs 3  --num_minibatches 16"
  "--lr 9.4141e-05 --clip_eps 0.2271471 --entropy_coeff 0.01021509 --vf_coeff 0.6425527 \
   --lambda1 0.9395234 --max_grad_norm 0.5358319 --num_envs 128 --num_steps 256 \
   --hidden_size 256 --update_epochs 6  --num_minibatches 8"
  "--lr 8.9078e-04 --clip_eps 0.31450995 --entropy_coeff 6.8547e-06 --vf_coeff 0.4235862 \
   --lambda1 0.9865203 --max_grad_norm 1.5992933 --num_envs 64  --num_steps 64  \
   --hidden_size 128 --update_epochs 6  --num_minibatches 8"
  "--lr 8.7294e-05 --clip_eps 0.3145981 --entropy_coeff 0.01012199 --vf_coeff 0.4771104 \
   --lambda1 0.9267482 --max_grad_norm 1.8952493 --num_envs 64  --num_steps 256 \
   --hidden_size 512 --update_epochs 8  --num_minibatches 16"
)

# ------------------------------------------------------------
# Activate virtualenv
# ------------------------------------------------------------
source /home/users/ucakir/Experiments_POMDP_DRL/.venv/bin/activate

# ------------------------------------------------------------
# Select environment & parameters for this array index
# ------------------------------------------------------------
ENV=${ENVS[$SLURM_ARRAY_TASK_ID]}
ARGS=${HPARAMS[$SLURM_ARRAY_TASK_ID]}

echo ">>> Training $ENV on GPU ${CUDA_VISIBLE_DEVICES}"
echo ">>> Hyper-parameters: $ARGS"

python -m pobax.algos.ppo \
  --env "$ENV" \
  $ARGS \
  --total_steps 1500000 \
  --n_seeds 5 \
  --num_eval_envs 100000 \
  --default_max_steps_in_episode 1000 \
  --debug --ld_weight 0 --action_concat --lambda0 0.1 \
  --anneal_lr \
  --platform gpu \
  --seed 2025 \
  --show_discounted

