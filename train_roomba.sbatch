#!/bin/bash

#SBATCH --job-name=roomba_train
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00           # double the original wall-time
#SBATCH --output=logs/roomba/%x_%A_%a.out
#SBATCH --error=logs/roomba/%x_%A_%a.err
#SBATCH --exclude=gpuhost015,gpuhost016     # keep if host is unreliable



# ------------------------------------------------------------
# Activate virtualenv
# ------------------------------------------------------------
source /home/users/ucakir/Experiments_POMDP_DRL/.venv/bin/activate

# ------------------------------------------------------------
# Select environment & parameters for this array index
# ------------------------------------------------------------
# 'lr': 0.00030883592922447117, 'clip_eps': 0.3106620952103814, 'entropy_coeff': 0.0563727004233995, 'vf_coeff': 0.4785348792117911, 'lambda1': 0.961625393680678, 'lambda0': 0.12767733335001702, 'max_grad_norm': 1.919566572991225, 'num_envs': 64, 'num_steps': 256, 'hidden_size': 256, 'update_epochs': 6, 'num_minibatches': 8




python -m pobax.algos.ppo \
  --env "roomba" \
  --total_steps 15000000 \
  --n_seeds 10 \
  --num_eval_envs 100000 \
  --default_max_steps_in_episode 1000 \
  --debug --ld_weight 0 --action_concat \
  --anneal_lr \
  --platform gpu \
  --seed 2025 \
  --show_discounted \
  --lr 0.00030883592922447117 \
  --clip_eps 0.3106620952103814 \
  --entropy_coeff 0.0563727004233995 \
  --vf_coeff 0.4785348792117911 \
  --lambda1 0.961625393680678 \
  --lambda0 0.12767733335001702 \
  --max_grad_norm 1.919566572991225 \
  --num_envs 64 \
  --num_steps 256 \
  --hidden_size 256 \
  --update_epochs 6 \
  --num_minibatches 8

# python -m pobax.algos.ppo \
#   --env "roomba" \
#   \
#   --total_steps 1500000 \
#   --n_seeds 5 \
#   --num_eval_envs 100000 \
#   --default_max_steps_in_episode 1000 \
#   --debug --ld_weight 0 --action_concat --lambda0 0.1 \
#   --anneal_lr \
#   --platform gpu \
#   --seed 2025 \
#   --show_discounted

