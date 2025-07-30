#!/bin/sh
#SBATCH --mem=40G
#SBATCH --job-name=GPU_OPTIM_LIGHTDARK
#SBATCH --output=logs/optim/%x_%j.out
#SBATCH --error=logs/optim/%x_%j.err
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --qos=orchid
#SBATCH --exclude=gpuhost015







source /home/users/ucakir/Experiments_POMDP_DRL/.venv/bin/activate


python -m pobax.algos.ppo \
  --env lightdark \
  --lr 8.9928e-4 \
  --clip_eps 0.1024 \
  --entropy_coeff 0.01258 \
  --vf_coeff 0.4793 \
  --lambda1 0.98705 \
  --max_grad_norm 0.5627 \
  --num_envs 64 \
  --num_steps 64 \
  --hidden_size 512 \
  --update_epochs 7 \
  --num_minibatches 16 \
  --total_steps 1200000 \
  --debug \
  --ld_weight 0 \
  --action_concat \
  --lambda0 0.1 \
  --anneal_lr \
  --default_max_steps_in_episode 100 \
  --platform gpu \
  --seed 2025 \
  --gamma 0.9 \
  --n_seeds 5 \
  --num_eval_envs 100000 \
  --show_discounted



