source .venv/bin/activate

# Then run the pobax PPO command
python -m pobax.algos.ppo \
  --env rocksample_11_11 \
  --num_envs 8 \
  --total_steps 5000000 \
  --hidden_size 256 \
  --entropy_coeff 0.2 \
  --lr 2.5e-4 \
  --lambda0 0.95 \
  --clip_eps 0.2 \
  --max_grad_norm 0.5 \
  --n_seeds 5 \
  --debug \
  --platform gpu \
  --num_eval_envs 10 \
  --save_runner_state