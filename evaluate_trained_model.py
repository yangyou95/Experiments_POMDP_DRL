import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import chex
import sys
import os
from pathlib import Path
import time
from functools import partial

# --- Dependencies from the original training script ---

# Add the project root to the Python path to allow imports like 'pobax.config'
# This assumes you run the script from the root of your project, or that 'pobax' is installed.
# sys.path.append(str(Path(__file__).parent.parent))

from pobax.config import PPOHyperparams
from pobax.envs import get_env
from pobax.models import ScannedRNN, get_network_fn
from pobax.algos.ppo import PPO, Transition # Assuming PPO and Transition are in this module

def get_results_path(args: PPOHyperparams, return_npy: bool = True) -> Path:
    """A helper function to create a standardized results path."""
    path_name = f"env={args.env},seed={args.seed},mem={args.memoryless},double={args.double_critic}"
    results_dir = Path("results") / path_name
    results_dir.mkdir(parents=True, exist_ok=True)
    if return_npy:
        return results_dir / "results.npy"
    return results_dir
    
def limit_jax_memory(limit: float = 0.8):
    """Sets the JAX memory fraction to prevent OOM errors."""
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{limit:.2f}'
    print(f"JAX memory fraction limited to {limit:.2f}")


# This function is required for the evaluation loop and was defined in the original script.
# It's included here to make the script self-contained.
def env_step(runner_state, unused, agent: PPO, env, env_params):
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)
    
    # In evaluation, we often want deterministic actions.
    # We can achieve this by using pi.mode() instead of pi.sample().
    # For simplicity here, we reuse the agent's default `act` method which samples.
    # For a purely deterministic evaluation, you would modify agent.act.
    value, action, log_prob, hstate = agent.act(_rng, train_state, hstate, last_obs, last_done)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    transition = Transition(
        last_done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (train_state, env_state, obsv, done, hstate, rng)
    return runner_state, transition

# --- Core Evaluation Functions ---

def load_trained_model(results_path: str):
    """Load the trained model from pobax results"""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results path does not exist: {results_path}")
    
    print(f"Loading trained model from: {results_path}")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = checkpointer.restore(results_path)
    
    final_train_state = data['final_train_state']
    args_dict = data['args']
    
    # --- FIX STARTS HERE ---
    
    # Load the arguments from the saved dictionary using the correct Tap method
    
    # Reconstruct the TrainState from the saved parameters
    # The saved final_train_state is a dictionary, we need to extract params and create a proper TrainState
    print(f"Final train state type: {type(final_train_state)}")
    print(f"Final train state keys: {final_train_state.keys() if isinstance(final_train_state, dict) else 'Not a dict'}")
    
    
    
    print(f"Args dict keys: {args_dict.keys()}")    
    
    # PPO hyperparams expects 
    
    
    
    args = PPOHyperparams().from_dict(args_dict)
    
    # --- FIX ENDS HERE ---
    
    print(f"Loaded model with args: {args.as_dict()}")
    return final_train_state, args

def run_large_evaluation(final_train_state, args: PPOHyperparams, num_eval_envs=100000):
    """Run evaluation with a large number of environments"""
    print(f"\nRunning evaluation with {num_eval_envs} environments...")
    
    # Create environment
    env_key = jax.random.PRNGKey(2025)
    env, env_params = get_env(args.env, env_key,
                              num_envs=num_eval_envs,
                              image_size=args.image_size,
                              gamma=args.gamma,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat)
    
    # Create agent
    network_fn, action_size, is_image, is_discrete = get_network_fn(env, env_params)
    network = network_fn(
        args.env,
        action_size,
        double_critic=args.double_critic,
        hidden_size=args.hidden_size,
        memoryless=args.memoryless,
        is_discrete=is_discrete,
        is_image=is_image
    )
    agent = PPO(network=network) # Other agent params are not needed for eval

    # Reconstruct TrainState if needed (final_train_state might be a dict)
    if isinstance(final_train_state, dict):
        # Extract the first set of parameters (assuming single seed/run)
        ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], final_train_state)
        
        # Create optimizer (we don't need it for evaluation, but TrainState requires it)
        import optax
        tx = optax.adam(args.lr[0] if isinstance(args.lr, (list, jnp.ndarray)) else args.lr)
        
        # Create the TrainState
        from flax.training.train_state import TrainState
        final_train_state = TrainState.create(
            apply_fn=network.apply,
            params=ts_dict['params'],
            tx=tx,
        )
        print(f"Reconstructed TrainState with params shape: {jax.tree.map(lambda x: x.shape, ts_dict['params'])}")

    # JIT the environment step function for performance
    _env_step_jit = jax.jit(partial(env_step, agent=agent, env=env, env_params=env_params))

    # Set up evaluation runner state
    eval_rng = jax.random.PRNGKey(2025)
    reset_rngs = jax.random.split(eval_rng, num_eval_envs)
    eval_obsv, eval_env_state = env.reset(reset_rngs, env_params)
    eval_init_hstate = ScannedRNN.initialize_carry(num_eval_envs, args.hidden_size)
    
    eval_runner_state = (
        final_train_state,
        eval_env_state,
        eval_obsv,
        jnp.zeros((num_eval_envs), dtype=bool),
        eval_init_hstate,
        eval_rng,
    )
    
    # Run evaluation
    print(f"Running evaluation for {env_params.max_steps_in_episode} steps...")
    start_time = time.time()
    
    # Use the JIT-compiled step function
    _, eval_traj_batch = jax.lax.scan(
        _env_step_jit, eval_runner_state, None, env_params.max_steps_in_episode
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Extract results
    final_eval_metrics = eval_traj_batch.info
    returns = final_eval_metrics['returned_discounted_episode_returns']
    mask = final_eval_metrics['returned_episode']
    
    # Get the shape of the arrays to understand the structure
    print(f"Returns shape: {returns.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # We need to extract only the final episode returns for each environment
    # The arrays have shape (max_steps, num_envs)
    # We want to get the final completed episode for each environment
    num_steps, num_envs = returns.shape
    
    # Find the last timestep where each environment completed an episode
    final_episode_mask = np.zeros(num_envs, dtype=bool)
    final_episode_returns = np.zeros(num_envs)
    
    for env_idx in range(num_envs):
        # Find the last timestep where this environment completed an episode
        episode_completed_timesteps = np.where(mask[:, env_idx])[0]
        if len(episode_completed_timesteps) > 0:
            # Get the last completed episode
            last_completed_step = episode_completed_timesteps[-1]
            final_episode_mask[env_idx] = True
            final_episode_returns[env_idx] = returns[last_completed_step, env_idx]
    
    # Filter for only the environments that actually completed at least one episode
    valid_returns = final_episode_returns[final_episode_mask]
    
    if valid_returns.size > 0:
        avg_return = np.mean(valid_returns)
        std_return = np.std(valid_returns)
        num_completed_episodes = len(valid_returns)
        
        print("\n" + "="*60)
        print("LARGE SCALE EVALUATION RESULTS")
        print("="*60)
        print(f"Number of completed episodes: {num_completed_episodes} / {num_eval_envs}")
        print(f"Average Discounted Return:  {avg_return:.4f}")
        print(f"Standard Deviation of Return: {std_return:.4f}")
        print(f"Evaluation time: {eval_time:.2f} seconds")
        print(f"Episodes per second: {num_completed_episodes/eval_time:.1f}")
        print("="*60)
        return {'avg_return': avg_return, 'std_return': std_return, 'num_episodes': num_completed_episodes}
    else:
        print("\nWarning: No completed episodes found during evaluation!")
        return None

def main():
    print("Starting evaluation script...")
    
    # Prevent JAX from gobbling up all the GPU memory, which is crucial for
    # running with a large number of environments. Adjust the fraction as needed.
    limit_jax_memory(0.8)

    if len(sys.argv) != 2:
        print("Usage: python evaluate_trained_model.py <path_to_results_directory>")
        print("Example: python evaluate_trained_model.py results/env=myenv,seed=42,...")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    try:
        final_train_state, args = load_trained_model(results_path)
        print(f"Loaded final train state: {final_train_state}")
        print(f"Loaded args: {args}")
        
        # Run the large-scale evaluation
        run_large_evaluation(final_train_state, args, num_eval_envs=100000)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# This is the crucial missing piece
if __name__ == "__main__":
    main()