#!/usr/bin/env python3
"""
A compact script to load, analyze, and plot PObax experiment results.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
import orbax.checkpoint

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from pobax.algos.ppo import PPO, PPOHyperparams, env_step
from pobax.utils.file_system import get_results_path, numpyify

from pobax.models import ScannedRNN, get_network_fn
from pobax.envs import get_env




def load_for_evaluation(results_path: str, num_eval_envs: int):
    """
    Loads a saved experiment for large-scale evaluation.

    This function is adapted from `load_train_state` but is specifically
    designed to initialize the environment with a potentially different number
    of parallel environments (`num_eval_envs`).

    Args:
        results_path: Path to the saved experiment directory.
        num_eval_envs: The number of parallel environments to use for evaluation.

    Returns:
        A tuple containing the env, env_params, args, network, and the loaded TrainState.
    """
    # 1. Load the saved data using Orbax
    print(f"Loading experiment from: {results_path}")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(results_path)
    
    # 2. Restore arguments and the TrainState
    # The 'args' are saved as a dict, not a PPOHyperparams object.
    args = PPOHyperparams().from_dict(restored['args'])
    unpacked_ts = restored['final_train_state'] # Use the final, saved train state
    print(f"ARGS: {args}")
    # 3. Re-create the environment, but with the new number of environments
    # We use a dummy key here since it's just for environment setup.
    env_key = jax.random.PRNGKey(0) 
    env, env_params = get_env(args.env, env_key,
                              num_envs=num_eval_envs,  # <-- CRITICAL CHANGE HERE
                              image_size=args.image_size,
                              gamma=args.gamma,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat)

    # 4. Re-create the network architecture
    network_fn, action_size, _, _ = get_network_fn(env, env_params)
    network = network_fn(args.env,
                         action_size,
                         double_critic=args.double_critic,
                         hidden_size=args.hidden_size,
                         memoryless=args.memoryless)
    
    # 5. Create the final, usable TrainState for a single agent
    # The saved parameters might have extra dimensions from vmapping over seeds/hyperparams.
    # The lambda function strips these dimensions away to get the core agent parameters.
    # Note: The exact number of indices [0] may depend on your vmap setup.
    # This assumes the first agent of the first run is the one to evaluate.



    print(f"Unpacked TrainState Parameters:")
    print(f"Total parameters: {len(unpacked_ts['params'])}")
    print(unpacked_ts['params'].keys())
    
    print(unpacked_ts['params'])
    
    print("Shapes of parameters:")
    for key, value in unpacked_ts["params"].items():
       print(f"{key}: {value.shape if isinstance(value, jnp.ndarray) else type(value)}")

    try:
        # Strip away all the vmapped dimensions.
        params = jax.tree_util.tree_map(lambda x: x[0][0][0][0][0][0][0], unpacked_ts['params'])
    except IndexError:
        # Fallback if the shape is not as expected
        print("Warning: Failed to strip 5 dimensions. Trying to strip just one.")
        try:
            params = jax.tree_util.tree_map(lambda x: x[0], unpacked_ts['params'])
        except IndexError:
            params = unpacked_ts['params']

    # The optimizer state doesn't matter for evaluation, so we can create a dummy one.
    tx = optax.adam(1e-4) 
    
    ts = TrainState.create(apply_fn=network.apply,
                           params=params,
                           tx=tx)

    return env, env_params, args, network, ts



def rerun_large_scale_evaluation(
    results_path: str,
    num_eval_envs: int = 100000,
    num_runs: int = 5,
    seed: int = 42,
):
    """
    Loads a saved training state and reruns evaluation on a large number of parallel environments.

    Args:
        results_path: Path to the saved experiment directory.
        num_eval_envs: The number of parallel environments for evaluation.
        num_runs: The number of times to repeat the evaluation with different seeds.
        seed: The base random seed.
    """
    # 1. Load all necessary components using the new helper function
    env, env_params, args, network, train_state = load_for_evaluation(results_path, num_eval_envs)

    # 2. Re-create the PPO agent object
    agent = PPO(network,
                double_critic=args.double_critic,
                ld_weight=args.ld_weight,
                alpha=args.alpha,
                vf_coeff=args.vf_coeff,
                entropy_coeff=args.entropy_coeff,
                clip_eps=args.clip_eps)

    # 3. Define and JIT-compile the evaluation function
    _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)

    @jax.jit
    def run_evaluation_scan(ts, rng):
        """Initializes environments and runs a full evaluation scan."""
        reset_rng = jax.random.split(rng, num_eval_envs)
        obsv, env_state = env.reset(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(num_eval_envs, args.hidden_size)
        
        eval_runner_state = (
            ts, # Use the loaded train_state
            env_state,
            obsv,
            jnp.zeros((num_eval_envs), dtype=bool),
            init_hstate,
            rng,
        )

        _, eval_traj_batch = jax.lax.scan(
            _env_step, eval_runner_state, None, env_params.max_steps_in_episode
        )
        return eval_traj_batch.info

    # 4. Run the evaluation multiple times with different seeds
    rng = jax.random.PRNGKey(seed)
    all_means = []
    all_stds = []

    print("\n" + "=" * 50)
    print(f"Starting Large-Scale Re-evaluation for {args.env}")
    print(f"Running {num_runs} times with {num_eval_envs} parallel environments each.")
    print("=" * 50)

    for i in range(num_runs):
        print(f"  > Starting run {i + 1}/{num_runs}...")
        rng, eval_rng = jax.random.split(rng)
        
        eval_info = run_evaluation_scan(train_state, eval_rng)
        
        # Use your numpyify helper for consistency
        eval_info = numpyify(eval_info)

        returns = eval_info['returned_discounted_episode_returns']
        mask = eval_info['returned_episode']
        valid_returns = returns[mask]
        
        if valid_returns.size > 0:
            run_mean = np.mean(valid_returns)
            run_std = np.std(valid_returns)
            all_means.append(run_mean)
            all_stds.append(run_std)
            print(f"    - Mean Return: {run_mean:.3f}, Std Dev: {run_std:.3f}")
        else:
            print("    - No completed episodes in this run.")

    # 5. Aggregate and print the final results
    if all_means:
        final_avg_mean = np.mean(all_means)
        final_avg_std = np.mean(all_stds)
        
        print("\n" + "="*50)
        print("Final Aggregated Results")
        print("="*50)
        print(f"Average of Mean Discounted Returns over {num_runs} runs: {final_avg_mean:.4f}")
        print(f"Average of Std Dev of Discounted Returns over {num_runs} runs: {final_avg_std:.4f}")
    else:
        print("\nNo metrics were collected across all runs.")

    print("\nLarge-scale evaluation finished.")









def analyze_and_plot(results_path: str):
    """
    Loads data, plots rewards and losses, and prints final evaluation metrics.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"Error: Path does not exist: {results_path}")
        return

    # 1. Load data using Orbax
    print(f"Loading experiment from: {results_path}")
    try:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        data = checkpointer.restore(results_path)
    except Exception as e:
        print(f"Failed to load data with Orbax: {e}")
        return

    args = data['args']
    out = data['out']
    
    # 2. Calculate and print final evaluation metrics
    print("\n" + "="*50)
    print("Final Evaluation Metrics")
    print("="*50)
    final_metrics = out.get('final_eval_metric', {})
    if final_metrics:
        # Use discounted returns instead of undiscounted returns
        returns = final_metrics.get('returned_discounted_episode_returns')
        mask = final_metrics.get('returned_episode', np.ones_like(returns, dtype=bool))
        
        valid_returns = returns[mask]
        if valid_returns.size > 0:
            avg_return = np.mean(valid_returns)
            std_return = np.std(valid_returns)
            print(f"Average Final Discounted Return: {avg_return:.2f} +/- {std_return:.2f}")
        else:
            print("No completed episodes found in final evaluation.")
    else:
        print("No final evaluation metrics found.")

    # 3. Plot training rewards and losses
    print("\nGenerating plots...")
    training_metrics = out.get('metric')
    if not training_metrics:
        print("No training metrics ('metric') found in the saved data. Cannot create plots.")
        return
        
    # Define metrics to plot
    plot_specs = {
        "Episodic Discounted Return": "returned_discounted_episode_returns",
        "Value Loss": "value_loss",
        "Actor Loss": "loss_actor",
        "Entropy": "entropy",
    }
    
    # Determine the number of updates for the x-axis
    num_seeds = training_metrics['returned_episode'].shape[0]
    num_updates = training_metrics['returned_episode'].shape[1]
    timesteps_per_log = args['num_steps'] * args['num_envs'] * args.get('update_log_freq', 1)
    timesteps = np.arange(num_updates) * timesteps_per_log

    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(10, 5 * len(plot_specs)), sharex=True)
    fig.suptitle(f"Training Analysis for: {args['env']}", fontsize=16)

    for i, (title, key) in enumerate(plot_specs.items()):
        ax = axes[i]
        if key not in training_metrics:
            ax.text(0.5, 0.5, f"Metric '{key}' not found in data", ha='center', va='center')
            ax.set_title(title)
            continue
            
        metric_data = np.array(training_metrics[key])

        # For returns, we only want to plot completed episodes
        if key == "returned_discounted_episode_returns":
            mask = training_metrics['returned_episode']
            # Calculate mean and std only on valid returns per timestep
            mean_values = [metric_data[s, t][mask[s, t]].mean() if mask[s, t].any() else np.nan 
                           for s in range(num_seeds) for t in range(num_updates)]
            mean_values = np.nanmean(np.array(mean_values).reshape(num_seeds, num_updates), axis=0)
            std_values = [metric_data[s, t][mask[s, t]].std() if mask[s, t].any() else np.nan 
                          for s in range(num_seeds) for t in range(num_updates)]
            std_values = np.nanmean(np.array(std_values).reshape(num_seeds, num_updates), axis=0)
        else:
            # For losses, average across seeds
            mean_values = np.mean(metric_data, axis=tuple(range(metric_data.ndim - 1)))
            std_values = np.std(metric_data, axis=tuple(range(metric_data.ndim - 1)))

        ax.plot(timesteps, mean_values, label=f"Mean {title}")
        ax.fill_between(timesteps, mean_values - std_values, mean_values + std_values, alpha=0.2, label="Std Dev")
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Timesteps")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plot_path = results_path.parent / f"{results_path.name}_analysis.png"
    plt.savefig(plot_path)
    print(f"Plots saved to: {plot_path}")
    plt.show()

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_experiment_directory>")
        # Example default path for convenience
        default_path = "results/brax-ant/ppo-20240516-103000" # Change to a valid path
        if Path(default_path).exists():
             print(f"Running with default path: {default_path}")
             analyze_and_plot(default_path)
             
       
        sys.exit(1)
        
    experiment_path = sys.argv[1]
    analyze_and_plot(experiment_path)
    
    
    print("running eval by self")
    rerun_large_scale_evaluation(
            results_path=experiment_path,
            num_eval_envs=100000,
            num_runs=5,
            seed=42
        )