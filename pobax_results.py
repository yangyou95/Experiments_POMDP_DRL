#!/usr/bin/env python3
"""
A compact script to load, analyze, and plot PObax experiment results.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import orbax.checkpoint

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