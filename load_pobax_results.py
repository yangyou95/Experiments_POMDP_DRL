#!/usr/bin/env python3
"""
Script to load and analyze PObax results from OCDBT format.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add pobax to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'pobax'))

try:
    import orbax.checkpoint
    orbax_available = True
except ImportError:
    print("Warning: Could not import orbax.checkpoint. You may need to install orbax.")
    orbax_available = False

def load_pobax_experiment(experiment_path):
    """
    Load a PObax experiment from the given path.
    
    Args:
        experiment_path: Path to the experiment directory
        
    Returns:
        Dictionary containing the loaded data
    """
    experiment_path = Path(experiment_path)
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")
    
    print(f"Loading experiment from: {experiment_path}")
    
    # Try to load using orbax if available
    if orbax_available:
        try:
            print("Attempting to load with orbax.checkpoint...")
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            data = orbax_checkpointer.restore(experiment_path)
            return data
        except Exception as e:
            print(f"Failed to load with orbax: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback: try to read metadata and understand structure
    print("Falling back to manual loading...")
    
    # Read metadata
    metadata_file = experiment_path / "_METADATA"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print("Found metadata file")
    else:
        print("No metadata file found")
        metadata = None
    
    # Look for other files
    files = list(experiment_path.rglob("*"))
    print(f"Found {len(files)} files in experiment directory")
    
    # Try to find any JSON files that might contain results
    json_files = [f for f in files if f.suffix == '.json']
    print(f"Found {len(json_files)} JSON files")
    
    # Try to read the strings.json file which might contain some data
    strings_file = experiment_path / "_strings.json"
    if strings_file.exists():
        with open(strings_file, 'r') as f:
            strings_data = json.load(f)
        print("Found strings data")
    else:
        strings_data = None
    
    return {
        'metadata': metadata,
        'strings_data': strings_data,
        'files': [str(f) for f in files],
        'experiment_path': str(experiment_path)
    }

def analyze_experiment(data):
    """
    Analyze the loaded experiment data.
    
    Args:
        data: Dictionary containing experiment data
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    if data.get('metadata'):
        metadata = data['metadata']
        tree_metadata = metadata.get('tree_metadata', {})
        
        # Extract key information
        analysis['available_keys'] = list(tree_metadata.keys())
        
        # Look for specific metrics
        metric_keys = [k for k in tree_metadata.keys() if 'metric' in k.lower()]
        analysis['metric_keys'] = metric_keys
        
        # Look for final evaluation metrics
        final_eval_keys = [k for k in tree_metadata.keys() if 'final_eval_metric' in k]
        analysis['final_eval_keys'] = final_eval_keys
        
        # Look for arguments
        arg_keys = [k for k in tree_metadata.keys() if k[0] == 'args']
        analysis['arg_keys'] = arg_keys
    
    return analysis

def print_experiment_summary(data, analysis):
    """
    Print a summary of the experiment.
    
    Args:
        data: Loaded experiment data
        analysis: Analysis results
    """
    print("\n" + "="*60)
    print("PObax Experiment Summary")
    print("="*60)
    
    print(f"Experiment path: {data.get('experiment_path', 'Unknown')}")
    
    if analysis.get('available_keys'):
        print(f"\nAvailable data keys: {len(analysis['available_keys'])}")
        for key in analysis['available_keys'][:10]:  # Show first 10
            print(f"  - {key}")
        if len(analysis['available_keys']) > 10:
            print(f"  ... and {len(analysis['available_keys']) - 10} more")
    
    if analysis.get('metric_keys'):
        print(f"\nMetric keys found: {len(analysis['metric_keys'])}")
        for key in analysis['metric_keys']:
            print(f"  - {key}")
    
    if analysis.get('final_eval_keys'):
        print(f"\nFinal evaluation keys found: {len(analysis['final_eval_keys'])}")
        for key in analysis['final_eval_keys']:
            print(f"  - {key}")
    
    if analysis.get('arg_keys'):
        print(f"\nArgument keys found: {len(analysis['arg_keys'])}")
        for key in analysis['arg_keys']:
            print(f"  - {key}")

def main():
    """Main function to load and analyze PObax results."""
    
    # Default path to PObax results
    default_path = "pobax/results/batch_ppo_test"
    
    # Get experiment path from command line or use default
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        # List available experiments
        base_path = Path(default_path)
        if base_path.exists():
            experiments = [d for d in base_path.iterdir() if d.is_dir()]
            if experiments:
                print("Available experiments:")
                for i, exp in enumerate(experiments):
                    print(f"  {i+1}. {exp.name}")
                
                # Use the first one as default
                experiment_path = str(experiments[0])
                print(f"\nUsing first experiment: {experiment_path}")
            else:
                print(f"No experiments found in {default_path}")
                return
        else:
            print(f"Default path {default_path} does not exist")
            return
    
    try:
        # Load the experiment
        data = load_pobax_experiment(experiment_path)
        
        # Analyze the data
        analysis = analyze_experiment(data)
        
        # Print summary
        print_experiment_summary(data, analysis)
        
        # If we have orbax available, try to extract actual metrics
        if orbax_available and 'out' in data:
            print("\n" + "="*60)
            print("Extracted Metrics")
            print("="*60)
            
            out_data = data['out']
            
            # Try to get training metrics
            if 'metric' in out_data:
                metrics = out_data['metric']
                print("Training metrics available:")
                for key in metrics.keys():
                    if isinstance(metrics[key], np.ndarray):
                        print(f"  {key}: shape {metrics[key].shape}, dtype {metrics[key].dtype}")
                        if len(metrics[key]) > 0:
                            print(f"    - Range: [{metrics[key].min()}, {metrics[key].max()}]")
                            print(f"    - Mean: {metrics[key].mean()}")
            
            # Try to get final evaluation metrics
            if 'final_eval_metric' in out_data:
                final_metrics = out_data['final_eval_metric']
                print("\nFinal evaluation metrics available:")
                for key in final_metrics.keys():
                    if isinstance(final_metrics[key], np.ndarray):
                        print(f"  {key}: shape {final_metrics[key].shape}, dtype {final_metrics[key].dtype}")
                        if len(final_metrics[key]) > 0:
                            # Handle multi-dimensional arrays properly
                            if final_metrics[key].size == 1:
                                print(f"    - Value: {final_metrics[key].item()}")
                            else:
                                print(f"    - Mean: {final_metrics[key].mean()}")
                                print(f"    - Shape: {final_metrics[key].shape}")
                                if final_metrics[key].ndim > 1:
                                    # For multi-dimensional arrays, show some sample values
                                    flat_array = final_metrics[key].flatten()
                                    if len(flat_array) > 0:
                                        print(f"    - Sample values: [{flat_array[0]}, ..., {flat_array[-1]}]")
                                else:
                                    # For 1D arrays, show first few values
                                    if len(final_metrics[key]) <= 5:
                                        print(f"    - Values: {final_metrics[key]}")
                                    else:
                                        print(f"    - First 5 values: {final_metrics[key][:5]}")
            
            # Print detailed metrics analysis
            print_detailed_metrics(data)
            
            # Plot training and evaluation metrics
            plot_training_metrics(data)
            plot_evaluation_rewards(data)
                            
                                    
    except Exception as e:
        print(f"Error loading experiment: {e}")
        import traceback
        traceback.print_exc()

def plot_training_metrics(data):
    """
    Plot training metrics like rewards over time.
    
    Args:
        data: Loaded experiment data
    """
    if 'out' not in data:
        print("No 'out' data found for plotting training metrics")
        return
    
    out_data = data['out']
    
    # Check for training metrics
    if 'metric' not in out_data:
        print("No training metrics found")
        return
    
    metrics = out_data['metric']
    
    # Look for reward-related metrics
    reward_keys = [k for k in metrics.keys() if 'reward' in k.lower() or 'return' in k.lower()]
    
    if not reward_keys:
        print("No reward metrics found in training data")
        print(f"Available metric keys: {list(metrics.keys())}")
        return
    
    # Create subplots for different metrics
    n_metrics = len(reward_keys)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    for i, key in enumerate(reward_keys):
        metric_data = metrics[key]
        if isinstance(metric_data, np.ndarray) and metric_data.size > 0:
            # Handle multi-dimensional arrays by flattening or taking appropriate slices
            if metric_data.ndim > 1:
                # For high-dimensional arrays, take the mean over certain dimensions
                # or reshape to get a meaningful training curve
                if metric_data.ndim == 10:  # (1, 1, 1, 1, 1, 1, 3, steps, envs, ?)
                    # Take mean over environments and other dimensions
                    plot_data = metric_data.mean(axis=tuple(range(metric_data.ndim-2)))
                    if plot_data.ndim == 2:  # (steps, remaining_dim)
                        plot_data = plot_data.mean(axis=-1)  # Average over last dimension
                else:
                    # For other cases, try to find a meaningful 1D representation
                    plot_data = metric_data.flatten()
                    if len(plot_data) > 1000:  # If too many points, subsample
                        indices = np.linspace(0, len(plot_data)-1, 1000, dtype=int)
                        plot_data = plot_data[indices]
            else:
                plot_data = metric_data
            
            if len(plot_data) > 0:
                axes[i].plot(plot_data)
                axes[i].set_title(f'Training {key}')
                axes[i].set_xlabel('Training Step')
                axes[i].set_ylabel(key)
                axes[i].grid(True)
                
                # Add some statistics to the plot
                if len(plot_data) > 1:
                    axes[i].axhline(y=plot_data.mean(), color='r', linestyle='--', alpha=0.7, label=f'Mean: {plot_data.mean()}')
                    axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f"training_metrics_{Path(data.get('experiment_path', 'unknown')).name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training metrics plot saved for {len(reward_keys)} metrics")

def plot_evaluation_rewards(data):
    """
    Plot evaluation rewards.
    
    Args:
        data: Loaded experiment data
    """
    if 'out' not in data:
        print("No 'out' data found for plotting evaluation rewards")
        return
    
    out_data = data['out']
    
    # Check for evaluation metrics
    eval_sources = []
    
    # Check final evaluation metrics
    if 'final_eval_metric' in out_data:
        eval_sources.append(('Final Evaluation', out_data['final_eval_metric']))
    
    # Check for periodic evaluation metrics
    eval_keys = [k for k in out_data.keys() if 'eval' in k.lower() and k != 'final_eval_metric']
    for key in eval_keys:
        eval_sources.append((key, out_data[key]))
    
    if not eval_sources:
        print("No evaluation metrics found")
        return
    
    # Plot evaluation metrics
    n_sources = len(eval_sources)
    fig, axes = plt.subplots(n_sources, 1, figsize=(12, 4*n_sources))
    if n_sources == 1:
        axes = [axes]
    
    for i, (source_name, eval_data) in enumerate(eval_sources):
        # Look for reward-related keys in evaluation data
        if isinstance(eval_data, dict):
            reward_keys = [k for k in eval_data.keys() if 'reward' in k.lower() or 'return' in k.lower()]
            
            if reward_keys:
                for j, key in enumerate(reward_keys):
                    metric_data = eval_data[key]
                    if isinstance(metric_data, np.ndarray) and metric_data.size > 0:
                        if metric_data.size == 1:  # scalar
                            # For scalar values, create a bar plot
                            axes[i].bar([key], [metric_data.item()])
                            axes[i].set_title(f'{source_name} - {key}')
                            axes[i].set_ylabel('Reward')
                        elif metric_data.ndim == 1:  # 1D array
                            axes[i].plot(metric_data, label=key)
                            axes[i].set_title(f'{source_name} - {key}')
                            axes[i].set_xlabel('Episode/Step')
                            axes[i].set_ylabel('Reward')
                            axes[i].legend()
                        else:  # Multi-dimensional array
                            # Flatten or take appropriate slice for plotting
                            if metric_data.ndim > 2:
                                # Take mean over extra dimensions except the last two
                                while metric_data.ndim > 2:
                                    metric_data = metric_data.mean(axis=0)
                            
                            if metric_data.ndim == 2:
                                # Plot mean and std over one dimension
                                plot_mean = metric_data.mean(axis=-1)
                                plot_std = metric_data.std(axis=-1)
                                x_vals = range(len(plot_mean))
                                axes[i].plot(x_vals, plot_mean, label=f'{key} (mean)')
                                axes[i].fill_between(x_vals, plot_mean - plot_std, plot_mean + plot_std, alpha=0.3)
                                axes[i].set_title(f'{source_name} - {key}')
                                axes[i].set_xlabel('Episode/Step')
                                axes[i].set_ylabel('Reward')
                                axes[i].legend()
                            else:
                                # Fallback: plot flattened data
                                flat_data = metric_data.flatten()
                                if len(flat_data) > 1000:  # Subsample if too many points
                                    indices = np.linspace(0, len(flat_data)-1, 1000, dtype=int)
                                    flat_data = flat_data[indices]
                                axes[i].plot(flat_data, label=key)
                                axes[i].set_title(f'{source_name} - {key}')
                                axes[i].set_xlabel('Data Point')
                                axes[i].set_ylabel('Reward')
                                axes[i].legend()
                        axes[i].grid(True)
            else:
                # If no reward keys, show all available metrics
                all_keys = list(eval_data.keys())[:5]  # Show first 5 metrics
                values = []
                labels = []
                for key in all_keys:
                    metric_data = eval_data[key]
                    if isinstance(metric_data, np.ndarray) and len(metric_data.shape) == 0:
                        values.append(metric_data.item())
                        labels.append(key)
                
                if values:
                    axes[i].bar(labels, values)
                    axes[i].set_title(f'{source_name} - All Metrics')
                    axes[i].set_ylabel('Value')
                    axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"evaluation_rewards_{Path(data.get('experiment_path', 'unknown')).name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Evaluation rewards plot saved for {len(eval_sources)} evaluation sources")

def print_detailed_metrics(data):
    """
    Print detailed metrics from the experiment.
    
    Args:
        data: Loaded experiment data
    """
    if 'out' not in data:
        print("No 'out' data found")
        return
    
    out_data = data['out']
    
    print("\n" + "="*60)
    print("DETAILED METRICS ANALYSIS")
    print("="*60)
    
    # Training metrics
    if 'metric' in out_data:
        metrics = out_data['metric']
        print("\n--- TRAINING METRICS ---")
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                if len(value) > 0:
                    if 'reward' in key.lower() or 'return' in key.lower():
                        print(f"\n{key}:")
                        print(f"  Shape: {value.shape}")
                        print(f"  Final value: {value[-1]}")
                        print(f"  Mean: {value.mean()}")
                        print(f"  Std: {value.std()}")
                        print(f"  Min: {value.min()}")
                        print(f"  Max: {value.max()}")
                        
                        # Print last 10 values
                        if len(value) >= 10:
                            print(f"  Last 10 values: {value[-10:]}")
                        else:
                            print(f"  All values: {value}")
    
    # Evaluation metrics
    if 'final_eval_metric' in out_data:
        final_metrics = out_data['final_eval_metric']
        #only get ended episode
        
        print("\n--- FINAL EVALUATION METRICS ---")
        for key, value in final_metrics.items():
            # only get the returned episode ['returned_episode']
            value = value['returned_episode']
            if isinstance(value, np.ndarray):
                if value.size == 1:  # scalar
                    print(f"{key}: {value.item()}")
                else:
                    print(f"{key}: shape {value.shape}, mean {value.mean()}, std {value.std()}")
                    # For reward-related metrics, show more details
                    if 'reward' in key.lower() or 'return' in key.lower():
                        print(f"  Min: {value.min()}, Max: {value.max()}")
                        if value.ndim == 1 and len(value) <= 10:
                            print(f"  Values: {value}")
                        elif value.ndim > 1:
                            # Show some sample episodes if it's episode-based data
                            flat_vals = value.flatten()
                            if len(flat_vals) > 0:
                                print(f"  Sample: [{flat_vals[0]}, ..., {flat_vals[-1]}]")
    
    # Look for other evaluation metrics
    eval_keys = [k for k in out_data.keys() if 'eval' in k.lower() and k != 'final_eval_metric']
    if eval_keys:
        print("\n--- OTHER EVALUATION METRICS ---")
        for eval_key in eval_keys:
            print(f"\n{eval_key}:")
            eval_data = out_data[eval_key]
            if isinstance(eval_data, dict):
                for key, value in eval_data.items():
                    if isinstance(value, np.ndarray):
                        if len(value.shape) == 0:  # scalar
                            print(f"  {key}: {value.item()}")
                        else:
                            print(f"  {key}: shape {value.shape}, mean {value.mean()}")

def compare_experiments(experiment_paths):
    """
    Compare multiple experiments and plot their performance.
    
    Args:
        experiment_paths: List of paths to experiment directories
    """
    experiments_data = []
    
    for path in experiment_paths:
        try:
            data = load_pobax_experiment(path)
            experiments_data.append((Path(path).name, data))
        except Exception as e:
            print(f"Failed to load experiment {path}: {e}")
    
    if len(experiments_data) < 2:
        print("Need at least 2 experiments to compare")
        return
    
    # Compare final evaluation rewards
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Final evaluation rewards comparison
    exp_names = []
    final_rewards = []
    
    for exp_name, data in experiments_data:
        if 'out' in data and 'final_eval_metric' in data['out']:
            final_metrics = data['out']['final_eval_metric']
            # Look for reward metrics
            reward_keys = [k for k in final_metrics.keys() if 'reward' in k.lower() or 'return' in k.lower()]
            if reward_keys:
                # Use the first reward metric found
                reward_value = final_metrics[reward_keys[0]]
                if isinstance(reward_value, np.ndarray):
                    exp_names.append(exp_name[:20] + '...' if len(exp_name) > 20 else exp_name)
                    final_rewards.append(reward_value.item() if len(reward_value.shape) == 0 else reward_value.mean())
    
    if final_rewards:
        ax1.bar(range(len(exp_names)), final_rewards)
        ax1.set_title('Final Evaluation Rewards Comparison')
        ax1.set_ylabel('Reward')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    
    # Plot 2: Training curves comparison
    for exp_name, data in experiments_data:
        if 'out' in data and 'metric' in data['out']:
            metrics = data['out']['metric']
            reward_keys = [k for k in metrics.keys() if 'reward' in k.lower() or 'return' in k.lower()]
            if reward_keys:
                # Use the first reward metric found
                reward_data = metrics[reward_keys[0]]
                if isinstance(reward_data, np.ndarray) and len(reward_data) > 0:
                    label = exp_name[:15] + '...' if len(exp_name) > 15 else exp_name
                    ax2.plot(reward_data, label=label, alpha=0.7)
    
    ax2.set_title('Training Curves Comparison')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved for {len(experiments_data)} experiments")

def load_and_analyze_specific_experiment():
    """
    Interactive function to select and analyze a specific experiment.
    """
    # List available experiments
    base_path = Path("pobax/results/batch_ppo_test")
    if not base_path.exists():
        print(f"Base path {base_path} does not exist")
        return
    
    experiments = [d for d in base_path.iterdir() if d.is_dir()]
    if not experiments:
        print(f"No experiments found in {base_path}")
        return
    
    print("Available experiments:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp.name}")
    
    try:
        choice = int(input(f"\nSelect experiment (1-{len(experiments)}): ")) - 1
        if 0 <= choice < len(experiments):
            experiment_path = str(experiments[choice])
            print(f"\nLoading experiment: {experiment_path}")
            
            # Load and analyze the selected experiment
            data = load_pobax_experiment(experiment_path)
            analysis = analyze_experiment(data)
            print_experiment_summary(data, analysis)
            
            if orbax_available and 'out' in data:
                print_detailed_metrics(data)
                plot_training_metrics(data)
                plot_evaluation_rewards(data)
                
        else:
            print("Invalid selection")
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")

def load_multiple_experiments_for_comparison():
    """
    Load multiple experiments and compare them.
    """
    base_path = Path("pobax/results/batch_ppo_test")
    if not base_path.exists():
        print(f"Base path {base_path} does not exist")
        return
    
    experiments = [d for d in base_path.iterdir() if d.is_dir()]
    if len(experiments) < 2:
        print(f"Need at least 2 experiments for comparison, found {len(experiments)}")
        return
    
    print("Available experiments:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp.name}")
    
    try:
        selections = input(f"\nSelect experiments to compare (e.g., 1,3,5): ").split(',')
        selected_paths = []
        
        for sel in selections:
            choice = int(sel.strip()) - 1
            if 0 <= choice < len(experiments):
                selected_paths.append(str(experiments[choice]))
            else:
                print(f"Invalid selection: {sel}")
                return
        
        if len(selected_paths) >= 2:
            compare_experiments(selected_paths)
        else:
            print("Need at least 2 valid experiments for comparison")
            
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            load_and_analyze_specific_experiment()
        elif sys.argv[1] == "--compare":
            load_multiple_experiments_for_comparison()
        else:
            # Use the provided path
            main()
    else:
        # Default behavior
        main() 