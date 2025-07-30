#!/usr/bin/env python3
"""
Enhanced Pobax PPO Training Script with Evaluation
Runs the pobax PPO command and captures output, similar to the Julia rs78.jl implementation
"""

import subprocess
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


def run_pobax_ppo_with_evaluation():
    """Run pobax PPO training with evaluation and capture output"""
    
    # Configuration matching your original command
    config = {
        'env': 'rocksample_11_11',
        'num_envs': 8,
        'total_steps': 100000,  # Reduced for testing, change to 5000000 for full training
        'hidden_size': 256,
        'entropy_coeff': 0.2,
        'lr': 2.5e-4,
        'lambda0': 0.95,
        'clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'n_seeds': 5,
        'debug': True,
        'platform': 'gpu',
        'num_eval_envs': 5,  # Number of environments for evaluation
        'save_runner_state': True  # Save the trained agent for later evaluation
    }
    
    # Build the command
    cmd = [
        sys.executable, '-m', 'pobax.algos.ppo',
        '--env', config['env'],
        '--num_envs', str(config['num_envs']),
        '--total_steps', str(config['total_steps']),
        '--hidden_size', str(config['hidden_size']),
        '--entropy_coeff', str(config['entropy_coeff']),
        '--lr', str(config['lr']),
        '--lambda0', str(config['lambda0']),
        '--clip_eps', str(config['clip_eps']),
        '--max_grad_norm', str(config['max_grad_norm']),
        '--n_seeds', str(config['n_seeds']),
        '--platform', config['platform'],
        '--num_eval_envs', str(config['num_eval_envs']),
        '--save_runner_state'  # Save the trained agent
    ]
    
    if config['debug']:
        cmd.append('--debug')
    
    print("=== Running Pobax PPO Training with Evaluation ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment: {config['env']}")
    print(f"Total steps: {config['total_steps']}")
    print(f"Number of environments: {config['num_envs']}")
    print(f"Evaluation environments: {config['num_eval_envs']}")
    print(f"Platform: {config['platform']}")
    
    # Create results directory
    results_dir = Path("results/pobax_ppo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the command and capture output
    start_time = time.time()
    
    try:
        # Run the pobax command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from the main directory
        )
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        # Save the output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save stdout
        stdout_path = results_dir / f"pobax_ppo_stdout_{timestamp}.txt"
        with open(stdout_path, 'w') as f:
            f.write(result.stdout)
        
        # Save stderr
        stderr_path = results_dir / f"pobax_ppo_stderr_{timestamp}.txt"
        with open(stderr_path, 'w') as f:
            f.write(result.stderr)
        
        # Save configuration
        config_path = results_dir / f"pobax_ppo_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Check if training was successful
        if result.returncode == 0:
            print("\n=== Training completed successfully! ===")
            
            # Extract metrics and evaluation results
            metrics = extract_enhanced_metrics(result.stdout, results_dir, timestamp)
            
            # Save summary
            summary_path = results_dir / f"pobax_ppo_summary_{timestamp}.txt"
            with open(summary_path, 'w') as f:
                f.write("=== Pobax PPO Training Summary ===\n\n")
                f.write(f"Training time: {training_time:.2f} seconds\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                
                f.write("Configuration:\n")
                for key, value in config.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nCommand:\n")
                f.write(f"  {' '.join(cmd)}\n")
                
                f.write("\nTraining completed successfully!\n")
                
                # Add evaluation summary
                if metrics:
                    f.write("\nEvaluation Results:\n")
                    if 'final_eval_return' in metrics:
                        f.write(f"  Final evaluation return: {metrics['final_eval_return']:.4f}\n")
                    if 'final_eval_std' in metrics:
                        f.write(f"  Final evaluation std: {metrics['final_eval_std']:.4f}\n")
                    if 'max_training_return' in metrics:
                        f.write(f"  Max training return: {metrics['max_training_return']:.4f}\n")
                    if 'num_episodes' in metrics:
                        f.write(f"  Number of episodes: {metrics['num_episodes']}\n")
            
            print(f"\nResults saved to:")
            print(f"  Stdout: {stdout_path}")
            print(f"  Stderr: {stderr_path}")
            print(f"  Config: {config_path}")
            print(f"  Summary: {summary_path}")
            
            # Print evaluation results
            if metrics:
                print("\n=== Evaluation Results ===")
                if 'final_eval_return' in metrics:
                    print(f"Final evaluation return: {metrics['final_eval_return']:.4f}")
                if 'final_eval_std' in metrics:
                    print(f"Final evaluation std: {metrics['final_eval_std']:.4f}")
                if 'max_training_return' in metrics:
                    print(f"Max training return: {metrics['max_training_return']:.4f}")
                if 'num_episodes' in metrics:
                    print(f"Number of episodes: {metrics['num_episodes']}")
            
            return True
            
        else:
            print(f"\nTraining failed with return code: {result.returncode}")
            
            # Save error summary
            summary_path = results_dir / f"pobax_ppo_error_{timestamp}.txt"
            with open(summary_path, 'w') as f:
                f.write("=== Pobax PPO Training Error ===\n\n")
                f.write(f"Training time: {training_time:.2f} seconds\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                
                f.write("Configuration:\n")
                for key, value in config.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nCommand:\n")
                f.write(f"  {' '.join(cmd)}\n")
                
                f.write(f"\nTraining failed with return code: {result.returncode}\n")
                
                if result.stderr:
                    f.write("\nError output:\n")
                    f.write(result.stderr)
            
            print(f"Error summary saved to: {summary_path}")
            return False
            
    except Exception as e:
        print(f"Error running pobax command: {e}")
        return False


def extract_enhanced_metrics(stdout: str, results_dir: Path, timestamp: str):
    """Extract enhanced metrics from stdout output including evaluation results"""
    
    metrics = {
        'timesteps': [],
        'returns': [],
        'episodes': [],
        'eval_returns': [],
        'eval_lengths': []
    }
    
    # Parse the output for timesteps and returns
    lines = stdout.split('\n')
    
    for line in lines:
        # Parse training progress
        if 'timesteps=' in line and 'avg episodic return=' in line:
            try:
                # Extract timestep and return values
                parts = line.split(',')
                timestep_part = parts[0]
                return_part = parts[1]
                
                # Extract timestep
                timestep_str = timestep_part.split('=')[1].strip()
                if '-' in timestep_str:
                    timestep = int(timestep_str.split('-')[0])
                else:
                    timestep = int(timestep_str)
                
                # Extract return
                return_str = return_part.split('=')[1].strip()
                return_val = float(return_str)
                
                metrics['timesteps'].append(timestep)
                metrics['returns'].append(return_val)
                metrics['episodes'].append(len(metrics['returns']))
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse training line: {line}")
                continue
        
        # Parse evaluation results (look for final evaluation output)
        elif 'final_eval' in line.lower() or 'evaluation' in line.lower():
            try:
                # Look for evaluation return values
                if 'return' in line.lower():
                    # Extract return value from evaluation line
                    import re
                    return_match = re.search(r'return[:\s]*([\d.-]+)', line, re.IGNORECASE)
                    if return_match:
                        eval_return = float(return_match.group(1))
                        metrics['eval_returns'].append(eval_return)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse evaluation line: {line}")
                continue
    
    # Calculate summary statistics
    if metrics['returns']:
        metrics['max_training_return'] = max(metrics['returns'])
        metrics['min_training_return'] = min(metrics['returns'])
        metrics['mean_training_return'] = np.mean(metrics['returns'])
        metrics['std_training_return'] = np.std(metrics['returns'])
        metrics['num_episodes'] = len(metrics['returns'])
    
    if metrics['eval_returns']:
        metrics['final_eval_return'] = np.mean(metrics['eval_returns'])
        metrics['final_eval_std'] = np.std(metrics['eval_returns'])
        metrics['final_eval_min'] = min(metrics['eval_returns'])
        metrics['final_eval_max'] = max(metrics['eval_returns'])
    
    # Save metrics
    if metrics['returns'] or metrics['eval_returns']:
        metrics_path = results_dir / f"pobax_ppo_metrics_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                metrics_serializable[key] = [float(x) if isinstance(x, (np.floating, float)) else int(x) if isinstance(x, (np.integer, int)) else x for x in value]
            elif isinstance(value, (np.floating, float)):
                metrics_serializable[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                metrics_serializable[key] = int(value)
            else:
                metrics_serializable[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"  Metrics: {metrics_path}")
        print(f"  Extracted {len(metrics['returns'])} training data points")
        if metrics['eval_returns']:
            print(f"  Extracted {len(metrics['eval_returns'])} evaluation data points")
        
        # Create enhanced plots
        try:
            create_enhanced_plots(metrics, results_dir, timestamp)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    return metrics


def create_enhanced_plots(metrics: dict, results_dir: Path, timestamp: str):
    """Create enhanced plots including evaluation results"""
    
    if not metrics['returns'] and not metrics['eval_returns']:
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training Returns
    if metrics['returns']:
        plt.subplot(2, 3, 1)
        plt.plot(metrics['episodes'], metrics['returns'])
        plt.title('Training Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Training Returns vs Timesteps
    if metrics['timesteps'] and metrics['returns']:
        plt.subplot(2, 3, 2)
        plt.plot(metrics['timesteps'], metrics['returns'])
        plt.title('Training Returns vs Timesteps')
        plt.xlabel('Timestep')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Moving Average Returns
    if len(metrics['returns']) > 10:
        window = min(10, len(metrics['returns']) // 4)
        moving_avg = np.convolve(metrics['returns'], np.ones(window)/window, mode='valid')
        episodes_avg = metrics['episodes'][window-1:]
        
        plt.subplot(2, 3, 3)
        plt.plot(episodes_avg, moving_avg)
        plt.title(f'Moving Average Returns (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Training Returns Distribution
    if metrics['returns']:
        plt.subplot(2, 3, 4)
        plt.hist(metrics['returns'], bins=20, alpha=0.7)
        plt.title('Training Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Evaluation Returns (if available)
    if metrics['eval_returns']:
        plt.subplot(2, 3, 5)
        plt.hist(metrics['eval_returns'], bins=20, alpha=0.7, color='orange')
        plt.title('Evaluation Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add evaluation statistics
        eval_mean = np.mean(metrics['eval_returns'])
        eval_std = np.std(metrics['eval_returns'])
        plt.axvline(eval_mean, color='red', linestyle='--', label=f'Mean: {eval_mean:.2f}')
        plt.legend()
    
    # Plot 6: Training vs Evaluation Comparison
    if metrics['returns'] and metrics['eval_returns']:
        plt.subplot(2, 3, 6)
        
        # Create box plot comparison
        data = [metrics['returns'], metrics['eval_returns']]
        labels = ['Training', 'Evaluation']
        colors = ['lightblue', 'lightcoral']
        
        bp = plt.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Training vs Evaluation Returns')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_dir / f"pobax_ppo_enhanced_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Enhanced Plot: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    success = run_pobax_ppo_with_evaluation()
    
    if success:
        print("\n=== Training and Evaluation completed successfully! ===")
        print("This is similar to the Julia rs78.jl evaluation approach.")
        print("The pobax package automatically runs evaluation at the end of training.")
    else:
        print("\n=== Training failed! ===")
        sys.exit(1) 