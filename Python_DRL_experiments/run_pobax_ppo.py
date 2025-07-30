#!/usr/bin/env python3
"""
Pobax PPO Training Script for RockSample Environment
Similar to the Julia rs78.jl implementation but using pobax package
"""

import os
import sys
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

# Add pobax to path
pobax_path = Path(__file__).parent.parent / "pobax"
sys.path.insert(0, str(pobax_path))

from pobax.config import PPOHyperparams
from pobax.algos.ppo import main as ppo_main
from pobax.utils.file_system import get_results_path


class PobaxPPOTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pobax PPO Trainer
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.results_dir = Path("results/pobax_ppo")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'timesteps': [],
            'episode_returns': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'learning_rates': [],
            'eval_returns': [],
            'eval_lengths': []
        }
        
        # Set up JAX platform
        if config.get('platform', 'gpu') == 'gpu':
            jax.config.update('jax_platform_name', 'gpu')
        else:
            jax.config.update('jax_platform_name', 'cpu')
    
    def create_hyperparams(self) -> PPOHyperparams:
        """Create PPOHyperparams object from config"""
        args = PPOHyperparams()
        
        # Environment settings
        args.env = self.config['env']
        args.num_envs = self.config['num_envs']
        args.total_steps = self.config['total_steps']
        
        # Network settings
        args.hidden_size = self.config['hidden_size']
        args.memoryless = self.config.get('memoryless', False)
        args.action_concat = self.config.get('action_concat', True)
        
        # PPO hyperparameters
        args.entropy_coeff = self.config['entropy_coeff']
        args.lr = [self.config['lr']]  # Pobax expects list for sweeping
        args.lambda0 = [self.config['lambda0']]
        args.clip_eps = self.config['clip_eps']
        args.max_grad_norm = self.config['max_grad_norm']
        
        # Training settings
        args.num_steps = self.config.get('num_steps', 128)
        args.num_epochs = self.config.get('num_epochs', 50)
        args.update_epochs = self.config.get('update_epochs', 4)
        args.num_minibatches = self.config.get('num_minibatches', 4)
        
        # Loss coefficients
        args.vf_coeff = [self.config.get('vf_coeff', 0.5)]
        args.lambda1 = [self.config.get('lambda1', 0.5)]
        args.alpha = [self.config.get('alpha', 1.0)]
        args.ld_weight = [self.config.get('ld_weight', 0.0)]
        
        # Evaluation settings
        args.num_eval_envs = self.config.get('num_eval_envs', 10)
        args.steps_log_freq = self.config.get('steps_log_freq', 1)
        args.update_log_freq = self.config.get('update_log_freq', 1)
        
        # Other settings
        args.seed = self.config.get('seed', 2020)
        args.n_seeds = self.config.get('n_seeds', 5)
        args.platform = self.config.get('platform', 'gpu')
        args.debug = self.config.get('debug', False)
        args.show_discounted = self.config.get('show_discounted', False)
        args.anneal_lr = self.config.get('anneal_lr', True)
        
        # Save settings
        args.save_checkpoints = self.config.get('save_checkpoints', False)
        args.save_runner_state = self.config.get('save_runner_state', False)
        args.study_name = self.config.get('study_name', 'pobax_ppo_rocksample')
        
        # Process arguments
        args.process_args()
        
        # Parse arguments to fix the Tap issue
        args.parse_args([])
        
        return args
    
    def extract_metrics(self, output: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract metrics from pobax output
        
        Args:
            output: Output from pobax PPO training
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Extract metric data
        metric_data = output['metric']
        
        # Extract timesteps and returns
        if 'timestep' in metric_data:
            timesteps = np.array(metric_data['timestep'])
            metrics['timesteps'] = timesteps.flatten().tolist()
        
        if 'returned_episode_returns' in metric_data:
            episode_returns = np.array(metric_data['returned_episode_returns'])
            metrics['episode_returns'] = episode_returns.flatten().tolist()
        
        if 'returned_episode_lengths' in metric_data:
            episode_lengths = np.array(metric_data['returned_episode_lengths'])
            metrics['episode_lengths'] = episode_lengths.flatten().tolist()
        
        # Extract final evaluation metrics
        if 'final_eval_metric' in output:
            eval_metric = output['final_eval_metric']
            
            if 'returned_episode_returns' in eval_metric:
                eval_returns = np.array(eval_metric['returned_episode_returns'])
                metrics['eval_returns'] = eval_returns.flatten().tolist()
            
            if 'returned_episode_lengths' in eval_metric:
                eval_lengths = np.array(eval_metric['returned_episode_lengths'])
                metrics['eval_lengths'] = eval_lengths.flatten().tolist()
        
        # Note: Pobax doesn't expose individual loss components in the same way
        # as the Julia implementation, so we'll need to modify the pobax code
        # to extract these if needed
        
        return metrics
    
    def extract_metrics_from_saved(self, loaded_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract metrics from saved pobax results
        
        Args:
            loaded_results: Results loaded from disk using orbax
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Extract metric data from the 'out' field
        if 'out' in loaded_results:
            output = loaded_results['out']
            
            # Extract timesteps and returns from metric data
            if 'metric' in output:
                metric_data = output['metric']
                
                # Extract timesteps and returns
                if 'timestep' in metric_data:
                    timesteps = np.array(metric_data['timestep'])
                    metrics['timesteps'] = timesteps.flatten().tolist()
                
                if 'returned_episode_returns' in metric_data:
                    episode_returns = np.array(metric_data['returned_episode_returns'])
                    metrics['episode_returns'] = episode_returns.flatten().tolist()
                
                if 'returned_episode_lengths' in metric_data:
                    episode_lengths = np.array(metric_data['returned_episode_lengths'])
                    metrics['episode_lengths'] = episode_lengths.flatten().tolist()
            
            # Extract final evaluation metrics
            if 'final_eval_metric' in output:
                eval_metric = output['final_eval_metric']
                
                if 'returned_episode_returns' in eval_metric:
                    eval_returns = np.array(eval_metric['returned_episode_returns'])
                    metrics['eval_returns'] = eval_returns.flatten().tolist()
                
                if 'returned_episode_lengths' in eval_metric:
                    eval_lengths = np.array(eval_metric['returned_episode_lengths'])
                    metrics['eval_lengths'] = eval_lengths.flatten().tolist()
        
        # Add training runtime if available
        if 'total_runtime' in loaded_results:
            metrics['total_runtime'] = [float(loaded_results['total_runtime'])]
        
        return metrics
    
    def extract_metrics_from_output(self, output: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract metrics from direct pobax training output
        
        Args:
            output: Direct output from pobax training function
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Extract metric data from the output
        if 'metric' in output:
            metric_data = output['metric']
            
            # Extract timesteps and returns
            if 'timestep' in metric_data:
                timesteps = np.array(metric_data['timestep'])
                metrics['timesteps'] = timesteps.flatten().tolist()
            
            if 'returned_episode_returns' in metric_data:
                episode_returns = np.array(metric_data['returned_episode_returns'])
                metrics['episode_returns'] = episode_returns.flatten().tolist()
            
            if 'returned_episode_lengths' in metric_data:
                episode_lengths = np.array(metric_data['returned_episode_lengths'])
                metrics['episode_lengths'] = episode_lengths.flatten().tolist()
        
        # Extract final evaluation metrics
        if 'final_eval_metric' in output:
            eval_metric = output['final_eval_metric']
            
            if 'returned_episode_returns' in eval_metric:
                eval_returns = np.array(eval_metric['returned_episode_returns'])
                metrics['eval_returns'] = eval_returns.flatten().tolist()
            
            if 'returned_episode_lengths' in eval_metric:
                eval_lengths = np.array(eval_metric['returned_episode_lengths'])
                metrics['eval_lengths'] = eval_lengths.flatten().tolist()
        
        return metrics
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run PPO training using pobax
        
        Returns:
            Dictionary containing training results and metrics
        """
        print("=== Pobax PPO Training ===")
        print(f"Environment: {self.config['env']}")
        print(f"Total steps: {self.config['total_steps']}")
        print(f"Number of environments: {self.config['num_envs']}")
        print(f"Hidden size: {self.config['hidden_size']}")
        print(f"Platform: {self.config.get('platform', 'gpu')}")
        
        # Create hyperparameters
        args = self.create_hyperparams()
        
        # Run training
        start_time = time.time()
        print("\nStarting training...")
        
        try:
            # Import the training function directly to capture output
            from pobax.algos.ppo import make_train
            import jax
            
            # Set up JAX platform
            jax.config.update('jax_platform_name', args.platform)
            
            # Create training function
            rng = jax.random.PRNGKey(args.seed)
            make_train_rng, rng = jax.random.split(rng)
            rngs = jax.random.split(rng, args.n_seeds)
            train_fn = make_train(args, make_train_rng)
            
            # Get training arguments
            import inspect
            train_args = list(inspect.signature(train_fn).parameters.keys())
            
            # Prepare arguments for training
            from collections import deque
            vmaps_train = train_fn
            swept_args = deque()
            
            # Set up vmapped training
            for i, arg in reversed(list(enumerate(train_args))):
                if arg == 'rng':
                    swept_args.appendleft(rngs)
                else:
                    assert hasattr(args, arg)
                    train_arg = getattr(args, arg)
                    swept_args.appendleft(train_arg[0])
            
            # Run training
            train_jit = jax.jit(vmaps_train)
            output = train_jit(*swept_args)
            
            training_time = time.time() - start_time
            print(f"\nTraining completed in {training_time:.2f} seconds")
            
            # Extract metrics from the output
            metrics = self.extract_metrics_from_output(output)
            
            # Calculate summary statistics
            summary = self.calculate_summary_stats(metrics)
            
            results = {
                'config': self.config,
                'metrics': metrics,
                'summary': summary,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat(),
                'pobax_output': output
            }
            
            return results
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
    
    def calculate_summary_stats(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate summary statistics from metrics"""
        summary = {}
        
        if 'episode_returns' in metrics and metrics['episode_returns']:
            returns = np.array(metrics['episode_returns'])
            summary['final_avg_return'] = float(np.mean(returns[-100:]))  # Last 100 episodes
            summary['max_return'] = float(np.max(returns))
            summary['min_return'] = float(np.min(returns))
            summary['std_return'] = float(np.std(returns))
        
        if 'episode_lengths' in metrics and metrics['episode_lengths']:
            lengths = np.array(metrics['episode_lengths'])
            summary['final_avg_length'] = float(np.mean(lengths[-100:]))
            summary['max_length'] = float(np.max(lengths))
            summary['min_length'] = float(np.min(lengths))
        
        if 'eval_returns' in metrics and metrics['eval_returns']:
            eval_returns = np.array(metrics['eval_returns'])
            summary['final_eval_return'] = float(np.mean(eval_returns))
            summary['eval_std_return'] = float(np.std(eval_returns))
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save training results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.results_dir / f"pobax_ppo_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics as numpy arrays
        np_path = self.results_dir / f"pobax_ppo_metrics_{timestamp}.npz"
        np.savez_compressed(np_path, **results['metrics'])
        
        # Save summary
        summary_path = self.results_dir / f"pobax_ppo_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=== Pobax PPO Training Summary ===\n\n")
            f.write(f"Training time: {results['training_time']:.2f} seconds\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            f.write("Configuration:\n")
            for key, value in results['config'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nSummary Statistics:\n")
            for key, value in results['summary'].items():
                f.write(f"  {key}: {value:.4f}\n")
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  NPZ: {np_path}")
        print(f"  Summary: {summary_path}")
    
    def plot_results(self, results: Dict[str, Any]) -> None:
        """Create plots of training results"""
        metrics = results['metrics']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Pobax PPO Training Results - {self.config["env"]}', fontsize=16)
        
        # Plot 1: Episode Returns
        if 'episode_returns' in metrics and metrics['episode_returns']:
            axes[0, 0].plot(metrics['episode_returns'])
            axes[0, 0].set_title('Episode Returns')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        if 'episode_lengths' in metrics and metrics['episode_lengths']:
            axes[0, 1].plot(metrics['episode_lengths'])
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Moving Average Returns
        if 'episode_returns' in metrics and metrics['episode_returns']:
            returns = np.array(metrics['episode_returns'])
            window = min(100, len(returns) // 10)
            if window > 1:
                moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(moving_avg)
                axes[1, 0].set_title(f'Moving Average Returns (window={window})')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Average Return')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Evaluation Returns (if available)
        if 'eval_returns' in metrics and metrics['eval_returns']:
            axes[1, 1].hist(metrics['eval_returns'], bins=20, alpha=0.7)
            axes[1, 1].set_title('Final Evaluation Returns Distribution')
            axes[1, 1].set_xlabel('Return')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"pobax_ppo_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Plot: {plot_path}")
        
        plt.show()


def main():
    """Main function to run Pobax PPO training"""
    
    # Configuration matching your command line arguments
    config = {
        'env': 'rocksample_11_11',
        'num_envs': 8,
        'total_steps': 100000, #5000000,
        'hidden_size': 256,
        'entropy_coeff': 0.2,
        'lr': 2.5e-4,
        'lambda0': 0.95,
        'clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'n_seeds': 5,
        'debug': True,
        
        # Additional settings
        'platform': 'gpu',
        'num_steps': 128,
        'num_epochs': 50,
        'update_epochs': 4,
        'num_minibatches': 4,
        'vf_coeff': 0.5,
        'lambda1': 0.5,
        'alpha': 1.0,
        'ld_weight': 0.0,
        'num_eval_envs': 10,
        'steps_log_freq': 1,
        'update_log_freq': 1,
        'save_checkpoints': False,
        'save_runner_state': False,
        'study_name': 'pobax_ppo_rocksample_11_11',
        'anneal_lr': True,
        'show_discounted': False,
        'memoryless': False,
        'action_concat': True,
        'seed': 2020
    }
    
    print(f"Config: {config}")
    
    # Create trainer and run training
    trainer = PobaxPPOTrainer(config)
    
    try:
        results = trainer.run_training()
        trainer.save_results(results)
        trainer.plot_results(results)
        
        # Print summary
        print("\n=== Training Summary ===")
        print(f"Final average return: {results['summary'].get('final_avg_return', 'N/A'):.4f}")
        print(f"Final evaluation return: {results['summary'].get('final_eval_return', 'N/A'):.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 