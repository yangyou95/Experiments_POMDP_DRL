import optuna
import subprocess
import sys
import re

# --- Configuration ---
ENV_NAME = "lightdark"
N_SEEDS = 3  # Use fewer seeds for faster optimization
TOTAL_STEPS = 800000  # Use fewer steps to speed up each trial
N_TRIALS = 50  # The number of hyperparameter combinations to test


def objective(trial: optuna.Trial) -> float:
    """
    Defines a single trial for Optuna.
    A trial consists of:
    1. Suggesting a set of hyperparameters.
    2. Running the PPO training script with these hyperparameters.
    3. Parsing the output to get the final score.
    4. Returning the score for Optuna to maximize.
    """
    # 1. Suggest hyperparameters
    # --- Learning Parameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.4)
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.0, 0.1)
    vf_coeff = trial.suggest_float("vf_coeff", 0.3, 0.7)
    lambda1 = trial.suggest_float("lambda1", 0.9, 0.99)  # GAE Lambda
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)

    # --- Architecture and Batching ---
    num_envs = trial.suggest_categorical("num_envs", [64, 128, 256])
    num_steps = trial.suggest_categorical("num_steps", [64, 128, 256])
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    update_epochs = trial.suggest_int("update_epochs", 2, 8)
    num_minibatches = trial.suggest_categorical("num_minibatches", [2, 4, 8, 16])

    print(f"\n--- Starting Trial {trial.number} ---")
    print(
        f"  Params: lr={lr:.6f}, num_steps={num_steps}, hidden_size={hidden_size}, ent_coeff={entropy_coeff:.4f}"
    )
    print(
        f"          clip_eps={clip_eps:.4f}, update_epochs={update_epochs}, num_envs={num_envs}, vf_coeff={vf_coeff:.4f}"
    )
    print(
        f"          lambda1={lambda1:.4f}, max_grad_norm={max_grad_norm:.4f}, num_minibatches={num_minibatches}"
    )

    # 2. Construct the command to run the training script
    command = [
        sys.executable,  # Use the same python interpreter
        "-m",
        "pobax.algos.ppo",
        "--env",
        ENV_NAME,
        # --- Tuned Hyperparameters ---
        "--lr",
        str(lr),
        "--num_steps",
        str(num_steps),
        "--hidden_size",
        str(hidden_size),
        "--entropy_coeff",
        str(entropy_coeff),
        "--clip_eps",
        str(clip_eps),
        "--update_epochs",
        str(update_epochs),
        "--num_envs",
        str(num_envs),
        "--lambda1",
        str(lambda1),
        "--vf_coeff",
        str(vf_coeff),
        "--max_grad_norm",
        str(max_grad_norm),
        "--num_minibatches",
        str(num_minibatches),
        # --- Fixed parameters from your script ---
        "--debug",
        "--ld_weight",
        "0",
        "--action_concat",
        "--lambda0",
        "0.1",  # This seems fixed in your scripts
        "--seed",
        "2024",
        "--n_seeds",
        str(N_SEEDS),
        "--total_steps",
        str(TOTAL_STEPS),
        "--anneal_lr",
        "--default_max_steps_in_episode",
        "1000",
        "--platform",
        "gpu",
        "--num_eval_envs",
        "5",
        "--show_discounted",
    ]

    # 3. Run the command and capture output
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        output = result.stdout

        print(output)  # Print the output for debugging

        # 4. Parse the output to find the final score
        match = re.search(r"Final Average Return: ([-+]?\d*\.\d+|\d+)", output)
        if match:
            final_return = float(match.group(1))
            print(f"  Trial {trial.number} finished. Final Return: {final_return}")
            return final_return
        else:
            print(
                f"  Trial {trial.number} failed: Could not parse final return from output."
            )
            # Return a very low value to penalize failed runs
            return -1e6

    except subprocess.CalledProcessError as e:
        print(f"  Trial {trial.number} failed with an error.")
        print(e.stderr)
        # Prune the trial if it fails
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"An unexpected error occurred during trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    # Create a study object and specify the direction is to maximize the return
    study = optuna.create_study(direction="maximize")

    # Start the optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # Print the results
    print("\n--- Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial value (average return): {best_trial.value}")

    print("Best hyperparameters found:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

