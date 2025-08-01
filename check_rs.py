# test_roomba_env.py

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from pobax.envs.jax.rocksample import RockSample, EnvParams # Assuming your code is in rocksample.py

def run_environment_test():
    """
    Initializes and runs a test episode of the RockSample environment to verify its functionality.
    """
    # 1. Setup Keys, Environment, and Parameters
    key = random.PRNGKey(42)  # Use a fixed seed for reproducibility
    key, env_key, reset_key = random.split(key, 3)
    
    print("--- Initializing rocksample Environment (Configuration 1) ---")
    config_path = "/home/users/ucakir/Experiments_POMDP_DRL/pobax/pobax/envs/configs/rocksample_7_8_config.json"
    env = RockSample(key=env_key, config_path=config_path)
    env_params = env.default_params

    # 2. Reset Environment to get initial state
    print("\n--- Resetting Environment ---")
    obs, state = env.reset_env(reset_key, env_params)

    print(f"Initial Observation: {obs}")
    print(f"Initial State: position=({state.position}), rock_morality={state.rock_morality}")
    
    # --- Assertion for Initial State ---
    # Check if the initial observation is within the defined observation space
    assert env.observation_space(env_params).contains(obs), f"Initial observation {obs} is outside the defined space."
    print("Initial observation is valid.")

    # 3. Simulation Loop
    print("\n--- Starting Simulation Loop ---")
    cumulative_reward = 0.0
    
    # Jit the step function for performance
    jitted_step = jax.jit(env.step_env)

    for step in range(env_params.max_steps_in_episode):
        # Generate new keys for action sampling and environment stepping
        key, action_key, step_key = random.split(key, 3)

        # Sample a random action from the action space
        action = env.action_space(env_params).sample(action_key)
        
        # Execute one step in the environment
        obs, state, reward, done, info = jitted_step(step_key, state, action, env_params)

        cumulative_reward += reward

        print(f"\n--- Step {step} ---")
        print(f"Action Taken:      {action}")
        print(f"New Observation:   {obs}")
        print(f"New State:         position=({state.position}), rock_morality={state.rock_morality}")
        print(f"Reward Received:   {reward:.2f}")
        print(f"Cumulative Reward: {cumulative_reward:.2f}")
        print(f"Episode Done:      {done}")
        
        # --- Assertions for Step Correctness ---
        assert env.observation_space(env_params).contains(obs), f"Observation {obs} at step {state.steps} is out of bounds!"
        assert isinstance(reward, jax.Array) and reward.shape == (), "Reward must be a JAX scalar."
        assert isinstance(done, jax.Array) and done.dtype == jnp.bool_, "Done flag must be a JAX boolean."

        if done:
            print("\n------------------------------------")
            print(f"EPISODE FINISHED at step {step}.")
            print("Final Cumulative Reward:", cumulative_reward)
            print("Final State:", state)
            print("Final Observation:", obs)
            print("Final Info:", info)
            print("Environment stepping and termination are functioning correctly.")
            
            
            print("------------------------------------")
            break

    if not done:
        print(f"\nEpisode finished after {env_params.max_steps_in_episode} steps because the step limit was reached.")
        
    print("\n--- Test Script Finished ---")
    print("The environment appears to be stepping and terminating correctly.")


if __name__ == "__main__":
    run_environment_test()