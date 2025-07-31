#!/usr/bin/env python3
"""
Smoke-test for the JAX Roomba environment.
"""

import jax, jax.numpy as jnp
from jax import random
from pobax.envs import get_env


def test_roomba_env():
    print("-- Roomba smoke-test --")

    # force CPU only for this quick test, you can drop this later
    jax.config.update("jax_platform_name", "cpu")

    # ------------------------------------------------------------------
    # 1. Master PRNG key and environment construction
    # ------------------------------------------------------------------
    master_key = random.PRNGKey(43)
    env_key, master_key = random.split(master_key)
    env, env_params = get_env("roomba", env_key, perfect_memory=False)

    print("Observation space:", env.observation_space(env_params))
    print("Action space     :", env.action_space(env_params))

    # ------------------------------------------------------------------
    # 2. Reset - POBAX environments are vectorized, need batch of keys
    # ------------------------------------------------------------------
    reset_key, master_key = random.split(master_key)
    # Create a batch of keys for vectorized environment (default is 1 environment)
    reset_keys = random.split(reset_key, 1)  # Single environment for testing
    print(f"DEBUG: Master key: {master_key}")
    print(f"type of master_key: {type(master_key)}")
    print(f"DEBUG: Reset keys shape: {reset_keys.shape}")
    print(f"type of reset_keys: {type(reset_keys)}")
    obs, state = env.reset(reset_keys, env_params)
    print("Initial obs:", obs, "\n")

    # ------------------------------------------------------------------
    # 3. Roll a few steps - also need batched keys
    # ------------------------------------------------------------------
    for t in range(10):
        step_key, master_key = random.split(master_key)
        # Create batch of keys for vectorized step
        step_keys = random.split(step_key, 1)  # Single environment for testing

        # sample a valid *continuous* action
        action_key, master_key = random.split(master_key)
        action_keys = random.split(action_key, 1)  # Single environment for testing
        # Vectorized sampling for vectorized environment
        sample_fn = jax.vmap(env.action_space(env_params).sample)
        action = sample_fn(action_keys)

        obs, state, reward, done, info = env.step(step_keys, state, action, env_params)
        print(f"t={t:02d} | a={jnp.round(action, 3)} | r={reward} | done={done}")

        if done:
            print("Episode finished early.")
            break


if __name__ == "__main__":
    test_roomba_env()