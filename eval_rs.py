import jax, jax.numpy as jnp
from flax.training import orbax_utils
import orbax.checkpoint
from types import SimpleNamespace

from pobax.config import PPOHyperparams
from pobax.envs import get_env
from pobax.models import ScannedRNN, ActorCritic
from pobax.algos.ppo import PPO, env_step, calculate_gae   # import symbols you already defined

from pobax.models import ScannedRNN, get_network_fn
from pobax.utils.file_system import get_results_path, numpyify
import functools
import optax 
from pathlib import Path
from flax.training.train_state import TrainState
from pobax.models import get_gymnax_network_fn
from gymnax.environments import environment, spaces
# ---------- 1.  Load checkpoint ----------

# chkpath for roomba
#ckpt_path = "/home/users/ucakir/Experiments_POMDP_DRL/pobax/results/batch_ppo_test/roomba_seed(2024)_time(20250731-020404)_43c7c57e55488e7016a575342b19d04c"           # directory created earlier

#ckpt_path = "/home/users/ucakir/Experiments_POMDP_DRL/pobax/results/batch_ppo_test/lightdark_seed(2025)_time(20250730-134529)_5824f9e29097cf1b61429303e9df7058"

ckpt_path = "/home/users/ucakir/Experiments_POMDP_DRL/pobax/results/batch_ppo_test/rocksample_7_8_seed(2024)_time(20250731-040136)_6333576792178f016af7ef2dcd8029fb"

ckptr      = orbax.checkpoint.PyTreeCheckpointer()
ckpt       = ckptr.restore(ckpt_path)         # returns the dict you saved
train_state_data = ckpt["final_train_state"]

#print(f"type(train_state_data): {type(train_state_data)}")
# Re‑wrap params so PPO’s code can access `.params`
args_dict   = ckpt["args"]                    # hyper-parameters

def get_gymnax_network_fn(env: environment.Environment, env_params: environment.EnvParams,
                          memoryless: bool = False):
    
    
    # check type of action space
    if isinstance(env.action_space(env_params), spaces.Discrete):
        print("Action space is discrete")
        action_size = env.action_space(env_params).n
    else:
        action_size = env.action_space(env_params).shape[0]
    print(f"type of action space: {type(env.action_space(env_params))}")
    print(f"env.action_space(env_params): {env.action_space(env_params).shape}")
    print(f"env.observation_space(env_params): {env.observation_space(env_params)}")
    
   
    
    network_fn = ActorCritic
    
        
    return network_fn, action_size

unpacked_ts = ckpt["final_train_state"]["params"]#ckpt['out']['runner_state'][0]

key = jax.random.PRNGKey(0)
env, env_params = get_env(args_dict['env'], key,
                        args_dict['gamma'],
                        action_concat=args_dict['action_concat'])

network_fn, action_size = get_gymnax_network_fn(env, env_params, memoryless=args_dict['memoryless'])

network = network_fn(action_size,
                    double_critic=args_dict['double_critic'],
                    hidden_size=args_dict['hidden_size'],
                    memoryless= args_dict['memoryless'],
                    is_discrete=False,
                    action_dim=action_size,
                    is_image=False)


tx = optax.adam(args_dict['lr'][0])
train_state = TrainState.create(apply_fn=network.apply,
                    params=jax.tree.map(lambda x: x[0, 0, 0, 0, 0,0,0], unpacked_ts),
                    tx=tx)





args = PPOHyperparams().from_dict(args_dict)

args.num_envs = int(1e4)

print(f"Loaded checkpoint from {ckpt_path}")
print(f"Hyperparameters: {args}")

# ---------- 2.  Recreate env/network ----------
env_key = jax.random.PRNGKey(0)

env, env_params = get_env(args.env, env_key,
                          num_envs=args.num_envs,
                          image_size=args.image_size,
                          gamma=args.gamma,
                          perfect_memory=args.perfect_memory,
                          action_concat=args.action_concat)

network_fn, action_size, is_image, is_discrete = get_network_fn(env, env_params)
network = network_fn(args.env, action_size,
                     double_critic=args.double_critic,
                     hidden_size=args.hidden_size,
                     memoryless=args.memoryless,
                     is_discrete=is_discrete,
                     is_image=is_image)

agent = PPO(network,
            double_critic=args.double_critic,
            ld_weight=args.ld_weight,
            alpha=args.alpha,
            vf_coeff=args.vf_coeff,
            entropy_coeff=args.entropy_coeff,
            clip_eps=args.clip_eps)

_env_step = functools.partial(env_step, agent=agent, env=env, env_params=env_params)

# ---------- 3.  Evaluate five episodes ----------
def one_eval(rng):
    reset_key, scan_key = jax.random.split(rng)
    obs, env_state = env.reset(jax.random.split(reset_key, args.num_envs), env_params)
    hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    runner_state = (train_state, env_state, obs, jnp.zeros(args.num_envs, bool), hstate, scan_key)
    runner_state, traj = jax.lax.scan(_env_step, runner_state, None, env_params.max_steps_in_episode)
    info = traj.info
    mask = info["returned_episode"]
    # Non-concrete boolean indexing is not allowed inside jax.vmap.
    # Instead, we use jnp.where to select the returns of finished episodes
    # and then calculate the mean.
    masked_returns = jnp.where(mask, info["returned_discounted_episode_returns"], 0.0)
    num_episodes = jnp.sum(mask)
    mean_return = jnp.sum(masked_returns) / jnp.maximum(num_episodes, 1)
    return mean_return, traj

rngs = jax.random.split(jax.random.PRNGKey(42), 5)
episode_returns, trajectories = jax.vmap(one_eval)(rngs)


print(f"NUM ENVS: {args.num_envs}")
for i, r in enumerate(episode_returns, 1):
    print(f"[Eval {i}/5] Avg discounted return: {r:.2f}")
    
    
print(f"average across 5 episodes: {jnp.mean(episode_returns):.2f}")
print(f" std across 5 episodes: {jnp.std(episode_returns):.2f}")





# I want to show information about trajectory, the actions taken the observation, etrc
# Example of how to print trajectory information
actions = trajectories.action
rewards = trajectories.reward
done = trajectories.done

observations = trajectories.obs
info = trajectories.info



returned_discounted_episode_returns = info["returned_discounted_episode_returns"]
returned_episode_lengths = info["returned_episode_lengths"]


# Print the shape of the arrays
print(f"Actions shape: {actions.shape}")
#print(f"Observations shape: {observations.shape}")
print(f"Rewards shape: {rewards.shape}")
print(f"Done flags shape: {done.shape}")
print(f"Returned Discounted Episode Returns shape: {returned_discounted_episode_returns.shape}")
print(f"Returned Episode Lengths shape: {returned_episode_lengths.shape}")



# look at one trajectory
trajectory_index = 0  # Change this to look at different trajectories
simulation_index = 2
print(f"\n--- Trajectory {trajectory_index + 1} ---")









# create a plot to show the reward at each step for all trajectories
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for i in range(actions.shape[0]):
    plt.plot(rewards[i, :, 0], label=f'Trajectory {i + 1}')
plt.title('Rewards per Step for Each Trajectory')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.grid()
plt.show()

for step in range(actions.shape[1]):
    print("==")
    print(f"\n--- Step {step + 1} ---")
    print(f"Action: {actions[trajectory_index][simulation_index][step]}")
    #print(f"Observation: {observations[trajectory_index][simulation_index][step]}")
    print(f"Reward: {rewards[trajectory_index][simulation_index][step]}")
    print(f"Done: {done[trajectory_index][simulation_index][step]}")
    print("-" * 30)
    if done[trajectory_index][simulation_index][step]:
        print("Episode finished.")
        break











