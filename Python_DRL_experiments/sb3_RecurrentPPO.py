import gym
import gym_pomdp
import torch
from sb3_contrib import RecurrentPPO  # Recurrent PPO from sb3_contrib
from stable_baselines3.common.env_util import make_vec_env
from evaluation import evaluate_agent  # Replace with your evaluation function or code

# Step 1: Check if GPU is available
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Training will run on CPU.")

# Step 2: Create the environment
env = gym.make("Rock-v0", board_size=7, num_rocks=8)  # Replace with your environment
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

# Step 3: Wrap the environment for vectorization
vec_env = make_vec_env(lambda: env, n_envs=4)

# Specify the directory for TensorBoard logs
tensorboard_log_dir = "./tensorboard_logs"

model = RecurrentPPO(
    "MlpLstmPolicy",  # LSTM-based policy
    vec_env,
    gamma=0.99,  # Discount factor
    verbose=1,  # Verbose output
    learning_rate=1e-4,  # Learning rate
    n_steps=2048,  # Larger number of steps per update
    batch_size=128,  # Batch size
    ent_coef=0.1,  # Higher entropy coefficient for exploration
    gae_lambda=0.95,  # GAE lambda for advantage estimation
    clip_range=0.2,  # PPO clipping range
    policy_kwargs={'net_arch': [64, 64], 'lstm_hidden_size': 64},  # Policy architecture
    device="cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
)

# Step 5: Train the model
print("Training started...")
model.learn(total_timesteps=1_000_000)  # 1 million timesteps
print("Training finished!")

# Step 6: Save the model
model.save("recurrent_ppo_rock_v0")

# Step 7: Load the trained model
print("Loading the trained model...")
model = RecurrentPPO.load("recurrent_ppo_rock_v0")

# Step 8: Evaluate the agent
print("Evaluating the agent...")
evaluate_agent(env, model, num_episodes=1000, gamma=0.95, max_steps=50)

# Step 9: Close the environment
env.close()

# Optional: Check the device used by the model
print("Model is running on:", model.device)
