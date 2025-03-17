import numpy as np
# Number of evaluation episodes
def evaluate_agent(env, model, num_episodes=20, gamma=0.95, max_steps=100):
    discounted_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done and step < max_steps:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            # Compute the discounted reward for this step
            episode_reward += (gamma ** step) * reward
            step += 1

        discounted_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Discounted Reward = {episode_reward:.2f}, Steps Taken = {step}")

    # Calculate and display the average discounted reward
    average_discounted_reward = np.mean(discounted_rewards)
    print(f"\nAverage Discounted Reward over {num_episodes} episodes: {average_discounted_reward:.2f}")

    return average_discounted_reward
