import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import numpy as np
import time

# ==========================
# Load trained model
# ==========================
model_path = "dqn_highway_lidar_vehicles.zip"   # your saved model
model = DQN.load(model_path, device="cpu")

# ==========================
# Create evaluation environment
# ==========================
env = gym.make("highway-v0", render_mode="human")  # for on-screen visualization

# Optional: match training config
config = {
    "observation": {
        "type": "LidarObservation",
        "cells": 64,
        "maximum_range": 200,
        "normalize": True,
    },
    "vehicles_count": 0,  # test on moderate traffic
}
env.unwrapped.configure(config)

# ==========================
# Run evaluation loop
# ==========================
n_eval_episodes = 10
episode_rewards = []

for ep in range(n_eval_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Model predicts action deterministically
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(3)
        done = terminated or truncated
        total_reward += reward

        env.render()
        time.sleep(0.02)  # small delay for smoother visualization

    episode_rewards.append(total_reward)
    print(f"✅ Episode {ep+1}: reward = {total_reward:.2f}")

env.close()

print("\n📊 Evaluation Results:")
print(f"Average reward over {n_eval_episodes} episodes: {np.mean(episode_rewards):.2f}")
print(f"Std dev: {np.std(episode_rewards):.2f}")
