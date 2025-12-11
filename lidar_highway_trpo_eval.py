import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor


MODEL_PATH = "./models/trpo_lidar_highway_opt.zip"   
EPISODES = 500
RENDER = False  


def make_eval_env(render=False):
    render_mode = "human" if render else None
    env = gym.make("highway-v0", render_mode=render_mode)

    config = {
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
        },

        # ---------- Traffic Setup ----------
        "vehicles_count": 15,
        "duration": 30,
        "simulation_frequency": 15,
        "policy_frequency": 5,

        # ---------- Reward Structure ----------
        "collision_reward": -8.0,
        "high_speed_reward": 0.6,
        "reward_speed_range": [20, 30],
        "lane_change_reward": 0.3,
        "right_lane_reward": 0.02,
        "normalize_reward": False,

        # ---------- Behavior ----------
        "lane_change_speed_ratio": 1.0,
        "offroad_terminal": False,
    }

    env.unwrapped.configure(config)
    env = Monitor(env)
    return env


def plot_violin(reward_list, save_path="trpo_lidar_violin.png"):
    plt.figure(figsize=(6, 4))
    plt.violinplot(reward_list, showmedians=True)
    plt.title("TRPO Highway LiDAR Reward Distribution")
    plt.ylabel("Episode Reward")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def evaluate(model_path, episodes=500, render=False):
    print(f"\nLoading TRPO model from: {model_path}")
    model = TRPO.load(model_path, device="cpu")

    env = make_eval_env(render)
    episode_rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward

            if render:
                env.render()

        episode_rewards.append(total_reward)
        print(f"✅ Episode {ep+1}: reward = {total_reward:.2f}")

    env.close()

    print("\n==================== SUMMARY =====================")
    print(f"Episodes: {episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Dev    : {np.std(episode_rewards):.2f}")
    print(f"Min/Max    : {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")

    return episode_rewards


if __name__ == "__main__":
    rewards = evaluate(MODEL_PATH, episodes=EPISODES, render=RENDER)
    plot_violin(rewards)
