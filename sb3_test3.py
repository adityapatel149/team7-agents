import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import time


def make_env():
    def _init():
        env = gym.make("highway-v0", disable_env_checker=False)
        config = {
            "observation": {"type": "LidarObservation", "cells": 32, "maximum_range": 100},
            "simulation_frequency": 10,
            "policy_frequency": 2,
            "duration": 40,
            "vehicles_count": 20,
        }
        env.unwrapped.configure(config)
        env.reset()  # Make sure configuration takes effect
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init

if __name__ == "__main__":
    # Only run multiprocessing code here
    env = SubprocVecEnv([make_env() for _ in range(4)]) # can use 8 for more envs in parallel if you have more cores.

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-5,
        buffer_size=1000000,
        batch_size=64,
        #train_freq=4,
        target_update_interval=3000,
        verbose=1,
        device="cpu",
        tensorboard_log="dqn_tensorboard_log"
    )

    print("Starting DQN training...")
    start = time.time()
    model.learn(total_timesteps=100_000)
    print(f"Done!")

    model.save("dqn_highway_fast")
    print("Model saved as dqn_highway_fast.zip")

    env.close()

