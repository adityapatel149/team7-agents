import os
import time

import gymnasium as gym
import highway_env
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


MODEL_NAME = "trpo_lidar_highway_opt"
MODELS_DIR = "./models"
TB_LOG_DIR = "./tb_logs_trpo_lidar_highway"

TOTAL_TIMESTEPS = 100_000     
NUM_ENVS = 4


def make_env(idx):
    def _init():
        env = gym.make("highway-v0")

        config = {
            "observation": {
                "type": "LidarObservation",
                "cells": 128,
                "maximum_range": 64
            },

            # ------ Traffic Settings ------
            "vehicles_count": 15,
            "duration": 30,
            "simulation_frequency": 15,
            "policy_frequency": 5,

            # ------ Reward Shaping (Tuned) ------
            "collision_reward": -4.0,
            "high_speed_reward": 0.6,            # strong incentive for overtaking
            "reward_speed_range": [20, 30],      # penalizes slow following
            "lane_change_reward": 2,           # big incentive → lane changes are worth it
            "right_lane_reward": 0.02,           # tiny preference, not dominant
            "normalize_reward": False,

            # ------ Behavior Tweaks ------
            "lane_change_speed_ratio": 1.0,      # allow lane changes even at low speeds
            "offroad_terminal": False,
        }

        env.unwrapped.configure(config)
        env.reset()
        env = Monitor(env)
        return env

    return _init


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    print("=" * 70)
    print("        TRPO Training – HighwayEnv Lidar (Optimized Config)")
    print("=" * 70)

    env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])

    model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")

    # Try loading existing model
    try:
        print(f"Loading existing model from {model_path} ...")
        model = TRPO.load(model_path, env=env, device="cpu")
        print("Loaded! Continuing training.")
    except:
        print("No existing model. Creating new TRPO model.")

        model = TRPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=5e-4,          # slightly higher for LiDAR
            gamma=0.99,
            gae_lambda=0.95,
            n_steps=4096,                 # large batch improves TRPO stability
            batch_size=512,
            cg_damping=0.1,
            cg_max_steps=20,
            n_critic_updates=10,
            target_kl=0.015,
            line_search_shrinking_factor=0.8,
            line_search_max_iter=10,
            normalize_advantage=True,

            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],       # TRPO benefits from deeper nets with LiDAR
                    vf=[256, 256]
                )
            ),

            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            device="cpu",
        )

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        tb_log_name="TRPO_Lidar_Optimized",
    )
    elapsed = (time.time() - start) / 60

    save_path = os.path.join(MODELS_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"\nTraining completed in {elapsed:.1f} minutes.")
    print(f"Model saved to: {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
