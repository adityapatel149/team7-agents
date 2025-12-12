import argparse
import time

import gymnasium as gym
import highway_env  # noqa: F401  # registers merge-v0 and others

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO


def make_env(action_type: str, obs_type: str):
    """
    Factory for merge-v0 with a given action and observation type.

    action_type:
        - "DiscreteMetaAction"
        - "ContinuousAction"

    obs_type:
        - "lidar"  -> LidarObservation (vector)
        - "gray"   -> GrayscaleObservation (stacked frames)
    """

    def _init():
        # No rendering during training (especially with SubprocVecEnv)
        env = gym.make("merge-v0", disable_env_checker=False)

        # Choose observation config
        if obs_type == "lidar":
            obs_config = {
                "type": "LidarObservation",
                "cells": 32,
                "maximum_range": 120,  # a bit larger to see merging traffic earlier
            }
        else:  # obs_type == "gray"
            obs_config = {
                "type": "GrayscaleObservation",
                "observation_shape": (84, 84),
                "stack_size": 4,  # helps infer motion
                "weights": [0.2989, 0.5870, 0.1140],  # RGB -> gray
                "scaling": 1.0,
            }

        config = {
            "observation": obs_config,
            "action": {"type": action_type},
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "duration": 40,
            "vehicles_count": 15,        # >0 so there is actual merging traffic
            "collision_reward": -5.0,    # harsh penalty
            "high_speed_reward": 0.5,
            "lane_change_reward": -0.1,
            "reward_speed_range": [20, 30],
            "right_lane_reward": 0.0,
            "reward_acceleration": -0.05,
        }

        env.unwrapped.configure(config)
        env.reset()  # make sure config takes effect

        # Monitor records episode rewards/lengths for SB3 (ep_rew_mean, ep_len_mean)
        env = Monitor(env)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=["dqn", "trpo"],
        default="trpo",
        help="Which algorithm to use.",
    )
    parser.add_argument(
        "--action",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Action type: discrete = DiscreteMetaAction, continuous = ContinuousAction.",
    )
    parser.add_argument(
        "--obs",
        choices=["lidar", "gray"],
        default="lidar",
        help="Observation type: lidar (LidarObservation) or gray (GrayscaleObservation).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (SubprocVecEnv).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Total training timesteps.",
    )
    args = parser.parse_args()

    # Map CLI choice to highway-env action type string
    action_type = "DiscreteMetaAction" if args.action == "discrete" else "ContinuousAction"

    # DQN only supports discrete actions
    if args.algo == "dqn" and action_type != "DiscreteMetaAction":
        raise ValueError("DQN only supports discrete actions. Use --action discrete with --algo dqn.")

    # Create vectorized env
    env = SubprocVecEnv([make_env(action_type, args.obs) for _ in range(args.num_envs)])

    # Choose policy based on observation type
    # - lidar: vector -> MlpPolicy
    # - gray: image -> CnnPolicy
    if args.obs == "gray":
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"

    # Try to load existing model (if any)
    model = None
    algo_name = args.algo

    if args.algo == "dqn":
        model_path = f"dqn_merge_{args.action}_{args.obs}.zip"
        try:
            print(f"Loading existing DQN model from {model_path} ...")
            model = DQN.load(model_path, env=env, device="cpu")
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No existing DQN model found. A new model will be created.")
    else:  # TRPO
        model_path = f"trpo_merge_{args.action}_{args.obs}.zip"
        try:
            print(f"Loading existing TRPO model from {model_path} ...")
            model = TRPO.load(model_path, env=env, device="cpu")
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No existing TRPO model found. A new model will be created.")

    # Build model only if we didn't load one
    if model is None:
        if args.algo == "dqn":
            model = DQN(
                policy,
                env,
                learning_rate=1e-5,
                buffer_size=1_000_000,
                batch_size=64,
                target_update_interval=3000,
                verbose=1,
                device="cpu",
                tensorboard_log="tb_logs",
            )
        else:  # TRPO
            model = TRPO(
                policy,
                env,
                gamma=0.99,
                learning_rate=5e-4,
                gae_lambda=0.98,
                n_steps=2048,
                batch_size=256,
                cg_max_steps=20,
                target_kl=0.01,
                verbose=1,
                tensorboard_log="tb_logs",
            )

    print(
        f"Starting {algo_name.upper()} training on merge-v0 "
        f"(action={action_type}, obs={args.obs})..."
    )

    start = time.time()
    tb_name = f"{algo_name}_merge_{args.action}_{args.obs}"
    model.learn(total_timesteps=args.steps, tb_log_name=tb_name)
    elapsed = time.time() - start
    print(f"Done! Training time: {elapsed:.1f}s")

    save_name = f"{algo_name}_merge_{args.action}_{args.obs}"
    model.save(save_name)
    print(f"Model saved as {save_name}.zip")

    env.close()


if __name__ == "__main__":
    main()
