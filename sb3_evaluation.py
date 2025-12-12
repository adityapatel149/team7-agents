import argparse
import time

import gymnasium as gym
import highway_env  # noqa: F401  # registers intersection-v0 and others

from stable_baselines3 import DQN, PPO
from sb3_contrib import TRPO
from matplotlib import pyplot as plt

def make_eval_env(action_type: str, obs_type: str):
    """
    Create a single intersection-v0 environment for evaluation with rendering.

    action_type:
        - "DiscreteMetaAction"
        - "ContinuousAction"

    obs_type:
        - "lidar"  -> LidarObservation
        - "gray"   -> GrayscaleObservation
    """
    # Render to screen during eval
    env = gym.make("intersection-v1", disable_env_checker=False, render_mode="human")

    # Choose observation config
    if obs_type == "lidar":
        obs_config = {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 120,
        }
    else:  # obs_type == "gray"
        obs_config = {
            "type": "GrayscaleObservation",
            "observation_shape": (5, 5),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.0,
        }

    config = {
        "observation": obs_config,
        "action": {"type": action_type},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,        
    }

    env.unwrapped.configure(config)
    env.reset()
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=["dqn", "trpo","ppo"],
        default="ppo",
        help="Which algorithm was used: dqn or trpo.",
    )
    parser.add_argument(
        "--action",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Action type used in training: discrete or continuous.",
    )
    parser.add_argument(
        "--obs",
        choices=["lidar", "gray"],
        default="lidar",
        help="Observation type used in training: lidar or gray.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes to run.",
    )
    args = parser.parse_args()

    action_type = "DiscreteMetaAction" if args.action == "discrete" else "ContinuousAction"

    # DQN only supports discrete actions
    if args.algo == "dqn" and action_type != "DiscreteMetaAction":
        raise ValueError("DQN models must have been trained with discrete actions.")

    # Figure out model path (must match train script naming)
    if args.algo == "dqn":
        model_path = f"dqn_intersection_{args.action}_{args.obs}.zip"
    elif args.algo == "ppo":
        model_path = f"ppo_intersection_{args.action}_{args.obs}.zip"
    else:
        model_path = f"trpo_intersection_{args.action}_{args.obs}.zip"
    #model_path = "model.zip"

    print(f"Loading model from {model_path} ...")
    if args.algo == "dqn":
        model = DQN.load(model_path, device="cpu")
    elif args.algo == "ppo":
        model = PPO.load(model_path, device="cpu")
    else:
        model = TRPO.load(model_path, device="cpu")
    
    #model = PPO.load(model_path, device="cpu")
    print("Model loaded.")

    # Create evaluation environment
    env = make_eval_env(action_type, args.obs)

    total_rewards = []
    episode_lengths = []

    print(
        f"Evaluating {args.algo.upper()} on intersection-v0 "
        f"(action={action_type}, obs={args.obs}) for {args.episodes} episodes..."
    )

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            # Deterministic to see actual learned policy (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            done = terminated or truncated

            # Small sleep just to slow things down visually if needed
            # (you can comment this out)
            time.sleep(0.02)

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        print(f"Episode {ep + 1}: return={ep_reward:.2f}, length={ep_len}")

    avg_rew = sum(total_rewards) / len(total_rewards)
    avg_len = sum(episode_lengths) / len(episode_lengths)
    print("====================================")
    print(f"Average return over {args.episodes} episodes: {avg_rew:.2f}")
    print(f"Average length over {args.episodes} episodes: {avg_len:.1f}")
    print("====================================")

    env.close()


    #generate violin plot for 500 episodes
    plt.violinplot(total_rewards,vert=True,showmeans=False,showmedians=True,showextrema=True)
    plt.title(f"{args.algo.upper()} Rewards over {args.episodes} Episodes") 
    plt.ylabel("Total Reward")
    plt.savefig(f"{args.algo}_rewards_violin_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
