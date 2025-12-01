import gymnasium as gym
import custom_env 
import numpy as np
from sb3_contrib import TRPO

MODEL_PATH = "./models/trpo_team7_lidar.zip"   # path to trained model

def make_eval_env(render: bool = False,):
    """
    Create a single Team7-v0 environment for evaluation.
    Uses the same config as training.
    """
    render_mode = "human" if render else None

    env = gym.make("Team7-v0",  render_mode=render_mode)

    env.reset()

    return env


def evaluate(model_path, episodes=10, render=False):
    print("   TRPO MODEL EVALUATION")

    print(f"Loading model from: {model_path}")
    model = TRPO.load(model_path, device="cpu")

    env = make_eval_env(True)
    episode_rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward:.2f}")

    env.close()

    print("\n========== SUMMARY ==========")
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std  reward: {np.std(episode_rewards):.2f}")
    print(f"Min / Max  : {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print("=============================\n")


if __name__ == "__main__":
    evaluate(MODEL_PATH, episodes=10, render=True)
