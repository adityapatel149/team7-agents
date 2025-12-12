import gymnasium as gym
import custom_env 
import numpy as np
from sb3_contrib import TRPO
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "./models/trpo_team7_lidar.zip"   # path to trained model

def make_eval_env(render: bool = False,):
    """
    Create a single Team7-v0 environment for evaluation.
    """
    render_mode = "human" if render else None

    env = gym.make("Team7-v0",  render_mode=render_mode)

    env.reset()

    return env


def evaluate(model_path, episodes=150, render=False):
    print("   TRPO MODEL EVALUATION")

    print(f"Loading model from: {model_path}")
    model = TRPO.load(model_path, device="cpu")

    env = make_eval_env(render)
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

    return episode_rewards

def plot_violin(reward_list, title="TRPO Reward Distribution", save_path=None):

    """
    Create a violin plot from a list of episode rewards.
    """
    # plt.figure(figsize=(8, 6))
    # plt.violinplot(data=reward_list, inner="quartile", color="skyblue")
    # plt.title(title, fontsize=14)
    # plt.ylabel("Episode Reward", fontsize=12)
    # plt.xticks([])
    plt.violinplot(reward_list)
    plt.title(f"TRPO Rewards over 500 Episodes") 
    plt.ylabel("Total Reward")
    plt.savefig(f"TRPO_rewards_violin_plot.png")
    plt.show()

    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
    #     print(f"Violin plot saved to {save_path}")

    # plt.show()



if __name__ == "__main__":
    rewards = evaluate(MODEL_PATH, episodes=500, render=True)

    plot_violin(
        rewards,
        title="TRPO: Episode Reward Distribution (500 Episodes)",
        save_path="trpo_violin_plot.png"
    )
