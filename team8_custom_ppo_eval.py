import narrow_safe_env
import narrow_street
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


class CleanObservation(gym.ObservationWrapper):
    """Replace NaN, +inf, -inf values in observations."""
    def observation(self, obs):
        return np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)


class CleanReward(gym.RewardWrapper):
    """Ensure reward is numeric and safe."""
    def reward(self, r):
        if np.isnan(r) or np.isinf(r):
            return 0.0
        return float(r)


ENV_ID = "narrow-safe-v0"        # or "narrow-street-v0"
MODEL_PATH = "./models/ppo_narrow_safe.zip"
RENDER_MODE = None               # "human" to visualize, None for faster eval
EPISODES = 500


def make_eval_env():
    env = gym.make(ENV_ID, render_mode=RENDER_MODE)

    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy", "heading"],
            "normalize": False,
            "vehicles_count": 10
        }
    })

    # Important: wrappers must match training order
    env = CleanObservation(env)
    env = FlattenObservation(env)   # (10,5) → (50,)
    env = CleanReward(env)

    return env

def plot_violin(rewards, save_path="ppo_narrow_violin.png"):
    plt.figure(figsize=(6, 4))
    plt.violinplot(rewards, showmedians=True)
    plt.title("PPO Narrow-Safe: Reward Distribution")
    plt.ylabel("Episode Reward")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate(model_path, episodes=10, render=False):
    print("Running PPO evaluation on narrow-safe v0")

    env = make_eval_env()
    model = PPO.load(model_path)

    rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        print(f"\n--- Episode {ep+1} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

            if render:
                env.render()

        rewards.append(total_reward)
        print(f"Episode {ep+1} Reward: {total_reward:.2f}")

    env.close()

    print("\n===== SUMMARY =====")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std dev:     {np.std(rewards):.2f}")
    print(f"Min/Max:     {np.min(rewards):.2f} / {np.max(rewards):.2f}")

    return rewards


if __name__ == "__main__":
    rewards = evaluate(MODEL_PATH, episodes=EPISODES, render=False)
    plot_violin(rewards)
