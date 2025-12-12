
import narrow_safe_env      # registers narrow-safe-v0
import narrow_street        # registers narrow-street-v0

import numpy as np
import gymnasium as gym

class CleanObservation(gym.ObservationWrapper):
    """Sanitize NaN, +inf, -inf values in observations."""
    def observation(self, obs):
        return np.nan_to_num(obs, nan=0.0, posinf=1000.0, neginf=-1000.0)


class CleanReward(gym.RewardWrapper):
    """Prevent NaN or infinite rewards from breaking PPO."""
    def reward(self, r):
        if np.isnan(r) or np.isinf(r):
            return 0.0
        return float(r)


from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import os
import time


ENV_ID = "narrow-safe-v0"      # You can change to narrow-street-v0
NUM_ENVS = 4                   # Good for Windows multiprocessing
TOTAL_TIMESTEPS = 1_000_000    # Enough for good behavior


def make_env(idx):
    def _init():
        env = gym.make(ENV_ID)

        # Force Kinematics observation (matrix shape)
        env.unwrapped.configure({
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "heading"],
                "normalize": True,
                "vehicles_count": 10
            }
        })

        # ----- SANITIZE OBSERVATION FIRST -----
        env = CleanObservation(env)

        # ----- FLATTEN (10,5) → (50,) -----
        env = FlattenObservation(env)

        # ----- SANITIZE REWARD -----
        env = CleanReward(env)

        env = Monitor(env)

        obs, _ = env.reset()
        print(f"[Env {idx}] Final obs shape: {obs.shape}")  # Should be (50,)
        return env

    return _init


def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tb_logs_ppo_narrow", exist_ok=True)

    print(" PPO Training – narrow-safe-v0")

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./tb_logs_ppo_narrow"
    )

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    end = (time.time() - start) / 60

    print(f"\nTraining complete! Time: {end:.1f} minutes")
    model.save("./models/ppo_narrow_safe")
    print("Model saved → ./models/ppo_narrow_safe.zip")

    env.close()


if __name__ == "__main__":
    main()
