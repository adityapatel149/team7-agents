import os
import time
import gymnasium as gym
import highway_env
from numpy import save
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


# ==========================
# Custom Wrapper
# ==========================
# class ProgressiveTrafficWrapper(gym.Wrapper):
#     """
#     Gradually increases the number of vehicles every N episodes.
#     """
#     def __init__(self, env, start_vehicles=0, increase_every=500, increment=2, max_vehicles=40):
#         super().__init__(env)
#         self.episode_counter = 0
#         self.base_vehicles = start_vehicles
#         self.increase_every = increase_every
#         self.increment = increment
#         self.max_vehicles = max_vehicles
#         self.current_vehicles = start_vehicles

#     def reset(self, **kwargs):
#         # Count episodes
#         self.episode_counter += 1

#         # Compute new vehicle count
#         new_vehicle_count = min(
#             self.base_vehicles + (self.episode_counter // self.increase_every) * self.increment,
#             self.max_vehicles,
#         )

#         if new_vehicle_count != self.current_vehicles:
#             print(f"[Episode {self.episode_counter}] Increasing vehicles_count from {self.current_vehicles} → {new_vehicle_count}")
#             self.current_vehicles = new_vehicle_count

#         # Reconfigure environment dynamically
#         new_config = {
#             "observation": {
#                 "type": "LidarObservation",
#                 "cells": 64,
#                 "maximum_range": 200,
#                 "normalize": True,
#             },
#             #"action": {"type": "ContinuousAction"},
#             "vehicles_count": new_vehicle_count,
#         }

#         self.env.unwrapped.configure(new_config)
#         obs, info = self.env.reset(**kwargs)
#         return obs, info


class ProgressiveTrafficWrapper(gym.Wrapper):
    """
    Gradually increases the number of vehicles every N timesteps (safe version).
    """
    def __init__(self, env, start_vehicles=0, increase_every=50_000, increment=2, max_vehicles=40):
        super().__init__(env)
        self.timestep_counter = 0
        self.base_vehicles = start_vehicles
        self.increase_every = increase_every
        self.increment = increment
        self.max_vehicles = max_vehicles
        self.current_vehicles = start_vehicles
        self.last_increase_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.timestep_counter += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Decide if vehicle count should increase
        if self.timestep_counter - self.last_increase_step >= self.increase_every:
            new_vehicle_count = min(
                self.current_vehicles + self.increment,
                self.max_vehicles,
            )
            if new_vehicle_count != self.current_vehicles:
                print(f"[Timestep {self.timestep_counter}] Increasing vehicles_count "
                      f"from {self.current_vehicles} → {new_vehicle_count}")
                self.current_vehicles = new_vehicle_count
                self.last_increase_step = self.timestep_counter

        # Apply new config safely at reset
        new_config = {
            "observation": {
                "type": "LidarObservation",
                "cells": 128,
                "maximum_range": 64,
                "normalise": True,
            },
            "vehicles_count": self.current_vehicles,
        }
        self.env.unwrapped.configure(new_config)
        obs, info = self.env.reset(**kwargs)
        return obs, info



# ==========================
# Custom Callback (Clean Logger)
# ==========================
from stable_baselines3.common.callbacks import BaseCallback


class ResetExplorationCallback(BaseCallback):
    """
    Resets exploration rate (epsilon) when vehicle count increases.
    Saves model at each vehicle increase.
    Logs everything to TensorBoard.
    """
    def __init__(self, envs, reset_eps=0.5, min_eps=0.05, decay_steps=50_000, verbose=0):
        super().__init__(verbose)
        self.envs = envs
        self.reset_eps = reset_eps
        self.min_eps = min_eps
        self.decay_steps = decay_steps
        self.last_vehicle_count = 0
        self.steps_since_reset = 0


    def _on_training_start(self):
        log_dir = self.logger.get_dir()
        print(f"🧠 Logging exploration + vehicle stats to TensorBoard at: {log_dir}")

    def _on_step(self):
        # Get current vehicle count from one of the parallel envs
        current_vehicles = self.envs.get_attr("current_vehicles")[0]

        # Detect vehicle count increase
        if current_vehicles > self.last_vehicle_count:
            self.model.exploration_rate = self.reset_eps
            self.last_vehicle_count = current_vehicles
            self.steps_since_reset = 0
            print(f"🔁 Reset exploration: ε={self.model.exploration_rate:.2f} (vehicles={current_vehicles})")

        # Linear decay of epsilon
        self.steps_since_reset += 1
        frac = min(self.steps_since_reset / self.decay_steps, 1.0)
        self.model.exploration_rate = self.reset_eps - frac * (self.reset_eps - self.min_eps)

        # Log to TensorBoard
        self.logger.record("exploration/epsilon", self.model.exploration_rate)
        self.logger.record("environment/vehicles_count", current_vehicles)
        return True



# ==========================
# Environment Factory
# ==========================
def make_env():
    def _init():
        env = gym.make("highway-v0", disable_env_checker=False)
        env = ProgressiveTrafficWrapper(
            env,
            start_vehicles=0,
            increase_every=40000//4, # divide by number of env
            increment=2,
            max_vehicles=50,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        obs, info = env.reset()
        return env
    return _init


# ==========================
# Main Training Loop
# ==========================
if __name__ == "__main__":
    model_path = "dqn_highway_lidar_vehicles2.zip"

    # Create parallel environments
    env = SubprocVecEnv([make_env() for _ in range(4)])

    # Model hyperparameters
    policy_kwargs = dict(net_arch=[64, 64])

    # Load or create model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} ...")
        model = DQN.load(model_path, env=env, device="cpu")
    else:
        print("No existing model found. Creating a new one ...")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-5,
            buffer_size=100_000,
            batch_size=128,
            target_update_interval=2_000,
            policy_kwargs=policy_kwargs,
            device="cpu",
            tensorboard_log="dqn_tensorboard_curriculum",
            exploration_initial_eps=0.0,  # 👈 disable SB3’s internal schedule
            exploration_final_eps=0.0,    # 👈
            exploration_fraction=0.0,     # 👈
            verbose=1,
        )
        model.exploration_schedule = lambda _: model.exploration_rate  # 👈 critical line


    # Initialize callback
    callback = ResetExplorationCallback(env, reset_eps=0.5, min_eps=0.05, decay_steps=10_000//4) # dividing by num of env

    print("🚀 Starting (or resuming) DQN training...")
    start = time.time()

    model.learn(
        total_timesteps=1_000_000,
        reset_num_timesteps=False,
        tb_log_name="DQN_curriculum",
        callback=callback
    )

    print(f"✅ Training complete in {time.time() - start:.2f} seconds")

    model.save(model_path)
    print(f"💾 Model saved as {model_path}")

    env.close()
