import os
import time
import gymnasium as gym
import highway_env
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import custom_env


ENV_IDS = ["Team7-v0", "Team7-v0", "Team7-v0"]
n_env = len(ENV_IDS)
# ==========================
# Custom Wrapper
# ==========================


class ProgressiveTrafficWrapper(gym.Wrapper):
    """
    Gradually increases the number of vehicles every N timesteps (safe version).
    """
    def __init__(self, env, start_vehicles=0, increase_every=50_000, increment=2, max_vehicles=40):
        super().__init__(env)
        self.timestep_counter = 0
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
                        
        # ----- Base config -----
        new_config = {
                
                "vehicles_count": self.current_vehicles,
                "reward_speed_range": [25,50],
                "potholes": {
                    "count": self.current_vehicles,
                }
            }
        

        self.env.unwrapped.configure(new_config)
        obs, info = self.env.reset(**kwargs)
        return obs, info


class ResetExplorationCallback(BaseCallback):
    """
    Resets exploration rate (epsilon) when vehicle count increases.
    Saves model at each vehicle increase.
    Logs everything to TensorBoard.
    """
    def __init__(self, envs, verbose=0):
        super().__init__(verbose)
        self.envs = envs
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
            self.last_vehicle_count = current_vehicles
            self.steps_since_reset = 0
            print(f"🔁 Reset exploration: (vehicles={current_vehicles})")
        self.steps_since_reset += 1
        self.logger.record("environment/vehicles_count", current_vehicles)
        return True



# ==========================
# Environment Factory
# ==========================

def make_env(env_id):
    def _init():
        env = gym.make(env_id, disable_env_checker=False)

        env = ProgressiveTrafficWrapper(
            env,
            start_vehicles=0,
            # increase_every=80000 // n_env,
            increase_every=20000,
            increment=3,
            max_vehicles=20,
        )
         
        env = gym.wrappers.RecordEpisodeStatistics(env)        
        obs, info = env.reset()
        return env
    return _init


# ==========================
# Main PPO Training Loop
# ==========================
if __name__ == "__main__":
    
    env = SubprocVecEnv([make_env(eid) for eid in ENV_IDS])
    env = VecFrameStack(env, n_stack=4) 


    # PPO policy network architecture
    model_path = "ppo_custom.zip"
    policy_kwargs = dict(
        #features_extractor_class=SmallCNN,
        #features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 128],   # small MLP after CNN
        #activation_fn=torch.nn.ReLU
    )
    # Load or create PPO model
    if os.path.exists(model_path):
        print(f"📦 Loading existing PPO model from {model_path} ...")
        model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("🆕 No existing PPO model found. Creating a new one ...")
        model = PPO(
            "MlpPolicy",
            #"CnnPolicy",
            env,            
            policy_kwargs=policy_kwargs,

            learning_rate=1e-5,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.25,
            ent_coef=0.05,
            # vf_coef=0.5,
            # max_grad_norm=0.5,
            tensorboard_log="ppo_custom",
            
            device="cuda", 
            verbose=1,
        )

    
    # ==========================
    # Train PPO Agent
    # ==========================
    print("🚀 Starting (or resuming) PPO training...")
    callback = [
        ResetExplorationCallback(env),
    ]

    model.learn(
        total_timesteps=500_000,
        reset_num_timesteps=False,
        tb_log_name="PPO_custom",
        callback=callback,
    )

    # Save model
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")

    # Clean up
    env.close()

