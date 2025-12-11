import os
import time
import gymnasium as gym
import highway_env
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ENV LIST. highway sampled more because merge/intersection produces more episodes because of shorter duration. This will balance out the env contributions
ENV_IDS = [
    "highway-v0", "highway-v0",
    "merge-v0", #"merge-v0",
    "intersection-v0", #"intersection-v0",
]
n_env = len(ENV_IDS)


class ProgressiveTrafficWrapper(gym.Wrapper):
    """
    Gradually increases the number of vehicles every N timesteps.
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

        env_name = self.env.unwrapped.__class__.__name__

        new_config = {
            "vehicles_count": self.current_vehicles, # for highway
            "initial_vehicle_count": self.current_vehicles, # for intersection
            # no way to change in merge

            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
            },

            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },
        }

        # ==========================================================
        # HIGHWAY CONFIG 
        # ==========================================================
        if env_name in ["HighwayEnv", "HighwayEnvFast"]:
            new_config.update({
                "reward_speed_range": [26, 30],
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True,
                    "target_speeds": [26, 28, 30], # Force faster speeds in training so agent learns to dodge/change lanes instead of slowing down and cruising at 20m/s
                },
            })

        # ==========================================================
        # INTERSECTION CONFIG 
        # ==========================================================
        if env_name == "IntersectionEnv":
            new_config.update({
                                                        
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True, # False by default in Intersection, but changed to True to make action space same for all envs. (Model will learn to ignore incompatible actions)
                    "target_speeds": [0, 4.5, 9], # default config for intersection, but have to mention explicitly again because setting lateral:True changes the action space
                },
            })

        # APPLY CONFIG UPDATE
        self.env.unwrapped.configure(new_config)

        obs, info = self.env.reset(**kwargs)
        return obs, info


class TrackVehicleCallback(BaseCallback):
    def __init__(self, envs, verbose=0):
        super().__init__(verbose)
        self.envs = envs
        self.last_vehicle_count = 0
        self.steps_since_reset = 0

    def _on_training_start(self):
        log_dir = self.logger.get_dir()
        print(f"🧠 Logging vehicle stats to TensorBoard at: {log_dir}")

    def _on_step(self):
        current_vehicles = self.envs.get_attr("current_vehicles")[0]
        if current_vehicles > self.last_vehicle_count:
            self.last_vehicle_count = current_vehicles
            self.steps_since_reset = 0
            print(f"🔁 Vehicles updated: (vehicles={current_vehicles})")
        self.steps_since_reset += 1
        self.logger.record("environment/vehicles_count", current_vehicles)
        return True


class PerEnvStatsCallback(BaseCallback):
    def __init__(self, env_ids, verbose=0):
        super().__init__(verbose)
        self.env_ids = env_ids
        self.num_envs = len(env_ids)
        self.ep_rewards = [[] for _ in range(self.num_envs)]
        self.ep_lengths = [[] for _ in range(self.num_envs)]

    def _on_step(self):
        infos = self.locals["infos"]
        for i, info in enumerate(infos):
            if "episode" in info:
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                env_name = self.env_ids[i]
                self.logger.record(f"{env_name}/ep_reward", reward)
                self.logger.record(f"{env_name}/ep_length", length)
        return True




def make_env(env_id):
    def _init():
        env = gym.make(env_id, disable_env_checker=False)

        env = ProgressiveTrafficWrapper(
            env,
            start_vehicles=4,
            increase_every=10000, # per env, so for 4 parallel envs, increase vehicles every 40000 total timesteps
            increment=2,
            max_vehicles=12,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        obs, info = env.reset()
        return env
    return _init



import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AttentionCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, num_heads=4):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            cnn_out = self.cnn(sample)
            _, C, H, W = cnn_out.shape

        self.seq_len = H * W
        self.embed_dim = C

        self.flatten = nn.Flatten(2, 3)
        self.transpose = lambda x: x.transpose(1, 2)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        x = self.flatten(x)
        x = self.transpose(x)
        attn_out, _ = self.attn(x, x, x)
        pooled = attn_out.mean(dim=1)
        return self.proj(pooled)




if __name__ == "__main__":

    env = SubprocVecEnv([make_env(eid) for eid in ENV_IDS])

    model_path = "ppo_all_0004.zip"
    policy_kwargs = dict(
        features_extractor_class=AttentionCNN, # Using Attention so that agent can easily differentiate between various environments, and also focus on vehicles ahead
        features_extractor_kwargs=dict(features_dim=256, num_heads=4),
        net_arch=[256, 256],
    )

    if os.path.exists(model_path):
        print(f"📦 Loading model from {model_path}")
        model = PPO.load(model_path, env=env, device="cuda")
        

    else:
        print("🆕 Creating new PPO model...")
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=7.5e-5, #lowered for stable learning, attention may cause high variance in beginning
            n_steps=1024, # frequent updates
            batch_size=128, # stable gradients
            n_epochs=8, # Reuse data but not overfit
            gamma=0.95, # make it greedy to favour speeding
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="ppo_all_0004",
            device="cuda",
            verbose=1,
        )

    print("🚀 Starting PPO training...")
    callback = [
        TrackVehicleCallback(env),
        PerEnvStatsCallback(ENV_IDS)
    ]

    model.learn(
        total_timesteps=900_000,
        reset_num_timesteps=False,
        tb_log_name="ppo_all_0004",
        callback=callback,
    )

    model.save(model_path)
    print(f"💾 Model saved to {model_path}")

    env.close()
