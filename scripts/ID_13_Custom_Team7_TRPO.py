"""
TRPO Training for Team7-v0 Custom Environment
Uses LidarObservation with parallel environments for faster training
"""

import gymnasium as gym
import custom_env
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import time
import os


def make_env(env_id: int):
    def _init():
        
        env = gym.make("Team7-v0", disable_env_checker=False)
        
        # Configure to EXACTLY match custom env config
        config = {
            "observation": {
                "type": "LidarObservation",
                "cells": 36,  # 
                "maximum_range": 100.0,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "vehicle_class": "custom_env.vehicle.customvehicle.CustomVehicle",  # ADDED custom vehicle
            },
            "lanes_count": 7, #reduced to 7 from 8
            "vehicles_count": 20, #reduced to 20 from 40
            "controlled_vehicles": 1,
            "initial_lane_id": None,  # Random initial lane
            "duration": 30,  # [s]
            "ego_spacing": 2,  # Spacing for ego vehicle
            "vehicles_density": 3,  # Vehicle density parameter
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "lane_change_reward": 0,
            "high_speed_reward": 0.4,
            "speed_limit": 50,
            "reward_speed_range": [25, 50],
            "normalize_reward": True,
            "offroad_terminal": False,  # Don't terminate on offroad
            "other_vehicles_type": "highway_env.vehicle.behavior.SuddenBrakingVehicle",  # ADDED custom vehicle type
            "anomaly_interval": 3,  # Ghost vehicle anomaly interval
            "potholes": {
                "enabled": True,
                "count": 15, #reduced to 15 from 20
                "spawn_ahead_min": 20.0,
                "spawn_ahead_max": 1000.0,
            }
        }
        
        env.unwrapped.configure(config)
        env.reset() 
        
        
        env = Monitor(env)
        
        return env
    
    return _init


def main():
    print("TRPO Training on Team7-v0 Custom Environment")
    print("Environment Features:")
    
    # Configuration
    NUM_ENVS = 4 
    TOTAL_TIMESTEPS = 200_000
    MODEL_NAME = "trpo_team7_lidar"
    TB_LOG_DIR = "./tb_logs"
    
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Parallel Environments: {NUM_ENVS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Model Name: {MODEL_NAME}")
    print(f"  TensorBoard Log: {TB_LOG_DIR}")
    
    # Create vectorized environment
    print(f"\nCreating {NUM_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    print("✓ Environments created successfully")
    
    # Try to load existing model
    model_path = f"./models/{MODEL_NAME}.zip"
    model = None
    
    try:
        print(f"\nAttempting to load existing model from {model_path}...")
        model = TRPO.load(model_path, env=env, device="cpu")
        print("Model loaded successfully! Continuing training...")
    except FileNotFoundError:
        print("No existing model found. Creating new model...")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model...")
    
    # Create new model if not loaded
    if model is None:
        print("\nCreating TRPO model...")
        model = TRPO(
            "MlpPolicy",  # Lidar is vector, so MLP
            env,
            learning_rate=5e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.98,
            cg_max_steps=20,
            cg_damping=0.1,
            line_search_shrinking_factor=0.8,
            line_search_max_iter=10,
            n_critic_updates=10,
            target_kl=0.01,
            normalize_advantage=True,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),
            verbose=1,
            device="cpu",
            tensorboard_log=TB_LOG_DIR
        )
        print("Model created successfully")
    
    # Train
    print("Starting TRPO training...")
    print("Monitor training progress:")
    print(f"  tensorboard --logdir={TB_LOG_DIR}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="TRPO_Team7_Lidar",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
    
    elapsed = time.time() - start_time
    
    # Save model
    save_path = f"./models/{MODEL_NAME}"
    model.save(save_path)
    
    print("Training Complete!")
    print(f"Training time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"Model saved as: {save_path}.zip")
    
    # Cleanup
    env.close()
    


if __name__ == "__main__":
    main()