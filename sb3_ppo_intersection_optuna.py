import argparse
import time

import gymnasium as gym
import highway_env  # noqa: F401  # registers intersection-v1 and others

import optuna

from highway_env.envs import IntersectionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(action_type: str, obs_type: str):
    """
    Factory for intersection-v1 with a given action and observation type.

    action_type:
        - "DiscreteMetaAction"
        - "ContinuousAction"

    obs_type:
        - "lidar"  -> LidarObservation (vector)
        - "gray"   -> GrayscaleObservation (stacked frames)
    """

    def _init():
        # No rendering during training (especially with SubprocVecEnv)
        env = gym.make("intersection-v1", disable_env_checker=False)

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
                "observation_shape": (84, 84),
                "stack_size": 4,  # helps infer motion
                "weights": [0.2989, 0.5870, 0.1140],  # RGB -> gray
                "scaling": 1.0,
            }

        config = IntersectionEnv.default_config()
        config.update({
            "observation": obs_config,
            "action": {"type": action_type},
            # You can also tweak these for speed:
            # "simulation_frequency": 15,
            # "policy_frequency": 1,
            # "duration": 40,
            # "vehicles_count": 15,
            # "collision_reward": -5.0,
            # "high_speed_reward": 0.5,
            # "lane_change_reward": -0.05,
            # "reward_speed_range": [20, 30],
            # "right_lane_reward": 0.2,
            # "reward_acceleration": -0.05,
        })

        env.unwrapped.configure(config)
        env.reset()  # make sure config takes effect

        # Monitor records episode rewards/lengths for SB3
        env = Monitor(env)
        return env

    return _init


def build_policy(obs_type: str) -> str:
    # lidar: vector -> MlpPolicy
    # gray: image -> CnnPolicy
    return "CnnPolicy" if obs_type == "gray" else "MlpPolicy"


def optimize_ppo(trial: optuna.Trial, args, action_type: str) -> float:
    """
    Optuna objective function:
    - Sample PPO hyperparameters.
    - Train for a limited number of timesteps.
    - Evaluate and return mean reward.
    """

    # ---- Hyperparameter search space (fairly diverse) ----
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.5)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])

    # PPO requirement: batch_size <= n_steps * n_envs
    if batch_size > n_steps * args.num_envs:
        # Penalize invalid configs without failing the trial
        raise optuna.TrialPruned()

    # ---- Vectorized training env ----
    env = SubprocVecEnv(
        [make_env(action_type, args.obs) for _ in range(args.num_envs)]
    )

    policy = build_policy(args.obs)

    model = PPO(
        policy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        verbose=0,              # keep Optuna output clean
        tensorboard_log="tb_logs",
        device="auto",          # use GPU if available
    )

    # ---- Train for a relatively short time per trial ----
    model.learn(total_timesteps=args.trial_steps)

    # ---- Evaluate on a single (non-vectorized) env ----
    eval_env = make_env(action_type, args.obs)()
    mean_reward, _ = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )

    eval_env.close()
    env.close()

    # Optionally save trial model (comment out if you don't want many files)
    # model.save(f"ppo_intersection_trial_{trial.number}")

    # We want to MAXIMIZE mean reward
    return mean_reward


def main():
    parser = argparse.ArgumentParser()
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
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials (different hyperparameter configs).",
    )
    parser.add_argument(
        "--trial-steps",
        type=int,
        default=50_000,
        help="Training timesteps per Optuna trial (shorter than final training).",
    )
    parser.add_argument(
        "--final-steps",
        type=int,
        default=200_000,
        help="Final training timesteps with best hyperparameters.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per trial.",
    )
    args = parser.parse_args()

    # Map CLI choice to highway-env action type string
    action_type = "DiscreteMetaAction" if args.action == "discrete" else "ContinuousAction"

    print(
        f"Starting Optuna + PPO on intersection-v1 "
        f"(action={action_type}, obs={args.obs})..."
    )

    start = time.time()

    # ---- Run Optuna study ----
    study = optuna.create_study(direction="maximize", study_name="ppo_intersection")
    study.optimize(lambda trial: optimize_ppo(trial, args, action_type),
                   n_trials=args.n_trials)

    elapsed = time.time() - start
    print(f"Hyperparameter search done in {elapsed:.1f}s")
    print(f"Best mean reward: {study.best_value}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ---- Optional: train a final PPO model with best hyperparameters ----
    print("Training final PPO model with best hyperparameters...")

    best_params = study.best_params

    # Ensure batch_size is not too big for final training, just in case
    if best_params["batch_size"] > best_params["n_steps"] * args.num_envs:
        best_params["batch_size"] = best_params["n_steps"] * args.num_envs

    env = SubprocVecEnv(
        [make_env(action_type, args.obs) for _ in range(args.num_envs)]
    )
    policy = build_policy(args.obs)

    final_model = PPO(
        policy,
        env,
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=best_params["clip_range"],
        ent_coef=best_params["ent_coef"],
        vf_coef=best_params["vf_coef"],
        max_grad_norm=best_params["max_grad_norm"],
        n_epochs=best_params["n_epochs"],
        verbose=1,
        tensorboard_log="tb_logs",
        device="auto",
    )

    start_final = time.time()
    tb_name = f"ppo_intersection_best_{args.action}_{args.obs}"
    final_model.learn(total_timesteps=args.final_steps, tb_log_name=tb_name)
    final_elapsed = time.time() - start_final
    print(f"Final training done in {final_elapsed:.1f}s")

    save_name = f"ppo_intersection_best_{args.action}_{args.obs}"
    final_model.save(save_name)
    print(f"Final model saved as {save_name}.zip")

    env.close()


if __name__ == "__main__":
    main()
