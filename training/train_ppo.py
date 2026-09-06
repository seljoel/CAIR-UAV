"""
PPO Training Script
====================
Main execution script for training the PPO models (Standard or CAIR-RL).

This script:
    1. Instantiates the vectorized Gymnasium environment.
    2. Builds the Stable-Baselines3 PPO model with `MultiInputPolicy`.
    3. Attaches the custom `MetricsLoggingCallback` for TensorBoard.
    4. Executes the training loop for the configured number of timesteps.
    5. Saves the final trained model to the `models/` directory.

Usage:
    python training/train_ppo.py --model_type cair
    python training/train_ppo.py --model_type standard

Author: CAIR-UAV Research Team
"""

import argparse
import os
import sys

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAINING, LOGS_DIR, MODELS_DIR
from algorithms.standard_rl import build_standard_rl_env, create_standard_ppo
from algorithms.proposed_cair_rl import build_cair_rl_env, create_cair_ppo
from training.callbacks import MetricsLoggingCallback


def train(model_type: str) -> None:
    """Train the specified PPO model.

    Args:
        model_type: "standard" or "cair".
    """
    print(f"==================================================")
    print(f"Starting Training for Model: {model_type.upper()}")
    print(f"==================================================")

    # Setup directories
    tb_log_dir = os.path.join(LOGS_DIR, model_type)
    model_save_dir = os.path.join(MODELS_DIR, model_type)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # 1. Build Vectorized Environment
    print("Building vectorized environments...")
    if model_type == "cair":
        env = build_cair_rl_env(seed=TRAINING.seed, n_envs=4)
    elif model_type == "standard":
        env = build_standard_rl_env(seed=TRAINING.seed, n_envs=4)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 2. Build Model
    print("Instantiating PPO with MultiInputPolicy...")
    if model_type == "cair":
        model = create_cair_ppo(env, config=TRAINING, tensorboard_log=tb_log_dir)
    else:
        model = create_standard_ppo(env, config=TRAINING, tensorboard_log=tb_log_dir)

    # 3. Setup Callbacks
    metrics_callback = MetricsLoggingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, TRAINING.save_freq // 4),  # Divide by n_envs
        save_path=model_save_dir,
        name_prefix=f"{model_type}_ppo_model"
    )
    callbacks = CallbackList([metrics_callback, checkpoint_callback])

    # 4. Train
    print(f"Training for {TRAINING.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=TRAINING.total_timesteps,
        callback=callbacks,
        tb_log_name="run",
        reset_num_timesteps=True,
    )

    # 5. Save Final Model
    final_model_path = os.path.join(model_save_dir, f"{model_type}_final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}.zip")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for CAIR-UAV")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["standard", "cair"],
        default="cair",
        help="Which model variant to train (standard or full cair)"
    )
    args = parser.parse_args()

    train(args.model_type)
