"""
Ablation Training Script
========================
Trains missing ablation configurations from scratch using standard PPO hyperparameters.

Usage:
    python training/train_ablation.py --group group_b --timesteps 100000
    python training/train_ablation.py --group group_c --timesteps 100000

Author: CAIR-UAV Research Team
"""

import argparse
import os
import sys
import time
import json
from dataclasses import asdict

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAINING, PROJECT_ROOT, LOGS_DIR, MODELS_DIR
from algorithms.ablation_rl import ABLATION_GROUPS, build_ablation_env, create_ablation_ppo
from training.callbacks import MetricsLoggingCallback


def train_ablation_group(
    group_id: str,
    total_timesteps: int = 100_000,
    seed: int = 42,
    n_envs: int = 4,
) -> str:
    """Train a specific ablation model from scratch.

    Args:
        group_id: One of 'group_b', 'group_c', 'group_d'.
        total_timesteps: Total environment timesteps.
        seed: Training random seed.
        n_envs: Parallel environments.

    Returns:
        Path to the saved final model zip.
    """
    if group_id not in ABLATION_GROUPS:
        raise ValueError(f"Unknown ablation group: {group_id}")

    cfg = ABLATION_GROUPS[group_id]
    print("=" * 60)
    print(f"Starting Training for Ablation: {cfg.name}")
    print(f"Features: IG={cfg.enable_information_gain}, Memory={cfg.enable_mission_memory}, Context={cfg.enable_context_features}")
    print(f"Budget: {total_timesteps} timesteps | Seed: {seed} | Parallel Envs: {n_envs}")
    print("=" * 60)

    # Directories
    model_save_dir = os.path.join(PROJECT_ROOT, cfg.checkpoint_dir)
    checkpoints_dir = os.path.join(model_save_dir, "checkpoints")
    tb_log_dir = os.path.join(LOGS_DIR, "ablation", group_id)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # 1. Build Vectorized Environment with Safety Wrapper
    print("[1/4] Building vectorized ablation environments...")
    env = build_ablation_env(
        group_id=group_id,
        seed=seed,
        n_envs=n_envs,
        use_safety_wrapper=True,
        safety_factor=1.60,
    )

    # 2. Build Model
    print("[2/4] Instantiating PPO model with MultiInputPolicy...")
    model = create_ablation_ppo(env, config=TRAINING, tensorboard_log=tb_log_dir)

    # 3. Setup Callbacks
    metrics_callback = MetricsLoggingCallback()
    save_freq = max(1, 25_000 // n_envs)  # Checkpoint every 25,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoints_dir,
        name_prefix=f"{group_id}_ppo",
    )
    callbacks = CallbackList([metrics_callback, checkpoint_callback])

    # 4. Train
    print(f"[3/4] Training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="run",
        reset_num_timesteps=True,
    )
    duration_sec = time.time() - start_time
    fps = total_timesteps / max(duration_sec, 1e-3)
    print(f"[3/4] Training completed in {duration_sec:.1f}s ({duration_sec/60:.2f} min) at {fps:.1f} FPS.")

    # 5. Save Final Model
    print("[4/4] Saving final checkpoint and metadata...")
    final_model_name = os.path.splitext(cfg.final_model_filename)[0]
    final_model_path = os.path.join(model_save_dir, final_model_name)
    model.save(final_model_path)
    final_zip = final_model_path + ".zip"

    # Save Configuration Snapshot
    config_snapshot = {
        "group_id": cfg.group_id,
        "name": cfg.name,
        "description": cfg.description,
        "features": {
            "enable_information_gain": cfg.enable_information_gain,
            "enable_mission_memory": cfg.enable_mission_memory,
            "enable_context_features": cfg.enable_context_features,
        },
        "ppo_hyperparameters": {
            "learning_rate": TRAINING.learning_rate,
            "n_steps": TRAINING.n_steps,
            "batch_size": TRAINING.batch_size,
            "n_epochs": TRAINING.n_epochs,
            "gamma": TRAINING.gamma,
            "gae_lambda": TRAINING.gae_lambda,
            "clip_range": TRAINING.clip_range,
            "ent_coef": TRAINING.ent_coef,
            "vf_coef": TRAINING.vf_coef,
            "max_grad_norm": TRAINING.max_grad_norm,
            "net_arch_pi": TRAINING.net_arch_pi,
            "net_arch_vf": TRAINING.net_arch_vf,
        },
        "training_run": {
            "total_timesteps": total_timesteps,
            "seed": seed,
            "n_envs": n_envs,
            "duration_seconds": round(duration_sec, 2),
            "fps": round(fps, 2),
            "final_model_path": final_zip,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    snapshot_path = os.path.join(model_save_dir, "config_snapshot.json")
    with open(snapshot_path, "w") as f:
        json.dump(config_snapshot, f, indent=4)
    print(f"Snapshot written: {snapshot_path}")
    print(f"Model saved: {final_zip}")

    env.close()
    return final_zip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAIR-UAV Ablation Models")
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        choices=["group_b", "group_c", "group_d"],
        help="Ablation group identifier ('group_b', 'group_c', 'group_d')",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    train_ablation_group(
        group_id=args.group,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )
