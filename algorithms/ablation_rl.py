"""
Ablation RL Environments & Model Factory
=========================================
Defines the five experimental ablation configurations for CAIR-UAV:
    - Group A: Standard PPO (No IG, No Memory, No Context)
    - Group B: Information Gain Only (IG On, Memory Off, Context Off)
    - Group C: Information Gain + Memory (IG On, Memory On, Context Off)
    - Group D: Information Gain + Memory + Context (Full CAIR-RL, re-trained or alias of E)
    - Group E: Existing Full CAIR-RL (Pre-trained reference checkpoint)

Ensures that the observation space, action space, sensor model,
hazard dynamics, and reward foundations remain strictly identical
across all ablation groups. Only the intended architectural features
are ablated.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAINING, TrainingConfig
from environment.disaster_env import DisasterResponseEnv
from environment.safety_wrapper import BatterySafetyWrapper


@dataclass(frozen=True)
class AblationConfig:
    """Explicit configuration definition for an ablation group."""
    group_id: str
    name: str
    description: str
    enable_information_gain: bool
    enable_mission_memory: bool
    enable_context_features: bool
    checkpoint_dir: str
    final_model_filename: str


# Canonical definition of the five ablation groups
ABLATION_GROUPS: Dict[str, AblationConfig] = {
    "group_a": AblationConfig(
        group_id="group_a",
        name="Group A (Standard PPO)",
        description="Baseline PPO with no information gain reward, no revisit memory decay, and no context features.",
        enable_information_gain=False,
        enable_mission_memory=False,
        enable_context_features=False,
        checkpoint_dir="models/standard",
        final_model_filename="standard_final.zip",
    ),
    "group_b": AblationConfig(
        group_id="group_b",
        name="Group B (IG Only)",
        description="PPO with Shannon entropy-based Information Gain reward, but no revisit memory and no context features.",
        enable_information_gain=True,
        enable_mission_memory=False,
        enable_context_features=False,
        checkpoint_dir="models/ablation/group_b_ig_only",
        final_model_filename="group_b_final.zip",
    ),
    "group_c": AblationConfig(
        group_id="group_c",
        name="Group C (IG + Memory)",
        description="PPO with Information Gain reward and Revisit-Aware Memory decay, but no context features.",
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=False,
        checkpoint_dir="models/ablation/group_c_ig_memory",
        final_model_filename="group_c_final.zip",
    ),
    "group_d": AblationConfig(
        group_id="group_d",
        name="Group D (Full CAIR-RL)",
        description="Full CAIR-RL configuration: Information Gain + Revisit Memory + 5 Context Features.",
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=True,
        checkpoint_dir="models/ablation/group_d_full_cair",
        final_model_filename="group_d_final.zip",
    ),
    "group_e": AblationConfig(
        group_id="group_e",
        name="Group E (Existing Full CAIR-RL)",
        description="Verified pre-trained reference checkpoint of Full CAIR-RL.",
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=True,
        checkpoint_dir="models/cair",
        final_model_filename="cair_final.zip",
    ),
}


def build_ablation_env(
    group_id: str,
    seed: int = 42,
    n_envs: int = 4,
    use_safety_wrapper: bool = True,
    safety_factor: float = 1.60,
) -> VecEnv:
    """Build a vectorized Gymnasium environment for a specific ablation group.

    Args:
        group_id: Key identifying the group ("group_a" through "group_e").
        seed: Random seed for initialization.
        n_envs: Number of parallel vectorized environments.
        use_safety_wrapper: Whether to apply BatterySafetyWrapper.
        safety_factor: Multiplier for battery safety return budget.

    Returns:
        Vectorized Gymnasium environment for Stable-Baselines3.
    """
    if group_id not in ABLATION_GROUPS:
        raise ValueError(f"Unknown ablation group: '{group_id}'. Choose from: {list(ABLATION_GROUPS.keys())}")

    cfg = ABLATION_GROUPS[group_id]

    def make_env(env_idx: int):
        def _init():
            env = DisasterResponseEnv(
                seed=seed + env_idx,
                enable_information_gain=cfg.enable_information_gain,
                enable_mission_memory=cfg.enable_mission_memory,
                enable_context_features=cfg.enable_context_features,
            )
            if use_safety_wrapper:
                env = BatterySafetyWrapper(env, safety_factor=safety_factor)
            return env
        return _init

    return make_vec_env(
        make_env(0),
        n_envs=n_envs,
        seed=seed,
    )


def create_ablation_ppo(
    env: VecEnv,
    config: TrainingConfig = TRAINING,
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """Create a PPO model using MultiInputPolicy matching CAIR-UAV hyperparameters.

    Args:
        env: Vectorized environment.
        config: TrainingConfig instance.
        tensorboard_log: Path for TensorBoard logs.

    Returns:
        Configured PPO model.
    """
    policy_kwargs = dict(
        net_arch=dict(pi=config.net_arch_pi, vf=config.net_arch_vf),
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=config.seed,
        verbose=1,
    )
    return model
