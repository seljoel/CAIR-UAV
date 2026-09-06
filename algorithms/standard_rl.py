"""
Standard RL Baseline (PPO)
===========================
A standard Deep Reinforcement Learning baseline using Proximal Policy
Optimization (PPO).

This agent uses the `MultiInputPolicy` to process the Dict observation space,
but the environment it trains in has Information Gain, Mission Memory, and
Context Features DISABLED via ablation toggles.

It relies entirely on spatial observations (UAV position, belief map, hazards)
and base telemetry (battery, time, position) to learn a policy, without the
benefit of structured context or exploration incentives.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TRAINING, TrainingConfig
from environment.disaster_env import DisasterResponseEnv


def build_standard_rl_env(
    seed: int = 42,
    n_envs: int = 4,
) -> VecEnv:
    """Build a vectorized environment with ablation toggles for Standard RL.

    Disables:
        - Information Gain Reward (enable_information_gain=False)
        - Mission Memory (enable_mission_memory=False)
        - Context Features (enable_context_features=False)

    Args:
        seed: Random seed.
        n_envs: Number of parallel environments.

    Returns:
        Vectorized Gymnasium environment for SB3.
    """
    def make_env():
        def _init():
            env = DisasterResponseEnv(
                seed=seed,
                enable_information_gain=False,
                enable_mission_memory=False,
                enable_context_features=False,
            )
            return env
        return _init

    return make_vec_env(make_env(), n_envs=n_envs, seed=seed)


def create_standard_ppo(
    env: VecEnv,
    config: TrainingConfig = TRAINING,
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """Instantiate the Standard PPO model using MultiInputPolicy.

    Args:
        env: Vectorized environment.
        config: Training hyperparameters.
        tensorboard_log: Path to tensorboard log directory.

    Returns:
        Configured SB3 PPO model.
    """
    policy_kwargs = dict(
        #features_extractor_kwargs=dict(features_dim=config.cnn_features_dim),
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
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        verbose=1,
    )

    return model
