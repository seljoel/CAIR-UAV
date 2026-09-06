"""
Random Policy Baseline
========================
Uniform random action selection for the UAV disaster response environment.

This serves as the weakest baseline — the agent selects each of the 5
actions with equal probability (1/5), regardless of the observation.
It establishes the lower bound of performance against which all other
policies are compared.

Usage:
    policy = RandomPolicy(seed=42)
    action = policy.select_action(observation)

    # Or run a full episode:
    results = policy.run_episode(env, seed=42)

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Action


class RandomPolicy:
    """Uniform random action policy baseline.

    Selects each of the 5 discrete actions with equal probability.
    The observation is completely ignored.

    Attributes:
        name: Human-readable policy name.
        rng: NumPy random number generator.
    """

    name: str = "Random Policy"

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the random policy.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Select a uniformly random action.

        Args:
            observation: The current observation (ignored).
            info: Optional info dict from the environment (ignored).

        Returns:
            Random action index in [0, 4].
        """
        return int(self.rng.integers(0, Action.NUM_ACTIONS))

    def run_episode(
        self,
        env,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a complete episode with the random policy.

        Args:
            env: Gymnasium-compatible environment instance.
            seed: Optional seed for both env and policy.
            max_steps: Override for maximum steps (uses env default if None).

        Returns:
            Dict containing episode metrics:
                total_reward, survivors_found, area_coverage_pct,
                total_information_gain, battery_remaining, time_steps_used,
                mission_success, hazard_encounters, unique_cells_visited,
                trajectory (list of actions).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        actions_taken: List[int] = []
        step_count = 0

        while not done:
            action = self.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions_taken.append(action)
            step_count += 1
            done = terminated or truncated

            if max_steps is not None and step_count >= max_steps:
                break

        return {
            "policy_name": self.name,
            "total_reward": total_reward,
            "survivors_found": info.get("survivors_found", 0),
            "total_survivors": info.get("total_survivors", 0),
            "area_coverage_pct": info.get("area_coverage_pct", 0.0),
            "total_information_gain": info.get("total_information_gain", 0.0),
            "battery_remaining": info.get("battery_remaining", 0.0),
            "time_steps_used": step_count,
            "mission_success": info.get("mission_success", False),
            "hazard_encounters": info.get("hazard_encounters", 0),
            "unique_cells_visited": info.get("unique_cells_visited", 0),
            "trajectory": actions_taken,
        }

    def run_multiple_episodes(
        self,
        env,
        num_episodes: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Run multiple episodes and collect results.

        Args:
            env: Environment instance.
            num_episodes: Number of episodes to run.
            seeds: Optional list of seeds (one per episode). If None,
                uses sequential seeds starting from 0.

        Returns:
            List of episode result dicts.
        """
        if seeds is None:
            seeds = list(range(num_episodes))

        results = []
        for i, seed in enumerate(seeds[:num_episodes]):
            result = self.run_episode(env, seed=seed)
            result["episode_idx"] = i
            result["seed"] = seed
            results.append(result)

        return results

    def __repr__(self) -> str:
        return f"RandomPolicy()"
