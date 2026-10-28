"""
Model Evaluation Script
========================
Evaluates all algorithms (Baselines + RL) on a fixed set of test seeds.
Computes and saves performance metrics to a CSV file.

Algorithms evaluated:
1. Random
2. Greedy Heuristic
3. Rule-Based
4. Standard RL (if model exists)
5. Proposed CAIR-RL (if model exists)

Safety behaviour (applied to ALL policies):
    The environment is wrapped with BatterySafetyWrapper (safety_factor=1.30).
    This overrides any action with RETURN_TO_BASE when remaining battery is
    <= 130% of the energy needed to return to base (distance-aware, 30% margin).
    The same wrapper is used in training, eliminating the train/eval gap.
"""

import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXPERIMENT, RESULTS_DIR, MODELS_DIR, Action
from environment.disaster_env import DisasterResponseEnv
from environment.safety_wrapper import BatterySafetyWrapper
from algorithms.random_policy import RandomPolicy
from algorithms.heuristic import GreedyHeuristicPolicy
from algorithms.rule_based import RuleBasedPolicy


class RLAgentWrapper:
    """Wraps an SB3 model to match the baseline policy interface.

    The environment is expected to already be wrapped with BatterySafetyWrapper,
    so this class contains NO battery override logic -- that is the wrapper's job.
    """
    def __init__(self, name: str, model: PPO):
        self.name = name
        self.model = model

    def run_multiple_episodes(self, env, num_episodes, seeds):
        results = []
        for i, seed in enumerate(seeds[:num_episodes]):
            obs, info = env.reset(seed=seed)
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # BatterySafetyWrapper on the env overrides action if battery low.
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                steps += 1
                done = terminated or truncated

            res = {
                "policy_name":            self.name,
                "episode_idx":            i,
                "seed":                   seed,
                "total_reward":           total_reward,
                "survivors_found":        info.get("survivors_found", 0),
                "total_survivors":        info.get("total_survivors", 0),
                "area_coverage_pct":      info.get("area_coverage_pct", 0.0),
                "total_information_gain": info.get("total_information_gain", 0.0),
                "battery_remaining":      info.get("battery_remaining", 0.0),
                "time_steps_used":        steps,
                "mission_success":        info.get("mission_success", False),
                "hazard_encounters":      info.get("hazard_encounters", 0),
                "at_base":                info.get("at_base", False),
                "safety_overrides":       info.get("safety_override_count", 0),
            }
            results.append(res)
        return results


def evaluate_models():
    print("==================================================")
    print("CAIR-UAV Evaluation Suite")
    print("==================================================")

    # Build wrapped environment
    _base_env = DisasterResponseEnv(
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=True,
    )
    env = BatterySafetyWrapper(_base_env, safety_factor=1.60)
    print(f"[+] Safety wrapper: factor={env.safety_factor} "
          f"(60% margin above bare return cost)")

    # Policies
    policies = [
        RandomPolicy(),
        GreedyHeuristicPolicy(),
        RuleBasedPolicy(),
    ]

    std_path  = os.path.join(MODELS_DIR, "standard", "standard_final.zip")
    cair_path = os.path.join(MODELS_DIR, "cair",     "cair_final.zip")

    if os.path.exists(std_path):
        print(f"Loading Standard RL from {std_path}...")
        policies.append(RLAgentWrapper("Standard RL", PPO.load(std_path)))
    else:
        print("[!] Standard RL model not found. Skipping.")

    if os.path.exists(cair_path):
        print(f"Loading CAIR-RL from {cair_path}...")
        policies.append(RLAgentWrapper("CAIR-RL", PPO.load(cair_path)))
    else:
        print("[!] CAIR-RL model not found. Skipping.")

    num_episodes = min(20, EXPERIMENT.num_episodes)
    seeds = list(EXPERIMENT.seeds) * (num_episodes // len(EXPERIMENT.seeds) + 1)

    print(f"\nEvaluating {len(policies)} policies over {num_episodes} episodes each...")
    all_results = []
    for policy in policies:
        print(f"  -> Running {policy.name}...")
        all_results.extend(
            policy.run_multiple_episodes(env, num_episodes=num_episodes, seeds=seeds)
        )

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\n--- Summary Statistics (Mean +/- Std) ---")
    summary = df.groupby("policy_name").agg({
        "survivors_found":        ["mean", "std"],
        "area_coverage_pct":      ["mean", "std"],
        "total_reward":           ["mean", "std"],
        "battery_remaining":      "mean",
        "mission_success":        "mean",
        "at_base":                "mean",
        "safety_overrides":       "mean",
    }).round(2)
    print(summary)
    print("==================================================")


if __name__ == "__main__":
    evaluate_models()
