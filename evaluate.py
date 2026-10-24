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
"""

import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXPERIMENT, RESULTS_DIR, MODELS_DIR, Action
from environment.disaster_env import DisasterResponseEnv
from algorithms.random_policy import RandomPolicy
from algorithms.heuristic import GreedyHeuristicPolicy
from algorithms.rule_based import RuleBasedPolicy


class RLAgentWrapper:
    """Wraps an SB3 model to match the baseline policy interface."""
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
                final_action = int(action)
                
                # 1. Check battery level from telemetry
                battery_norm = obs["telemetry"][0]
                
                # 2. Safety override if battery < 20%
                if battery_norm < 0.20:
                    uav_x_norm = float(obs["telemetry"][2])
                    uav_y_norm = float(obs["telemetry"][3])
                    
                    grid_width = env.grid_cfg.width
                    grid_height = env.grid_cfg.height
                    
                    uav_col = int(round(uav_x_norm * (grid_width - 1)))
                    uav_row = int(round(uav_y_norm * (grid_height - 1)))
                    uav_col = np.clip(uav_col, 0, grid_width - 1)
                    uav_row = np.clip(uav_row, 0, grid_height - 1)
                    
                    base_row, base_col = env.grid_cfg.base_position
                    
                    if uav_row == base_row and uav_col == base_col:
                        # Already at base, just hover
                        final_action = Action.HOVER_AND_SCAN
                    else:
                        # 3. Force action toward base
                        dy = base_row - uav_row
                        dx = base_col - uav_col
                        
                        if abs(dy) >= abs(dx):
                            final_action = Action.MOVE_NORTH if dy < 0 else Action.MOVE_SOUTH
                        else:
                            final_action = Action.MOVE_WEST if dx < 0 else Action.MOVE_EAST
                            
                obs, reward, terminated, truncated, info = env.step(final_action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            res = {
                "policy_name": self.name,
                "episode_idx": i,
                "seed": seed,
                "total_reward": total_reward,
                "survivors_found": info.get("survivors_found", 0),
                "total_survivors": info.get("total_survivors", 0),
                "area_coverage_pct": info.get("area_coverage_pct", 0.0),
                "total_information_gain": info.get("total_information_gain", 0.0),
                "battery_remaining": info.get("battery_remaining", 0.0),
                "time_steps_used": steps,
                "mission_success": info.get("mission_success", False),
                "hazard_encounters": info.get("hazard_encounters", 0)
            }
            results.append(res)
        return results


def evaluate_models():
    print("==================================================")
    print("CAIR-UAV Evaluation Suite")
    print("==================================================")

    # Initialize Environment
    env = DisasterResponseEnv(
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=True
    )

    # Collect Policies
    policies = [
        RandomPolicy(),
        GreedyHeuristicPolicy(),
        RuleBasedPolicy()
    ]

    # Load RL models if they exist
    std_path = os.path.join(MODELS_DIR, "standard", "standard_final.zip")
    cair_path = os.path.join(MODELS_DIR, "cair", "cair_final.zip")

    if os.path.exists(std_path):
        print(f"Loading Standard RL from {std_path}...")
        std_model = PPO.load(std_path)
        policies.append(RLAgentWrapper("Standard RL", std_model))
    else:
        print("[!] Standard RL model not found. Skipping.")

    if os.path.exists(cair_path):
        print(f"Loading CAIR-RL from {cair_path}...")
        cair_model = PPO.load(cair_path)
        policies.append(RLAgentWrapper("CAIR-RL", cair_model))
    else:
        print("[!] CAIR-RL model not found. Skipping.")

    # Run Evaluation
    all_results = []
    num_episodes = min(20, EXPERIMENT.num_episodes) # Evaluate on 20 for speed by default
    seeds = list(EXPERIMENT.seeds) * (num_episodes // len(EXPERIMENT.seeds) + 1)

    print(f"\nEvaluating {len(policies)} policies over {num_episodes} episodes each...")
    
    for policy in policies:
        print(f"  -> Running {policy.name}...")
        results = policy.run_multiple_episodes(env, num_episodes=num_episodes, seeds=seeds)
        all_results.extend(results)

    # Save to CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print Summary Statistics
    print("\n--- Summary Statistics (Mean +/- Std) ---")
    summary = df.groupby("policy_name").agg({
        "survivors_found": ["mean", "std"],
        "area_coverage_pct": ["mean", "std"],
        "total_reward": ["mean", "std"],
        "mission_success": "mean"
    }).round(2)
    
    print(summary)
    print("==================================================")

if __name__ == "__main__":
    evaluate_models()
