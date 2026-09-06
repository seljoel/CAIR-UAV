"""
Ablation Study Experiment
==========================
Evaluates the contribution of each architectural component (Information Gain,
Mission Memory, Context Features) to the overall CAIR-UAV performance.

Note: This script requires you to have trained models for each ablation group:
- Group A: Standard PPO (No IG, No Memory, No Context)
- Group B: PPO + IG only
- Group C: PPO + IG + Memory
- Group D: PPO + IG + Memory + Context (Full CAIR-UAV)

Since training takes time, this script uses a mock evaluation structure
that relies on the `evaluate.py` wrapper for the loaded models.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EXPERIMENT, RESULTS_DIR, PLOTS_DIR, MODELS_DIR
from environment.disaster_env import DisasterResponseEnv
from evaluate import RLAgentWrapper

def run_ablation_study():
    print("==================================================")
    print("CAIR-UAV Ablation Study")
    print("==================================================")
    
    ablation_models = {
        "Group A (Standard RL)": os.path.join(MODELS_DIR, "standard", "standard_final.zip"),
        "Group E (Full CAIR-RL)": os.path.join(MODELS_DIR, "cair", "cair_final.zip")
    }
    
    # Note: Groups B, C, D require you to train intermediate models.
    # For this script, we'll plot what's available.
    
    policies = []
    
    # Initialize full environment for evaluation
    env = DisasterResponseEnv(
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=True
    )
    
    for name, path in ablation_models.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            model = PPO.load(path)
            policies.append(RLAgentWrapper(name, model))
        else:
            print(f"[!] Model {name} not found. Skipping.")
            
    if len(policies) < 2:
        print("[!] Not enough trained models to run a meaningful ablation comparison.")
        print("Please run `python train.py --model_type standard` and `python train.py --model_type cair` first.")
        return
        
    all_results = []
    num_episodes = EXPERIMENT.num_episodes
    seeds = list(EXPERIMENT.seeds) * (num_episodes // len(EXPERIMENT.seeds) + 1)
    
    print(f"\nRunning {num_episodes} episodes for each ablation group...")
    for policy in policies:
        print(f"  -> Running {policy.name}...")
        results = policy.run_multiple_episodes(env, num_episodes=num_episodes, seeds=seeds)
        all_results.extend(results)
        
    # Save to CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAblation results saved to {csv_path}")
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    metrics = [
        ("survivors_found", "Survivors Found (Count)"),
        ("total_reward", "Total Episodic Reward")
    ]
    
    for metric_col, title in metrics:
        plt.figure(figsize=(8, 5))
        
        sns.boxplot(
            data=df, 
            x="policy_name", 
            y=metric_col, 
            palette="Set2"
        )
        
        plt.title(f"Ablation Study: {title}")
        plt.ylabel(title)
        plt.xlabel("Ablation Group")
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f"ablation_{metric_col}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    run_ablation_study()
