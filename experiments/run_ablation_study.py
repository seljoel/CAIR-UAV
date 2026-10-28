"""
Complete Ablation Study Evaluation Suite
========================================
Scientifically rigorous evaluation of all five ablation configurations:
    - Group A: Standard PPO (No IG, No Memory, No Context)
    - Group B: Information Gain Only (IG On, Memory Off, Context Off)
    - Group C: Information Gain + Memory (IG On, Memory On, Context Off)
    - Group D/E: Full CAIR-RL (IG On, Memory On, Context On)

Key Features:
1. Strict seed alignment across all groups (identical disaster maps).
2. Evaluation under BOTH conditions:
   - "Policy-Only": Raw policy execution without safety override.
   - "Safety-Guarded": Deployment with BatterySafetyWrapper.
3. Tracks all 8 required metrics:
   - Average episode reward
   - Survivors discovered
   - Area coverage (%)
   - Information gain (bits)
   - Energy consumption
   - Return-to-base success rate
   - Stranded / depletion rate
   - Episode length (timesteps)
4. Exports detailed CSV, aggregate summary CSV, and publication-ready charts.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROJECT_ROOT, RESULTS_DIR, PLOTS_DIR, MODELS_DIR, RESOURCE
from environment.disaster_env import DisasterResponseEnv
from environment.safety_wrapper import BatterySafetyWrapper
from algorithms.ablation_rl import ABLATION_GROUPS


def evaluate_single_episode(
    env,
    model: PPO,
    seed: int,
) -> Dict[str, Any]:
    """Execute one evaluation episode deterministically."""
    obs, info = env.reset(seed=seed)
    done = False
    episode_reward = 0.0
    steps = 0
    overrides = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        episode_reward += float(reward)
        steps += 1
        done = terminated or truncated
        if info.get("safety_override_active", False):
            overrides += 1

    battery_remaining = float(info.get("battery_remaining", 0.0))
    at_base = bool(info.get("at_base", False))
    mission_success = bool(info.get("mission_success", False) or (at_base and battery_remaining > 0))
    stranded = bool(not at_base and (battery_remaining <= 0.0 or terminated))
    energy_consumed = float(RESOURCE.max_battery - battery_remaining)

    return {
        "seed": seed,
        "total_reward": episode_reward,
        "survivors_found": int(info.get("survivors_found", 0)),
        "total_survivors": int(info.get("total_survivors", 0)),
        "survivor_find_rate": float(info.get("survivor_find_rate", 0.0)),
        "area_coverage_pct": float(info.get("area_coverage_pct", 0.0)),
        "total_information_gain": float(info.get("total_information_gain", 0.0)),
        "battery_remaining": battery_remaining,
        "energy_consumed": energy_consumed,
        "time_steps_used": steps,
        "at_base": at_base,
        "mission_success": mission_success,
        "stranded": stranded,
        "safety_overrides": overrides,
        "hazard_encounters": int(info.get("hazard_encounters", 0)),
    }


def run_full_ablation_evaluation(
    eval_seeds: List[int],
    models_dict: Dict[str, str],
    eval_modes: List[str] = ["policy_only", "safety_guarded"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run evaluation across all models, seeds, and evaluation modes.

    Args:
        eval_seeds: List of unique random seeds for map evaluation.
        models_dict: Mapping from group display name to model checkpoint path.
        eval_modes: List containing 'policy_only', 'safety_guarded', or both.

    Returns:
        Tuple of (detailed_df, summary_df).
    """
    print("=" * 70)
    print("CAIR-UAV COMPREHENSIVE ABLATION STUDY EVALUATION")
    print(f"Total Unique Evaluation Seeds: {len(eval_seeds)}")
    print(f"Evaluation Modes: {eval_modes}")
    print("=" * 70)

    all_records: List[Dict[str, Any]] = []

    # Map groups to their environment configurations
    group_env_features = {
        "Group A (Standard PPO)": {"enable_information_gain": False, "enable_mission_memory": False, "enable_context_features": False},
        "Group B (IG Only)": {"enable_information_gain": True, "enable_mission_memory": False, "enable_context_features": False},
        "Group C (IG + Memory)": {"enable_information_gain": True, "enable_mission_memory": True, "enable_context_features": False},
        "Group D/E (Full CAIR-RL)": {"enable_information_gain": True, "enable_mission_memory": True, "enable_context_features": True},
    }

    for model_name, model_path in models_dict.items():
        if not os.path.exists(model_path):
            print(f"[!] Warning: Model path not found for {model_name} at '{model_path}'. Skipping.")
            continue

        print(f"\n[*] Loading model: {model_name} from {model_path}...")
        model = PPO.load(model_path)
        features = group_env_features.get(model_name, {
            "enable_information_gain": True,
            "enable_mission_memory": True,
            "enable_context_features": True
        })

        for mode in eval_modes:
            use_wrapper = (mode == "safety_guarded")
            mode_label = "Safety-Guarded" if use_wrapper else "Policy-Only"
            print(f"  -> Testing {model_name} [{mode_label}] over {len(eval_seeds)} seeds...")

            # Instantiate test environment
            base_env = DisasterResponseEnv(
                enable_information_gain=features["enable_information_gain"],
                enable_mission_memory=features["enable_mission_memory"],
                enable_context_features=features["enable_context_features"],
            )
            if use_wrapper:
                env = BatterySafetyWrapper(base_env, safety_factor=1.60)
            else:
                env = base_env

            for ep_idx, seed in enumerate(eval_seeds):
                metrics = evaluate_single_episode(env, model, seed=seed)
                metrics["group"] = model_name
                metrics["eval_mode"] = mode_label
                metrics["episode_idx"] = ep_idx
                metrics["ig_enabled"] = features["enable_information_gain"]
                metrics["memory_enabled"] = features["enable_mission_memory"]
                metrics["context_enabled"] = features["enable_context_features"]
                all_records.append(metrics)

            env.close()

    detailed_df = pd.DataFrame(all_records)
    
    # Save detailed per-episode records
    detailed_csv = os.path.join(RESULTS_DIR, "ablation_per_episode.csv")
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"\n[+] Detailed results saved to: {detailed_csv}")

    # Compute Aggregate Summary (Group x Mode)
    summary_cols = [
        "total_reward",
        "survivors_found",
        "area_coverage_pct",
        "total_information_gain",
        "energy_consumed",
        "battery_remaining",
        "mission_success",
        "stranded",
        "time_steps_used",
        "safety_overrides",
    ]
    summary_df = detailed_df.groupby(["eval_mode", "group"])[summary_cols].agg(["mean", "std", "count"]).round(2)
    summary_csv = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    summary_df.to_csv(summary_csv)
    print(f"[+] Summary results saved to: {summary_csv}")

    return detailed_df, summary_df


def generate_ablation_plots(df: pd.DataFrame) -> None:
    """Generate high-quality comparison plots for the ablation study."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Mean Reward Comparison (Barplot by Mode and Group)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="group",
        y="total_reward",
        hue="eval_mode",
        palette="viridis",
        capsize=0.1,
    )
    plt.title("Ablation Study: Episodic Reward (Policy-Only vs Safety-Guarded)")
    plt.xlabel("Ablation Group")
    plt.ylabel("Mean Episodic Reward")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Evaluation Mode")
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "ablation_reward_comparison.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[+] Saved plot: {p1}")

    # 2. Survivors Found
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="group",
        y="survivors_found",
        hue="eval_mode",
        palette="crest",
        capsize=0.1,
    )
    plt.title("Ablation Study: Survivors Discovered per Episode")
    plt.xlabel("Ablation Group")
    plt.ylabel("Survivors Found (Count)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "ablation_survivors_found.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[+] Saved plot: {p2}")

    # 3. Area Coverage (%)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="group",
        y="area_coverage_pct",
        hue="eval_mode",
        palette="magma",
        capsize=0.1,
    )
    plt.title("Ablation Study: Disaster Area Coverage (%)")
    plt.xlabel("Ablation Group")
    plt.ylabel("Coverage Percentage (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "ablation_area_coverage.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[+] Saved plot: {p3}")

    # 4. Return to Base & Stranded Rates (Safety Guarded vs Policy-Only)
    plt.figure(figsize=(11, 6))
    agg_rates = df.groupby(["group", "eval_mode"])[["mission_success", "stranded"]].mean().reset_index()
    agg_rates_melt = agg_rates.melt(id_vars=["group", "eval_mode"], value_vars=["mission_success", "stranded"], var_name="Metric", value_name="Rate")
    agg_rates_melt["Rate"] *= 100.0
    agg_rates_melt["Metric"] = agg_rates_melt["Metric"].map({"mission_success": "Return Success Rate (%)", "stranded": "Stranded Rate (%)"})

    ax = sns.barplot(
        data=agg_rates_melt,
        x="group",
        y="Rate",
        hue="Metric",
        palette=["#2ecc71", "#e74c3c"],
    )
    plt.title("Return-to-Base Success vs Stranded Rate Across Groups")
    plt.xlabel("Ablation Group")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 105)
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, "ablation_return_vs_stranded.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[+] Saved plot: {p4}")

    # 5. Total Information Gain
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="group",
        y="total_information_gain",
        hue="eval_mode",
        palette="rocket",
        capsize=0.1,
    )
    plt.title("Ablation Study: Total Information Gain (Bits)")
    plt.xlabel("Ablation Group")
    plt.ylabel("Information Gain (Bits)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p5 = os.path.join(PLOTS_DIR, "ablation_information_gain.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"[+] Saved plot: {p5}")


if __name__ == "__main__":
    # 100 distinct held-out evaluation seeds
    eval_seeds = [1000 + i for i in range(100)]
    
    models = {
        "Group A (Standard PPO)": os.path.join(MODELS_DIR, "standard", "standard_final.zip"),
        "Group B (IG Only)": os.path.join(MODELS_DIR, "ablation", "group_b_ig_only", "group_b_final.zip"),
        "Group C (IG + Memory)": os.path.join(MODELS_DIR, "ablation", "group_c_ig_memory", "group_c_final.zip"),
        "Group D/E (Full CAIR-RL)": os.path.join(MODELS_DIR, "cair", "cair_final.zip"),
    }

    df, summary = run_full_ablation_evaluation(
        eval_seeds=eval_seeds,
        models_dict=models,
        eval_modes=["policy_only", "safety_guarded"],
    )

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY STATISTICS (Mean +/- Std across 100 seeds)")
    print("=" * 60)
    print(summary)
    
    generate_ablation_plots(df)
    print("\n[+] Ablation Study Execution & Visualization Complete.")

