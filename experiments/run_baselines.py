"""
Run Baselines Experiment
=========================
Executes all baseline algorithms against the Standard RL and CAIR-RL models.
Generates performance CSVs and Matplotlib bar charts comparing key metrics.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluate import evaluate_models
from config import RESULTS_DIR, PLOTS_DIR

def run_and_plot_baselines():
    print("Running Baseline Benchmark Experiment...")
    
    # Run evaluations (this will generate evaluation_results.csv)
    # We call the evaluate module's main function
    evaluate_models()
    
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    if not os.path.exists(csv_path):
        print("[!] Evaluation failed to produce CSV.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    metrics = [
        ("survivors_found", "Survivors Found (Count)"),
        ("area_coverage_pct", "Area Coverage (%)"),
        ("total_information_gain", "Total Information Gain (Bits)"),
        ("total_reward", "Total Episodic Reward")
    ]
    
    for metric_col, title in metrics:
        plt.figure(figsize=(10, 6))
        
        # Barplot with error bars (std)
        sns.barplot(
            data=df, 
            x="policy_name", 
            y=metric_col, 
            capsize=.1,
            err_kws={'linewidth': 2},
            palette="viridis"
        )
        
        plt.title(f"Baseline Comparison: {title}")
        plt.ylabel(title)
        plt.xlabel("Policy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f"baseline_{metric_col}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    run_and_plot_baselines()
