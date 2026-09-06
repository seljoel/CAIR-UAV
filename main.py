"""
CAIR-UAV Main Entry Point
==========================
Interactive CLI menu for the CAIR-UAV project.

Author: CAIR-UAV Research Team
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_menu():
    print("\n" + "="*50)
    print(" CAIR-UAV: Autonomous Disaster Response UAV")
    print("="*50)
    print(" 1. Run Visual Demo (demo.py)")
    print(" 2. Train Proposed RL Model (train.py --model_type cair)")
    print(" 3. Evaluate Models (evaluate.py)")
    print(" 4. Run Baseline Experiments")
    print(" 5. Run Ablation Study")
    print(" 6. Exit")
    print("="*50)

def main():
    while True:
        print_menu()
        choice = input("Select an option (1-6): ").strip()
        
        if choice == '1':
            from demo import run_demo
            run_demo()
        elif choice == '2':
            from training.train_ppo import train
            train("cair")
        elif choice == '3':
            from evaluate import evaluate_models
            evaluate_models()
        elif choice == '4':
            from experiments.run_baselines import run_and_plot_baselines
            run_and_plot_baselines()
        elif choice == '5':
            from experiments.ablation import run_ablation_study
            run_ablation_study()
        elif choice == '6':
            print("Exiting CAIR-UAV. Goodbye!")
            sys.exit(0)
        else:
            print("[!] Invalid option. Please try again.")

if __name__ == "__main__":
    main()
