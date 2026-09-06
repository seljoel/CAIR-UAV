"""
Root Training Wrapper
======================
Convenience script to launch PPO training from the project root.

Usage:
    python train.py --model_type cair
    python train.py --model_type standard
"""

import sys
import os

# Ensure the root directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_ppo import train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for CAIR-UAV")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["standard", "cair"],
        default="cair",
        help="Which model variant to train (standard or full cair)"
    )
    args = parser.parse_args()
    
    train(args.model_type)
