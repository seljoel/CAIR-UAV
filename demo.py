"""
Interactive Pygame Demo
========================
Runs a visual simulation episode. Attempts to load the trained CAIR-RL
model. If no trained model is found, it falls back to the Rule-Based baseline.
"""

import os
import sys
import time
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, Action
from environment.disaster_env import DisasterResponseEnv
from algorithms.rule_based import RuleBasedPolicy
from visualization.simulator import VisualSimulator


def run_demo():
    print("Initializing CAIR-UAV Visual Demo...")
    
    # Create environment
    env = DisasterResponseEnv(
        seed=42,
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=True
    )
    
    # Check for trained model
    model_path = os.path.join(MODELS_DIR, "cair", "cair_final.zip")
    use_rl = os.path.exists(model_path)
    
    if use_rl:
        print(f"[*] Found trained CAIR-RL model at {model_path}")
        model = PPO.load(model_path)
    else:
        print("[!] No trained CAIR-RL model found. Falling back to Rule-Based Policy.")
        print("[!] Train the model using `python train.py` first for full capability.")
        model = RuleBasedPolicy()

    sim = VisualSimulator(fps=8)
    
    # Run episodes loop
    while sim.running:
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0.0
        
        while not done and sim.running:
            # Handle Pygame events (Pause, Restart, Quit)
            if not sim.handle_events():
                break
                
            if sim.restart_requested:
                sim.restart_requested = False
                break
                
            if sim.paused:
                sim.render(env, episode_reward, step, "PAUSED")
                time.sleep(0.1)
                continue

            # Select action
            if use_rl:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = model.select_action(obs, info)
                
            action_name = Action.ACTION_NAMES[action]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            step += 1
            done = terminated or truncated
            
            # Render
            sim.render(env, episode_reward, step, action_name)
            
        if sim.running:
            print(f"Episode Finished! Reward: {episode_reward:.2f} | Found: {info['survivors_found']}/{info['total_survivors']}")
            print("Restarting in 2 seconds...")
            time.sleep(2)
            
    sim.close()
    env.close()

if __name__ == "__main__":
    run_demo()
