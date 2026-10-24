import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR
from environment.disaster_env import DisasterResponseEnv

def run_diagnostic():
    print("--- Diagnostic Script Started ---")
    
    # Initialize Environment
    env = DisasterResponseEnv(
        enable_information_gain=True,
        enable_mission_memory=True,
        enable_context_features=True
    )
    
    cair_path = os.path.join(MODELS_DIR, "cair", "cair_final.zip")
    if not os.path.exists(cair_path):
        print(f"Model not found at {cair_path}")
        return
        
    print(f"Loading CAIR-RL from {cair_path}...")
    model = PPO.load(cair_path)
    
    # Run one episode until death by battery
    obs, info = env.reset(seed=42)
    done = False
    step = 0
    
    history = []
    
    while not done:
        # Save obs for the history before stepping
        current_obs = {
            "spatial_grid": obs["spatial_grid"].copy(),
            "telemetry": obs["telemetry"].copy()
        }
        
        # 1. Observation Space State (print once)
        if step == 5: # just an arbitrary early step
            print("\n==================================================")
            print("1. OBSERVATION SPACE STATE (Passed to model.predict())")
            print("==================================================")
            print("Dictionary keys:", current_obs.keys())
            print(f"spatial_grid shape: {current_obs['spatial_grid'].shape}, dtype: {current_obs['spatial_grid'].dtype}")
            print(f"telemetry shape: {current_obs['telemetry'].shape}, dtype: {current_obs['telemetry'].dtype}")
            print("\nExact telemetry vector values:")
            print(np.array2string(current_obs['telemetry'], precision=4, separator=', ', suppress_small=True))
            print("\nINDEX 0 is the UAV's current normalized battery level:")
            print(f"Battery Level (Telemetry Index 0): {current_obs['telemetry'][0]:.4f}")
            print("==================================================\n")
            
        action, _ = model.predict(obs, deterministic=True)
        
        # Capture current battery before step
        batt_before = env.unwrapped._battery
        
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        batt_after = env.unwrapped._battery
        
        done = terminated or truncated
        
        # Keep track of history
        transition = {
            "step": step,
            "action": int(action),
            "battery_before": batt_before,
            "battery_after": batt_after,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated
        }
        history.append(transition)
        
        # Keep only last 3
        if len(history) > 3:
            history.pop(0)
            
        step += 1
        
        # Stop early if episode took too long (just in case, but battery should deplete)
        if step > 2000:
            break

    print("\n==================================================")
    print("4. TERMINAL STATE TRACE (Final 3 Steps)")
    print("==================================================")
    for t in history:
        print(f"Step {t['step']}:")
        print(f"  - Action Taken: {t['action']}")
        print(f"  - Battery Level (Before): {t['battery_before']:.2f}")
        print(f"  - Battery Level (After): {t['battery_after']:.2f}")
        print(f"  - Reward Received: {t['reward']:.4f}")
        print(f"  - Terminated: {t['terminated']}, Truncated: {t['truncated']}")
    print("==================================================\n")

if __name__ == '__main__':
    run_diagnostic()
