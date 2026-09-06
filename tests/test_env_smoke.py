"""Smoke test for DisasterResponseEnv Gymnasium compatibility."""
import sys
sys.path.insert(0, ".")

from environment.disaster_env import DisasterResponseEnv
import numpy as np

def test_env():
    env = DisasterResponseEnv(seed=42, render_mode="ansi")

    # --- Reset ---
    obs, info = env.reset(seed=42)
    assert "spatial_grid" in obs, "Missing spatial_grid key"
    assert "telemetry" in obs, "Missing telemetry key"
    assert obs["spatial_grid"].shape == (5, 20, 20), f"Bad spatial shape: {obs['spatial_grid'].shape}"
    assert obs["telemetry"].shape == (11,), f"Bad telemetry shape: {obs['telemetry'].shape}"
    assert env.observation_space.contains(obs), "Obs not in obs space"
    print("Reset OK. Spatial:", obs["spatial_grid"].shape, "Telemetry:", obs["telemetry"].shape)

    # --- Step through 50 random actions ---
    total_reward = 0.0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert env.observation_space.contains(obs), f"Step {i}: obs not in space"
        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    print(f"50 steps done. Reward: {total_reward:.2f}")
    print(f"Info: survivors={info['survivors_found']}/{info['total_survivors']}, "
          f"coverage={info['area_coverage_pct']:.1f}%, battery={info['battery_remaining']:.1f}")

    # --- Render ---
    text = env.render()
    print(text)

    # --- Test full episode ---
    obs, info = env.reset(seed=123)
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    print(f"\nFull episode: {steps} steps, reward={info['episode_reward']:.2f}, "
          f"success={info['mission_success']}")

    # --- Observation space check ---
    print(f"\nObs space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_env()
