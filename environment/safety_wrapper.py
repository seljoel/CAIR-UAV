"""
Battery Safety Wrapper
======================
A gymnasium.Wrapper that intercepts every action and overrides it with
RETURN_TO_BASE (action 5) when the UAV's remaining battery is insufficient
to guarantee reaching the base position.

This wrapper is the SINGLE authoritative source of the return-to-base safety
rule.  Apply it to the environment in both training and evaluation so that
the policy is trained and tested under identical conditions.

Why it belongs here (not inside DisasterResponseEnv):
  - The environment already has a soft reward shaping signal (lines 426-431 of
    disaster_env.py) and a hard-termination guard (line 434).  Those two
    mechanisms proved insufficient:
      * Soft reward +5/-10 is dominated by the IG reward (~+10 per scan).
      * Hard termination fires only after the drone is ALREADY stranded.
  - This wrapper adds a third, earlier layer: it *prevents* the stranding
    event from occurring in the first place by overriding actions before they
    are passed to the environment.

Safety formula (same formula used in disaster_env.py line 417):
    energy_to_return = manhattan_distance * (move_cost + idle_cost)
                     = manhattan_distance * 1.3

    safety_budget    = energy_to_return * safety_factor  (default 1.30)
    trigger          = battery <= safety_budget

With safety_factor = 1.30:
    - A drone at distance 15 needs 19.5 to return; trigger fires at 25.4.
    - A drone at distance 38 (worst case) needs 49.4; trigger fires at 64.2.
    - This always leaves a >=30% margin above the bare minimum.

Return controller (action override):
    Moves the drone one step along the axis with the larger gap to base,
    exactly mirroring the RTB logic in disaster_env.py lines 338-343.
    The override only passes action 5 (RETURN_TO_BASE) to the env, which
    already knows how to execute it correctly.

Usage (training):
    from environment.safety_wrapper import BatterySafetyWrapper
    env = DisasterResponseEnv(...)
    env = BatterySafetyWrapper(env, safety_factor=1.30)

Usage (evaluation — RLAgentWrapper):
    Same wrapping. The model's predict() output is then checked by the wrapper
    before reaching the env.  The wrapper's override is logged in info.

Author: CAIR-UAV Research Team (fix: battery-depletion / return-to-base)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple, SupportsFloat

from config import Action, RESOURCE, GRID


class BatterySafetyWrapper(gym.Wrapper):
    """Gymnasium wrapper that enforces a distance-aware battery safety rule.

    The wrapper inspects the environment's internal state after every action
    and, if the drone's battery is at or below the safety budget, overrides
    the *next* action with RETURN_TO_BASE before passing it to the underlying
    environment.

    This means the wrapper acts on the ACTION of the CURRENT step, not the
    previous one.  Specifically:

        wrapped_env.step(action):
            1.  Compute safety budget from current battery & distance.
            2.  If battery <= safety_budget:  action = RETURN_TO_BASE (5)
            3.  Delegate to env.step(action).
            4.  Append override flag to info dict.

    Args:
        env:            A DisasterResponseEnv instance (or compatible env).
        safety_factor:  Multiplier applied to the bare energy-to-return cost.
                        1.30 means 30% headroom above the bare minimum.
                        Must be >= 1.0.
    """

    def __init__(self, env: gym.Env, safety_factor: float = 1.60) -> None:
        super().__init__(env)
        if safety_factor < 1.0:
            raise ValueError(
                f"safety_factor must be >= 1.0, got {safety_factor}"
            )
        # Minimum recommended value to absorb the env's hard-termination guard.
        # Setting below 1.30 may cause emergency termination during return walks.
        self.safety_factor = safety_factor
        self._override_active: bool = False
        self._override_count: int = 0
        # Once safety zone is entered, latch to RTB until drone reaches base.
        self._returning: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the wrapped environment and clear override counters."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._override_active = False
        self._override_count = 0
        self._returning = False
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step with battery-safety action override.

        Args:
            action: Proposed action from the policy (0-5).

        Returns:
            Standard Gymnasium (obs, reward, terminated, truncated, info) tuple.
            info always contains:
                'safety_override_active': bool — was this step overridden?
                'safety_override_count':  int  — total overrides this episode.
        """
        # ---- 1. Read current state from the wrapped environment ----
        battery  = float(self.env.unwrapped._battery)
        uav_pos  = self.env.unwrapped._uav_pos          # [row, col]
        base_pos = self.env.unwrapped.grid_cfg.base_position  # (row, col)

        move_cost = self.env.unwrapped.resource_cfg.move_energy_cost
        idle_cost = self.env.unwrapped.resource_cfg.hover_idle_energy
        cost_per_step = move_cost + idle_cost            # 1.3

        # Subtract the passive idle drain that will occur at the very start
        # of this env.step() call (disaster_env.py line 317: battery -= idle).
        # This makes the trigger one step earlier, preventing the env's hard
        # emergency-termination guard (line 434) from firing mid-return-walk.
        effective_battery = battery - idle_cost

        # ---- 2. Compute Manhattan distance to base ----
        dist = int(abs(uav_pos[0] - base_pos[0]) + abs(uav_pos[1] - base_pos[1]))

        # ---- 3. Compute safety budget ----
        energy_to_return = dist * cost_per_step          # bare minimum
        safety_budget    = energy_to_return * self.safety_factor  # with margin

        # ---- 4. Override decision ----
        # Case A: Already latched — keep overriding every step.
        # Case B: Safety budget first crossed — latch into return mode.
        # Case C: Drone reaches base while latched — hover in place (keep latch)
        #         so the model cannot walk away from base again.
        if self._returning or (dist > 0 and effective_battery <= safety_budget):
            if dist == 0:
                # At base: RTB costs only 0.3 passive drain (env skips move
                # cost when dy==dx==0, see disaster_env.py lines 333-335).
                # HOVER_AND_SCAN would cost 2.3 — wasteful when already home.
                action = Action.RETURN_TO_BASE
            else:
                action = Action.RETURN_TO_BASE
            overridden = True
            self._returning = True
            self._override_active = True
            self._override_count += 1
        else:
            overridden = False
            self._override_active = False

        # ---- 5. Delegate to the underlying environment ----
        obs, reward, terminated, truncated, info = self.env.step(action)

        # ---- 6. Annotate info ----
        info["safety_override_active"] = overridden
        info["safety_override_count"]  = self._override_count
        info["safety_battery"]         = battery
        info["safety_effective_battery"] = effective_battery
        info["safety_dist"]            = dist
        info["safety_budget"]          = safety_budget

        return obs, reward, terminated, truncated, info

    def __repr__(self) -> str:
        return (
            f"BatterySafetyWrapper(env={self.env!r}, "
            f"safety_factor={self.safety_factor})"
        )
