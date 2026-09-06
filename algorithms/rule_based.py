"""
Rule-Based Policy Baseline
===========================
A hierarchical rule-based policy for the UAV disaster response environment.

This baseline uses fixed IF-THEN logic to make decisions based on
the current observation. It attempts to balance exploration, exploitation,
and resource management using hand-crafted thresholds.

Decision Hierarchy:
    1. EMERGENCY: If battery is critically low (< 20%), move toward base.
    2. INVESTIGATE: If current cell has high survivor belief (> 0.6) and
       hasn't been scanned recently, HOVER_AND_SCAN.
    3. EXPLORE: If current cell uncertainty is high (entropy > 0.8), HOVER_AND_SCAN.
    4. GREEDY MOVE: Otherwise, move toward the highest belief cell, avoiding hazards if possible.

This policy performs better than the Random and Greedy baselines but is rigid.
It cannot adapt its thresholds dynamically or learn complex spatial relationships,
which is why the RL approach is proposed.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Action, GRID


class RuleBasedPolicy:
    """Hierarchical rule-based policy baseline.

    Uses fixed thresholds to decide when to return to base, scan, or move.

    Attributes:
        name: Human-readable policy name.
        grid_height: Grid height.
        grid_width: Grid width.
        emergency_battery_threshold: Fraction of max battery to trigger return.
        high_belief_threshold: Minimum belief to trigger an immediate scan.
        high_uncertainty_threshold: Minimum entropy to trigger an exploratory scan.
    """

    name: str = "Rule-Based Policy"

    def __init__(
        self,
        grid_height: int = GRID.height,
        grid_width: int = GRID.width,
        emergency_battery_threshold: float = 0.20,
        high_belief_threshold: float = 0.60,
        high_uncertainty_threshold: float = 0.80,
    ) -> None:
        """Initialize the rule-based policy.

        Args:
            grid_height: Grid height.
            grid_width: Grid width.
            emergency_battery_threshold: Battery fraction to trigger return to base.
            high_belief_threshold: Belief probability to trigger scan.
            high_uncertainty_threshold: Entropy value to trigger scan.
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.emergency_battery_threshold = emergency_battery_threshold
        self.high_belief_threshold = high_belief_threshold
        self.high_uncertainty_threshold = high_uncertainty_threshold

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Select an action based on hierarchical rules.

        Args:
            observation: Dict with 'spatial_grid' and 'telemetry'.
            info: Optional info dict.

        Returns:
            Action index in [0, 4].
        """
        spatial = observation["spatial_grid"]
        telemetry = observation["telemetry"]

        # Parse telemetry
        battery_norm = float(telemetry[0])
        uav_x_norm = float(telemetry[2])
        uav_y_norm = float(telemetry[3])

        uav_col = int(round(uav_x_norm * (self.grid_width - 1)))
        uav_row = int(round(uav_y_norm * (self.grid_height - 1)))
        uav_row = np.clip(uav_row, 0, self.grid_height - 1)
        uav_col = np.clip(uav_col, 0, self.grid_width - 1)

        # Parse spatial grid
        belief_map = spatial[1]
        entropy_map = spatial[2]
        hazard_map = spatial[3]

        current_belief = float(belief_map[uav_row, uav_col])
        current_entropy = float(entropy_map[uav_row, uav_col])

        # RULE 1: EMERGENCY RETURN
        # If battery is critically low, head straight back to base (0,0)
        if battery_norm < self.emergency_battery_threshold:
            if uav_row == 0 and uav_col == 0:
                # Already at base, just hover (idle) to save energy
                return Action.HOVER_AND_SCAN
            return self._move_toward(uav_row, uav_col, 0, 0, hazard_map)

        # RULE 2: INVESTIGATE HIGH BELIEF
        # If we think there's a survivor here, scan to confirm
        if current_belief > self.high_belief_threshold:
            return Action.HOVER_AND_SCAN

        # RULE 3: EXPLORE HIGH UNCERTAINTY
        # If we don't know much about this area, scan to gain information
        if current_entropy > self.high_uncertainty_threshold:
            return Action.HOVER_AND_SCAN

        # RULE 4: GREEDY MOVE TOWARD HIGHEST BELIEF
        # Find the most promising cell and move toward it, trying to avoid hazards
        target_row, target_col = self._find_best_target(
            belief_map, entropy_map, uav_row, uav_col
        )

        if uav_row == target_row and uav_col == target_col:
            return Action.HOVER_AND_SCAN

        return self._move_toward(uav_row, uav_col, target_row, target_col, hazard_map)

    def _find_best_target(
        self,
        belief_map: np.ndarray,
        entropy_map: np.ndarray,
        uav_row: int,
        uav_col: int,
    ) -> Tuple[int, int]:
        """Find the most promising cell to move toward.

        Prioritizes high belief. If beliefs are mostly uniform,
        falls back to high entropy.

        Args:
            belief_map: Current beliefs.
            entropy_map: Current uncertainty.
            uav_row: UAV row.
            uav_col: UAV column.

        Returns:
            (row, col) of the target cell.
        """
        max_belief = float(belief_map.max())

        # If there's a clear belief peak, go there
        if max_belief > 0.55:
            targets = np.argwhere(belief_map >= max_belief - 1e-5)
        else:
            # Otherwise, chase high entropy (exploration)
            max_entropy = float(entropy_map.max())
            targets = np.argwhere(entropy_map >= max_entropy - 1e-5)

        if len(targets) == 0:
            return uav_row, uav_col

        # Break ties by Manhattan distance
        distances = np.abs(targets[:, 0] - uav_row) + np.abs(targets[:, 1] - uav_col)
        closest_idx = np.argmin(distances)

        return int(targets[closest_idx, 0]), int(targets[closest_idx, 1])

    def _move_toward(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
        hazard_map: np.ndarray,
    ) -> int:
        """Compute the best movement action, avoiding immediate hazards.

        Args:
            from_row: Current row.
            from_col: Current column.
            to_row: Target row.
            to_col: Target column.
            hazard_map: Observed hazards.

        Returns:
            Action index.
        """
        dy = to_row - from_row
        dx = to_col - from_col

        # Determine desired primary and secondary directions
        primary = -1
        secondary = -1

        if abs(dy) >= abs(dx):
            primary = Action.MOVE_NORTH if dy < 0 else Action.MOVE_SOUTH
            if dx != 0:
                secondary = Action.MOVE_EAST if dx > 0 else Action.MOVE_WEST
        else:
            primary = Action.MOVE_EAST if dx > 0 else Action.MOVE_WEST
            if dy != 0:
                secondary = Action.MOVE_NORTH if dy < 0 else Action.MOVE_SOUTH

        if primary == -1:
            return Action.HOVER_AND_SCAN

        # Check if primary move leads into a hazard
        primary_safe = self._is_move_safe(from_row, from_col, primary, hazard_map)

        if primary_safe:
            return primary
        elif secondary != -1 and self._is_move_safe(from_row, from_col, secondary, hazard_map):
            return secondary
        else:
            # Both dangerous or secondary doesn't exist. Just take primary and tank the hazard.
            return primary

    def _is_move_safe(
        self,
        row: int,
        col: int,
        action: int,
        hazard_map: np.ndarray,
        hazard_threshold: float = 0.3,
    ) -> bool:
        """Check if a move leads into a known hazard zone.

        Args:
            row: Current row.
            col: Current column.
            action: Movement action.
            hazard_map: Observed hazards.
            hazard_threshold: Intensity considered dangerous.

        Returns:
            True if the target cell is within bounds and has low hazard.
        """
        if action not in Action.DIRECTION_DELTAS:
            return True

        dy, dx = Action.DIRECTION_DELTAS[action]
        nr, nc = row + dy, col + dx

        if not (0 <= nr < self.grid_height and 0 <= nc < self.grid_width):
            return False  # Edge of map

        return hazard_map[nr, nc] < hazard_threshold

    def run_episode(
        self,
        env,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a complete episode with the rule-based policy.

        Args:
            env: Gymnasium-compatible environment instance.
            seed: Optional seed for env reset.
            max_steps: Override for maximum steps.

        Returns:
            Dict containing episode metrics.
        """
        obs, info = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        actions_taken: List[int] = []
        step_count = 0

        while not done:
            action = self.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions_taken.append(action)
            step_count += 1
            done = terminated or truncated

            if max_steps is not None and step_count >= max_steps:
                break

        return {
            "policy_name": self.name,
            "total_reward": total_reward,
            "survivors_found": info.get("survivors_found", 0),
            "total_survivors": info.get("total_survivors", 0),
            "area_coverage_pct": info.get("area_coverage_pct", 0.0),
            "total_information_gain": info.get("total_information_gain", 0.0),
            "battery_remaining": info.get("battery_remaining", 0.0),
            "time_steps_used": step_count,
            "mission_success": info.get("mission_success", False),
            "hazard_encounters": info.get("hazard_encounters", 0),
            "unique_cells_visited": info.get("unique_cells_visited", 0),
            "trajectory": actions_taken,
        }

    def run_multiple_episodes(
        self,
        env,
        num_episodes: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Run multiple episodes and collect results.

        Args:
            env: Environment instance.
            num_episodes: Number of episodes.
            seeds: Optional list of seeds.

        Returns:
            List of episode result dicts.
        """
        if seeds is None:
            seeds = list(range(num_episodes))

        results = []
        for i, seed in enumerate(seeds[:num_episodes]):
            result = self.run_episode(env, seed=seed)
            result["episode_idx"] = i
            result["seed"] = seed
            results.append(result)

        return results

    def __repr__(self) -> str:
        return (
            f"RuleBasedPolicy("
            f"batt_th={self.emergency_battery_threshold}, "
            f"belief_th={self.high_belief_threshold})"
        )
