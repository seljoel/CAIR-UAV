"""
Greedy Heuristic Policy Baseline
==================================
A greedy policy that always moves toward the cell with the highest
survivor belief probability, ignoring hazards, battery constraints,
and information gain.

This baseline represents a "naively optimal" strategy that only
maximizes the immediate expected survivor detection probability
without considering resource management, risk, or exploration value.

Decision logic:
    1. Find the cell with the highest belief value in the grid.
    2. If the UAV is already at that cell, perform HOVER_AND_SCAN.
    3. Otherwise, move one step toward that cell (Manhattan/axis-aligned).

The greedy heuristic ignores:
    - Battery remaining (may strand itself far from base)
    - Hazard zones (walks straight through them)
    - Information gain (doesn't value uncertainty reduction)
    - Historical visits (will re-scan the same high-belief cell)

This demonstrates why greedy exploitation without context awareness
performs poorly compared to the proposed CAIR-UAV approach.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Action, GRID


class GreedyHeuristicPolicy:
    """Greedy policy that chases the highest survivor belief.

    At each step, the agent identifies the cell with the maximum
    belief value and moves toward it. When it reaches the target,
    it performs a scan. If the belief map is uniform, it defaults to
    scanning at its current position.

    This is a deterministic policy (no randomness in action selection),
    though the environment itself is stochastic.

    Attributes:
        name: Human-readable policy name.
        grid_height: Height of the grid.
        grid_width: Width of the grid.
    """

    name: str = "Greedy Heuristic"

    def __init__(
        self,
        grid_height: int = GRID.height,
        grid_width: int = GRID.width,
    ) -> None:
        """Initialize the greedy heuristic.

        Args:
            grid_height: Grid height.
            grid_width: Grid width.
        """
        self.grid_height = grid_height
        self.grid_width = grid_width

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Select the greedy action toward the highest-belief cell.

        Decision process:
            1. Extract belief map from spatial_grid channel 1.
            2. Find the cell with maximum belief.
            3. If UAV is at that cell → HOVER_AND_SCAN.
            4. Otherwise → move one step toward it (prefer axis with
               larger distance gap to break ties).

        Args:
            observation: Dict with 'spatial_grid' and 'telemetry'.
            info: Optional info dict (unused by this policy).

        Returns:
            Action index in [0, 4].
        """
        spatial_grid = observation["spatial_grid"]
        telemetry = observation["telemetry"]

        # Extract belief map (channel 1) and UAV position
        belief_map = spatial_grid[1]  # Shape: (H, W)

        # Get UAV position from telemetry (normalized coords)
        uav_x_norm = float(telemetry[2])  # x_norm = col / (W-1)
        uav_y_norm = float(telemetry[3])  # y_norm = row / (H-1)

        uav_col = int(round(uav_x_norm * (self.grid_width - 1)))
        uav_row = int(round(uav_y_norm * (self.grid_height - 1)))

        # Clamp to grid bounds
        uav_row = np.clip(uav_row, 0, self.grid_height - 1)
        uav_col = np.clip(uav_col, 0, self.grid_width - 1)

        # Find the cell with maximum belief
        target_row, target_col = self._find_max_belief_cell(
            belief_map, uav_row, uav_col
        )

        # If at target, scan
        if uav_row == target_row and uav_col == target_col:
            return Action.HOVER_AND_SCAN

        # Move toward target
        return self._move_toward(uav_row, uav_col, target_row, target_col)

    def _find_max_belief_cell(
        self,
        belief_map: np.ndarray,
        uav_row: int,
        uav_col: int,
    ) -> Tuple[int, int]:
        """Find the cell with the highest belief value.

        If multiple cells share the maximum belief (e.g., uniform map),
        the one closest to the UAV in Manhattan distance is selected.

        Args:
            belief_map: (H, W) array of current beliefs.
            uav_row: Current UAV row.
            uav_col: Current UAV column.

        Returns:
            (row, col) of the target cell.
        """
        max_belief = float(belief_map.max())

        # Get all cells with the maximum belief
        max_cells = np.argwhere(belief_map >= max_belief - 1e-8)

        if len(max_cells) == 0:
            # Fallback: stay put and scan
            return uav_row, uav_col

        # Among max-belief cells, pick the closest one
        distances = np.abs(max_cells[:, 0] - uav_row) + np.abs(
            max_cells[:, 1] - uav_col
        )
        closest_idx = np.argmin(distances)

        return int(max_cells[closest_idx, 0]), int(max_cells[closest_idx, 1])

    @staticmethod
    def _move_toward(
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
    ) -> int:
        """Compute the movement action to step toward a target cell.

        Moves along the axis with the larger distance gap first.
        Ties are broken by preferring vertical (N/S) over horizontal (E/W).

        Args:
            from_row: Current row.
            from_col: Current column.
            to_row: Target row.
            to_col: Target column.

        Returns:
            Action index for the best movement direction.
        """
        dy = to_row - from_row
        dx = to_col - from_col

        # Move along the axis with larger gap
        if abs(dy) >= abs(dx):
            if dy < 0:
                return Action.MOVE_NORTH
            elif dy > 0:
                return Action.MOVE_SOUTH
            else:
                # Same row, move horizontally
                if dx > 0:
                    return Action.MOVE_EAST
                else:
                    return Action.MOVE_WEST
        else:
            if dx > 0:
                return Action.MOVE_EAST
            elif dx < 0:
                return Action.MOVE_WEST
            else:
                # Same column, move vertically
                if dy > 0:
                    return Action.MOVE_SOUTH
                else:
                    return Action.MOVE_NORTH

    def run_episode(
        self,
        env,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a complete episode with the greedy heuristic.

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
            f"GreedyHeuristicPolicy("
            f"grid={self.grid_height}x{self.grid_width})"
        )
