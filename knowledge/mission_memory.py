"""
Mission Memory Module
======================
Tracks the UAV's historical search activity and accumulated information
gain across the grid over the course of a single episode.

Mission memory serves as the UAV's "experience log" — it records where
the UAV has been, how many times each cell was visited/scanned, and how
much information was gained from each scan. This enables:

    1. Diminishing IG rewards: Repeated scans on well-understood cells
       yield progressively less reward, encouraging the agent to explore
       new territory.
    2. Coverage tracking: Quantifying how much of the grid has been explored.
    3. Historical IG as observation: The accumulated IG per cell can be
       fed into the observation space so the agent learns to avoid
       redundant scans.
    4. Visit heatmaps: Visualization of the agent's search trajectory.

This module is separate from the belief map (which tracks WHAT the UAV
believes) — mission memory tracks WHERE the UAV has searched and WHAT
it learned.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, List, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GRID, REWARD, GridConfig, RewardConfig


class MissionMemory:
    """Tracks historical search activity and accumulated information gain.

    Maintains per-cell counters for visits, scans, and accumulated IG.
    Provides utilities for coverage analysis, diminishing-return computation,
    and trajectory logging.

    Attributes:
        height: Grid height.
        width: Grid width.
        visit_count: (H, W) array counting visits per cell.
        scan_count: (H, W) array counting scans per cell.
        accumulated_ig: (H, W) array of total IG collected per cell.
        revisit_decay: Multiplicative decay factor for IG on re-scans.
        trajectory: List of (row, col, time_step, action) tuples.
    """

    def __init__(
        self,
        grid_cfg: GridConfig = GRID,
        reward_cfg: RewardConfig = REWARD,
    ) -> None:
        """Initialize mission memory.

        Args:
            grid_cfg: Grid configuration for dimensions.
            reward_cfg: Reward configuration for decay parameters.
        """
        self.height = grid_cfg.height
        self.width = grid_cfg.width
        self.revisit_decay = reward_cfg.revisit_ig_decay
        self.min_ig_threshold = reward_cfg.min_ig_threshold

        # Per-cell tracking arrays
        self.visit_count = np.zeros(
            (self.height, self.width), dtype=np.float32
        )
        self.scan_count = np.zeros(
            (self.height, self.width), dtype=np.float32
        )
        self.accumulated_ig = np.zeros(
            (self.height, self.width), dtype=np.float32
        )

        # Last-visit timestamp per cell (-1 = never visited)
        self.last_visit_time = np.full(
            (self.height, self.width), fill_value=-1, dtype=np.int32
        )
        self.last_scan_time = np.full(
            (self.height, self.width), fill_value=-1, dtype=np.int32
        )

        # Ordered trajectory log
        self.trajectory: List[Tuple[int, int, int, int]] = []

        # Episode-level accumulators
        self._total_ig: float = 0.0
        self._total_steps: int = 0

    # ==================================================================
    # RECORDING EVENTS
    # ==================================================================

    def record_visit(
        self,
        row: int,
        col: int,
        time_step: int,
        action: int,
    ) -> None:
        """Record a cell visit (any movement action).

        Args:
            row: Row of the visited cell.
            col: Column of the visited cell.
            time_step: Current mission time step.
            action: Action that led to this visit (0-4).
        """
        self._validate_coords(row, col)
        self.visit_count[row, col] += 1.0
        self.last_visit_time[row, col] = time_step
        self.trajectory.append((row, col, time_step, action))
        self._total_steps = max(self._total_steps, time_step)

    def record_scan(
        self,
        row: int,
        col: int,
        time_step: int,
        ig_value: float,
    ) -> None:
        """Record a scan event and its information gain.

        Args:
            row: Row of the scanned cell.
            col: Column of the scanned cell.
            time_step: Current mission time step.
            ig_value: Information gain obtained from this scan.
        """
        self._validate_coords(row, col)
        self.scan_count[row, col] += 1.0
        self.last_scan_time[row, col] = time_step
        self.accumulated_ig[row, col] += ig_value
        self._total_ig += ig_value

    def record_scan_neighbourhood(
        self,
        center_row: int,
        center_col: int,
        time_step: int,
        per_cell_ig: Dict[Tuple[int, int], float],
        sensor_range: int = 1,
    ) -> None:
        """Record scan events for all cells in a scan neighbourhood.

        Args:
            center_row: Row of the scan center (UAV position).
            center_col: Column of the scan center.
            time_step: Current mission time step.
            per_cell_ig: Dict of (row, col) → IG for each scanned cell.
            sensor_range: Range of the scan around center.
        """
        for (r, c), ig in per_cell_ig.items():
            if 0 <= r < self.height and 0 <= c < self.width:
                self.record_scan(r, c, time_step, ig)

    # ==================================================================
    # DIMINISHING RETURNS
    # ==================================================================

    def get_ig_decay_factor(self, row: int, col: int) -> float:
        """Get the diminishing-return multiplier for a cell's next scan.

        The decay factor is: revisit_decay ^ scan_count

        A cell scanned 0 times gets factor 1.0 (full reward).
        A cell scanned once gets factor revisit_decay (e.g., 0.5).
        A cell scanned twice gets factor revisit_decay^2 (e.g., 0.25).

        Args:
            row: Cell row.
            col: Cell column.

        Returns:
            Decay multiplier in (0, 1].
        """
        self._validate_coords(row, col)
        count = int(self.scan_count[row, col])
        return float(self.revisit_decay ** count)

    def get_decay_map(self) -> np.ndarray:
        """Compute the decay factor map for the entire grid.

        Returns:
            (H, W) array where each cell holds revisit_decay ^ scan_count.
        """
        return np.power(
            self.revisit_decay, self.scan_count
        ).astype(np.float32)

    def apply_diminishing_ig(
        self,
        row: int,
        col: int,
        raw_ig: float,
    ) -> float:
        """Apply diminishing returns to a raw IG value.

        Args:
            row: Cell row.
            col: Cell column.
            raw_ig: Information gain before decay.

        Returns:
            Decayed IG. Returns 0.0 if below the minimum threshold.
        """
        decay = self.get_ig_decay_factor(row, col)
        decayed_ig = raw_ig * decay
        if decayed_ig < self.min_ig_threshold:
            return 0.0
        return decayed_ig

    # ==================================================================
    # COVERAGE ANALYSIS
    # ==================================================================

    def get_coverage_fraction(self) -> float:
        """Compute fraction of grid cells visited at least once.

        Returns:
            Coverage in [0, 1].
        """
        total = self.height * self.width
        visited = int(np.count_nonzero(self.visit_count))
        return float(visited / total)

    def get_scan_coverage_fraction(self) -> float:
        """Compute fraction of grid cells scanned at least once.

        Returns:
            Scan coverage in [0, 1].
        """
        total = self.height * self.width
        scanned = int(np.count_nonzero(self.scan_count))
        return float(scanned / total)

    def get_unvisited_cells(self) -> List[Tuple[int, int]]:
        """Get all cells that have never been visited.

        Returns:
            List of (row, col) tuples for unvisited cells.
        """
        cells = []
        for r in range(self.height):
            for c in range(self.width):
                if self.visit_count[r, c] == 0:
                    cells.append((r, c))
        return cells

    def get_unscanned_cells(self) -> List[Tuple[int, int]]:
        """Get all cells that have never been scanned.

        Returns:
            List of (row, col) tuples for unscanned cells.
        """
        cells = []
        for r in range(self.height):
            for c in range(self.width):
                if self.scan_count[r, c] == 0:
                    cells.append((r, c))
        return cells

    def get_normalized_visit_map(self) -> np.ndarray:
        """Get the visit count map normalized to [0, 1].

        Returns:
            (H, W) array where 1.0 is the most-visited cell.
        """
        max_val = max(float(self.visit_count.max()), 1.0)
        return (self.visit_count / max_val).astype(np.float32)

    def get_normalized_ig_map(self) -> np.ndarray:
        """Get the accumulated IG map normalized to [0, 1].

        Returns:
            (H, W) array where 1.0 is the cell with most accumulated IG.
        """
        max_val = max(float(self.accumulated_ig.max()), 1e-10)
        return (self.accumulated_ig / max_val).astype(np.float32)

    # ==================================================================
    # STALENESS
    # ==================================================================

    def get_staleness_map(self, current_time: int) -> np.ndarray:
        """Compute how 'stale' each cell's information is.

        Staleness = (current_time - last_scan_time) for scanned cells,
        or current_time for never-scanned cells. Normalized to [0, 1]
        by dividing by current_time.

        Args:
            current_time: Current mission time step.

        Returns:
            (H, W) array of staleness values in [0, 1].
        """
        if current_time <= 0:
            return np.ones(
                (self.height, self.width), dtype=np.float32
            )

        staleness = np.where(
            self.last_scan_time >= 0,
            (current_time - self.last_scan_time).astype(np.float32),
            float(current_time),
        )
        staleness = staleness / float(current_time)
        return np.clip(staleness, 0.0, 1.0).astype(np.float32)

    def get_time_since_last_visit(
        self,
        row: int,
        col: int,
        current_time: int,
    ) -> int:
        """Get the number of time steps since the cell was last visited.

        Args:
            row: Cell row.
            col: Cell column.
            current_time: Current mission time step.

        Returns:
            Time steps since last visit, or current_time if never visited.
        """
        self._validate_coords(row, col)
        last = self.last_visit_time[row, col]
        if last < 0:
            return current_time
        return current_time - last

    # ==================================================================
    # TRAJECTORY ANALYSIS
    # ==================================================================

    def get_trajectory_length(self) -> int:
        """Get the total number of recorded trajectory points.

        Returns:
            Number of entries in the trajectory log.
        """
        return len(self.trajectory)

    def get_unique_cells_in_trajectory(self) -> int:
        """Count unique cells visited in the trajectory.

        Returns:
            Number of distinct (row, col) pairs.
        """
        if not self.trajectory:
            return 0
        unique = set((r, c) for r, c, _, _ in self.trajectory)
        return len(unique)

    def get_trajectory_as_array(self) -> np.ndarray:
        """Convert trajectory to a numpy array.

        Returns:
            Array of shape (N, 4) with columns [row, col, time, action].
            Empty array of shape (0, 4) if no trajectory data.
        """
        if not self.trajectory:
            return np.empty((0, 4), dtype=np.int32)
        return np.array(self.trajectory, dtype=np.int32)

    def get_revisit_ratio(self) -> float:
        """Compute ratio of total visits to unique cells visited.

        A ratio of 1.0 means every visit was to a new cell.
        Higher ratios indicate more revisiting.

        Returns:
            Revisit ratio >= 1.0, or 0.0 if no visits.
        """
        unique = self.get_unique_cells_in_trajectory()
        total = self.get_trajectory_length()
        if unique == 0:
            return 0.0
        return float(total / unique)

    # ==================================================================
    # SUMMARY STATISTICS
    # ==================================================================

    def get_summary(self) -> Dict[str, float]:
        """Compile summary statistics for the current mission.

        Returns:
            Dict of metric name → value.
        """
        return {
            "total_steps": float(self._total_steps),
            "total_ig": float(self._total_ig),
            "coverage_fraction": self.get_coverage_fraction(),
            "scan_coverage_fraction": self.get_scan_coverage_fraction(),
            "unique_cells_visited": float(
                self.get_unique_cells_in_trajectory()
            ),
            "trajectory_length": float(self.get_trajectory_length()),
            "revisit_ratio": self.get_revisit_ratio(),
            "mean_visit_count": float(np.mean(self.visit_count)),
            "max_visit_count": float(np.max(self.visit_count)),
            "mean_scan_count": float(np.mean(self.scan_count)),
            "max_scan_count": float(np.max(self.scan_count)),
            "mean_accumulated_ig": float(np.mean(self.accumulated_ig)),
            "total_accumulated_ig": float(np.sum(self.accumulated_ig)),
        }

    # ==================================================================
    # STATE MANAGEMENT
    # ==================================================================

    def reset(self) -> None:
        """Reset all memory to initial state for a new episode."""
        self.visit_count = np.zeros(
            (self.height, self.width), dtype=np.float32
        )
        self.scan_count = np.zeros(
            (self.height, self.width), dtype=np.float32
        )
        self.accumulated_ig = np.zeros(
            (self.height, self.width), dtype=np.float32
        )
        self.last_visit_time = np.full(
            (self.height, self.width), fill_value=-1, dtype=np.int32
        )
        self.last_scan_time = np.full(
            (self.height, self.width), fill_value=-1, dtype=np.int32
        )
        self.trajectory = []
        self._total_ig = 0.0
        self._total_steps = 0

    def get_state(self) -> Dict[str, object]:
        """Serialize memory state for checkpointing.

        Returns:
            Dict with all tracking arrays and trajectory.
        """
        return {
            "visit_count": self.visit_count.copy(),
            "scan_count": self.scan_count.copy(),
            "accumulated_ig": self.accumulated_ig.copy(),
            "last_visit_time": self.last_visit_time.copy(),
            "last_scan_time": self.last_scan_time.copy(),
            "trajectory": list(self.trajectory),
            "total_ig": self._total_ig,
            "total_steps": self._total_steps,
        }

    def load_state(self, state: Dict[str, object]) -> None:
        """Restore memory state from a checkpoint.

        Args:
            state: Dict from get_state().
        """
        self.visit_count = state["visit_count"].copy()
        self.scan_count = state["scan_count"].copy()
        self.accumulated_ig = state["accumulated_ig"].copy()
        self.last_visit_time = state["last_visit_time"].copy()
        self.last_scan_time = state["last_scan_time"].copy()
        self.trajectory = list(state["trajectory"])
        self._total_ig = state["total_ig"]
        self._total_steps = state["total_steps"]

    # ==================================================================
    # INTERNAL
    # ==================================================================

    def _validate_coords(self, row: int, col: int) -> None:
        """Validate coordinates are within grid bounds."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(
                f"Cell ({row}, {col}) out of bounds for "
                f"grid ({self.height}, {self.width})."
            )

    def __repr__(self) -> str:
        coverage = self.get_coverage_fraction()
        return (
            f"MissionMemory({self.height}x{self.width}, "
            f"coverage={coverage:.1%}, "
            f"total_ig={self._total_ig:.3f}, "
            f"steps={self._total_steps})"
        )
