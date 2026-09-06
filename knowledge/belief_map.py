"""
Belief Map Module
==================
Manages the UAV's Bayesian belief state over the disaster environment grid.

The belief map is a 2D array of shape (H, W) where each cell holds
P(survivor_present | observations_so_far). The UAV starts with a uniform
prior (maximum entropy) and updates beliefs using Bayes' rule each time
a SCAN action is performed.

This module encapsulates:
    - Belief initialization (uniform prior)
    - Single-cell and neighbourhood Bayesian updates
    - Batch updates from sensor observations
    - Belief querying utilities (high-belief cells, mean belief, etc.)
    - Belief decay for temporal uncertainty (optional)

The belief map is the PRIMARY source of the agent's knowledge about the
environment. Ground truth is never exposed to the agent.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GRID, GridConfig

# Epsilon guard for numerical stability in log/division operations
_EPSILON = 1e-10


class BeliefMap:
    """Bayesian belief state manager for survivor detection.

    Maintains a probability grid where each cell represents the UAV's
    current belief that a survivor is present at that location.

    The belief map supports:
        - Bayesian update with noisy binary sensor readings
        - Neighbourhood-aware batch updates
        - Querying for high-probability regions
        - Entropy computation per cell
        - Optional temporal decay to model increasing uncertainty over time

    Attributes:
        height: Grid height.
        width: Grid width.
        initial_belief: Prior probability for each cell.
        sensor_tpr: Sensor true positive rate P(detect | survivor).
        sensor_fpr: Sensor false positive rate P(detect | no survivor).
        belief: np.ndarray of shape (H, W) with current beliefs.
    """

    def __init__(
        self,
        grid_cfg: GridConfig = GRID,
        initial_belief: Optional[float] = None,
    ) -> None:
        """Initialize the belief map.

        Args:
            grid_cfg: Grid configuration for dimensions and sensor parameters.
            initial_belief: Override for the initial uniform prior. If None,
                uses grid_cfg.initial_survivor_belief.
        """
        self.height = grid_cfg.height
        self.width = grid_cfg.width
        self.initial_belief = (
            initial_belief if initial_belief is not None
            else grid_cfg.initial_survivor_belief
        )
        self.sensor_tpr = grid_cfg.sensor_true_positive_rate
        self.sensor_fpr = grid_cfg.sensor_false_positive_rate
        self.sensor_range = grid_cfg.sensor_range

        # Initialize belief grid
        self.belief = np.full(
            (self.height, self.width),
            fill_value=self.initial_belief,
            dtype=np.float32,
        )

        # Track number of updates per cell for diagnostics
        self._update_count = np.zeros(
            (self.height, self.width), dtype=np.int32
        )

    # ==================================================================
    # CORE BAYESIAN UPDATE
    # ==================================================================

    def update_cell(self, row: int, col: int, detected: bool) -> float:
        """Update belief for a single cell using Bayes' rule.

        Computes:
            If detected:
                P(S|D) = P(D|S) * P(S) / [P(D|S)*P(S) + P(D|¬S)*P(¬S)]
            If not detected:
                P(S|¬D) = P(¬D|S) * P(S) / [P(¬D|S)*P(S) + P(¬D|¬S)*P(¬S)]

        Args:
            row: Grid row index.
            col: Grid column index.
            detected: Whether the sensor reported a positive detection.

        Returns:
            The updated posterior belief value for the cell.

        Raises:
            IndexError: If (row, col) is outside grid bounds.
        """
        self._validate_coords(row, col)

        prior = float(np.clip(self.belief[row, col], _EPSILON, 1.0 - _EPSILON))

        if detected:
            numerator = self.sensor_tpr * prior
            denominator = (
                self.sensor_tpr * prior
                + self.sensor_fpr * (1.0 - prior)
            )
        else:
            numerator = (1.0 - self.sensor_tpr) * prior
            denominator = (
                (1.0 - self.sensor_tpr) * prior
                + (1.0 - self.sensor_fpr) * (1.0 - prior)
            )

        denominator = max(denominator, _EPSILON)
        posterior = numerator / denominator
        posterior = float(np.clip(posterior, _EPSILON, 1.0 - _EPSILON))

        self.belief[row, col] = posterior
        self._update_count[row, col] += 1

        return posterior

    def update_neighbourhood(
        self,
        center_row: int,
        center_col: int,
        detections: Dict[Tuple[int, int], bool],
    ) -> Dict[Tuple[int, int], float]:
        """Update beliefs for multiple cells around a center position.

        Typically called after a SCAN action, which produces noisy readings
        for all cells within the sensor range.

        Args:
            center_row: Row of the scan center (UAV position).
            center_col: Column of the scan center (UAV position).
            detections: Dict mapping (row, col) → detected (bool) for each
                cell that received a sensor reading.

        Returns:
            Dict mapping (row, col) → updated posterior belief for each
            updated cell.
        """
        updated: Dict[Tuple[int, int], float] = {}

        for (r, c), detected in detections.items():
            if 0 <= r < self.height and 0 <= c < self.width:
                posterior = self.update_cell(r, c, detected)
                updated[(r, c)] = posterior

        return updated

    def batch_update_from_scan(
        self,
        uav_row: int,
        uav_col: int,
        ground_truth: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], bool]]:
        """Simulate a scan and perform Bayesian updates.

        Generates noisy sensor readings based on ground truth and the
        sensor model, then updates beliefs for all cells in range.

        Args:
            uav_row: UAV's current row position.
            uav_col: UAV's current column position.
            ground_truth: Binary (H, W) array of true survivor locations.
            rng: NumPy random generator for sensor noise.

        Returns:
            Tuple of:
                - Dict of (row, col) → posterior belief
                - Dict of (row, col) → detection result (bool)
        """
        detections: Dict[Tuple[int, int], bool] = {}

        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                r, c = uav_row + dy, uav_col + dx
                if 0 <= r < self.height and 0 <= c < self.width:
                    has_survivor = ground_truth[r, c] > 0.5
                    if has_survivor:
                        detected = rng.random() < self.sensor_tpr
                    else:
                        detected = rng.random() < self.sensor_fpr
                    detections[(r, c)] = detected

        posteriors = self.update_neighbourhood(uav_row, uav_col, detections)
        return posteriors, detections

    # ==================================================================
    # QUERYING
    # ==================================================================

    def get_high_belief_cells(
        self,
        threshold: float = 0.6,
    ) -> List[Tuple[int, int, float]]:
        """Return all cells with belief above a threshold.

        Args:
            threshold: Minimum belief value to include.

        Returns:
            List of (row, col, belief) tuples, sorted by belief descending.
        """
        cells = []
        for r in range(self.height):
            for c in range(self.width):
                if self.belief[r, c] >= threshold:
                    cells.append((r, c, float(self.belief[r, c])))
        cells.sort(key=lambda x: x[2], reverse=True)
        return cells

    def get_most_uncertain_cells(
        self,
        top_k: int = 10,
    ) -> List[Tuple[int, int, float]]:
        """Return the top-k cells with highest Shannon entropy.

        High entropy means belief is close to 0.5 (maximum uncertainty).

        Args:
            top_k: Number of cells to return.

        Returns:
            List of (row, col, entropy) tuples, sorted by entropy descending.
        """
        entropy_map = self.compute_entropy_map()
        flat_indices = np.argsort(entropy_map.ravel())[::-1][:top_k]
        cells = []
        for idx in flat_indices:
            r, c = divmod(int(idx), self.width)
            cells.append((r, c, float(entropy_map[r, c])))
        return cells

    def get_belief_at(self, row: int, col: int) -> float:
        """Get the current belief value at a specific cell.

        Args:
            row: Grid row.
            col: Grid column.

        Returns:
            Belief probability in [epsilon, 1-epsilon].
        """
        self._validate_coords(row, col)
        return float(self.belief[row, col])

    def get_mean_belief(self) -> float:
        """Compute the mean belief across the entire grid.

        Returns:
            Scalar mean belief value.
        """
        return float(np.mean(self.belief))

    def get_max_belief_position(self) -> Tuple[int, int]:
        """Find the cell with the highest current belief.

        Returns:
            (row, col) of the maximum-belief cell.
        """
        idx = np.argmax(self.belief)
        r, c = divmod(int(idx), self.width)
        return (r, c)

    def get_belief_in_radius(
        self,
        center_row: int,
        center_col: int,
        radius: int,
    ) -> np.ndarray:
        """Extract the belief sub-grid within a given radius.

        Args:
            center_row: Center row.
            center_col: Center column.
            radius: Number of cells around center to include.

        Returns:
            2D numpy array of beliefs for the neighbourhood.
        """
        r_min = max(0, center_row - radius)
        r_max = min(self.height, center_row + radius + 1)
        c_min = max(0, center_col - radius)
        c_max = min(self.width, center_col + radius + 1)
        return self.belief[r_min:r_max, c_min:c_max].copy()

    # ==================================================================
    # ENTROPY COMPUTATION
    # ==================================================================

    def compute_entropy_map(self) -> np.ndarray:
        """Compute Shannon entropy for every cell in the belief map.

        H(p) = -p * log2(p) - (1-p) * log2(1-p)

        Returns:
            np.ndarray of shape (H, W) with entropy in [0, 1].
        """
        p = np.clip(self.belief, _EPSILON, 1.0 - _EPSILON)
        entropy = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        return entropy.astype(np.float32)

    def compute_total_entropy(self) -> float:
        """Compute the sum of Shannon entropy across all cells.

        Returns:
            Total entropy (higher = more overall uncertainty).
        """
        return float(np.sum(self.compute_entropy_map()))

    def compute_mean_entropy(self) -> float:
        """Compute the mean Shannon entropy per cell.

        Returns:
            Mean entropy value.
        """
        return float(np.mean(self.compute_entropy_map()))

    # ==================================================================
    # TEMPORAL DECAY (OPTIONAL)
    # ==================================================================

    def apply_temporal_decay(
        self,
        decay_rate: float = 0.01,
        exclude_confirmed: Optional[set] = None,
    ) -> None:
        """Apply temporal decay to beliefs, moving them toward 0.5.

        Models increasing uncertainty over time for cells that haven't
        been recently scanned. Confirmed survivor cells can be excluded.

        Args:
            decay_rate: How much to shift beliefs toward 0.5 per call.
                A value of 0.01 means beliefs move 1% toward 0.5.
            exclude_confirmed: Set of (row, col) tuples for confirmed
                survivors whose beliefs should not decay.
        """
        if exclude_confirmed is None:
            exclude_confirmed = set()

        for r in range(self.height):
            for c in range(self.width):
                if (r, c) in exclude_confirmed:
                    continue
                current = self.belief[r, c]
                # Move toward 0.5 (maximum entropy prior)
                self.belief[r, c] = current + decay_rate * (0.5 - current)

    # ==================================================================
    # STATE MANAGEMENT
    # ==================================================================

    def reset(self) -> None:
        """Reset the belief map to the uniform prior."""
        self.belief = np.full(
            (self.height, self.width),
            fill_value=self.initial_belief,
            dtype=np.float32,
        )
        self._update_count = np.zeros(
            (self.height, self.width), dtype=np.int32
        )

    def copy(self) -> "BeliefMap":
        """Create a deep copy of this belief map.

        Returns:
            New BeliefMap instance with copied state.
        """
        new_bm = BeliefMap.__new__(BeliefMap)
        new_bm.height = self.height
        new_bm.width = self.width
        new_bm.initial_belief = self.initial_belief
        new_bm.sensor_tpr = self.sensor_tpr
        new_bm.sensor_fpr = self.sensor_fpr
        new_bm.sensor_range = self.sensor_range
        new_bm.belief = self.belief.copy()
        new_bm._update_count = self._update_count.copy()
        return new_bm

    def get_state(self) -> Dict[str, np.ndarray]:
        """Serialize the belief map state for checkpointing.

        Returns:
            Dict with 'belief' and 'update_count' arrays.
        """
        return {
            "belief": self.belief.copy(),
            "update_count": self._update_count.copy(),
        }

    def load_state(self, state: Dict[str, np.ndarray]) -> None:
        """Restore belief map state from a checkpoint.

        Args:
            state: Dict from get_state().
        """
        self.belief = state["belief"].copy()
        self._update_count = state["update_count"].copy()

    # ==================================================================
    # INTERNAL
    # ==================================================================

    def _validate_coords(self, row: int, col: int) -> None:
        """Validate that coordinates are within grid bounds.

        Args:
            row: Row index to validate.
            col: Column index to validate.

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(
                f"Cell ({row}, {col}) out of bounds for "
                f"grid ({self.height}, {self.width})."
            )

    def __repr__(self) -> str:
        """Human-readable representation."""
        mean_b = self.get_mean_belief()
        mean_e = self.compute_mean_entropy()
        total_updates = int(self._update_count.sum())
        return (
            f"BeliefMap({self.height}x{self.width}, "
            f"mean_belief={mean_b:.3f}, mean_entropy={mean_e:.3f}, "
            f"total_updates={total_updates})"
        )
