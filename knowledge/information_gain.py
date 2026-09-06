"""
Information Gain Module
========================
Shannon entropy-based information gain calculations for the CAIR-UAV system.

Information gain (IG) quantifies how much a sensor observation reduces the
UAV's uncertainty about a cell. It is defined as:

    IG = H(belief_before) - H(belief_after)

where H is the Shannon entropy of a Bernoulli random variable:

    H(p) = -p * log2(p) - (1 - p) * log2(1 - p)

This module provides:
    - Cell-level entropy and IG computation
    - Grid-wide IG maps (expected IG for scanning each cell)
    - Batch IG computation from scan results
    - Expected information gain (EIG) under the sensor model
    - Diminishing-return scaling for repeated scans

The IG values are used in two ways:
    1. As a component of the RL reward function (incentivizes exploration)
    2. As a channel in the observation space (uncertainty map)

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional, List

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GRID, REWARD, GridConfig, RewardConfig

# Epsilon guard for log(0) safety
_EPSILON = 1e-10


class InformationGainCalculator:
    """Computes Shannon entropy and information gain for belief maps.

    This calculator handles all entropy-related mathematics for the CAIR-UAV
    system, including per-cell entropy, information gain from observations,
    expected information gain for scan planning, and grid-wide IG maps.

    Attributes:
        height: Grid height.
        width: Grid width.
        sensor_tpr: True positive rate of the sensor.
        sensor_fpr: False positive rate of the sensor.
        revisit_decay: Multiplicative decay for IG on re-scanned cells.
        min_ig_threshold: Minimum IG below which reward is zero.
    """

    def __init__(
        self,
        grid_cfg: GridConfig = GRID,
        reward_cfg: RewardConfig = REWARD,
    ) -> None:
        """Initialize the IG calculator.

        Args:
            grid_cfg: Grid configuration for dimensions and sensor model.
            reward_cfg: Reward configuration for IG thresholds and decay.
        """
        self.height = grid_cfg.height
        self.width = grid_cfg.width
        self.sensor_tpr = grid_cfg.sensor_true_positive_rate
        self.sensor_fpr = grid_cfg.sensor_false_positive_rate
        self.sensor_range = grid_cfg.sensor_range
        self.revisit_decay = reward_cfg.revisit_ig_decay
        self.min_ig_threshold = reward_cfg.min_ig_threshold

    # ==================================================================
    # CORE ENTROPY FUNCTIONS
    # ==================================================================

    @staticmethod
    def shannon_entropy(p: float) -> float:
        """Compute Shannon entropy for a binary random variable.

        H(p) = -p * log2(p) - (1 - p) * log2(1 - p)

        The function is symmetric around p = 0.5, where it reaches its
        maximum value of 1.0 bit. It equals 0.0 at p = 0 and p = 1.

        Args:
            p: Probability in [0, 1].

        Returns:
            Entropy in [0, 1] bits.
        """
        p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
        return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

    @staticmethod
    def shannon_entropy_vectorized(p_array: np.ndarray) -> np.ndarray:
        """Compute Shannon entropy for an array of probabilities.

        Vectorized version for efficient grid-wide computation.

        Args:
            p_array: Array of probabilities with values in [0, 1].

        Returns:
            Array of same shape with entropy values in [0, 1].
        """
        p = np.clip(p_array, _EPSILON, 1.0 - _EPSILON)
        return (-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)).astype(
            np.float32
        )

    # ==================================================================
    # INFORMATION GAIN
    # ==================================================================

    def compute_ig(
        self,
        belief_before: float,
        belief_after: float,
    ) -> float:
        """Compute information gain from a single belief update.

        IG = H(before) - H(after), clamped to non-negative.

        Args:
            belief_before: Prior belief before observation.
            belief_after: Posterior belief after observation.

        Returns:
            Information gain (non-negative, in bits).
        """
        h_before = self.shannon_entropy(belief_before)
        h_after = self.shannon_entropy(belief_after)
        return max(0.0, h_before - h_after)

    def compute_batch_ig(
        self,
        beliefs_before: Dict[Tuple[int, int], float],
        beliefs_after: Dict[Tuple[int, int], float],
        scan_counts: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """Compute total IG from a batch of belief updates (e.g., one scan).

        Applies diminishing returns based on per-cell scan counts if provided.

        Args:
            beliefs_before: Dict of (row, col) → prior belief.
            beliefs_after: Dict of (row, col) → posterior belief.
            scan_counts: Optional (H, W) array of previous scan counts per cell.
                Used for diminishing-return decay.

        Returns:
            Tuple of:
                - Total information gain (scalar)
                - Dict of (row, col) → per-cell IG values
        """
        total_ig = 0.0
        per_cell_ig: Dict[Tuple[int, int], float] = {}

        for (r, c), p_after in beliefs_after.items():
            p_before = beliefs_before.get((r, c), 0.5)
            ig = self.compute_ig(p_before, p_after)

            # Apply diminishing returns
            if scan_counts is not None and ig > 0:
                count = int(scan_counts[r, c])
                decay = self.revisit_decay ** count
                ig *= decay

            # Threshold filter
            if ig < self.min_ig_threshold:
                ig = 0.0

            per_cell_ig[(r, c)] = ig
            total_ig += ig

        return total_ig, per_cell_ig

    # ==================================================================
    # EXPECTED INFORMATION GAIN (EIG)
    # ==================================================================

    def expected_ig_for_cell(self, belief: float) -> float:
        """Compute the Expected Information Gain if we were to scan a cell.

        This is the expected entropy reduction, averaged over the two
        possible sensor outcomes (detect / no-detect), weighted by their
        probabilities under the current belief:

            P(detect) = TPR * belief + FPR * (1 - belief)
            P(¬detect) = 1 - P(detect)

            EIG = H(belief)
                  - P(detect) * H(posterior_if_detect)
                  - P(¬detect) * H(posterior_if_no_detect)

        Args:
            belief: Current belief P(survivor) for the cell.

        Returns:
            Expected information gain (non-negative, in bits).
        """
        p = np.clip(belief, _EPSILON, 1.0 - _EPSILON)
        h_current = self.shannon_entropy(p)

        # Probability of detection under current belief
        p_detect = self.sensor_tpr * p + self.sensor_fpr * (1.0 - p)
        p_no_detect = 1.0 - p_detect

        # Posterior if detected
        num_det = self.sensor_tpr * p
        den_det = max(p_detect, _EPSILON)
        posterior_det = np.clip(num_det / den_det, _EPSILON, 1.0 - _EPSILON)

        # Posterior if not detected
        num_ndet = (1.0 - self.sensor_tpr) * p
        den_ndet = max(p_no_detect, _EPSILON)
        posterior_ndet = np.clip(num_ndet / den_ndet, _EPSILON, 1.0 - _EPSILON)

        h_det = self.shannon_entropy(posterior_det)
        h_ndet = self.shannon_entropy(posterior_ndet)

        eig = h_current - (p_detect * h_det + p_no_detect * h_ndet)
        return max(0.0, eig)

    def compute_eig_map(
        self,
        belief_map: np.ndarray,
        scan_counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Expected Information Gain for every cell in the grid.

        Creates a heatmap of how much information we'd expect to gain
        by scanning each cell. Useful for exploration planning.

        Args:
            belief_map: (H, W) array of current beliefs.
            scan_counts: Optional (H, W) array of scan counts for decay.

        Returns:
            (H, W) array of EIG values.
        """
        eig_map = np.zeros((self.height, self.width), dtype=np.float32)

        for r in range(self.height):
            for c in range(self.width):
                eig = self.expected_ig_for_cell(float(belief_map[r, c]))

                # Apply diminishing returns
                if scan_counts is not None:
                    count = int(scan_counts[r, c])
                    decay = self.revisit_decay ** count
                    eig *= decay

                eig_map[r, c] = eig

        return eig_map

    def compute_scan_eig_at_position(
        self,
        uav_row: int,
        uav_col: int,
        belief_map: np.ndarray,
        scan_counts: Optional[np.ndarray] = None,
    ) -> float:
        """Compute total EIG for a scan at a specific UAV position.

        Sums the EIG over all cells within sensor range of the position.

        Args:
            uav_row: UAV row position.
            uav_col: UAV column position.
            belief_map: Current belief map.
            scan_counts: Optional scan counts for decay.

        Returns:
            Total expected information gain for scanning at this position.
        """
        total_eig = 0.0
        sr = self.sensor_range

        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                r, c = uav_row + dy, uav_col + dx
                if 0 <= r < self.height and 0 <= c < self.width:
                    eig = self.expected_ig_for_cell(float(belief_map[r, c]))
                    if scan_counts is not None:
                        count = int(scan_counts[r, c])
                        decay = self.revisit_decay ** count
                        eig *= decay
                    total_eig += eig

        return total_eig

    # ==================================================================
    # GRID-WIDE ANALYSIS
    # ==================================================================

    def compute_entropy_map(self, belief_map: np.ndarray) -> np.ndarray:
        """Compute per-cell Shannon entropy for the entire belief map.

        Args:
            belief_map: (H, W) array of beliefs.

        Returns:
            (H, W) array of entropy values.
        """
        return self.shannon_entropy_vectorized(belief_map)

    def compute_total_uncertainty(self, belief_map: np.ndarray) -> float:
        """Compute the total entropy (sum) across the entire grid.

        Args:
            belief_map: (H, W) array of beliefs.

        Returns:
            Sum of all cell entropies.
        """
        return float(np.sum(self.compute_entropy_map(belief_map)))

    def compute_uncertainty_reduction(
        self,
        initial_belief: np.ndarray,
        current_belief: np.ndarray,
    ) -> float:
        """Compute how much total uncertainty has been reduced.

        Args:
            initial_belief: Belief map at episode start.
            current_belief: Current belief map.

        Returns:
            Fraction of initial uncertainty that has been resolved,
            in range [0, 1].
        """
        initial_entropy = self.compute_total_uncertainty(initial_belief)
        current_entropy = self.compute_total_uncertainty(current_belief)

        if initial_entropy < _EPSILON:
            return 1.0

        reduction = (initial_entropy - current_entropy) / initial_entropy
        return float(np.clip(reduction, 0.0, 1.0))

    # ==================================================================
    # UTILITY
    # ==================================================================

    def get_best_scan_positions(
        self,
        belief_map: np.ndarray,
        scan_counts: Optional[np.ndarray] = None,
        top_k: int = 5,
        exclude_positions: Optional[set] = None,
    ) -> List[Tuple[int, int, float]]:
        """Find the top-k positions that would yield highest EIG if scanned.

        Useful for heuristic and rule-based baselines.

        Args:
            belief_map: Current belief map.
            scan_counts: Optional scan count array for decay.
            top_k: Number of positions to return.
            exclude_positions: Set of (row, col) tuples to exclude.

        Returns:
            List of (row, col, eig) sorted by EIG descending.
        """
        if exclude_positions is None:
            exclude_positions = set()

        candidates: List[Tuple[int, int, float]] = []

        for r in range(self.height):
            for c in range(self.width):
                if (r, c) in exclude_positions:
                    continue
                eig = self.compute_scan_eig_at_position(
                    r, c, belief_map, scan_counts
                )
                candidates.append((r, c, eig))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def __repr__(self) -> str:
        return (
            f"InformationGainCalculator("
            f"grid={self.height}x{self.width}, "
            f"TPR={self.sensor_tpr}, FPR={self.sensor_fpr}, "
            f"decay={self.revisit_decay})"
        )
