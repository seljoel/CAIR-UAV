"""
Map Generator Module
=====================
Procedural generation of ground-truth disaster maps and initial belief states
for the CAIR-UAV simulation environment.

Ground truth is known ONLY to the simulator. The UAV agent receives a separate
belief map that starts at maximum entropy (uniform prior) and is updated via
noisy sensor observations during the mission.

Key responsibilities:
    1. Generate ground-truth survivor locations using Gaussian mixture clusters.
    2. Generate ground-truth hazard zones as rectangular regions with intensity.
    3. Create the initial belief map (uniform prior = 0.5 everywhere).
    4. Provide reset / regeneration for each new episode.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GRID, GridConfig


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class SurvivorCluster:
    """Represents a cluster of survivors on the ground-truth map.

    Attributes:
        center_y: Row index of cluster center.
        center_x: Column index of cluster center.
        num_survivors: Number of survivors placed from this cluster.
        survivor_positions: List of (row, col) tuples with actual positions.
    """
    center_y: int
    center_x: int
    num_survivors: int
    survivor_positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class HazardZone:
    """Represents a rectangular hazard zone on the ground-truth map.

    Attributes:
        top_left: (row, col) of the top-left corner.
        height: Number of rows the zone spans.
        width: Number of columns the zone spans.
        intensity: Hazard severity in [0, 1]. Higher = more dangerous.
    """
    top_left: Tuple[int, int]
    height: int
    width: int
    intensity: float


# ==============================================================================
# MAP GENERATOR
# ==============================================================================

class MapGenerator:
    """Procedural generator for disaster environment maps.

    Creates ground-truth maps with survivor locations and hazard zones,
    as well as initial belief maps for the UAV agent.

    The generator is deterministic given a fixed NumPy random generator,
    enabling reproducible experiments across seeds.

    Attributes:
        grid_cfg: Grid configuration parameters.
        rng: NumPy random number generator for reproducibility.
    """

    def __init__(
        self,
        grid_cfg: GridConfig = GRID,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize the map generator.

        Args:
            grid_cfg: Grid configuration (dimensions, cluster params, etc.).
            rng: NumPy random Generator. If None, a default unseeded one is used.
        """
        self.grid_cfg = grid_cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate(self) -> Dict[str, np.ndarray | list]:
        """Generate a complete set of ground-truth and initial belief maps.

        Returns:
            Dictionary containing:
                "survivor_map"       : np.ndarray (H, W) binary, 1 = survivor present
                "survivor_prob_map"  : np.ndarray (H, W) float [0,1], soft probability
                "hazard_map"         : np.ndarray (H, W) float [0,1], hazard intensity
                "belief_map"         : np.ndarray (H, W) float, initial belief (0.5)
                "survivor_clusters"  : List[SurvivorCluster]
                "hazard_zones"       : List[HazardZone]
                "total_survivors"    : int
        """
        survivor_map, survivor_prob_map, clusters = self._generate_survivors()
        hazard_map, hazard_zones = self._generate_hazards()
        belief_map = self._generate_initial_belief()

        # Ensure base position (0,0) is always safe
        hazard_map[
            self.grid_cfg.base_position[0],
            self.grid_cfg.base_position[1],
        ] = 0.0

        total_survivors = int(survivor_map.sum())

        return {
            "survivor_map": survivor_map,
            "survivor_prob_map": survivor_prob_map,
            "hazard_map": hazard_map,
            "belief_map": belief_map,
            "survivor_clusters": clusters,
            "hazard_zones": hazard_zones,
            "total_survivors": total_survivors,
        }

    def regenerate_with_seed(self, seed: int) -> Dict[str, np.ndarray | list]:
        """Regenerate maps with a specific seed for reproducibility.

        Args:
            seed: Integer seed for the random number generator.

        Returns:
            Same dictionary structure as `generate()`.
        """
        self.rng = np.random.default_rng(seed)
        return self.generate()

    # ------------------------------------------------------------------
    # SURVIVOR GENERATION
    # ------------------------------------------------------------------

    def _generate_survivors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[SurvivorCluster]]:
        """Generate ground-truth survivor locations using Gaussian clusters.

        Survivors are placed by sampling positions from 2D Gaussian
        distributions centered at randomly chosen cluster centers. Each
        cluster contributes a random number of survivors within the
        configured range.

        Returns:
            survivor_map: Binary (H, W) array. 1 where a survivor exists.
            survivor_prob_map: Soft probability (H, W) array derived from
                Gaussian density, useful for visualization but NOT exposed
                to the agent.
            clusters: List of SurvivorCluster metadata objects.
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        survivor_map = np.zeros((h, w), dtype=np.float32)
        survivor_prob_map = np.zeros((h, w), dtype=np.float32)
        clusters: List[SurvivorCluster] = []

        for _ in range(self.grid_cfg.num_survivor_clusters):
            # Random cluster center, avoiding edges for realism
            cy = self.rng.integers(2, h - 2)
            cx = self.rng.integers(2, w - 2)

            num_survivors = self.rng.integers(
                self.grid_cfg.survivors_per_cluster_range[0],
                self.grid_cfg.survivors_per_cluster_range[1] + 1,
            )

            cluster = SurvivorCluster(
                center_y=int(cy),
                center_x=int(cx),
                num_survivors=int(num_survivors),
            )

            # Sample survivor positions from 2D Gaussian
            for _ in range(num_survivors):
                sy = int(
                    np.clip(
                        self.rng.normal(cy, self.grid_cfg.survivor_cluster_std),
                        0,
                        h - 1,
                    )
                )
                sx = int(
                    np.clip(
                        self.rng.normal(cx, self.grid_cfg.survivor_cluster_std),
                        0,
                        w - 1,
                    )
                )
                survivor_map[sy, sx] = 1.0
                cluster.survivor_positions.append((sy, sx))

            clusters.append(cluster)

            # Build soft probability heatmap from Gaussian kernel
            for r in range(h):
                for c in range(w):
                    dist_sq = (r - cy) ** 2 + (c - cx) ** 2
                    gaussian_val = np.exp(
                        -dist_sq / (2 * self.grid_cfg.survivor_cluster_std ** 2)
                    )
                    survivor_prob_map[r, c] = max(
                        survivor_prob_map[r, c], gaussian_val
                    )

        # Normalize soft probability map to [0, 1]
        max_val = survivor_prob_map.max()
        if max_val > 0:
            survivor_prob_map /= max_val

        return survivor_map, survivor_prob_map, clusters

    # ------------------------------------------------------------------
    # HAZARD GENERATION
    # ------------------------------------------------------------------

    def _generate_hazards(self) -> Tuple[np.ndarray, List[HazardZone]]:
        """Generate ground-truth hazard zones as rectangles with intensity.

        Each hazard zone is a randomly placed rectangle with a random
        intensity value. Hazards can overlap; overlapping cells take the
        maximum intensity.

        Returns:
            hazard_map: Float (H, W) array with hazard intensity in [0, 1].
            hazard_zones: List of HazardZone metadata objects.
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        hazard_map = np.zeros((h, w), dtype=np.float32)
        hazard_zones: List[HazardZone] = []

        for _ in range(self.grid_cfg.num_hazard_zones):
            zone_h = self.rng.integers(
                self.grid_cfg.hazard_zone_size_range[0],
                self.grid_cfg.hazard_zone_size_range[1] + 1,
            )
            zone_w = self.rng.integers(
                self.grid_cfg.hazard_zone_size_range[0],
                self.grid_cfg.hazard_zone_size_range[1] + 1,
            )

            top_r = self.rng.integers(0, max(1, h - zone_h))
            top_c = self.rng.integers(0, max(1, w - zone_w))

            intensity = self.rng.uniform(
                self.grid_cfg.hazard_intensity_range[0],
                self.grid_cfg.hazard_intensity_range[1],
            )

            zone = HazardZone(
                top_left=(int(top_r), int(top_c)),
                height=int(zone_h),
                width=int(zone_w),
                intensity=float(intensity),
            )
            hazard_zones.append(zone)

            # Paint hazard onto the map (max-merge for overlaps)
            r_end = min(top_r + zone_h, h)
            c_end = min(top_c + zone_w, w)
            hazard_map[top_r:r_end, top_c:c_end] = np.maximum(
                hazard_map[top_r:r_end, top_c:c_end],
                intensity,
            )

        return hazard_map, hazard_zones

    # ------------------------------------------------------------------
    # BELIEF MAP INITIALIZATION
    # ------------------------------------------------------------------

    def _generate_initial_belief(self) -> np.ndarray:
        """Generate the initial belief map for the UAV agent.

        The belief map represents the UAV's subjective probability that
        each cell contains a survivor. It starts as a uniform prior
        (maximum entropy), reflecting complete initial uncertainty.

        Returns:
            belief_map: Float (H, W) array initialized to the configured
                initial belief value (default 0.5).
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        belief_map = np.full(
            (h, w),
            fill_value=self.grid_cfg.initial_survivor_belief,
            dtype=np.float32,
        )
        return belief_map

    # ------------------------------------------------------------------
    # DYNAMIC EVENT HELPERS
    # ------------------------------------------------------------------

    def expand_hazard(
        self,
        hazard_map: np.ndarray,
        hazard_zones: List[HazardZone],
        expansion_cells: int = 1,
        expansion_intensity: float = 0.4,
    ) -> Tuple[np.ndarray, List[HazardZone]]:
        """Expand a randomly chosen existing hazard zone by `expansion_cells`.

        Used by the dynamic event system to simulate worsening conditions.

        Args:
            hazard_map: Current hazard map (H, W).
            hazard_zones: List of existing hazard zone metadata.
            expansion_cells: Number of cells to expand in each direction.
            expansion_intensity: Intensity of the expanded region.

        Returns:
            Updated hazard_map and hazard_zones list.
        """
        if not hazard_zones:
            return hazard_map, hazard_zones

        h, w = hazard_map.shape
        idx = self.rng.integers(0, len(hazard_zones))
        zone = hazard_zones[idx]

        new_top_r = max(0, zone.top_left[0] - expansion_cells)
        new_top_c = max(0, zone.top_left[1] - expansion_cells)
        new_bot_r = min(h, zone.top_left[0] + zone.height + expansion_cells)
        new_bot_c = min(w, zone.top_left[1] + zone.width + expansion_cells)

        hazard_map[new_top_r:new_bot_r, new_top_c:new_bot_c] = np.maximum(
            hazard_map[new_top_r:new_bot_r, new_top_c:new_bot_c],
            expansion_intensity,
        )

        # Update the zone metadata
        expanded_zone = HazardZone(
            top_left=(new_top_r, new_top_c),
            height=new_bot_r - new_top_r,
            width=new_bot_c - new_top_c,
            intensity=max(zone.intensity, expansion_intensity),
        )
        hazard_zones[idx] = expanded_zone

        # Keep base safe
        base_r, base_c = GRID.base_position
        hazard_map[base_r, base_c] = 0.0

        return hazard_map, hazard_zones

    def create_new_hazard(
        self,
        hazard_map: np.ndarray,
        hazard_zones: List[HazardZone],
    ) -> Tuple[np.ndarray, List[HazardZone]]:
        """Create a brand-new small hazard zone at a random location.

        Args:
            hazard_map: Current hazard map (H, W).
            hazard_zones: List of existing hazard zone metadata.

        Returns:
            Updated hazard_map and hazard_zones list.
        """
        h, w = hazard_map.shape
        zone_h = self.rng.integers(1, 3)
        zone_w = self.rng.integers(1, 3)
        top_r = self.rng.integers(0, max(1, h - zone_h))
        top_c = self.rng.integers(0, max(1, w - zone_w))
        intensity = self.rng.uniform(0.2, 0.6)

        zone = HazardZone(
            top_left=(int(top_r), int(top_c)),
            height=int(zone_h),
            width=int(zone_w),
            intensity=float(intensity),
        )
        hazard_zones.append(zone)

        r_end = min(top_r + zone_h, h)
        c_end = min(top_c + zone_w, w)
        hazard_map[top_r:r_end, top_c:c_end] = np.maximum(
            hazard_map[top_r:r_end, top_c:c_end],
            intensity,
        )

        # Keep base safe
        base_r, base_c = GRID.base_position
        hazard_map[base_r, base_c] = 0.0

        return hazard_map, hazard_zones


# ==============================================================================
# MODULE-LEVEL CONVENIENCE
# ==============================================================================

def create_map(
    seed: Optional[int] = None,
    grid_cfg: GridConfig = GRID,
) -> Dict[str, np.ndarray | list]:
    """Convenience function to generate a complete map.

    Args:
        seed: Optional random seed for reproducibility.
        grid_cfg: Grid configuration to use.

    Returns:
        Dictionary of generated maps (see MapGenerator.generate()).
    """
    rng = np.random.default_rng(seed)
    generator = MapGenerator(grid_cfg=grid_cfg, rng=rng)
    return generator.generate()
