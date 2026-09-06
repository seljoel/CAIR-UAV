"""
Disaster Response Environment
==============================
Gymnasium-compatible environment for a single UAV performing autonomous
search-and-rescue in a partially observable, dynamic disaster scenario.

This is a POMDP formulation. The simulator maintains ground truth internally.
The UAV agent observes only its belief state, sensor readings, resource levels,
and context features. The agent must learn an effective sequential policy
using Reinforcement Learning (PPO with MultiInputPolicy).

Observation Space (gymnasium.spaces.Dict):
    spatial_grid : Box(5, 20, 20) float32
        Channel 0: UAV position one-hot
        Channel 1: Survivor belief map
        Channel 2: Uncertainty (Shannon entropy of belief)
        Channel 3: Hazard map (as observed / revealed)
        Channel 4: Historical visit count (normalized)
    telemetry : Box(11,) float32
        [battery, time, x, y, compute, comm,
         ctx_explore, ctx_investigate, ctx_resource, ctx_risk, ctx_emergency]

Action Space (gymnasium.spaces.Discrete(5)):
    0: MOVE_NORTH
    1: MOVE_SOUTH
    2: MOVE_EAST
    3: MOVE_WEST
    4: HOVER_AND_SCAN

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    GRID,
    RESOURCE,
    OBSERVATION,
    REWARD,
    CONTEXT,
    DYNAMIC,
    Action,
    GridConfig,
    ResourceConfig,
    ObservationConfig,
    RewardConfig,
    ContextConfig,
    DynamicEventConfig,
)
from environment.map_generator import MapGenerator, HazardZone


# ==============================================================================
# CONSTANTS
# ==============================================================================

_EPSILON = 1e-10  # Guard for log(0) in entropy calculations


# ==============================================================================
# MAIN ENVIRONMENT
# ==============================================================================

class DisasterResponseEnv(gym.Env):
    """Gymnasium environment for UAV disaster response.

    This environment implements a single-agent POMDP on a 2D grid where the
    UAV must search for survivors under resource constraints and dynamic hazards.

    The environment strictly separates ground truth (internal) from the agent's
    observations (belief-based). The agent never sees the true survivor map
    directly; it only receives noisy sensor updates when performing SCAN actions.

    Attributes:
        metadata: Gymnasium metadata dict.
        grid_cfg: Grid configuration.
        resource_cfg: Resource limits and costs.
        obs_cfg: Observation space structure.
        reward_cfg: Reward function weights.
        context_cfg: Context feature thresholds.
        dynamic_cfg: Dynamic event probabilities.
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_cfg: GridConfig = GRID,
        resource_cfg: ResourceConfig = RESOURCE,
        obs_cfg: ObservationConfig = OBSERVATION,
        reward_cfg: RewardConfig = REWARD,
        context_cfg: ContextConfig = CONTEXT,
        dynamic_cfg: DynamicEventConfig = DYNAMIC,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        enable_dynamic_events: bool = True,
        enable_information_gain: bool = True,
        enable_mission_memory: bool = True,
        enable_context_features: bool = True,
    ) -> None:
        """Initialize the disaster response environment.

        Args:
            grid_cfg: Grid dimensions and map generation parameters.
            resource_cfg: UAV resource limits and action costs.
            obs_cfg: Observation space channel/dimension configuration.
            reward_cfg: Reward function weights and penalties.
            context_cfg: Context feature activation thresholds.
            dynamic_cfg: Dynamic event probabilities and parameters.
            render_mode: One of None, "human", "ansi", "rgb_array".
            seed: Random seed for reproducibility.
            enable_dynamic_events: Whether stochastic events occur mid-episode.
            enable_information_gain: Include IG in reward (ablation toggle).
            enable_mission_memory: Track historical IG memory (ablation toggle).
            enable_context_features: Compute context features (ablation toggle).
        """
        super().__init__()

        # Store configs
        self.grid_cfg = grid_cfg
        self.resource_cfg = resource_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.context_cfg = context_cfg
        self.dynamic_cfg = dynamic_cfg
        self.render_mode = render_mode

        # Ablation toggles
        self.enable_dynamic_events = enable_dynamic_events
        self.enable_information_gain = enable_information_gain
        self.enable_mission_memory = enable_mission_memory
        self.enable_context_features = enable_context_features

        # Random number generator
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Map generator
        self.map_generator = MapGenerator(grid_cfg=grid_cfg, rng=self.rng)

        # ----- Define Gymnasium spaces -----

        h, w = grid_cfg.height, grid_cfg.width
        c = obs_cfg.num_spatial_channels

        self.observation_space = spaces.Dict({
            "spatial_grid": spaces.Box(
                low=0.0,
                high=np.float32(np.inf),
                shape=(c, h, w),
                dtype=np.float32,
            ),
            "telemetry": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_cfg.telemetry_dim,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Discrete(Action.NUM_ACTIONS)

        # ----- Internal state (initialized in reset) -----
        self._ground_truth_survivor_map: np.ndarray = np.zeros((h, w), dtype=np.float32)
        self._ground_truth_hazard_map: np.ndarray = np.zeros((h, w), dtype=np.float32)
        self._belief_map: np.ndarray = np.zeros((h, w), dtype=np.float32)
        self._observed_hazard_map: np.ndarray = np.zeros((h, w), dtype=np.float32)
        self._visit_count: np.ndarray = np.zeros((h, w), dtype=np.float32)
        self._scan_count: np.ndarray = np.zeros((h, w), dtype=np.float32)
        self._historical_ig: np.ndarray = np.zeros((h, w), dtype=np.float32)

        self._uav_pos: np.ndarray = np.array(grid_cfg.base_position, dtype=np.int32)
        self._battery: float = resource_cfg.max_battery
        self._time_step: int = 0
        self._computation_budget: float = resource_cfg.max_computation_budget
        self._communication_budget: float = resource_cfg.max_communication_budget

        self._survivors_found: int = 0
        self._found_survivor_cells: set = set()
        self._total_survivors: int = 0
        self._total_ig: float = 0.0
        self._hazard_encounters: int = 0
        self._dynamic_events_count: int = 0
        self._hazard_zones: List[HazardZone] = []

        self._episode_reward: float = 0.0
        self._done: bool = False

    # ==================================================================
    # GYMNASIUM API
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to a new episode.

        Generates a fresh ground-truth map and resets all UAV state.

        Args:
            seed: Optional seed override for this episode.
            options: Additional options (unused, for Gymnasium compatibility).

        Returns:
            observation: Dict with 'spatial_grid' and 'telemetry'.
            info: Dict with episode metadata.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.map_generator = MapGenerator(grid_cfg=self.grid_cfg, rng=self.rng)

        # Generate ground-truth maps
        map_data = self.map_generator.generate()

        h, w = self.grid_cfg.height, self.grid_cfg.width

        self._ground_truth_survivor_map = map_data["survivor_map"].copy()
        self._ground_truth_hazard_map = map_data["hazard_map"].copy()
        self._belief_map = map_data["belief_map"].copy()
        self._hazard_zones = list(map_data["hazard_zones"])
        self._total_survivors = map_data["total_survivors"]

        # Agent's observed hazard map starts empty (revealed by proximity)
        self._observed_hazard_map = np.zeros((h, w), dtype=np.float32)

        # Visit and scan tracking
        self._visit_count = np.zeros((h, w), dtype=np.float32)
        self._scan_count = np.zeros((h, w), dtype=np.float32)
        self._historical_ig = np.zeros((h, w), dtype=np.float32)

        # UAV state
        self._uav_pos = np.array(self.grid_cfg.base_position, dtype=np.int32)
        self._battery = self.resource_cfg.max_battery
        self._time_step = 0
        self._computation_budget = self.resource_cfg.max_computation_budget
        self._communication_budget = self.resource_cfg.max_communication_budget

        # Episode tracking
        self._survivors_found = 0
        self._found_survivor_cells = set()
        self._total_ig = 0.0
        self._hazard_encounters = 0
        self._dynamic_events_count = 0
        self._episode_reward = 0.0
        self._done = False

        # Mark initial position as visited
        self._visit_count[self._uav_pos[0], self._uav_pos[1]] = 1.0

        # Reveal hazards near starting position
        self._reveal_nearby_hazards()

        obs = self._build_observation()
        info = self._build_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, np.ndarray], SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one action in the environment.

        Args:
            action: Integer action index (0-4).

        Returns:
            observation: Updated Dict observation.
            reward: Scalar reward for this step.
            terminated: True if episode ended by terminal condition.
            truncated: True if episode ended by time limit.
            info: Dict with step metadata.
        """
        assert self.action_space.contains(action), (
            f"Invalid action {action}. Must be in [0, {Action.NUM_ACTIONS - 1}]."
        )

        if self._done:
            # If already done, return zero-reward terminal observation
            obs = self._build_observation()
            return obs, 0.0, True, False, self._build_info()

        # --- Initialize step reward components ---
        reward = 0.0
        survivor_reward = 0.0
        ig_reward = 0.0
        energy_cost = 0.0
        hazard_penalty = 0.0
        new_cell_bonus = 0.0
        terminated = False
        truncated = False

        # --- Passive resource drain ---
        self._battery -= self.resource_cfg.hover_idle_energy
        self._time_step += 1

        # --- Execute action ---
        if action == Action.HOVER_AND_SCAN:
            ig_reward, survivor_reward = self._execute_scan()
            energy_cost = self.resource_cfg.scan_energy_cost
            self._battery -= self.resource_cfg.scan_energy_cost
            self._computation_budget -= self.resource_cfg.scan_computation_cost
            self._communication_budget -= self.resource_cfg.scan_communication_cost
        elif action in Action.DIRECTION_DELTAS:
            dy, dx = Action.DIRECTION_DELTAS[action]
            new_y = int(np.clip(self._uav_pos[0] + dy, 0, self.grid_cfg.height - 1))
            new_x = int(np.clip(self._uav_pos[1] + dx, 0, self.grid_cfg.width - 1))

            self._uav_pos[0] = new_y
            self._uav_pos[1] = new_x

            energy_cost = self.resource_cfg.move_energy_cost
            self._battery -= self.resource_cfg.move_energy_cost
            self._computation_budget -= self.resource_cfg.move_computation_cost

            # Track visit
            if self._visit_count[new_y, new_x] == 0:
                new_cell_bonus = self.reward_cfg.w_new_cell_bonus
            self._visit_count[new_y, new_x] += 1.0

            # Reveal hazards near new position
            self._reveal_nearby_hazards()

            # Check hazard at new position
            hazard_intensity = self._ground_truth_hazard_map[new_y, new_x]
            if hazard_intensity > 0:
                hazard_penalty = self.reward_cfg.w_hazard_penalty * hazard_intensity
                self._hazard_encounters += 1

        # --- Dynamic events ---
        if self.enable_dynamic_events:
            self._process_dynamic_events()

        # --- Compute composite reward ---
        time_penalty = self.reward_cfg.w_time_penalty

        reward = (
            survivor_reward
            + ig_reward
            + new_cell_bonus
            - self.reward_cfg.w_move_energy * energy_cost
            - hazard_penalty
            - time_penalty
        )

        # --- Check terminal conditions ---
        # Battery depleted
        if self._battery <= 0:
            self._battery = 0.0
            terminated = True
            # Check if UAV is at base
            at_base = (
                self._uav_pos[0] == self.grid_cfg.base_position[0]
                and self._uav_pos[1] == self.grid_cfg.base_position[1]
            )
            if not at_base:
                reward += self.reward_cfg.w_stranded_penalty
            else:
                reward += self.reward_cfg.w_mission_complete_bonus

        # Time limit exceeded
        if self._time_step >= self.resource_cfg.max_mission_time:
            truncated = True
            at_base = (
                self._uav_pos[0] == self.grid_cfg.base_position[0]
                and self._uav_pos[1] == self.grid_cfg.base_position[1]
            )
            if at_base:
                reward += self.reward_cfg.w_mission_complete_bonus * 0.5
            else:
                reward += self.reward_cfg.w_out_of_time_penalty

        # Computation / communication exhausted (soft termination)
        if self._computation_budget <= 0:
            self._computation_budget = 0.0
        if self._communication_budget <= 0:
            self._communication_budget = 0.0

        self._done = terminated or truncated
        self._episode_reward += reward

        obs = self._build_observation()
        info = self._build_info()

        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    # SCAN / OBSERVATION UPDATE
    # ==================================================================

    def _execute_scan(self) -> Tuple[float, float]:
        """Execute a HOVER_AND_SCAN action.

        Updates the belief map using Bayesian inference with the sensor model.
        Computes information gain as the reduction in Shannon entropy.

        Returns:
            ig_reward: Weighted information gain reward.
            survivor_reward: Reward for newly confirmed survivors.
        """
        y, x = int(self._uav_pos[0]), int(self._uav_pos[1])
        h, w = self.grid_cfg.height, self.grid_cfg.width
        sensor_range = self.grid_cfg.sensor_range

        total_ig = 0.0
        survivor_reward = 0.0

        for dy in range(-sensor_range, sensor_range + 1):
            for dx in range(-sensor_range, sensor_range + 1):
                cy, cx = y + dy, x + dx
                if 0 <= cy < h and 0 <= cx < w:
                    # Get ground truth
                    ground_truth = self._ground_truth_survivor_map[cy, cx]

                    # Generate noisy observation
                    if ground_truth > 0.5:
                        # Survivor present
                        detected = (
                            self.rng.random()
                            < self.grid_cfg.sensor_true_positive_rate
                        )
                    else:
                        # No survivor
                        detected = (
                            self.rng.random()
                            < self.grid_cfg.sensor_false_positive_rate
                        )

                    # --- Compute pre-scan entropy ---
                    p_before = float(self._belief_map[cy, cx])
                    entropy_before = self._shannon_entropy(p_before)

                    # --- Bayesian update ---
                    p_after = self._bayesian_update(p_before, detected)
                    self._belief_map[cy, cx] = p_after

                    # --- Compute post-scan entropy ---
                    entropy_after = self._shannon_entropy(p_after)

                    # --- Information gain ---
                    ig = max(0.0, entropy_before - entropy_after)

                    # Apply diminishing returns for repeated scans
                    if self.enable_mission_memory:
                        scan_count = self._scan_count[cy, cx]
                        decay = self.reward_cfg.revisit_ig_decay ** scan_count
                        ig *= decay

                    if ig < self.reward_cfg.min_ig_threshold:
                        ig = 0.0

                    total_ig += ig

                    # Update memory
                    self._scan_count[cy, cx] += 1.0
                    self._historical_ig[cy, cx] += ig

                    # Check for newly confirmed survivor
                    cell_key = (cy, cx)
                    if (
                        p_after > 0.85
                        and ground_truth > 0.5
                        and cell_key not in self._found_survivor_cells
                    ):
                        self._found_survivor_cells.add(cell_key)
                        self._survivors_found += 1
                        survivor_reward += self.reward_cfg.w_survivor_found

        self._total_ig += total_ig

        if self.enable_information_gain:
            ig_reward = self.reward_cfg.w_information_gain * total_ig
        else:
            ig_reward = 0.0

        return ig_reward, survivor_reward

    # ==================================================================
    # BAYESIAN INFERENCE
    # ==================================================================

    @staticmethod
    def _bayesian_update(prior: float, detected: bool) -> float:
        """Update belief using Bayes' rule given a noisy sensor reading.

        P(survivor | detected) = P(detected | survivor) * P(survivor) / P(detected)

        Args:
            prior: Prior belief P(survivor) in [0, 1].
            detected: Whether the sensor reported a detection.

        Returns:
            Posterior belief P(survivor | observation) in [0, 1].
        """
        p = np.clip(prior, _EPSILON, 1.0 - _EPSILON)
        tp = GRID.sensor_true_positive_rate
        fp = GRID.sensor_false_positive_rate

        if detected:
            # P(det|surv)*P(surv)  /  P(det)
            numerator = tp * p
            denominator = tp * p + fp * (1.0 - p)
        else:
            # P(¬det|surv)*P(surv) / P(¬det)
            numerator = (1.0 - tp) * p
            denominator = (1.0 - tp) * p + (1.0 - fp) * (1.0 - p)

        denominator = max(denominator, _EPSILON)
        posterior = numerator / denominator
        return float(np.clip(posterior, _EPSILON, 1.0 - _EPSILON))

    @staticmethod
    def _shannon_entropy(p: float) -> float:
        """Compute Shannon entropy for a binary random variable.

        H(p) = -p * log2(p) - (1-p) * log2(1-p)

        Uses epsilon guards to handle p ≈ 0 or p ≈ 1 safely.

        Args:
            p: Probability in [0, 1].

        Returns:
            Entropy in [0, 1] (bits).
        """
        p = np.clip(p, _EPSILON, 1.0 - _EPSILON)
        return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

    # ==================================================================
    # OBSERVATION CONSTRUCTION
    # ==================================================================

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Construct the Dict observation for the agent.

        Returns:
            Dict with keys:
                'spatial_grid': np.ndarray of shape (C, H, W)
                'telemetry': np.ndarray of shape (D,)
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        c = self.obs_cfg.num_spatial_channels

        spatial = np.zeros((c, h, w), dtype=np.float32)

        # Channel 0: UAV position one-hot
        spatial[0, self._uav_pos[0], self._uav_pos[1]] = 1.0

        # Channel 1: Survivor belief map
        spatial[1] = self._belief_map.copy()

        # Channel 2: Uncertainty (entropy) map
        spatial[2] = self._compute_entropy_map()

        # Channel 3: Observed hazard map
        spatial[3] = self._observed_hazard_map.copy()

        # Channel 4: Normalized visit count
        max_visits = max(self._visit_count.max(), 1.0)
        spatial[4] = self._visit_count / max_visits

        # Telemetry vector
        telemetry = self._build_telemetry()

        return {
            "spatial_grid": spatial,
            "telemetry": telemetry,
        }

    def _compute_entropy_map(self) -> np.ndarray:
        """Compute cell-wise Shannon entropy of the belief map.

        Returns:
            np.ndarray of shape (H, W) with entropy values in [0, 1].
        """
        p = np.clip(self._belief_map, _EPSILON, 1.0 - _EPSILON)
        # 1. Catch any existing NaNs and default them to max uncertainty (0.5)
        p_clean = np.nan_to_num(p, nan=0.5)

        # 2. Use a larger epsilon (1e-4) to survive float32 rounding
        eps = 1e-4
        p_safe = np.clip(p_clean, eps, 1.0 - eps)

        # 3. Calculate entropy safely
        entropy = -p_safe * np.log2(p_safe) - (1.0 - p_safe) * np.log2(1.0 - p_safe)
        return entropy.astype(np.float32)

    def _build_telemetry(self) -> np.ndarray:
        """Build the 1D telemetry observation vector.

        All values are normalized to [0, 1].

        Returns:
            np.ndarray of shape (telemetry_dim,).
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width

        # Normalized resource levels
        battery_norm = np.clip(
            self._battery / self.resource_cfg.max_battery, 0.0, 1.0
        )
        time_norm = np.clip(
            1.0 - self._time_step / self.resource_cfg.max_mission_time, 0.0, 1.0
        )
        x_norm = self._uav_pos[1] / max(w - 1, 1)
        y_norm = self._uav_pos[0] / max(h - 1, 1)
        compute_norm = np.clip(
            self._computation_budget / self.resource_cfg.max_computation_budget,
            0.0,
            1.0,
        )
        comm_norm = np.clip(
            self._communication_budget / self.resource_cfg.max_communication_budget,
            0.0,
            1.0,
        )

        # Context features
        if self.enable_context_features:
            ctx = self._compute_context_features()
        else:
            ctx = np.zeros(5, dtype=np.float32)

        telemetry = np.array(
            [
                battery_norm,
                time_norm,
                x_norm,
                y_norm,
                compute_norm,
                comm_norm,
                ctx[0],  # exploration
                ctx[1],  # investigation
                ctx[2],  # resource constrained
                ctx[3],  # high risk
                ctx[4],  # emergency return
            ],
            dtype=np.float32,
        )

        return telemetry

    # ==================================================================
    # CONTEXT FEATURE COMPUTATION
    # ==================================================================

    def _compute_context_features(self) -> np.ndarray:
        """Compute 5 continuous context features in [0, 1].

        These are appended to the telemetry vector so the RL agent
        receives structured awareness of the mission context.

        Features:
            0. EXPLORATION       — fraction of unvisited cells
            1. INVESTIGATION     — max survivor belief within radius
            2. RESOURCE_CONSTRAINED — how resource-limited the UAV is
            3. HIGH_RISK         — max hazard intensity within radius
            4. EMERGENCY_RETURN  — urgency of returning to base

        Returns:
            np.ndarray of shape (5,) with values in [0, 1].
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        y, x = int(self._uav_pos[0]), int(self._uav_pos[1])

        # --- EXPLORATION ---
        total_cells = h * w
        visited_cells = np.count_nonzero(self._visit_count)
        unvisited_ratio = 1.0 - (visited_cells / total_cells)
        ctx_exploration = float(np.clip(
            unvisited_ratio / max(self.context_cfg.exploration_unvisited_ratio_threshold, _EPSILON),
            0.0,
            1.0,
        ))

        # --- INVESTIGATION ---
        inv_radius = self.context_cfg.investigation_radius
        r_min = max(0, y - inv_radius)
        r_max = min(h, y + inv_radius + 1)
        c_min = max(0, x - inv_radius)
        c_max = min(w, x + inv_radius + 1)
        local_belief = self._belief_map[r_min:r_max, c_min:c_max]
        max_local_belief = float(local_belief.max()) if local_belief.size > 0 else 0.0
        ctx_investigation = float(np.clip(
            max_local_belief / max(self.context_cfg.investigation_belief_threshold, _EPSILON),
            0.0,
            1.0,
        ))

        # --- RESOURCE_CONSTRAINED ---
        battery_frac = self._battery / self.resource_cfg.max_battery
        time_frac = 1.0 - self._time_step / self.resource_cfg.max_mission_time
        compute_frac = self._computation_budget / self.resource_cfg.max_computation_budget

        resource_stress = 0.0
        if battery_frac < self.context_cfg.resource_battery_threshold:
            resource_stress = max(
                resource_stress,
                1.0 - battery_frac / self.context_cfg.resource_battery_threshold,
            )
        if time_frac < self.context_cfg.resource_time_threshold:
            resource_stress = max(
                resource_stress,
                1.0 - time_frac / self.context_cfg.resource_time_threshold,
            )
        if compute_frac < self.context_cfg.resource_compute_threshold:
            resource_stress = max(
                resource_stress,
                1.0 - compute_frac / self.context_cfg.resource_compute_threshold,
            )
        ctx_resource = float(np.clip(resource_stress, 0.0, 1.0))

        # --- HIGH_RISK ---
        risk_radius = self.context_cfg.high_risk_radius
        r_min_risk = max(0, y - risk_radius)
        r_max_risk = min(h, y + risk_radius + 1)
        c_min_risk = max(0, x - risk_radius)
        c_max_risk = min(w, x + risk_radius + 1)
        local_hazard = self._observed_hazard_map[r_min_risk:r_max_risk, c_min_risk:c_max_risk]
        max_local_hazard = float(local_hazard.max()) if local_hazard.size > 0 else 0.0
        ctx_high_risk = float(np.clip(
            max_local_hazard / max(self.context_cfg.high_risk_hazard_threshold, _EPSILON),
            0.0,
            1.0,
        ))

        # --- EMERGENCY_RETURN ---
        base_y, base_x = self.grid_cfg.base_position
        manhattan_dist = abs(y - base_y) + abs(x - base_x)
        # How many steps of battery remain (each move costs move_energy_cost)
        steps_remaining = self._battery / max(self.resource_cfg.move_energy_cost, _EPSILON)
        # Emergency = distance is large relative to remaining steps
        if steps_remaining > 0:
            urgency = (
                manhattan_dist * self.context_cfg.emergency_distance_factor
            ) / steps_remaining
        else:
            urgency = 1.0
        # Also activate if battery is critically low
        if battery_frac < self.context_cfg.emergency_battery_threshold:
            urgency = max(urgency, 1.0)
        ctx_emergency = float(np.clip(urgency, 0.0, 1.0))

        return np.array(
            [ctx_exploration, ctx_investigation, ctx_resource, ctx_high_risk, ctx_emergency],
            dtype=np.float32,
        )

    # ==================================================================
    # HAZARD REVELATION
    # ==================================================================

    def _reveal_nearby_hazards(self) -> None:
        """Reveal ground-truth hazard intensity for cells near the UAV.

        The UAV can observe hazard zones within its sensor range. This
        simulates visual / thermal detection of hazards (fires, floods, etc.).
        """
        y, x = int(self._uav_pos[0]), int(self._uav_pos[1])
        h, w = self.grid_cfg.height, self.grid_cfg.width
        sr = self.grid_cfg.sensor_range + 1  # Slightly larger for hazard visibility

        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                cy, cx = y + dy, x + dx
                if 0 <= cy < h and 0 <= cx < w:
                    # Reveal ground truth hazard
                    self._observed_hazard_map[cy, cx] = max(
                        self._observed_hazard_map[cy, cx],
                        self._ground_truth_hazard_map[cy, cx],
                    )

    # ==================================================================
    # DYNAMIC EVENTS
    # ==================================================================

    def _process_dynamic_events(self) -> None:
        """Process stochastic environmental events.

        Each timestep has a small probability of triggering a dynamic event
        (hazard expansion, new hazard, or survivor drift). This models the
        unpredictable evolution of real disaster scenarios.
        """
        if self._dynamic_events_count >= self.dynamic_cfg.max_events_per_episode:
            return

        if self.rng.random() >= self.dynamic_cfg.event_probability:
            return

        self._dynamic_events_count += 1

        if self.rng.random() < self.dynamic_cfg.new_hazard_probability:
            # Spawn a brand-new hazard
            self._ground_truth_hazard_map, self._hazard_zones = (
                self.map_generator.create_new_hazard(
                    self._ground_truth_hazard_map,
                    self._hazard_zones,
                )
            )
        else:
            # Expand an existing hazard
            self._ground_truth_hazard_map, self._hazard_zones = (
                self.map_generator.expand_hazard(
                    self._ground_truth_hazard_map,
                    self._hazard_zones,
                    expansion_cells=self.dynamic_cfg.hazard_expansion_cells,
                    expansion_intensity=self.dynamic_cfg.hazard_expansion_intensity,
                )
            )

        # Survivor drift (rare)
        if self.rng.random() < self.dynamic_cfg.survivor_drift_probability:
            self._drift_survivors()

    def _drift_survivors(self) -> None:
        """Randomly shift some survivors to adjacent cells.

        Simulates movement of mobile survivors (people trying to escape,
        moving to higher ground, etc.). Only unfound survivors can drift.
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        new_map = self._ground_truth_survivor_map.copy()

        survivor_locs = np.argwhere(self._ground_truth_survivor_map > 0.5)
        for loc in survivor_locs:
            sy, sx = int(loc[0]), int(loc[1])
            if (sy, sx) in self._found_survivor_cells:
                continue  # Don't move already-found survivors

            if self.rng.random() < 0.3:  # 30% chance each survivor moves
                dy = self.rng.integers(-1, 2)
                dx = self.rng.integers(-1, 2)
                ny = int(np.clip(sy + dy, 0, h - 1))
                nx = int(np.clip(sx + dx, 0, w - 1))

                # Don't move into hazard
                if self._ground_truth_hazard_map[ny, nx] < 0.3:
                    new_map[sy, sx] = 0.0
                    new_map[ny, nx] = 1.0

        self._ground_truth_survivor_map = new_map

    # ==================================================================
    # INFO DICT
    # ==================================================================

    def _build_info(self) -> Dict[str, Any]:
        """Build the info dictionary returned with each step.

        Returns:
            Dict containing episode metrics and diagnostic information.
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        total_cells = h * w
        unique_visited = int(np.count_nonzero(self._visit_count))

        at_base = (
            self._uav_pos[0] == self.grid_cfg.base_position[0]
            and self._uav_pos[1] == self.grid_cfg.base_position[1]
        )

        return {
            "time_step": self._time_step,
            "battery_remaining": float(self._battery),
            "battery_fraction": float(self._battery / self.resource_cfg.max_battery),
            "computation_remaining": float(self._computation_budget),
            "communication_remaining": float(self._communication_budget),
            "uav_position": (int(self._uav_pos[0]), int(self._uav_pos[1])),
            "survivors_found": self._survivors_found,
            "total_survivors": self._total_survivors,
            "survivor_find_rate": (
                self._survivors_found / max(self._total_survivors, 1)
            ),
            "total_information_gain": float(self._total_ig),
            "unique_cells_visited": unique_visited,
            "area_coverage_pct": float(unique_visited / total_cells * 100.0),
            "hazard_encounters": self._hazard_encounters,
            "dynamic_events": self._dynamic_events_count,
            "episode_reward": float(self._episode_reward),
            "at_base": at_base,
            "mission_success": at_base and self._battery > 0,
        }

    # ==================================================================
    # RENDERING
    # ==================================================================

    def render(self) -> Optional[str | np.ndarray]:
        """Render the environment.

        Returns:
            String representation (ansi mode) or RGB array, or None.
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render a text-based view of the environment.

        Legend:
            U = UAV position
            S = Known survivor (found)
            H = Hazard
            . = Unvisited
            ~ = Visited
            B = Base

        Returns:
            Multi-line string representation.
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        lines = [f"Step: {self._time_step}  Battery: {self._battery:.1f}  "
                 f"Found: {self._survivors_found}/{self._total_survivors}  "
                 f"IG: {self._total_ig:.2f}"]
        lines.append("+" + "-" * w + "+")

        for r in range(h):
            row_chars = []
            for c in range(w):
                if r == self._uav_pos[0] and c == self._uav_pos[1]:
                    row_chars.append("U")
                elif (r, c) == self.grid_cfg.base_position:
                    row_chars.append("B")
                elif (r, c) in self._found_survivor_cells:
                    row_chars.append("S")
                elif self._observed_hazard_map[r, c] > 0.3:
                    row_chars.append("H")
                elif self._visit_count[r, c] > 0:
                    row_chars.append("~")
                else:
                    row_chars.append(".")
            lines.append("|" + "".join(row_chars) + "|")

        lines.append("+" + "-" * w + "+")
        return "\n".join(lines)

    def _render_rgb(self) -> np.ndarray:
        """Render an RGB image of the environment.

        Returns:
            np.ndarray of shape (H*cell_size, W*cell_size, 3) uint8.
        """
        h, w = self.grid_cfg.height, self.grid_cfg.width
        cell_size = 20
        img = np.ones((h * cell_size, w * cell_size, 3), dtype=np.uint8) * 40

        for r in range(h):
            for c in range(w):
                y0, y1 = r * cell_size, (r + 1) * cell_size
                x0, x1 = c * cell_size, (c + 1) * cell_size

                # Background: visited cells get lighter
                if self._visit_count[r, c] > 0:
                    img[y0:y1, x0:x1] = [60, 70, 80]

                # Hazard overlay (red tint)
                hazard_val = self._observed_hazard_map[r, c]
                if hazard_val > 0:
                    red_intensity = int(hazard_val * 180)
                    img[y0:y1, x0:x1, 0] = min(
                        255, img[y0:y1, x0:x1, 0].astype(int) + red_intensity
                    ).astype(np.uint8)

                # Belief intensity (green channel for high belief)
                belief_val = self._belief_map[r, c]
                if belief_val > 0.6:
                    green_intensity = int((belief_val - 0.5) * 300)
                    img[y0:y1, x0:x1, 1] = np.minimum(
                        255,
                        img[y0:y1, x0:x1, 1].astype(int) + green_intensity,
                    ).astype(np.uint8)

                # Found survivors (bright green square)
                if (r, c) in self._found_survivor_cells:
                    img[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = [0, 255, 100]

                # Base (blue square)
                if (r, c) == self.grid_cfg.base_position:
                    img[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = [50, 100, 255]

        # UAV position (white dot)
        uy, ux = int(self._uav_pos[0]), int(self._uav_pos[1])
        cy = uy * cell_size + cell_size // 2
        cx = ux * cell_size + cell_size // 2
        radius = cell_size // 3
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                        img[py, px] = [255, 255, 255]

        return img

    # ==================================================================
    # UTILITY PROPERTIES
    # ==================================================================

    @property
    def uav_position(self) -> Tuple[int, int]:
        """Current UAV grid position (row, col)."""
        return (int(self._uav_pos[0]), int(self._uav_pos[1]))

    @property
    def battery_fraction(self) -> float:
        """Remaining battery as a fraction of maximum."""
        return float(self._battery / self.resource_cfg.max_battery)

    @property
    def time_fraction(self) -> float:
        """Remaining time as a fraction of maximum."""
        return float(1.0 - self._time_step / self.resource_cfg.max_mission_time)

    @property
    def coverage_fraction(self) -> float:
        """Fraction of grid cells visited at least once."""
        total = self.grid_cfg.height * self.grid_cfg.width
        visited = int(np.count_nonzero(self._visit_count))
        return float(visited / total)

    def get_ground_truth(self) -> Dict[str, np.ndarray]:
        """Return ground-truth maps (for visualization / evaluation only).

        WARNING: This should NEVER be used by the agent during training
        or evaluation. It is provided solely for diagnostic plots.

        Returns:
            Dict with 'survivor_map' and 'hazard_map'.
        """
        return {
            "survivor_map": self._ground_truth_survivor_map.copy(),
            "hazard_map": self._ground_truth_hazard_map.copy(),
        }
