"""
CAIR-UAV Configuration Module
==============================
Central configuration for the Context-Adaptive Information-Gain-Driven UAV
system. All tunable hyperparameters, environment settings, reward weights,
and experiment configurations are defined here.

This module serves as the single source of truth for all configurable
parameters across the entire project. Modifying values here propagates
to every component that imports this module.

Author: CAIR-UAV Research Team
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Resolve project root relative to this config file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Ensure output directories exist
for _dir in [MODELS_DIR, LOGS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ==============================================================================
# GRID / MAP CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class GridConfig:
    """Immutable configuration for the 2D grid world."""

    width: int = 20
    height: int = 20
    base_position: Tuple[int, int] = (0, 0)  # UAV start / return-to-base cell

    # Ground-truth map generation
    num_survivor_clusters: int = 4       # Number of Gaussian survivor clusters
    survivor_cluster_std: float = 2.5    # Spatial spread (cells) of each cluster
    survivors_per_cluster_range: Tuple[int, int] = (3, 8)  # Min/max per cluster

    num_hazard_zones: int = 3            # Number of rectangular hazard zones
    hazard_zone_size_range: Tuple[int, int] = (2, 5)  # Side length min/max
    hazard_intensity_range: Tuple[float, float] = (0.3, 1.0)

    # Initial belief map
    initial_survivor_belief: float = 0.5  # Uniform prior (max entropy)

    # Observation / sensor model
    sensor_true_positive_rate: float = 0.85   # P(detect | survivor present)
    sensor_false_positive_rate: float = 0.10  # P(detect | survivor absent)
    sensor_range: int = 1  # Cells around UAV that get updated on SCAN

    @property
    def size(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def total_cells(self) -> int:
        return self.height * self.width


# ==============================================================================
# UAV RESOURCE CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class ResourceConfig:
    """Immutable configuration for UAV resource constraints."""

    max_battery: float = 100.0
    max_mission_time: int = 300          # Maximum discrete time steps
    max_computation_budget: float = 100.0
    max_communication_budget: float = 100.0

    # Energy costs per action
    move_energy_cost: float = 1.0        # Battery consumed per MOVE action
    scan_energy_cost: float = 2.0        # Battery consumed per HOVER_AND_SCAN
    hover_idle_energy: float = 0.3       # Passive drain per timestep

    # Computation costs
    scan_computation_cost: float = 1.5   # Computation consumed per SCAN
    move_computation_cost: float = 0.2   # Minor compute for movement planning

    # Communication costs
    scan_communication_cost: float = 1.0 # Bandwidth consumed per SCAN result
    report_communication_cost: float = 2.0  # Bandwidth for periodic reporting


# ==============================================================================
# OBSERVATION SPACE CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for the Dict observation space structure.

    The observation space is composed of:
        spatial_grid : Box(Channels, H, W)  — 3D CNN-compatible tensor
        telemetry    : Box(D,)              — 1D vector for MLP branch

    Spatial channels (order matters for indexing):
        0 : UAV position one-hot       (1.0 at UAV cell, 0.0 elsewhere)
        1 : Survivor belief map         (probability in [0, 1])
        2 : Uncertainty / entropy map   (Shannon entropy of belief)
        3 : Hazard map                  (hazard intensity in [0, 1])
        4 : Historical visit count      (normalized by max visits)

    Telemetry vector layout:
        [battery_norm, time_norm, x_norm, y_norm, compute_norm, comm_norm,
         ctx_exploration, ctx_investigation, ctx_resource_constrained,
         ctx_high_risk, ctx_emergency_return]
    """

    num_spatial_channels: int = 5
    telemetry_dim: int = 11  # 6 resource + 5 context features

    # Names for reference / logging
    spatial_channel_names: Tuple[str, ...] = (
        "uav_position",
        "survivor_belief",
        "uncertainty_entropy",
        "hazard_intensity",
        "visit_count_normalized",
    )
    telemetry_feature_names: Tuple[str, ...] = (
        "battery_remaining",
        "time_remaining",
        "current_x",
        "current_y",
        "computation_budget",
        "communication_budget",
        "ctx_exploration",
        "ctx_investigation",
        "ctx_resource_constrained",
        "ctx_high_risk",
        "ctx_emergency_return",
    )


# ==============================================================================
# REWARD CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class RewardConfig:
    """Weights and penalties for the composite reward function.

    Reward = (w_survivor * survivor_value)
           + (w_ig * information_gain)
           - (w_move_energy * movement_cost)
           - (w_scan_energy * scan_cost)
           - (w_hazard * hazard_penalty)
           - (w_time * time_penalty_per_step)
           + Terminal bonuses / penalties
    """

    # Positive reward weights
    w_survivor_found: float = 50.0       # Large reward for confirming a survivor
    w_information_gain: float = 10.0     # Reward proportional to entropy reduction
    w_new_cell_bonus: float = 1.0        # Small bonus for visiting unvisited cells

    # Negative cost weights
    w_move_energy: float = 0.5           # Per-move energy penalty
    w_scan_energy: float = 0.8           # Per-scan energy penalty
    w_hazard_penalty: float = 15.0       # Penalty for entering hazard cells
    w_time_penalty: float = 0.1          # Small penalty per time step to encourage speed

    # Terminal rewards / penalties
    w_mission_complete_bonus: float = 20.0   # Bonus if returned to base with survivors found
    w_stranded_penalty: float = -50.0        # Penalty if battery dies away from base
    w_out_of_time_penalty: float = -10.0     # Penalty for running out of mission time

    # Diminishing returns
    revisit_ig_decay: float = 0.5        # Multiplicative decay on IG for re-scanned cells
    min_ig_threshold: float = 0.01       # Below this IG, no IG reward is given


# ==============================================================================
# CONTEXT ANALYZER CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class ContextConfig:
    """Thresholds for computing context feature activations.

    Context features are continuous values in [0, 1] that are concatenated
    into the telemetry observation vector. They are NOT used for rule-based
    branching; they provide the RL agent with structured context awareness.
    """

    # EXPLORATION: High when large portions of map remain unvisited
    exploration_unvisited_ratio_threshold: float = 0.4

    # TARGET_INVESTIGATION: High when nearby cells have high survivor belief
    investigation_belief_threshold: float = 0.6
    investigation_radius: int = 3  # Cells around UAV to check

    # RESOURCE_CONSTRAINED: Activates when resources are running low
    resource_battery_threshold: float = 0.3   # Fraction of max
    resource_time_threshold: float = 0.3      # Fraction of max
    resource_compute_threshold: float = 0.2

    # HIGH_RISK: Activates near hazard zones
    high_risk_hazard_threshold: float = 0.3   # Min hazard intensity to flag
    high_risk_radius: int = 2

    # EMERGENCY_RETURN: Activates when battery is critically low
    emergency_battery_threshold: float = 0.15
    emergency_distance_factor: float = 1.5  # Multiplier on Manhattan distance


# ==============================================================================
# DYNAMIC ENVIRONMENT CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class DynamicEventConfig:
    """Configuration for stochastic environmental events."""

    event_probability: float = 0.05      # Probability of an event each timestep
    hazard_expansion_cells: int = 1      # How many cells a hazard can expand
    hazard_expansion_intensity: float = 0.4
    new_hazard_probability: float = 0.3  # Among events, chance of brand-new hazard
    survivor_drift_probability: float = 0.1  # Chance survivors shift cells
    max_events_per_episode: int = 10     # Cap to prevent chaos


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """PPO and training loop hyperparameters."""

    # PPO core
    learning_rate: float = 3e-4
    n_steps: int = 2048               # Steps per rollout collection
    batch_size: int = 64
    n_epochs: int = 10                # Optimization epochs per rollout
    gamma: float = 0.99               # Discount factor
    gae_lambda: float = 0.95          # GAE lambda
    clip_range: float = 0.2           # PPO clipping
    ent_coef: float = 0.01            # Entropy bonus coefficient
    vf_coef: float = 0.5              # Value function loss coefficient
    max_grad_norm: float = 0.5        # Gradient clipping

    # Training schedule
    total_timesteps: int = 1_000_000  # Total environment steps for training
    eval_freq: int = 10_000           # Evaluate every N steps
    eval_episodes: int = 20           # Episodes per evaluation
    save_freq: int = 50_000           # Checkpoint every N steps
    log_interval: int = 10            # Log every N rollouts

    # Feature extractor
    cnn_features_dim: int = 128       # Output dim of CNN feature extractor
    net_arch_pi: List[int] = field(default_factory=lambda: [256, 128])
    net_arch_vf: List[int] = field(default_factory=lambda: [256, 128])

    # Seeds
    seed: int = 42
    num_eval_seeds: int = 5           # Different seeds for evaluation


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """Settings for the full experimental evaluation suite."""

    num_episodes: int = 100            # Episodes per baseline/ablation
    num_seeds: int = 5                 # Random seeds for statistical significance
    seeds: Tuple[int, ...] = (42, 123, 456, 789, 1024)

    # Metrics to track
    tracked_metrics: Tuple[str, ...] = (
        "total_reward",
        "survivors_found",
        "area_coverage_pct",
        "total_information_gain",
        "battery_remaining",
        "time_steps_used",
        "mission_success",           # Returned to base before battery died
        "hazard_encounters",
        "unique_cells_visited",
    )

    # Ablation groups
    ablation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "A": ["standard_ppo"],
        "B": ["ppo_with_ig"],
        "C": ["ppo_with_ig_memory"],
        "D": ["ppo_with_ig_memory_context"],
        "E": ["full_cair_uav"],
    })


# ==============================================================================
# SINGLETON CONFIG INSTANCES
# ==============================================================================
# These are the canonical configuration objects used throughout the project.
# Import them directly: `from config import GRID, RESOURCE, REWARD, ...`

GRID = GridConfig()
RESOURCE = ResourceConfig()
OBSERVATION = ObservationConfig()
REWARD = RewardConfig()
CONTEXT = ContextConfig()
DYNAMIC = DynamicEventConfig()
TRAINING = TrainingConfig()
EXPERIMENT = ExperimentConfig()


# ==============================================================================
# ACTION SPACE CONSTANTS
# ==============================================================================

class Action:
    """Enumeration of discrete UAV actions.

    These map directly to Gymnasium Discrete(5) action indices.
    """
    MOVE_NORTH: int = 0   # y - 1 (up in grid)
    MOVE_SOUTH: int = 1   # y + 1 (down in grid)
    MOVE_EAST: int = 2    # x + 1 (right in grid)
    MOVE_WEST: int = 3    # x - 1 (left in grid)
    HOVER_AND_SCAN: int = 4

    NUM_ACTIONS: int = 5

    # Direction vectors (dy, dx) for movement actions
    DIRECTION_DELTAS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),   # NORTH: row decreases
        1: (1, 0),    # SOUTH: row increases
        2: (0, 1),    # EAST:  col increases
        3: (0, -1),   # WEST:  col decreases
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "MOVE_NORTH",
        1: "MOVE_SOUTH",
        2: "MOVE_EAST",
        3: "MOVE_WEST",
        4: "HOVER_AND_SCAN",
    }


# ==============================================================================
# UTILITY HELPERS
# ==============================================================================

def get_config_summary() -> str:
    """Return a human-readable summary of all active configuration values.

    Returns:
        str: Formatted multi-line string with all configuration parameters.
    """
    sections = [
        ("Grid", GRID),
        ("Resources", RESOURCE),
        ("Observation", OBSERVATION),
        ("Reward", REWARD),
        ("Context", CONTEXT),
        ("Dynamic Events", DYNAMIC),
        ("Training", TRAINING),
        ("Experiment", EXPERIMENT),
    ]
    lines = ["=" * 60, "CAIR-UAV Configuration Summary", "=" * 60]
    for name, cfg in sections:
        lines.append(f"\n--- {name} ---")
        for k, v in cfg.__dict__.items() if hasattr(cfg, '__dict__') else []:
            lines.append(f"  {k}: {v}")
        # dataclass fields
        if hasattr(cfg, '__dataclass_fields__'):
            for fname in cfg.__dataclass_fields__:
                lines.append(f"  {fname}: {getattr(cfg, fname)}")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
