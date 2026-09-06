"""
Training Callbacks Module
==========================
Custom Stable-Baselines3 callbacks for the CAIR-UAV project.

These callbacks are invoked during PPO training to log domain-specific
metrics (e.g., survivors found, information gain, hazard encounters) to
TensorBoard. This allows researchers to monitor the agent's emergent
behavior and performance beyond just the raw RL reward.

Author: CAIR-UAV Research Team
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict


class MetricsLoggingCallback(BaseCallback):
    """Custom callback to log CAIR-UAV specific metrics to TensorBoard.

    This callback extracts information from the `info` dictionary returned
    by the environment at the end of each episode and logs it to the
    TensorBoard `rollout/` namespace.

    Attributes:
        verbose: Verbosity level (0: no output, 1: info, 2: debug).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Buffers for smoothing metrics over multiple episodes between logs
        self.survivors_found_buffer = []
        self.coverage_buffer = []
        self.total_ig_buffer = []
        self.battery_remaining_buffer = []
        self.success_buffer = []
        self.hazard_encounters_buffer = []

    def _on_step(self) -> bool:
        """Called at each environment step.

        We intercept the `info` dicts from all parallel environments. If an
        environment just finished an episode (`dones` is True), we extract
        and buffer its final metrics.

        Returns:
            Always returns True to continue training.
        """
        # Ensure we have access to info dicts and done flags
        if "infos" in self.locals and "dones" in self.locals:
            infos = self.locals["infos"]
            dones = self.locals["dones"]

            for idx, done in enumerate(dones):
                if done:
                    info = infos[idx]

                    # Extract custom metrics from the final step's info dict
                    self.survivors_found_buffer.append(info.get("survivors_found", 0))
                    self.coverage_buffer.append(info.get("area_coverage_pct", 0.0))
                    self.total_ig_buffer.append(info.get("total_information_gain", 0.0))
                    self.battery_remaining_buffer.append(info.get("battery_remaining", 0.0))
                    self.success_buffer.append(1.0 if info.get("mission_success", False) else 0.0)
                    self.hazard_encounters_buffer.append(info.get("hazard_encounters", 0))

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each PPO rollout (before optimization).

        We compute the mean of the buffered metrics and log them to TensorBoard.
        """
        if len(self.survivors_found_buffer) > 0:
            # Compute means
            mean_survivors = np.mean(self.survivors_found_buffer)
            mean_coverage = np.mean(self.coverage_buffer)
            mean_ig = np.mean(self.total_ig_buffer)
            mean_battery = np.mean(self.battery_remaining_buffer)
            mean_success = np.mean(self.success_buffer)
            mean_hazards = np.mean(self.hazard_encounters_buffer)

            # Log to TensorBoard
            self.logger.record("rollout/mean_survivors_found", mean_survivors)
            self.logger.record("rollout/mean_area_coverage_pct", mean_coverage)
            self.logger.record("rollout/mean_total_ig", mean_ig)
            self.logger.record("rollout/mean_battery_remaining", mean_battery)
            self.logger.record("rollout/mission_success_rate", mean_success)
            self.logger.record("rollout/mean_hazard_encounters", mean_hazards)

            # Clear buffers for the next rollout
            self.survivors_found_buffer.clear()
            self.coverage_buffer.clear()
            self.total_ig_buffer.clear()
            self.battery_remaining_buffer.clear()
            self.success_buffer.clear()
            self.hazard_encounters_buffer.clear()
