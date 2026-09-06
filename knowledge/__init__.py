"""
CAIR-UAV Knowledge Package
============================
Modules for managing the UAV's epistemic state — belief representation,
information gain computation, and historical mission memory.

These modules provide the mathematical backbone for context-adaptive
decision-making under partial observability.

Modules:
    belief_map       — Bayesian belief state management and update logic
    information_gain — Shannon entropy and information gain calculations
    mission_memory   — Historical visit tracking and accumulated IG memory
"""

from knowledge.belief_map import BeliefMap
from knowledge.information_gain import InformationGainCalculator
from knowledge.mission_memory import MissionMemory

__all__ = ["BeliefMap", "InformationGainCalculator", "MissionMemory"]
