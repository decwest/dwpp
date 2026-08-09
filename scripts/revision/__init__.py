"""Kinematic simulation studies for the DWPP journal-paper revision."""

from .simulator import DEFAULT_CONFIG, SimulationConfig, TrialResult, make_paths, simulate_trial

__all__ = [
    "DEFAULT_CONFIG",
    "SimulationConfig",
    "TrialResult",
    "make_paths",
    "simulate_trial",
]
