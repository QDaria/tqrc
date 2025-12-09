"""Utility functions for TQRC experiments.

Submodules:
    metrics: Performance metrics (NMSE, NRMSE, VPT, R², memory capacity)
    visualization: Plotting utilities (Lorenz attractor, predictions, memory capacity)
"""

from .metrics import (
    nmse,
    nrmse,
    valid_prediction_time,
    r_squared,
    pearson_correlation,
    memory_capacity,
)

# Visualization functions (optional dependency)
try:
    from .visualization import (
        plot_lorenz_attractor,
        plot_prediction,
        plot_memory_capacity,
        plot_phase_space,
        plot_vpt_analysis,
    )
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

__all__ = [
    # Metrics (always available)
    "nmse",
    "nrmse",
    "valid_prediction_time",
    "r_squared",
    "pearson_correlation",
    "memory_capacity",
]

# Add visualization functions if matplotlib available
if _HAS_VIZ:
    __all__.extend([
        "plot_lorenz_attractor",
        "plot_prediction",
        "plot_memory_capacity",
        "plot_phase_space",
        "plot_vpt_analysis",
    ])
