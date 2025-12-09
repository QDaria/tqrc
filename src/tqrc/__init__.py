"""Topological Quantum Reservoir Computing.

A Python implementation of reservoir computing using Fibonacci anyons
with topological protection.

Modules:
    core: Core TQRC components (anyons, braiding, encoding, reservoir, readout)
    benchmarks: Standard benchmark tasks (Lorenz-63, Mackey-Glass, memory capacity)
    utils: Utilities (metrics, visualization)

Example:
    >>> from tqrc import TQRCReservoir, Lorenz63
    >>>
    >>> # Create TQRC reservoir
    >>> reservoir = TQRCReservoir(n_anyons=8, delta_phi=0.1)
    >>>
    >>> # Generate benchmark data
    >>> lorenz = Lorenz63()
    >>> trajectory = lorenz.generate_trajectory(n_steps=10000)
    >>>
    >>> # Train and evaluate
    >>> reservoir.fit(trajectory[:8000])
    >>> predictions = reservoir.predict(trajectory[8000:])

References:
    See docs/mo-papers/tqrc-paper/ for full technical documentation.
"""

# Core components
from .core.anyons import FibonacciHilbertSpace
from .core.braiding import BraidingOperators
from .core.encoding import InputEncoder
from .core.reservoir import TQRCReservoir
from .core.readout import RidgeReadout

# Benchmarks
from .benchmarks.lorenz import Lorenz63
from .benchmarks.mackey_glass import MackeyGlass
from .benchmarks.memory_capacity import MemoryCapacityTest

# Metrics
from .utils.metrics import (
    nmse,
    nrmse,
    valid_prediction_time,
    r_squared,
    pearson_correlation,
    memory_capacity,
)

__version__ = "0.1.0"
__author__ = "Daniel Mo Houshmand"

__all__ = [
    # Core
    "FibonacciHilbertSpace",
    "BraidingOperators",
    "InputEncoder",
    "TQRCReservoir",
    "RidgeReadout",
    # Benchmarks
    "Lorenz63",
    "MackeyGlass",
    "MemoryCapacityTest",
    # Metrics
    "nmse",
    "nrmse",
    "valid_prediction_time",
    "r_squared",
    "pearson_correlation",
    "memory_capacity",
]
