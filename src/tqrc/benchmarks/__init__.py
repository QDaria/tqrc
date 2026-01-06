"""TQRC Benchmark Systems.

This module provides verified implementations of benchmark systems for testing
Topological Quantum Reservoir Computing (TQRC).

Available benchmarks:
- Lorenz63: Lorenz-63 chaotic attractor
- MackeyGlass: Mackey-Glass delay differential equation
- MemoryCapacityTest: Linear memory capacity measurement
- NARMA: NARMA-N benchmark (memory-focused)
- ESN: Echo State Network baseline

All parameters verified against primary sources (zero hallucination tolerance).

Source: verification/03_benchmark_systems.md
"""

from .lorenz import Lorenz63
from .mackey_glass import MackeyGlass
from .memory_capacity import MemoryCapacityTest, evaluate_tqrc_memory_capacity
from .narma import NARMA, narma10_benchmark
from .esn import ESN, esn_benchmark

__all__ = [
    'Lorenz63',
    'MackeyGlass',
    'MemoryCapacityTest',
    'evaluate_tqrc_memory_capacity',
    'NARMA',
    'narma10_benchmark',
    'ESN',
    'esn_benchmark',
]

__version__ = '0.1.0'
