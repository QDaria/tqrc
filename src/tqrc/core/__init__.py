"""
TQRC Core Module
================

Core implementation of Topological Quantum Reservoir Computing (TQRC)
using Fibonacci anyons.

Components:
    - anyons: Hilbert space for Fibonacci anyons
    - braiding: Braiding operators and topological gates
    - encoding: Input encoding (amplitude encoding)
    - reservoir: TQRC reservoir dynamics
    - readout: Ridge regression readout and training

Example Usage:
    >>> from tqrc.core import (
    ...     FibonacciHilbertSpace,
    ...     BraidingOperators,
    ...     InputEncoder,
    ...     TQRCReservoir,
    ...     RidgeReadout
    ... )
    >>>
    >>> # Create TQRC system
    >>> reservoir = TQRCReservoir(n_anyons=4, input_dim=1, braid_length=10)
    >>>
    >>> # Run dynamics
    >>> import numpy as np
    >>> input_seq = np.random.uniform(-1, 1, (1000, 1))
    >>> states = reservoir.run_dynamics(input_seq, washout=500)
    >>>
    >>> # Train readout
    >>> readout = RidgeReadout(state_dim=2, output_dim=1)
    >>> X = states[:800].T
    >>> Y = np.random.randn(1, 800)  # Target outputs
    >>> readout.train(X, Y)
    >>>
    >>> # Predict
    >>> y_pred = readout.predict(states[800])
"""

from .anyons import FibonacciHilbertSpace
from .braiding import BraidingOperators
from .encoding import InputEncoder, encode_scalar, encode_vector
from .reservoir import TQRCReservoir
from .readout import RidgeReadout, cross_validate_beta

__all__ = [
    'FibonacciHilbertSpace',
    'BraidingOperators',
    'InputEncoder',
    'encode_scalar',
    'encode_vector',
    'TQRCReservoir',
    'RidgeReadout',
    'cross_validate_beta',
]
