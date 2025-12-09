# TQRC Python Package

Core implementation for Topological Quantum Reservoir Computing simulations.

## Modules

- `core/` - Fibonacci anyon braiding, fusion rules, and reservoir dynamics
- `utils/` - Helper functions and mathematical utilities
- `benchmarks/` - Performance benchmarking tools
- `constants.py` - Physical constants (golden ratio, braiding matrices)

## Usage

```python
from tqrc.core import FibonacciReservoir, BraidingOperator
from tqrc.constants import PHI, BRAIDING_SIGMA

# Initialize 6-anyon reservoir
reservoir = FibonacciReservoir(n_anyons=6)

# Apply braiding sequence
reservoir.apply_braiding([1, 2, 1, 3, 2])

# Measure reservoir state
output = reservoir.measure()
```
