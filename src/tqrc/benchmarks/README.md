# TQRC Benchmark Systems

**Phase 1C Implementation - Complete**
**Date:** 2025-12-07
**Status:** ✅ ALL TESTS PASSING

---

## Overview

This module provides verified implementations of benchmark systems for evaluating Topological Quantum Reservoir Computing (TQRC) performance. All parameters are verified against primary sources with **zero hallucination tolerance**.

**Source:** `verification/03_benchmark_systems.md`

---

## Implemented Benchmarks

### 1. Lorenz-63 Chaotic Attractor (`lorenz.py`)

**Description:** The canonical chaotic system for testing long-term prediction.

**Verified Parameters:**
- σ (Prandtl number): 10.0
- ρ (Rayleigh number): 28.0
- β (Geometric factor): 8/3 = 2.666...
- Lyapunov exponent: λ₁ = 0.9056
- Lyapunov time: T_λ = 1.104 time units

**Source:** Lorenz (1963), Sprott (2003) Table 4.1

**Usage:**
```python
from tqrc.benchmarks import Lorenz63

# Initialize system
lorenz = Lorenz63()

# Generate trajectory
trajectory = lorenz.generate_trajectory(T=5000, transient=1000)

# Create training data (normalized to [-1, 1])
train_in, train_out, test_in, test_out, stats = \
    lorenz.create_training_data(T_train=5000, T_test=2000, steps_ahead=1)

# Get attractor statistics
stats = lorenz.get_attractor_statistics(T=10000)
```

**Evaluation Metric:**
- **VPT (Valid Prediction Time):** Duration before prediction error exceeds threshold
- **Target:** VPT ≥ 8 Lyapunov times (competitive with state-of-art hybrid RC)

---

### 2. Mackey-Glass Delay Differential Equation (`mackey_glass.py`)

**Description:** Standard time series prediction benchmark with tunable difficulty.

**Verified Parameters:**
- a (production rate): 0.2
- b (decay rate): 0.1
- n (Hill coefficient): 10
- τ (time delay): 17 (mild chaos), 30 (strong chaos)

**Source:** Mackey & Glass (1977), Scholarpedia

**Equation:**
```
dx/dt = a·x(t-τ)/(1 + x(t-τ)^n) - b·x(t)
```

**Usage:**
```python
from tqrc.benchmarks import MackeyGlass

# Mild chaos (standard benchmark)
mg_mild = MackeyGlass(tau=17)
series = mg_mild.generate_series(T=5000, transient=500)

# Strong chaos (harder benchmark)
mg_strong = MackeyGlass(tau=30)

# Create prediction task (1-step ahead)
train_in, train_out, test_in, test_out, stats = \
    mg_mild.create_prediction_task(T_train=5000, T_test=2000, steps_ahead=1)

# Multi-horizon prediction (1, 10, 84 steps)
multistep = mg_mild.create_multistep_task(
    T_train=5000, T_test=2000,
    horizons=[1, 10, 84]
)
```

**Evaluation Metric:**
- **NRMSE (Normalized Root Mean Square Error)**
- **Target (τ=17):** NRMSE < 0.05
- **Target (τ=30):** NRMSE < 0.10

---

### 3. Memory Capacity Test (`memory_capacity.py`)

**Description:** Measures how many past time steps a reservoir can linearly recall.

**Verified Protocol:**
1. Generate random input u(t) ~ Uniform[-1, 1]
2. For each delay k, train linear readout to predict u(t-k) from reservoir state
3. Compute r_k² (squared correlation)
4. Sum: MC = Σ r_k²

**Source:** Jaeger (2001) Section 4, Lukoševičius (2009)

**Theoretical Bound:**
- Classical RC: MC ≤ N (reservoir dimension)
- TQRC: MC ≤ F_{n-1} ~ φ^n (exponential scaling!)

**Usage:**
```python
from tqrc.benchmarks import MemoryCapacityTest, evaluate_tqrc_memory_capacity

# Create test instance
test = MemoryCapacityTest(
    reservoir_evolve_fn=reservoir.evolve,
    readout_train_fn=reservoir.train_readout,
    reservoir_dim=F_n_minus_1
)

# Measure capacity
MC, r_squared = test.measure_capacity(T=10000, k_max=50)

# Evaluate efficiency
efficiency = MC / F_n_minus_1
print(f"Memory Capacity: {MC:.2f}")
print(f"Efficiency: {efficiency:.2%}")

# Plot capacity curve
test.plot_capacity_curve(r_squared, save_path='mc_curve.png')
```

**For TQRC:**
```python
# Convenience function for TQRC reservoirs
results = evaluate_tqrc_memory_capacity(
    tqrc_reservoir=my_tqrc,
    n_anyons=6,  # F_5 = 5 dimensional Hilbert space
    T=10000,
    k_max=50
)

print(f"MC = {results['memory_capacity']:.2f}")
print(f"Bound = {results['theoretical_bound']}")
print(f"Efficiency = {results['efficiency']:.2%}")
print(f"Meets target (≥80%): {results['meets_target']}")
```

**Evaluation Metric:**
- **MC/F_{n-1} ratio**
- **Target:** ≥ 0.8 (80% efficiency, competitive with well-tuned ESN)

---

## Performance Targets

From `verification/03_benchmark_systems.md`:

| Benchmark | Metric | Target | Classical Baseline |
|-----------|--------|--------|-------------------|
| **Lorenz-63** | VPT (Lyapunov times) | ≥ 8 | 5-6 (large ESN), 8-10 (hybrid RC) |
| **Mackey-Glass τ=17** | NRMSE (1-step) | < 0.05 | 0.01-0.05 (well-tuned ESN) |
| **Mackey-Glass τ=30** | NRMSE (1-step) | < 0.10 | 0.05-0.15 (well-tuned ESN) |
| **Memory Capacity** | MC/F_{n-1} | ≥ 0.8 | 0.8-0.95 (state-of-art ESN) |

**TQRC Novel Hypothesis:**
- Memory capacity scales as **MC ~ φ^n** (exponential in anyon number)
- 12 anyons (F_11 = 89) ≈ 89 classical neurons in memory capacity!

---

## File Organization

```
src/tqrc/benchmarks/
├── __init__.py                # Package initialization
├── lorenz.py                  # Lorenz-63 implementation (~150 lines)
├── mackey_glass.py            # Mackey-Glass implementation (~150 lines)
├── memory_capacity.py         # Memory capacity test (~100 lines)
├── test_benchmarks.py         # Verification tests
└── README.md                  # This file
```

---

## Running Tests

**Verify all implementations:**
```bash
cd src/tqrc/benchmarks
python3 test_benchmarks.py
```

**Expected Output:**
```
======================================================================
TQRC BENCHMARK SYSTEMS - VERIFICATION TESTS
======================================================================

TEST 1: Lorenz-63 System
✅ Lorenz-63 test PASSED

TEST 2: Mackey-Glass System
✅ Mackey-Glass test PASSED

TEST 3: Memory Capacity Test
✅ Memory capacity test PASSED

TEST 4: Integration Test
✅ Integration test PASSED

======================================================================
ALL TESTS PASSED ✅
======================================================================
```

---

## Data Normalization

All benchmarks normalize inputs to **[-1, 1]** range following Section 3.2.4:

**Method: Z-Score with Clipping**
```python
u = (x - mean) / (3 × std)
u = clip(u, -1, 1)
```

**Why:**
- Preserves dynamics (unlike min-max normalization)
- Prevents extreme outliers
- Compatible with TQRC input encoding: θ = π·u

---

## Example: Complete Benchmark Workflow

```python
from tqrc.benchmarks import Lorenz63, MackeyGlass, MemoryCapacityTest

# 1. Lorenz-63 Prediction Task
lorenz = Lorenz63()
train_in, train_out, test_in, test_out, stats = \
    lorenz.create_training_data(T_train=5000, T_test=2000)

# Train TQRC
tqrc = TQRCReservoir(n_anyons=6)  # To be implemented
tqrc.train(train_in, train_out)

# Evaluate VPT
predictions = tqrc.predict(test_in, horizon=8.8)  # 8 Lyapunov times
vpt = compute_vpt(predictions, test_out, threshold=0.4)
print(f"Lorenz VPT: {vpt:.2f} Lyapunov times")

# 2. Mackey-Glass Prediction
mg = MackeyGlass(tau=17)
mg_train_in, mg_train_out, mg_test_in, mg_test_out, _ = \
    mg.create_prediction_task(T_train=5000, T_test=2000)

tqrc.train(mg_train_in, mg_train_out)
predictions = tqrc.predict(mg_test_in)
nrmse = compute_nrmse(predictions, mg_test_out)
print(f"Mackey-Glass NRMSE: {nrmse:.4f}")

# 3. Memory Capacity Measurement
mc_test = MemoryCapacityTest(
    reservoir_evolve_fn=tqrc.evolve,
    readout_train_fn=tqrc.train_readout,
    reservoir_dim=5  # F_5 for n=6 anyons
)

MC, r_squared = mc_test.measure_capacity(T=10000, k_max=50)
print(f"Memory Capacity: {MC:.2f} / 5 = {MC/5:.2%}")
```

---

## Verification Status

**Phase 1C: COMPLETE** ✅

| Component | Status | Lines | Tests |
|-----------|--------|-------|-------|
| Lorenz-63 | ✅ Verified | 153 | Pass |
| Mackey-Glass | ✅ Verified | 156 | Pass |
| Memory Capacity | ✅ Verified | 183 | Pass |
| Integration | ✅ Verified | - | Pass |

**All parameters verified against:**
- Lorenz (1963) "Deterministic Nonperiodic Flow"
- Mackey & Glass (1977) "Oscillation and chaos"
- Sprott (2003) *Chaos and Time-Series Analysis*
- Jaeger (2001) "Echo State Approach to Analysis"

**Zero hallucination tolerance maintained.**

---

## Dependencies

**Required:**
- `numpy` - Numerical arrays
- `scipy` - Integration (ODE solver for Lorenz)

**Optional:**
- `matplotlib` - Plotting memory capacity curves

**Install:**
```bash
pip install numpy scipy matplotlib
```

---

## Next Steps (Phase 2+)

**Ready for:**
1. TQRC reservoir implementation (`src/tqrc/reservoir.py`)
2. Performance evaluation on all benchmarks
3. Comparison with classical RC baselines
4. Paper results generation

**Integration with TQRC:**
- Input encoding: Benchmarks provide u(t) ∈ [-1, 1]
- TQRC maps: u → θ = π·u → B_i(θ) (fractional braiding)
- Output: Trained readout W_out from ridge regression

---

## References

1. **Lorenz, E. N.** "Deterministic Nonperiodic Flow" *J. Atmos. Sci.* **20**(2), 130-141 (1963)
2. **Mackey, M. C. & Glass, L.** "Oscillation and chaos" *Science* **197**(4300), 287-289 (1977)
3. **Sprott, J. C.** *Chaos and Time-Series Analysis* Oxford (2003)
4. **Jaeger, H.** "Echo State Approach" GMD Report 148 (2001)
5. **Pathak, J. et al.** "Model-Free Prediction" *Phys. Rev. Lett.* **120**, 024102 (2018)

---

**Document Status:** Phase 1C Complete
**Last Updated:** 2025-12-07
**Verification:** All tests passing ✅
