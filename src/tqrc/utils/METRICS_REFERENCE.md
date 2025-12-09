# TQRC Metrics - Quick Reference

**All formulas verified against primary sources**
**See `verification/03_benchmark_systems.md` for complete documentation**

---

## Performance Metrics

### 1. NMSE (Normalized Mean Squared Error)

```python
from tqrc.utils import nmse

error = nmse(y_pred, y_true)
```

**Formula:**
```
NMSE = <(y_pred - y_true)²> / <y_true²>
```

**Interpretation:**
- `NMSE = 0`: Perfect prediction
- `NMSE = 1`: Prediction variance equals signal variance
- `NMSE > 1`: Worse than mean prediction

**Source:** verification/03 Section 3

---

### 2. NRMSE (Normalized Root Mean Squared Error)

```python
from tqrc.utils import nrmse

error = nrmse(y_pred, y_true)
```

**Formula:**
```
NRMSE = √NMSE
```

**Targets:**
- `NRMSE < 0.05`: Excellent (Mackey-Glass τ=17 target)
- `NRMSE < 0.10`: Good
- `NRMSE > 0.30`: Poor

**Source:** verification/03 Section 3

---

### 3. VPT (Valid Prediction Time)

```python
from tqrc.utils import valid_prediction_time

vpt = valid_prediction_time(
    y_pred, y_true,
    lyapunov_time=1.104,  # T_λ for Lorenz-63
    dt=0.01,
    threshold=0.4
)
```

**Formula:**
```
VPT = t* / T_λ

where t* = first time when NMSE(t) > threshold
```

**Targets (Lorenz-63):**
- Small ESN: `VPT ~ 3-5 LT`
- Large ESN: `VPT ~ 5-6 LT`
- Hybrid RC: `VPT ~ 8-10 LT`
- **TQRC Target: `VPT ≥ 8 LT`**

**Source:** Pathak et al. (2018), verification/03 Section 3

---

### 4. R² (Coefficient of Determination)

```python
from tqrc.utils import r_squared

r2 = r_squared(y_pred, y_true)
```

**Formula:**
```
R² = 1 - SS_res / SS_tot

where:
  SS_res = Σ(y_true - y_pred)²
  SS_tot = Σ(y_true - ȳ_true)²
```

**Range:** `R² ∈ (-∞, 1]`
- `R² = 1`: Perfect prediction
- `R² > 0.9`: Strong correlation
- `R² < 0.5`: Weak correlation

**Source:** Standard statistical definition

---

### 5. Pearson Correlation

```python
from tqrc.utils import pearson_correlation

r = pearson_correlation(y_pred, y_true)
```

**Formula:**
```
r = <(y_pred - ȳ_pred)(y_true - ȳ_true)> / (σ_pred · σ_true)
```

**Range:** `r ∈ [-1, +1]`
- `r = +1`: Perfect positive correlation
- `r = -1`: Perfect negative correlation
- `r ≈ 0`: No correlation

**Relationship:** `r² = R²` (for linear regression)

**Source:** verification/03 Section 3

---

### 6. Memory Capacity

```python
from tqrc.utils import memory_capacity

mc = memory_capacity(r_squared_values)
mc_ratio = mc / reservoir_dim
```

**Formula:**
```
MC = Σ_{k=1}^{K_max} r_k²
```

**Theoretical Bound:** `MC ≤ N` (reservoir dimension)

**Performance Levels:**
- Poor: `MC/N ~ 0.3-0.5`
- Good: `MC/N ~ 0.6-0.8`
- Excellent: `MC/N ~ 0.8-0.95`
- Ideal: `MC/N = 1.0`

**TQRC Hypothesis:** `MC ~ φⁿ` (exponential in anyons!)

**Source:** Jaeger (2001), verification/03 Section 4

---

## Visualization Functions

### 1. Plot Lorenz Attractor

```python
from tqrc.utils import plot_lorenz_attractor

plot_lorenz_attractor(trajectory)  # trajectory.shape = (n_steps, 3)
```

### 2. Plot Prediction Quality

```python
from tqrc.utils import plot_prediction

nrmse_val = nrmse(y_pred, y_true)
plot_prediction(y_true, y_pred, nrmse_val, "NRMSE")
```

### 3. Plot Memory Capacity

```python
from tqrc.utils import plot_memory_capacity

plot_memory_capacity(r_squared_values, mc, theoretical_max=100)
```

### 4. Plot Phase Space Comparison

```python
from tqrc.utils import plot_phase_space

plot_phase_space(y_true, y_pred)  # 3D phase space
```

### 5. Plot VPT Analysis

```python
from tqrc.utils import plot_vpt_analysis

plot_vpt_analysis(y_true, y_pred, lyapunov_time=1.104, dt=0.01)
```

---

## System Parameters

### Lorenz-63

```python
# Standard chaotic regime
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

# Lyapunov exponents
lambda_1 = 0.9056  # Positive (chaos)
lambda_2 = 0.0     # Neutral
lambda_3 = -14.57  # Negative (dissipation)

# Lyapunov time
T_lambda = 1/lambda_1  # = 1.104 time units
```

**Source:** Lorenz (1963), Sprott (2003)

### Mackey-Glass

```python
# Standard parameters
a = 0.2
b = 0.1
n = 10

# Time delays
tau_mild = 17    # Mildly chaotic
tau_hard = 30    # Strongly chaotic
```

**Source:** Mackey & Glass (1977), Scholarpedia

---

## Complete Example

```python
import numpy as np
from tqrc import TQRCReservoir, Lorenz63
from tqrc.utils import nrmse, valid_prediction_time, plot_prediction

# 1. Generate benchmark data
lorenz = Lorenz63()
trajectory = lorenz.generate_trajectory(n_steps=10000)

# 2. Train TQRC
reservoir = TQRCReservoir(n_anyons=8)
reservoir.fit(trajectory[:8000])

# 3. Predict
y_pred = reservoir.predict(trajectory[8000:])
y_true = trajectory[8000:]

# 4. Evaluate
nrmse_val = nrmse(y_pred, y_true)
vpt = valid_prediction_time(
    y_pred, y_true,
    lyapunov_time=1.104,
    dt=0.01
)

print(f"NRMSE: {nrmse_val:.4f}")
print(f"VPT: {vpt:.2f} Lyapunov times")

# 5. Visualize
plot_prediction(y_true, y_pred, nrmse_val, "NRMSE")
```

---

## Error Handling

All metrics include robust error handling:

```python
# Shape mismatch
try:
    nmse(y_pred, y_true)
except ValueError as e:
    print(f"Error: {e}")  # "Shape mismatch: ..."

# NaN/Inf values
try:
    nmse(y_pred_with_nan, y_true)
except ValueError as e:
    print(f"Error: {e}")  # "y_pred contains NaN or Inf values"

# Zero variance (constant signal)
try:
    nmse(y_pred, np.ones(100))
except ValueError as e:
    print(f"Error: {e}")  # "y_true has zero variance"
```

---

## Testing

Run comprehensive test suite:

```bash
pytest tests/test_utils_metrics.py -v
```

Expected: **31 tests PASSED**

---

## References

1. **Lorenz, E. N.** (1963) "Deterministic Nonperiodic Flow"
2. **Mackey & Glass** (1977) "Oscillation and chaos in physiological control systems"
3. **Sprott, J. C.** (2003) *Chaos and Time-Series Analysis*
4. **Jaeger, H.** (2001) "The echo state approach to analysing and training recurrent neural networks"
5. **Pathak et al.** (2018) "Model-Free Prediction of Large Spatiotemporally Chaotic Systems"

---

*Quick reference for TQRC metrics*
*Full documentation: `verification/03_benchmark_systems.md`*
*Implementation: `src/tqrc/utils/metrics.py`*
