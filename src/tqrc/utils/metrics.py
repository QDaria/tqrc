"""Performance metrics for TQRC benchmarks.

All metric definitions verified against primary sources.
See verification/03_benchmark_systems.md for references.
"""

import numpy as np
from typing import Optional


def nmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Normalized Mean Squared Error.

    Definition from verification/03_benchmark_systems.md:
        NMSE = <(y_pred - y_true)²> / <y_true²>

    where <·> denotes time average.

    Args:
        y_pred: Predicted time series, shape (n_steps,) or (n_steps, n_features)
        y_true: Ground truth time series, same shape as y_pred

    Returns:
        NMSE value in [0, ∞)
        - NMSE = 0: Perfect prediction
        - NMSE = 1: Prediction variance equals signal variance
        - NMSE > 1: Worse than predicting the mean

    Raises:
        ValueError: If arrays have different total elements or contain NaN/Inf

    Source:
        Standard definition, verified in verification/03
    """
    # Flatten arrays for consistent comparison
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN or Inf values")
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or Inf values")

    # Time average <·>
    numerator = np.mean((y_pred - y_true) ** 2)
    denominator = np.mean(y_true ** 2)

    if denominator == 0:
        raise ValueError("y_true has zero variance (constant signal)")

    return numerator / denominator


def nrmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Normalized Root Mean Squared Error.

    Definition from verification/03_benchmark_systems.md:
        NRMSE = √NMSE = √(<(y_pred - y_true)²>) / √(<y_true²>)

    Relationship:
        NRMSE = √NMSE

    Args:
        y_pred: Predicted time series, shape (n_steps,) or (n_steps, n_features)
        y_true: Ground truth time series, same shape as y_pred

    Returns:
        NRMSE value in [0, ∞)
        - NRMSE < 0.05: Excellent prediction (Mackey-Glass τ=17 target)
        - NRMSE < 0.1: Good prediction
        - NRMSE > 0.3: Poor prediction

    Raises:
        ValueError: If arrays have different shapes or contain NaN/Inf

    Notes:
        More intuitive than NMSE (same units as signal RMS).

    Source:
        verification/03 Section 3 (Evaluation Metrics)
    """
    return np.sqrt(nmse(y_pred, y_true))


def valid_prediction_time(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    lyapunov_time: float,
    dt: float = 0.01,
    threshold: float = 0.4,
    metric: str = "nmse"
) -> float:
    """Valid Prediction Time in Lyapunov times.

    Definition from verification/03_benchmark_systems.md:
        VPT = t* / T_λ

    where:
        - t* is first time when error exceeds threshold
        - T_λ = Lyapunov time = 1/λ₁

    Args:
        y_pred: Predicted time series, shape (n_steps,) or (n_steps, n_features)
        y_true: Ground truth time series, same shape as y_pred
        lyapunov_time: T_λ in time units (e.g., 1.104 for Lorenz-63)
        dt: Time step between samples (default: 0.01)
        threshold: Error threshold (default: 0.4, typical 0.3-0.4)
        metric: Error metric to use ("nmse" or "mse")

    Returns:
        VPT in Lyapunov times
        - VPT = 1: Valid for 1 Lyapunov time
        - VPT = 5: Good long-term prediction
        - VPT ≥ 8: Excellent (competitive with hybrid RC for Lorenz-63)

    Raises:
        ValueError: If inputs are invalid

    Notes:
        For Lorenz-63 (λ₁ = 0.9056, T_λ = 1.104):
            - Small ESN: VPT ~ 3-5 LT
            - Large ESN: VPT ~ 5-6 LT
            - Hybrid RC: VPT ~ 8-10 LT
            - TQRC Target: VPT ≥ 8 LT

    Source:
        Pathak et al. (2018), standard practice in chaos prediction
        verification/03 Section 3 (Evaluation Metrics)
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
        )

    if lyapunov_time <= 0:
        raise ValueError(f"lyapunov_time must be positive, got {lyapunov_time}")

    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    n_steps = len(y_pred)

    # Compute error at each time step
    if metric == "nmse":
        # Compute cumulative NMSE up to each time step
        cumsum_squared_error = np.cumsum((y_pred - y_true) ** 2)
        cumsum_true_squared = np.cumsum(y_true ** 2)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            error_curve = cumsum_squared_error / cumsum_true_squared
        error_curve = np.nan_to_num(error_curve, nan=0.0, posinf=threshold + 1)
    elif metric == "mse":
        # Instantaneous MSE
        error_curve = (y_pred - y_true) ** 2
    else:
        raise ValueError(f"Unknown metric '{metric}', use 'nmse' or 'mse'")

    # Find first time when error exceeds threshold
    exceed_idx = np.where(error_curve > threshold)[0]

    if len(exceed_idx) == 0:
        # Error never exceeded threshold
        t_star = n_steps * dt
    else:
        t_star = exceed_idx[0] * dt

    # Convert to Lyapunov times
    vpt = t_star / lyapunov_time

    return vpt


def r_squared(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination R².

    Definition:
        R² = 1 - SS_res / SS_tot

    where:
        SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
        SS_tot = Σ(y_true - ȳ_true)²  (total sum of squares)

    Equivalently (for Pearson correlation r):
        R² = r²

    Args:
        y_pred: Predicted values, shape (n_samples,)
        y_true: Ground truth values, shape (n_samples,)

    Returns:
        R² value in (-∞, 1]
        - R² = 1: Perfect prediction
        - R² > 0.9: Strong correlation
        - R² < 0.5: Weak correlation
        - R² < 0: Model worse than mean prediction

    Raises:
        ValueError: If arrays have different shapes or lengths

    Notes:
        Used in memory capacity calculation:
            MC = Σ_{k=1}^{K_max} r_k²

    Source:
        Jaeger (2001) Section 4, standard statistical definition
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
        )

    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")

    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Total sum of squares
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    if ss_tot == 0:
        # Constant target (no variance to explain)
        if ss_res == 0:
            return 1.0  # Perfect prediction of constant
        else:
            return 0.0  # Can't predict constant correctly

    r2 = 1 - (ss_res / ss_tot)

    return r2


def pearson_correlation(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Pearson correlation coefficient.

    Definition from verification/03_benchmark_systems.md:
        r = <(y_pred - ȳ_pred)(y_true - ȳ_true)> / (σ_pred · σ_true)

    where:
        ȳ = time average
        σ = standard deviation

    Args:
        y_pred: Predicted values, shape (n_samples,)
        y_true: Ground truth values, shape (n_samples,)

    Returns:
        Correlation coefficient in [-1, +1]
        - r = 1: Perfect positive correlation
        - r > 0.9: Strong correlation
        - r < 0.5: Weak correlation
        - r = -1: Perfect negative correlation

    Raises:
        ValueError: If arrays have different shapes

    Notes:
        Relationship to R²: r² = R² for linear regression

    Source:
        Standard definition, verification/03 Section 3
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
        )

    if len(y_true) < 2:
        raise ValueError("Need at least 2 samples for correlation")

    # Center the data
    y_pred_centered = y_pred - np.mean(y_pred)
    y_true_centered = y_true - np.mean(y_true)

    # Standard deviations
    sigma_pred = np.std(y_pred, ddof=1)
    sigma_true = np.std(y_true, ddof=1)

    if sigma_pred == 0 or sigma_true == 0:
        # One or both signals are constant
        if sigma_pred == 0 and sigma_true == 0:
            # Both constant: perfect correlation if equal
            return 1.0 if np.allclose(y_pred, y_true) else 0.0
        else:
            # One constant, one varying: no correlation
            return 0.0

    # Pearson correlation
    r = np.mean(y_pred_centered * y_true_centered) / (sigma_pred * sigma_true)

    # Clamp to [-1, 1] to handle numerical errors
    r = np.clip(r, -1.0, 1.0)

    return r


def memory_capacity(
    r_squared_values: np.ndarray,
    max_delay: Optional[int] = None
) -> float:
    """Calculate total memory capacity.

    Definition from verification/03_benchmark_systems.md:
        MC = Σ_{k=1}^{K_max} r_k²

    where r_k is the correlation for k-step delayed recall.

    Args:
        r_squared_values: Array of R² values for each delay k, shape (K_max,)
        max_delay: Maximum delay to sum over (default: all)

    Returns:
        Total memory capacity

    Notes:
        Theoretical bound: MC ≤ N (reservoir dimension)

        Typical performance:
            - Poorly tuned ESN: MC/N ~ 0.3-0.5
            - Well-tuned ESN: MC/N ~ 0.6-0.8
            - Optimized ESN: MC/N ~ 0.8-0.95
            - Ideal (orthogonal): MC/N = 1.0

        TQRC hypothesis:
            MC ≤ F_{n-1} ~ φⁿ (exponential in number of anyons!)

    Source:
        Jaeger (2001) Section 4
        verification/03 Section 4 (Memory Capacity Task)
    """
    if max_delay is None:
        max_delay = len(r_squared_values)
    else:
        max_delay = min(max_delay, len(r_squared_values))

    mc = np.sum(r_squared_values[:max_delay])

    return mc
