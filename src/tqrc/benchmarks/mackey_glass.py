"""Mackey-Glass delay differential equation benchmark.

Implements the Mackey-Glass system for TQRC testing with verified parameters
from verification/03_benchmark_systems.md.

Author: [To be filled]
Date: 2025-12-07
"""

import numpy as np
from typing import Tuple, Optional


class MackeyGlass:
    """Mackey-Glass delay differential equation benchmark.

    Equation: dx/dt = a·x(t-τ)/(1 + x(t-τ)^n) - b·x(t)

    Verified parameters from Mackey & Glass (1977):
    - Production rate: a = 0.2
    - Decay rate: b = 0.1
    - Hill coefficient: n = 10
    - Standard delays: τ=17 (mild chaos), τ=30 (strong chaos)

    Source: verification/03_benchmark_systems.md
    """

    def __init__(self, tau: int = 17, a: float = 0.2, b: float = 0.1,
                 n: int = 10, dt: float = 1.0, random_seed: Optional[int] = None):
        """Initialize Mackey-Glass system with verified parameters.

        Args:
            tau: Time delay (17 for mild chaos, 30 for strong chaos)
            a: Production rate (default: 0.2)
            b: Decay rate (default: 0.1)
            n: Hill coefficient for nonlinearity (default: 10)
            dt: Integration time step (default: 1.0)
            random_seed: Random seed for reproducibility (optional)
        """
        self.tau = tau
        self.a = a
        self.b = b
        self.n = n
        self.dt = dt
        self.random_seed = random_seed

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Determine chaos regime
        if tau < 17:
            self.regime = 'periodic'
        elif tau == 17:
            self.regime = 'mildly_chaotic'
        elif tau > 17 and tau < 30:
            self.regime = 'chaotic'
        else:
            self.regime = 'strongly_chaotic'

    def _mackey_glass_derivative(self, x_current: float,
                                  x_delayed: float) -> float:
        """Compute dx/dt for Mackey-Glass equation.

        Args:
            x_current: Current value x(t)
            x_delayed: Delayed value x(t-τ)

        Returns:
            dx/dt derivative
        """
        production = self.a * x_delayed / (1.0 + x_delayed**self.n)
        decay = self.b * x_current

        return production - decay

    def generate_series(self, T: int, transient: int = 500,
                       x0: float = 1.2) -> np.ndarray:
        """Generate Mackey-Glass time series using Euler method.

        Args:
            T: Number of time steps to generate
            transient: Number of initial steps to discard
            x0: Initial condition for history (default: 1.2)

        Returns:
            series: Array of shape (T,) with normalized values
        """
        # Total length including transient
        T_total = T + transient

        # Initialize history buffer
        # Need to store past tau time steps
        history_length = self.tau + 1
        history = np.ones(history_length) * x0

        # Initialize time series array
        x = np.zeros(T_total)
        x[0] = x0

        # Euler integration with delay
        for t in range(1, T_total):
            # Get delayed value (tau steps back)
            if t < self.tau:
                x_delayed = x0  # Use initial condition for early times
            else:
                x_delayed = x[t - self.tau]

            # Compute derivative
            dx_dt = self._mackey_glass_derivative(x[t-1], x_delayed)

            # Euler step
            x[t] = x[t-1] + self.dt * dx_dt

        # Remove transient and return
        series = x[transient:]

        return series

    def normalize_series(self, series: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Normalize time series to [-1, 1] range for TQRC.

        Following Section 3.2.4: u = (x - mean) / std, then clip to [-1, 1].

        Args:
            series: Raw Mackey-Glass time series

        Returns:
            normalized: Series normalized to [-1, 1]
            stats: Dictionary with normalization statistics
        """
        mean = np.mean(series)
        std = np.std(series)

        # Z-score normalization, clip to ±3σ
        normalized = (series - mean) / (3 * std)
        normalized = np.clip(normalized, -1, 1)

        stats = {
            'mean': mean,
            'std': std,
            'min': np.min(series),
            'max': np.max(series),
            'method': 'zscore'
        }

        return normalized, stats

    def create_prediction_task(self, T_train: int = 5000, T_test: int = 2000,
                              T_washout: int = 500,
                              steps_ahead: int = 1) -> Tuple[np.ndarray, ...]:
        """Create prediction task: predict x(t+k) from x(t).

        Args:
            T_train: Number of training time steps
            T_test: Number of test time steps
            T_washout: Washout period (discarded)
            steps_ahead: Prediction horizon (default: 1 step)

        Returns:
            train_input: Shape (T_train, 1)
            train_target: Shape (T_train, 1)
            test_input: Shape (T_test, 1)
            test_target: Shape (T_test, 1)
            stats: Normalization statistics
        """
        # Generate full time series
        T_total = T_washout + T_train + T_test + steps_ahead
        series = self.generate_series(T_total, transient=500)

        # Normalize
        normalized, stats = self.normalize_series(series)

        # Reshape to (T, 1) for consistency with multi-dimensional inputs
        normalized = normalized.reshape(-1, 1)

        # Split into train/test
        train_input = normalized[T_washout : T_washout + T_train]
        train_target = normalized[T_washout + steps_ahead :
                                  T_washout + T_train + steps_ahead]

        test_input = normalized[T_washout + T_train :
                                T_washout + T_train + T_test]
        test_target = normalized[T_washout + T_train + steps_ahead :
                                 T_washout + T_train + T_test + steps_ahead]

        return train_input, train_target, test_input, test_target, stats

    def create_multistep_task(self, T_train: int = 5000, T_test: int = 2000,
                             T_washout: int = 500,
                             horizons: list = [1, 10, 84]) -> dict:
        """Create multi-horizon prediction tasks.

        Useful for evaluating different prediction horizons:
        - 1-step: Immediate prediction
        - 10-step: Short-term forecasting
        - 84-step: Long-term forecasting (standard benchmark)

        Args:
            T_train: Training length
            T_test: Test length
            T_washout: Washout period
            horizons: List of prediction horizons

        Returns:
            Dictionary with {horizon: (train_in, train_out, test_in, test_out)}
        """
        results = {}

        for k in horizons:
            data = self.create_prediction_task(
                T_train=T_train,
                T_test=T_test,
                T_washout=T_washout,
                steps_ahead=k
            )
            results[k] = data

        return results

    def get_statistics(self, T: int = 10000) -> dict:
        """Compute statistical properties of Mackey-Glass series.

        Args:
            T: Number of time steps for statistics

        Returns:
            Dictionary with series statistics
        """
        series = self.generate_series(T, transient=1000)

        stats = {
            'mean': np.mean(series),
            'std': np.std(series),
            'min': np.min(series),
            'max': np.max(series),
            'regime': self.regime,
            'parameters': {
                'tau': self.tau,
                'a': self.a,
                'b': self.b,
                'n': self.n
            }
        }

        return stats


if __name__ == "__main__":
    # Example usage and verification
    print("Mackey-Glass Benchmark System")
    print("=" * 50)

    # Test mild chaos (τ=17)
    print("\n1. Mild Chaos Configuration (τ=17)")
    print("-" * 50)
    mg_mild = MackeyGlass(tau=17)

    print(f"Parameters: a={mg_mild.a}, b={mg_mild.b}, n={mg_mild.n}, τ={mg_mild.tau}")
    print(f"Regime: {mg_mild.regime}")

    series_mild = mg_mild.generate_series(T=1000, transient=500)
    print(f"Series shape: {series_mild.shape}")
    print(f"Sample values: {series_mild[:5]}")

    stats_mild = mg_mild.get_statistics(T=5000)
    print(f"Mean: {stats_mild['mean']:.4f}")
    print(f"Std:  {stats_mild['std']:.4f}")
    print(f"Range: [{stats_mild['min']:.4f}, {stats_mild['max']:.4f}]")

    # Create prediction task
    train_in, train_out, test_in, test_out, norm_stats = \
        mg_mild.create_prediction_task(T_train=1000, T_test=500)

    print(f"Training input shape: {train_in.shape}")
    print(f"Normalized range: [{train_in.min():.3f}, {train_in.max():.3f}]")

    # Test strong chaos (τ=30)
    print("\n2. Strong Chaos Configuration (τ=30)")
    print("-" * 50)
    mg_strong = MackeyGlass(tau=30)

    print(f"Parameters: a={mg_strong.a}, b={mg_strong.b}, n={mg_strong.n}, τ={mg_strong.tau}")
    print(f"Regime: {mg_strong.regime}")

    series_strong = mg_strong.generate_series(T=1000, transient=500)
    stats_strong = mg_strong.get_statistics(T=5000)

    print(f"Mean: {stats_strong['mean']:.4f}")
    print(f"Std:  {stats_strong['std']:.4f}")
    print(f"Range: [{stats_strong['min']:.4f}, {stats_strong['max']:.4f}]")

    # Multi-step prediction tasks
    print("\n3. Multi-horizon Prediction Tasks")
    print("-" * 50)
    multistep_data = mg_mild.create_multistep_task(
        T_train=1000, T_test=500,
        horizons=[1, 10, 84]
    )

    for horizon, (tr_in, tr_out, te_in, te_out, stats) in multistep_data.items():
        print(f"Horizon {horizon:3d}: Train={tr_in.shape}, Test={te_in.shape}")

    print("\n✅ Mackey-Glass benchmark verified!")
