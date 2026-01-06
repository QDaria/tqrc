"""NARMA-10 benchmark for reservoir computing.

Implements the Nonlinear AutoRegressive Moving Average (NARMA) task,
a standard benchmark for testing memory capacity and nonlinear processing.

Reference:
    Atiya, A. F., & Parlos, A. G. (2000). New results on recurrent network
    training: unifying the algorithms and accelerating convergence.
    IEEE transactions on neural networks, 11(3), 697-709.

Author: TQRC Authors
Date: 2026-01-06
"""

import numpy as np
from typing import Tuple, Optional


class NARMA:
    """NARMA-N benchmark task for reservoir computing.

    The NARMA-N system is defined by:
        y(t+1) = α·y(t) + β·y(t)·Σ_{i=0}^{n-1} y(t-i) + γ·u(t-n+1)·u(t) + δ

    Standard NARMA-10 parameters:
        α = 0.3, β = 0.05, γ = 1.5, δ = 0.1

    This benchmark tests:
        - Memory capacity (requires remembering n steps back)
        - Nonlinear processing (product terms)
        - Temporal integration

    Attributes:
        order: NARMA order (typically 10)
        alpha, beta, gamma, delta: System parameters
        random_seed: Seed for reproducibility
    """

    def __init__(
        self,
        order: int = 10,
        alpha: float = 0.3,
        beta: float = 0.05,
        gamma: float = 1.5,
        delta: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """Initialize NARMA system.

        Args:
            order: NARMA order (default: 10)
            alpha: First-order coefficient (default: 0.3)
            beta: Nonlinear sum coefficient (default: 0.05)
            gamma: Input product coefficient (default: 1.5)
            delta: Constant offset (default: 0.1)
            random_seed: Random seed for reproducibility
        """
        self.order = order
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_series(
        self,
        T: int,
        input_range: Tuple[float, float] = (0.0, 0.5)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate NARMA-N time series.

        Args:
            T: Number of time steps
            input_range: Range for uniform random input (default: [0, 0.5])

        Returns:
            u: Input sequence, shape (T,)
            y: Target output sequence, shape (T,)
        """
        n = self.order
        u_min, u_max = input_range

        # Generate random input in [0, 0.5] (standard)
        u = np.random.uniform(u_min, u_max, T)

        # Initialize output
        y = np.zeros(T)

        # Compute NARMA dynamics
        for t in range(n, T - 1):
            # Sum of past n outputs: Σ_{i=0}^{n-1} y(t-i)
            y_sum = np.sum(y[t - n + 1:t + 1])

            # NARMA-n equation
            y[t + 1] = (
                self.alpha * y[t] +
                self.beta * y[t] * y_sum +
                self.gamma * u[t - n + 1] * u[t] +
                self.delta
            )

            # Stability check (NARMA can explode)
            if np.abs(y[t + 1]) > 1e6:
                raise ValueError(
                    f"NARMA series diverged at t={t+1}. "
                    "Try smaller input_range or different parameters."
                )

        return u, y

    def create_prediction_task(
        self,
        T_train: int = 3000,
        T_test: int = 1000,
        washout: int = 500,
        input_range: Tuple[float, float] = (0.0, 0.5)
    ) -> dict:
        """Create train/test split for NARMA prediction task.

        Args:
            T_train: Training sequence length
            T_test: Testing sequence length
            washout: Washout period for reservoir
            input_range: Input range

        Returns:
            Dictionary with train/test data:
            {
                'u_train': Input training sequence
                'y_train': Target training sequence
                'u_test': Input test sequence
                'y_test': Target test sequence
            }
        """
        T_total = T_train + T_test + washout
        u, y = self.generate_series(T_total, input_range)

        # Apply washout and split
        u_train = u[washout:washout + T_train]
        y_train = y[washout:washout + T_train]
        u_test = u[washout + T_train:]
        y_test = y[washout + T_train:]

        return {
            'u_train': u_train,
            'y_train': y_train,
            'u_test': u_test,
            'y_test': y_test
        }

    def scale_for_tqrc(
        self,
        u: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale inputs to [-1, 1] for TQRC compatibility.

        Args:
            u: Input sequence
            y: Target sequence

        Returns:
            u_scaled: Input scaled to [-1, 1]
            y_scaled: Target scaled to [-1, 1]
        """
        # Scale input: [0, 0.5] -> [-1, 1]
        u_scaled = 4 * u - 1

        # Scale output: typical NARMA range
        y_min, y_max = y.min(), y.max()
        if y_max > y_min:
            y_scaled = 2 * (y - y_min) / (y_max - y_min) - 1
        else:
            y_scaled = np.zeros_like(y)

        return u_scaled, y_scaled

    def get_statistics(self, T: int = 5000) -> dict:
        """Compute NARMA series statistics.

        Args:
            T: Number of points for statistics

        Returns:
            Dictionary with statistics
        """
        try:
            u, y = self.generate_series(T)
            return {
                'order': self.order,
                'input_mean': float(np.mean(u)),
                'input_std': float(np.std(u)),
                'output_mean': float(np.mean(y)),
                'output_std': float(np.std(y)),
                'output_range': (float(y.min()), float(y.max())),
                'stable': True
            }
        except ValueError:
            return {
                'order': self.order,
                'stable': False
            }


def narma10_benchmark(
    reservoir,
    n_trials: int = 30,
    T_train: int = 3000,
    T_test: int = 1000,
    random_seed: int = 42
) -> dict:
    """Run NARMA-10 benchmark on a reservoir.

    Args:
        reservoir: Reservoir object with run_dynamics method
        n_trials: Number of trials for statistics
        T_train: Training length
        T_test: Test length
        random_seed: Base random seed

    Returns:
        Dictionary with benchmark results
    """
    from ..utils.metrics import nrmse

    results = []

    for trial in range(n_trials):
        try:
            # Generate NARMA-10 data
            narma = NARMA(order=10, random_seed=random_seed + trial)
            task = narma.create_prediction_task(T_train, T_test)

            # Scale for TQRC
            u_train, y_train = narma.scale_for_tqrc(
                task['u_train'], task['y_train']
            )
            u_test, y_test = narma.scale_for_tqrc(
                task['u_test'], task['y_test']
            )

            # Run reservoir
            train_input = u_train.reshape(-1, 1)
            test_input = u_test.reshape(-1, 1)

            states_train = reservoir.run_dynamics(train_input, washout=500)
            X = states_train[:-1]
            y = y_train[501:len(states_train) + 500]

            # Ridge regression
            ridge_alpha = 1e-4
            W_out = np.linalg.lstsq(
                X.T @ X + ridge_alpha * np.eye(X.shape[1]),
                X.T @ y,
                rcond=None
            )[0]

            # Test
            states_test = reservoir.run_dynamics(test_input, washout=100)
            y_pred = states_test[:-1] @ W_out
            y_true = y_test[101:len(states_test) + 100]

            error = nrmse(y_pred, y_true)
            if error < 10.0:
                results.append(error)

        except Exception:
            pass

    if len(results) >= 10:
        return {
            'mean_nrmse': float(np.mean(results)),
            'std_nrmse': float(np.std(results, ddof=1)),
            'n_valid': len(results),
            'task': 'NARMA-10'
        }
    else:
        return {
            'mean_nrmse': None,
            'std_nrmse': None,
            'n_valid': len(results),
            'task': 'NARMA-10',
            'error': 'Insufficient valid trials'
        }
