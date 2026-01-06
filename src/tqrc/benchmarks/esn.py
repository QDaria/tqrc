"""Echo State Network (ESN) baseline for reservoir computing comparisons.

Implements the classical ESN architecture as a performance baseline.
ESN serves as the gold standard for comparing TQRC performance.

Reference:
    Jaeger, H. (2001). The "echo state" approach to analysing and training
    recurrent neural networks. GMD Report 148.

Author: TQRC Authors
Date: 2026-01-06
"""

import numpy as np
from typing import Optional, Tuple


class ESN:
    """Echo State Network baseline reservoir.

    A classical recurrent neural network reservoir with:
    - Random sparse reservoir connectivity
    - Tanh nonlinearity for fading memory
    - Spectral radius < 1 for echo state property

    Attributes:
        n_reservoir: Number of reservoir neurons
        spectral_radius: Spectral radius of weight matrix (< 1 for ESP)
        input_scaling: Input weight scaling factor
        sparsity: Reservoir connection sparsity (0-1)
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_reservoir: int = 100,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.1,
        sparsity: float = 0.9,
        leak_rate: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """Initialize Echo State Network.

        Args:
            n_reservoir: Number of reservoir neurons
            spectral_radius: Target spectral radius (< 1 for ESP)
            input_scaling: Input weight scaling
            sparsity: Fraction of zero weights in reservoir
            leak_rate: Leaky integrator rate (1.0 = standard ESN)
            random_seed: Random seed for reproducibility
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.random_seed = random_seed

        # Initialize random state
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize reservoir and input weights."""
        N = self.n_reservoir

        # Input weights: uniform in [-input_scaling, input_scaling]
        self.W_in = np.random.uniform(-1, 1, (N, 1)) * self.input_scaling

        # Reservoir weights: sparse random matrix
        W = np.random.randn(N, N)

        # Apply sparsity mask
        mask = np.random.rand(N, N) > self.sparsity
        W = W * mask

        # Scale to target spectral radius
        if np.any(W):
            current_radius = np.max(np.abs(np.linalg.eigvals(W)))
            if current_radius > 0:
                W = W * (self.spectral_radius / current_radius)

        self.W_reservoir = W

        # Bias (optional, small random)
        self.bias = np.random.uniform(-0.1, 0.1, N)

        # Initial state
        self.state = np.zeros(N)

    def reset_state(self):
        """Reset reservoir state to zeros."""
        self.state = np.zeros(self.n_reservoir)

    def step(self, u: np.ndarray) -> np.ndarray:
        """Perform single reservoir update step.

        Args:
            u: Input vector, shape (input_dim,)

        Returns:
            Updated reservoir state
        """
        # Ensure u is 1D
        u = np.atleast_1d(u).flatten()

        # Reservoir update: x(t+1) = (1-α)x(t) + α·tanh(W_in·u + W·x + bias)
        pre_activation = (
            self.W_in @ u.reshape(-1, 1) +
            self.W_reservoir @ self.state.reshape(-1, 1) +
            self.bias.reshape(-1, 1)
        )
        new_state = np.tanh(pre_activation.flatten())

        # Leaky integration
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state

        return self.state.copy()

    def run_dynamics(
        self,
        inputs: np.ndarray,
        washout: int = 0
    ) -> np.ndarray:
        """Run reservoir dynamics on input sequence.

        Args:
            inputs: Input sequence, shape (T, input_dim)
            washout: Number of initial steps to discard

        Returns:
            Reservoir states after washout, shape (T - washout, n_reservoir)
        """
        self.reset_state()

        T = len(inputs)
        states = []

        for t in range(T):
            state = self.step(inputs[t])
            if t >= washout:
                states.append(state.copy())

        return np.array(states)

    def get_feature_dim(self) -> int:
        """Return reservoir state dimension."""
        return self.n_reservoir

    def compute_memory_capacity(
        self,
        max_delay: int = 100,
        sequence_length: int = 5000
    ) -> Tuple[np.ndarray, float]:
        """Compute linear memory capacity.

        Args:
            max_delay: Maximum delay to test
            sequence_length: Length of random test sequence

        Returns:
            (mc_per_delay, total_mc)
        """
        # Generate random input sequence
        np.random.seed(self.random_seed if self.random_seed else 42)
        u = np.random.uniform(-1, 1, sequence_length)

        # Collect reservoir states
        self.reset_state()
        states = self.run_dynamics(u.reshape(-1, 1), washout=100)

        # Memory capacity for each delay
        mc_delays = np.zeros(max_delay)

        for k in range(1, max_delay + 1):
            # Target: delayed input
            target = u[100:-k] if k > 0 else u[100:]
            X = states[:len(target)]

            if len(X) < 10:
                continue

            # Ridge regression
            ridge_alpha = 1e-6
            try:
                W = np.linalg.lstsq(
                    X.T @ X + ridge_alpha * np.eye(X.shape[1]),
                    X.T @ target,
                    rcond=None
                )[0]
                y_pred = X @ W

                # Correlation coefficient squared
                if np.std(target) > 0 and np.std(y_pred) > 0:
                    corr = np.corrcoef(target, y_pred)[0, 1]
                    mc_delays[k-1] = corr ** 2
            except Exception:
                pass

        total_mc = np.sum(mc_delays)
        return mc_delays, total_mc


def esn_benchmark(
    task_type: str = 'mackey_glass',
    n_reservoir: int = 100,
    n_trials: int = 30,
    random_seed: int = 42
) -> dict:
    """Run ESN benchmark for comparison with TQRC.

    Args:
        task_type: 'mackey_glass' or 'lorenz'
        n_reservoir: Number of reservoir neurons
        n_trials: Number of trials
        random_seed: Base random seed

    Returns:
        Dictionary with benchmark results
    """
    from .mackey_glass import MackeyGlass
    from ..utils.metrics import nrmse

    results = []

    for trial in range(n_trials):
        try:
            # Generate data
            mg = MackeyGlass(tau=17, random_seed=random_seed + trial)
            data = mg.generate_series(3000, transient=500)
            data_min, data_max = data.min(), data.max()
            data_norm = 2 * (data - data_min) / (data_max - data_min) - 1

            train_data = data_norm[:2000].reshape(-1, 1)
            test_data = data_norm[2000:].reshape(-1, 1)

            # Create ESN
            esn = ESN(
                n_reservoir=n_reservoir,
                spectral_radius=0.95,
                input_scaling=0.1,
                random_seed=random_seed + trial + 1000
            )

            # Train
            states = esn.run_dynamics(train_data, washout=500)
            X = states[:-1]
            y = train_data[501:len(states)+500].flatten()

            ridge_alpha = 1e-4
            W_out = np.linalg.lstsq(
                X.T @ X + ridge_alpha * np.eye(X.shape[1]),
                X.T @ y,
                rcond=None
            )[0]

            # Test
            test_states = esn.run_dynamics(test_data, washout=100)
            y_pred = test_states[:-1] @ W_out
            y_true = test_data[101:len(test_states)+100].flatten()

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
            'n_reservoir': n_reservoir,
            'task': task_type
        }
    else:
        return {
            'mean_nrmse': None,
            'std_nrmse': None,
            'n_valid': len(results),
            'n_reservoir': n_reservoir,
            'task': task_type,
            'error': 'Insufficient valid trials'
        }
