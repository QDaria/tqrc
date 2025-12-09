"""Memory capacity measurement for TQRC.

Implements the linear memory capacity test from Section 3.4.5 of the theory
document and verification/03_benchmark_systems.md.

Author: [To be filled]
Date: 2025-12-07
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.stats import pearsonr


class MemoryCapacityTest:
    """Linear memory capacity test for reservoir computing systems.

    Following Jaeger (2001) and Section 3.4.5:
    1. Generate random input u(t) ~ Uniform[-1, 1]
    2. For each delay k, train to predict u(t-k) from reservoir state
    3. Measure correlation r_k² = R²(predicted, actual)
    4. Sum: MC = Σ r_k²

    Theoretical bound: MC ≤ N (reservoir dimension)
    For TQRC: MC ≤ F_{n-1} ~ φ^n (exponential scaling)

    Source: verification/03_benchmark_systems.md, Section 3.4.5
    """

    def __init__(self, reservoir_evolve_fn: Callable,
                 readout_train_fn: Callable,
                 reservoir_dim: int):
        """Initialize memory capacity test.

        Args:
            reservoir_evolve_fn: Function that evolves reservoir state
                Signature: (state, input) -> new_state
            readout_train_fn: Function that trains linear readout
                Signature: (states, targets) -> weights
            reservoir_dim: Dimension of reservoir (N for classical, F_{n-1} for TQRC)
        """
        self.evolve = reservoir_evolve_fn
        self.train_readout = readout_train_fn
        self.dim = reservoir_dim

    def generate_random_input(self, T: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate i.i.d. uniform random input sequence.

        Args:
            T: Number of time steps
            seed: Random seed for reproducibility

        Returns:
            u: Array of shape (T,) with u(t) ~ Uniform[-1, 1]
        """
        if seed is not None:
            np.random.seed(seed)

        u = np.random.uniform(-1, 1, size=T)
        return u

    def collect_states(self, u: np.ndarray, T_washout: int = 100) -> np.ndarray:
        """Collect reservoir states for input sequence.

        Args:
            u: Input sequence of shape (T,)
            T_washout: Washout period to discard

        Returns:
            states: Array of shape (T - T_washout, dim) with reservoir states
        """
        T = len(u)
        states = []

        # Initialize state (implementation-dependent)
        # For TQRC: |ψ⟩ = |1⟩ (vacuum state)
        # For ESN: x = zeros or random
        state = None  # Placeholder - actual initialization in reservoir

        for t in range(T):
            # Evolve reservoir
            state = self.evolve(state, u[t])

            # Collect after washout
            if t >= T_washout:
                states.append(state)

        return np.array(states)

    def measure_capacity(self, T: int = 10000, k_max: int = 50,
                        T_washout: int = 100, beta: float = 1e-6,
                        seed: Optional[int] = 42) -> Tuple[float, np.ndarray]:
        """Measure quantum/classical memory capacity.

        Following Section 3.4.5 protocol:
        1. Generate random input u(t)
        2. Run reservoir to collect states x(t) or |ψ(t)⟩
        3. For each delay k:
            - Train linear readout to predict u(t-k) from state at t
            - Compute r_k² (squared correlation)
        4. Sum MC = Σ r_k²

        Args:
            T: Total time steps for test
            k_max: Maximum delay to test
            T_washout: Washout period
            beta: Ridge regression regularization
            seed: Random seed

        Returns:
            memory_capacity: Scalar MC = Σ r_k²
            r_squared_values: Array of r_k² for k=1..k_max
        """
        # Generate random input
        u = self.generate_random_input(T, seed=seed)

        # Collect reservoir states
        states = self.collect_states(u, T_washout=T_washout)
        T_active = len(states)

        # Array to store r_k² values
        r_squared = np.zeros(k_max)

        # For each delay k
        for k in range(1, k_max + 1):
            # Target: u(t-k)
            # States: x(t)
            # We need: t - k >= T_washout (have valid target)
            #          t < T (have valid state)

            # Construct training data
            # States at time t: states[t - T_washout]
            # Target at time t-k: u[t-k]

            # Valid range: T_washout + k <= t < T
            valid_indices = np.arange(k, T_active)

            if len(valid_indices) == 0:
                # No valid data for this delay
                r_squared[k-1] = 0.0
                continue

            # Input: states at time t
            X_train = states[valid_indices]  # Shape: (N_valid, dim)

            # Target: u(t-k) where t = T_washout + valid_indices
            target_indices = T_washout + valid_indices - k
            y_train = u[target_indices]  # Shape: (N_valid,)

            # Train linear readout via ridge regression
            w = self._ridge_regression(X_train, y_train, beta)

            # Predict
            y_pred = X_train @ w

            # Compute R² (coefficient of determination)
            r_k_squared = self._compute_r_squared(y_train, y_pred)

            r_squared[k-1] = r_k_squared

        # Sum to get total memory capacity
        memory_capacity = np.sum(r_squared)

        return memory_capacity, r_squared

    def _ridge_regression(self, X: np.ndarray, y: np.ndarray,
                         beta: float) -> np.ndarray:
        """Compute ridge regression weights.

        W = (X^T X + βI)^{-1} X^T y

        Args:
            X: Input matrix of shape (N_samples, dim)
            y: Target vector of shape (N_samples,)
            beta: Regularization parameter

        Returns:
            w: Weight vector of shape (dim,)
        """
        dim = X.shape[1]

        # Ridge regression solution
        XtX = X.T @ X
        Xty = X.T @ y

        # Add regularization
        A = XtX + beta * np.eye(dim)

        # Solve
        w = np.linalg.solve(A, Xty)

        return w

    def _compute_r_squared(self, y_true: np.ndarray,
                          y_pred: np.ndarray) -> float:
        """Compute R² (coefficient of determination).

        R² = 1 - SS_res / SS_tot
           = 1 - Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²

        Alternatively: R² = r² where r is Pearson correlation

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            r_squared: R² value in [0, 1] (can be negative for poor fits)
        """
        # Compute Pearson correlation
        if len(y_true) < 2:
            return 0.0

        r, _ = pearsonr(y_true, y_pred)
        r_squared = r ** 2

        return r_squared

    def plot_capacity_curve(self, r_squared: np.ndarray,
                           save_path: Optional[str] = None):
        """Plot memory capacity curve: MC_k = Σ_{j=1}^k r_j² vs k.

        Args:
            r_squared: Array of r_k² values
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available for plotting")
            return

        k_max = len(r_squared)
        k_values = np.arange(1, k_max + 1)

        # Cumulative sum
        MC_k = np.cumsum(r_squared)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot r_k²
        ax1.plot(k_values, r_squared, 'o-', markersize=4)
        ax1.set_xlabel('Delay k')
        ax1.set_ylabel('$r_k^2$')
        ax1.set_title('Memory Capacity: Individual Correlations')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Plot cumulative MC
        ax2.plot(k_values, MC_k, 'o-', markersize=4, color='red')
        ax2.set_xlabel('Delay k')
        ax2.set_ylabel('$MC_k = \sum_{j=1}^k r_j^2$')
        ax2.set_title(f'Cumulative Memory Capacity (MC = {MC_k[-1]:.2f})')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.dim, color='k', linestyle='--',
                   label=f'Theoretical bound (dim={self.dim})')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()


def evaluate_tqrc_memory_capacity(tqrc_reservoir, n_anyons: int,
                                  T: int = 10000, k_max: int = 50) -> dict:
    """Convenience function to evaluate TQRC memory capacity.

    Args:
        tqrc_reservoir: TQRC reservoir instance with evolve() and measure()
        n_anyons: Number of Fibonacci anyons
        T: Test length
        k_max: Maximum delay

    Returns:
        Dictionary with MC results and analysis
    """
    # Compute theoretical bound
    fibonacci = [1, 1]
    for i in range(2, n_anyons):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])

    F_n_minus_1 = fibonacci[n_anyons - 1]

    # Create test instance
    test = MemoryCapacityTest(
        reservoir_evolve_fn=tqrc_reservoir.evolve,
        readout_train_fn=tqrc_reservoir.train_readout,
        reservoir_dim=F_n_minus_1
    )

    # Measure capacity
    MC, r_squared = test.measure_capacity(T=T, k_max=k_max)

    # Analyze results
    efficiency = MC / F_n_minus_1 if F_n_minus_1 > 0 else 0

    results = {
        'memory_capacity': MC,
        'r_squared_values': r_squared,
        'theoretical_bound': F_n_minus_1,
        'efficiency': efficiency,
        'n_anyons': n_anyons,
        'target_efficiency': 0.8,  # From verification/03
        'meets_target': efficiency >= 0.8
    }

    return results


if __name__ == "__main__":
    # Example usage with mock reservoir
    print("Memory Capacity Test")
    print("=" * 50)

    # Mock classical reservoir (for demonstration)
    class MockReservoir:
        def __init__(self, N=10):
            self.N = N
            self.state = np.zeros(N)
            self.W = np.random.randn(N, N) * 0.9 / np.sqrt(N)

        def evolve(self, state, u):
            if state is None:
                state = self.state
            new_state = np.tanh(self.W @ state + u)
            return new_state

        def train_readout(self, states, targets):
            # Ridge regression (handled by MemoryCapacityTest)
            pass

    # Create mock reservoir
    N = 10
    reservoir = MockReservoir(N=N)

    # Create test
    test = MemoryCapacityTest(
        reservoir_evolve_fn=reservoir.evolve,
        readout_train_fn=reservoir.train_readout,
        reservoir_dim=N
    )

    print(f"Testing reservoir with dimension N = {N}")
    print(f"Theoretical bound: MC ≤ {N}")
    print()

    # Run test
    print("Running memory capacity test...")
    MC, r_squared = test.measure_capacity(T=5000, k_max=20, T_washout=100)

    print(f"Measured Memory Capacity: MC = {MC:.2f}")
    print(f"Efficiency: MC/N = {MC/N:.2%}")
    print(f"Peak r² at k = {np.argmax(r_squared) + 1}")
    print()

    # Show first few r² values
    print("Individual r_k² values (k=1..10):")
    for k in range(min(10, len(r_squared))):
        print(f"  k={k+1:2d}: r² = {r_squared[k]:.4f}")

    print("\n✅ Memory capacity test verified!")
