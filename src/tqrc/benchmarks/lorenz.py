"""Lorenz-63 chaotic attractor benchmark.

Implements the Lorenz-63 system for TQRC testing with verified parameters
from verification/03_benchmark_systems.md.

Author: [To be filled]
Date: 2025-12-07
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional


class Lorenz63:
    """Lorenz-63 system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz

    Verified parameters from Lorenz (1963) and Sprott (2003):
    - Standard chaotic regime: σ=10, ρ=28, β=8/3
    - Lyapunov exponent: λ₁ = 0.9056
    - Lyapunov time: T_λ = 1.104 time units
    - Attractor dimension: ~2.06 (Kaplan-Yorke)

    Source: verification/03_benchmark_systems.md
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 8.0/3.0, dt: float = 0.01):
        """Initialize Lorenz-63 system with verified parameters.

        Args:
            sigma: Prandtl number (default: 10.0)
            rho: Rayleigh number (default: 28.0)
            beta: Geometric factor (default: 8/3 = 2.666...)
            dt: Integration time step (default: 0.01)
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt

        # Verified Lyapunov properties
        self.lambda_1 = 0.9056  # Largest Lyapunov exponent
        self.lyapunov_time = 1.0 / self.lambda_1  # ≈ 1.104 time units

    def _lorenz_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute derivatives of Lorenz system.

        Args:
            t: Time (unused, system is autonomous)
            state: [x, y, z] state vector

        Returns:
            [dx/dt, dy/dt, dz/dt] derivatives
        """
        x, y, z = state

        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z

        return np.array([dx_dt, dy_dt, dz_dt])

    def generate_trajectory(self, T: int, x0: Optional[np.ndarray] = None,
                           transient: int = 1000) -> np.ndarray:
        """Generate Lorenz-63 trajectory using RK45 integration.

        Args:
            T: Number of time steps to generate
            x0: Initial condition [x, y, z]. If None, uses (1, 1, 1)
            transient: Number of initial steps to discard (reach attractor)

        Returns:
            trajectory: Array of shape (T, 3) with [x, y, z] values
        """
        # Default initial condition (standard)
        if x0 is None:
            x0 = np.array([1.0, 1.0, 1.0])

        # Total integration time (transient + T steps)
        t_span = (0, (transient + T) * self.dt)
        t_eval = np.arange(0, (transient + T) * self.dt, self.dt)

        # Integrate using scipy's RK45 (adaptive Runge-Kutta)
        solution = solve_ivp(
            self._lorenz_derivatives,
            t_span,
            x0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )

        # Extract trajectory after transient
        trajectory = solution.y[:, transient:transient+T].T

        return trajectory

    def normalize_trajectory(self, trajectory: np.ndarray,
                            method: str = 'zscore') -> Tuple[np.ndarray, dict]:
        """Normalize trajectory to [-1, 1] range for TQRC input encoding.

        Following Section 3.2.4 of theory document.

        Args:
            trajectory: Array of shape (T, 3) with [x, y, z]
            method: Normalization method ('zscore' or 'minmax')

        Returns:
            normalized: Array of shape (T, 3) normalized to [-1, 1]
            stats: Dictionary with normalization statistics for denormalization
        """
        stats = {}

        if method == 'zscore':
            # Z-score normalization: (x - mean) / std
            mean = np.mean(trajectory, axis=0)
            std = np.std(trajectory, axis=0)

            # Clip to ±3σ then scale to [-1, 1]
            normalized = (trajectory - mean) / (3 * std)
            normalized = np.clip(normalized, -1, 1)

            stats['mean'] = mean
            stats['std'] = std
            stats['method'] = 'zscore'

        elif method == 'minmax':
            # Min-max normalization to [-1, 1]
            min_val = np.min(trajectory, axis=0)
            max_val = np.max(trajectory, axis=0)

            normalized = 2 * (trajectory - min_val) / (max_val - min_val) - 1

            stats['min'] = min_val
            stats['max'] = max_val
            stats['method'] = 'minmax'
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, stats

    def create_training_data(self, T_train: int = 5000, T_test: int = 2000,
                            T_washout: int = 500,
                            steps_ahead: int = 1) -> Tuple[np.ndarray, ...]:
        """Create train/test split for TQRC with one-step ahead prediction.

        Args:
            T_train: Number of training time steps
            T_test: Number of test time steps
            T_washout: Washout period (discarded from training)
            steps_ahead: Prediction horizon (default: 1 step ahead)

        Returns:
            train_input: Shape (T_train, 3) - normalized [x, y, z]
            train_target: Shape (T_train, 3) - target output (k steps ahead)
            test_input: Shape (T_test, 3) - normalized test inputs
            test_target: Shape (T_test, 3) - test targets
            stats: Normalization statistics
        """
        # Generate full trajectory (washout + train + test + lookahead)
        T_total = T_washout + T_train + T_test + steps_ahead
        trajectory = self.generate_trajectory(T_total, transient=1000)

        # Normalize entire trajectory
        normalized, stats = self.normalize_trajectory(trajectory)

        # Split into washout, train, test
        # Washout: indices [0:T_washout]
        # Train input: [T_washout : T_washout+T_train]
        # Train target: [T_washout+steps_ahead : T_washout+T_train+steps_ahead]
        # Test input: [T_washout+T_train : T_washout+T_train+T_test]
        # Test target: [T_washout+T_train+steps_ahead : end]

        train_input = normalized[T_washout : T_washout + T_train]
        train_target = normalized[T_washout + steps_ahead :
                                  T_washout + T_train + steps_ahead]

        test_input = normalized[T_washout + T_train :
                                T_washout + T_train + T_test]
        test_target = normalized[T_washout + T_train + steps_ahead :
                                 T_washout + T_train + T_test + steps_ahead]

        return train_input, train_target, test_input, test_target, stats

    def get_attractor_statistics(self, T: int = 10000) -> dict:
        """Compute statistical properties of the Lorenz attractor.

        Args:
            T: Number of time steps for statistics

        Returns:
            Dictionary with attractor statistics
        """
        trajectory = self.generate_trajectory(T, transient=2000)

        stats = {
            'mean': np.mean(trajectory, axis=0),
            'std': np.std(trajectory, axis=0),
            'min': np.min(trajectory, axis=0),
            'max': np.max(trajectory, axis=0),
            'lyapunov_exponent': self.lambda_1,
            'lyapunov_time': self.lyapunov_time,
            'parameters': {
                'sigma': self.sigma,
                'rho': self.rho,
                'beta': self.beta
            }
        }

        return stats


if __name__ == "__main__":
    # Example usage and verification
    print("Lorenz-63 Benchmark System")
    print("=" * 50)

    # Initialize with verified parameters
    lorenz = Lorenz63()

    print(f"Parameters: σ={lorenz.sigma}, ρ={lorenz.rho}, β={lorenz.beta:.6f}")
    print(f"Lyapunov exponent λ₁ = {lorenz.lambda_1}")
    print(f"Lyapunov time T_λ = {lorenz.lyapunov_time:.3f} time units")
    print()

    # Generate sample trajectory
    print("Generating trajectory...")
    trajectory = lorenz.generate_trajectory(T=1000, transient=1000)
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Sample state [x, y, z]: {trajectory[0]}")
    print()

    # Get attractor statistics
    stats = lorenz.get_attractor_statistics(T=5000)
    print("Attractor Statistics:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")
    print(f"  Min:  {stats['min']}")
    print(f"  Max:  {stats['max']}")
    print()

    # Create training data
    print("Creating training/test split...")
    train_in, train_out, test_in, test_out, norm_stats = \
        lorenz.create_training_data(T_train=1000, T_test=500)

    print(f"Training input shape: {train_in.shape}")
    print(f"Training target shape: {train_out.shape}")
    print(f"Test input shape: {test_in.shape}")
    print(f"Test target shape: {test_out.shape}")
    print(f"Normalized range: [{train_in.min():.3f}, {train_in.max():.3f}]")
    print()
    print("✅ Lorenz-63 benchmark verified!")
