"""
Input Encoding for TQRC
========================

Maps classical time-series data u(t) into quantum braiding operations
using amplitude encoding.

Theory Reference: Section 3.2 (Input Encoding)
"""

import numpy as np
from typing import Union
from .braiding import BraidingOperators


class InputEncoder:
    """
    Encodes classical signals into braiding operations.

    Implements amplitude encoding strategy where scalar inputs directly
    parameterize braiding angles:

        θ(u) = π · u(t)   for u(t) ∈ [-1, +1]

    For multi-dimensional inputs u = [u₁, u₂, ..., u_m]ᵀ:

        U_in(u) = B_m(θ_m) · B_{m-1}(θ_{m-1}) · ... · B_1(θ_1)

    where θ_j = π · u_j.

    Constraint: input_dim ≤ n-1 (number of available braids)

    Attributes:
        braiding_ops: BraidingOperators instance
        input_dim: Dimension of input signal
        n_anyons: Number of anyons
        max_braids: Maximum number of braids available (n-1)

    Example:
        >>> from .anyons import FibonacciHilbertSpace
        >>> from .braiding import BraidingOperators
        >>> hilbert = FibonacciHilbertSpace(4)
        >>> braiding = BraidingOperators(hilbert)
        >>> encoder = InputEncoder(braiding, input_dim=1)
        >>> u = np.array([0.5])  # Scalar input
        >>> U_in = encoder.encode(u)
        >>> U_in.shape
        (2, 2)
    """

    def __init__(self, braiding_ops: BraidingOperators, input_dim: int, logger=None):
        """
        Initialize input encoder.

        Args:
            braiding_ops: BraidingOperators instance
            input_dim: Dimension of input signal
            logger: Optional EncodingLogger for instrumentation (default: None)

        Raises:
            ValueError: If input_dim > n-1 (insufficient braids)

        Example:
            >>> encoder = InputEncoder(braiding_ops, input_dim=3)
            >>> # For n=4 anyons, input_dim=3 uses all 3 available braids

            >>> # With logging for debugging
            >>> from experiments.debug_logs.encoding_instrumentation import EncodingLogger
            >>> logger = EncodingLogger()
            >>> encoder = InputEncoder(braiding_ops, input_dim=3, logger=logger)
        """
        self.braiding_ops = braiding_ops
        self.input_dim = input_dim
        self.n_anyons = braiding_ops.n
        self.max_braids = self.n_anyons - 1
        self.logger = logger  # Optional instrumentation

        # Verify constraint: input_dim ≤ n-1
        if input_dim > self.max_braids:
            raise ValueError(
                f"Input dimension {input_dim} exceeds available braids "
                f"{self.max_braids} for {self.n_anyons} anyons"
            )

        self.dim = braiding_ops.dim

    def encode(self, u: np.ndarray) -> np.ndarray:
        """
        Encode input u(t) into unitary U_in(u).

        Args:
            u: Input vector, shape (input_dim,), values in [-1, 1]

        Returns:
            U_in: Unitary matrix (dim × dim)

        Raises:
            ValueError: If u has wrong shape or out-of-range values

        Theory:
            From Section 3.2.3:

            U_in(u) = B_m(θ_m) · B_{m-1}(θ_{m-1}) · ... · B_1(θ_1)

            where θ_j = π · u_j for j = 1, ..., input_dim

        Example:
            >>> # Scalar input (Mackey-Glass)
            >>> encoder = InputEncoder(braiding_ops, input_dim=1)
            >>> u = np.array([0.5])
            >>> U_in = encoder.encode(u)

            >>> # Vector input (Lorenz-63)
            >>> encoder = InputEncoder(braiding_ops, input_dim=3)
            >>> u = np.array([0.2, -0.5, 0.8])  # [x, y, z] normalized
            >>> U_in = encoder.encode(u)
        """
        # Validate input
        u = np.asarray(u)
        if u.shape != (self.input_dim,):
            raise ValueError(
                f"Expected input shape ({self.input_dim},), got {u.shape}"
            )

        # Verify input range [-1, +1]
        if np.any(np.abs(u) > 1.0):
            raise ValueError(
                f"Input values must be in [-1, +1], got min={np.min(u)}, "
                f"max={np.max(u)}"
            )

        # Compute braiding angles: θ_j = π · u_j
        thetas = np.pi * u

        # Build composite unitary: U_in = B_m(θ_m) · ... · B_1(θ_1)
        # Start with identity
        U_in = np.eye(self.dim, dtype=complex)

        # Apply braids sequentially (right-to-left composition)
        for j in range(self.input_dim):
            braid_index = j + 1  # Braid indices start at 1
            theta_j = thetas[j]

            # Get fractional braid operator
            B_j = self.braiding_ops.fractional_braid(braid_index, theta_j)

            # Compose: U_in ← B_j · U_in
            U_in = B_j @ U_in

        # Verify unitarity
        assert self.braiding_ops.verify_unitarity(U_in), \
            "Encoded U_in not unitary!"

        # === INSTRUMENTATION: Log encoding metrics ===
        if self.logger is not None:
            # Compute identity distance: ||U_in - I||_F
            identity = np.eye(self.dim, dtype=complex)
            identity_distance = np.linalg.norm(U_in - identity, ord='fro')

            # Compute trace
            trace = np.trace(U_in)

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(U_in)

            # Log all metrics
            self.logger.log_encoding_operation(
                u=u,
                thetas=thetas,
                U_in=U_in,
                identity_distance=identity_distance,
                trace=trace,
                eigenvalues=eigenvalues
            )

        return U_in

    def encode_fast_n4(self, u: np.ndarray) -> np.ndarray:
        """
        Fast encoding for n=4 anyons (diagonal braids).

        Args:
            u: Input vector (shape must match input_dim)

        Returns:
            Diagonal unitary matrix

        Note:
            For n=4, all braiding operators are diagonal, so composition
            is just element-wise multiplication of eigenvalues.
        """
        if self.n_anyons != 4:
            raise ValueError("Fast encoding only for n=4")

        u = np.asarray(u)
        if u.shape != (self.input_dim,):
            raise ValueError(f"Expected shape ({self.input_dim},), got {u.shape}")

        # Compute angles
        thetas = np.pi * u

        # For n=4, we can compose diagonal matrices efficiently
        # U_in = product of diagonal braids
        U_in = np.eye(self.dim, dtype=complex)

        for j in range(self.input_dim):
            braid_index = j + 1
            B_j = self.braiding_ops.fractional_braid_n4_fast(braid_index, thetas[j])
            U_in = B_j @ U_in

        return U_in

    def normalize_lorenz(self, state: np.ndarray,
                        x_max: float = 20.0,
                        y_max: float = 30.0,
                        z_max: float = 50.0) -> np.ndarray:
        """
        Normalize Lorenz system state [x, y, z] to [-1, +1].

        Args:
            state: Lorenz state [x, y, z]
            x_max, y_max, z_max: Scaling factors

        Returns:
            Normalized input u = [u₁, u₂, u₃] ∈ [-1, 1]³

        Theory:
            From Section 3.2.4:
                u₁ = x / x_max
                u₂ = y / y_max
                u₃ = z / z_max

        Example:
            >>> encoder = InputEncoder(braiding_ops, input_dim=3)
            >>> lorenz_state = np.array([10.0, 15.0, 25.0])
            >>> u = encoder.normalize_lorenz(lorenz_state)
            >>> u
            array([0.5, 0.5, 0.5])
        """
        if len(state) != 3:
            raise ValueError("Lorenz state must be 3-dimensional")

        x, y, z = state
        u = np.array([x / x_max, y / y_max, z / z_max])

        # Clip to [-1, +1] to handle occasional overshoots
        u = np.clip(u, -1.0, 1.0)

        return u

    def normalize_mackey_glass(self, x: float,
                               x_min: float = 0.4,
                               x_max: float = 1.6) -> np.ndarray:
        """
        Normalize Mackey-Glass signal to [-1, +1].

        Args:
            x: Mackey-Glass value (typically in [0.4, 1.6])
            x_min, x_max: Expected range

        Returns:
            Normalized scalar input u ∈ [-1, 1]

        Theory:
            From Section 3.2.4:
                u(t) = 2 · (x(t) - x_min) / (x_max - x_min) - 1

        Example:
            >>> encoder = InputEncoder(braiding_ops, input_dim=1)
            >>> x = 1.0  # Mackey-Glass value
            >>> u = encoder.normalize_mackey_glass(x)
            >>> u
            array([0.])
        """
        # Map [x_min, x_max] → [-1, +1]
        u = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

        # Clip to handle numerical errors
        u = np.clip(u, -1.0, 1.0)

        return np.array([u])

    def __repr__(self) -> str:
        """String representation."""
        return (f"InputEncoder(input_dim={self.input_dim}, "
                f"n_anyons={self.n_anyons}, "
                f"available_braids={self.max_braids})")


# Convenience functions for common encodings

def encode_scalar(braiding_ops: BraidingOperators, u: float) -> np.ndarray:
    """
    Encode single scalar input (convenience function).

    Args:
        braiding_ops: BraidingOperators instance
        u: Scalar input in [-1, 1]

    Returns:
        Unitary matrix U_in(u)

    Example:
        >>> U = encode_scalar(braiding_ops, 0.5)
    """
    encoder = InputEncoder(braiding_ops, input_dim=1)
    return encoder.encode(np.array([u]))


def encode_vector(braiding_ops: BraidingOperators, u: np.ndarray) -> np.ndarray:
    """
    Encode vector input (convenience function).

    Args:
        braiding_ops: BraidingOperators instance
        u: Input vector

    Returns:
        Unitary matrix U_in(u)

    Example:
        >>> u = np.array([0.2, -0.5, 0.8])
        >>> U = encode_vector(braiding_ops, u)
    """
    input_dim = len(u)
    encoder = InputEncoder(braiding_ops, input_dim=input_dim)
    return encoder.encode(u)
