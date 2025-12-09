"""
Fibonacci Anyon Hilbert Space and State Representation
=======================================================

Implements the Hilbert space for n Fibonacci anyons with total charge τ.
Basis states correspond to fusion tree diagrams.

Theory Reference: Section 3 of theory (Hilbert Space Structure)
Verification: verification/01_fibonacci_anyon_mathematics.md
"""

import numpy as np
from typing import List, Tuple
from ..constants import hilbert_space_dim_tau, _fibonacci


class FibonacciHilbertSpace:
    """
    Hilbert space for n Fibonacci anyons with total charge τ.

    The Hilbert space dimension is given by the Fibonacci number:
        dim(ℋ_n^τ) = F_{n-1}

    For small n:
        n=4 → dim=2
        n=6 → dim=5
        n=8 → dim=13
        n=10 → dim=34

    Attributes:
        n_anyons: Number of Fibonacci anyons
        dim: Hilbert space dimension = F_{n-1}
        basis_labels: List of fusion tree basis state labels

    Example:
        >>> hilbert = FibonacciHilbertSpace(n_anyons=4)
        >>> hilbert.dim
        2
        >>> state = hilbert.random_state()
        >>> np.abs(np.vdot(state, state) - 1.0) < 1e-14
        True
    """

    def __init__(self, n_anyons: int):
        """
        Initialize Hilbert space for n Fibonacci anyons.

        Args:
            n_anyons: Number of anyons (must be ≥ 2)

        Raises:
            ValueError: If n_anyons < 2
        """
        if n_anyons < 2:
            raise ValueError(f"Need at least 2 anyons, got {n_anyons}")

        self.n_anyons = n_anyons

        # Calculate dimension using Fibonacci number F_{n-1}
        # Source: Trebst (2008) Section 3
        self.dim = hilbert_space_dim_tau(n_anyons)

        # Generate basis labels for fusion trees
        # For n=4: basis = ['|1⟩', '|τ⟩']  (2 fusion channels)
        # For n=6: basis = ['|1⟩', '|τ₁⟩', '|τ₂⟩', '|τ₃⟩', '|τ₄⟩']  (5 channels)
        self.basis_labels = self._generate_basis_labels()

    def _generate_basis_labels(self) -> List[str]:
        """
        Generate fusion tree basis state labels.

        Returns:
            List of basis state labels

        Note:
            For n=4 (dim=2): ['|1⟩', '|τ⟩']
            For n≥6: Label states by index for simplicity
        """
        if self.dim == 1:
            return ['|1⟩']  # Total charge vacuum only
        elif self.dim == 2:
            # n=4 case: Two fusion channels (vacuum and τ)
            return ['|1⟩', '|τ⟩']
        else:
            # General case: Label by index
            return [f'|{i}⟩' for i in range(self.dim)]

    def random_state(self, seed: int = None) -> np.ndarray:
        """
        Generate random normalized quantum state.

        Args:
            seed: Random seed for reproducibility (optional)

        Returns:
            Random quantum state |ψ⟩ ∈ ℂ^dim with ||ψ|| = 1

        Example:
            >>> hilbert = FibonacciHilbertSpace(4)
            >>> state = hilbert.random_state(seed=42)
            >>> state.shape
            (2,)
            >>> np.abs(np.linalg.norm(state) - 1.0) < 1e-14
            True
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random complex amplitudes
        real_part = np.random.randn(self.dim)
        imag_part = np.random.randn(self.dim)
        state = real_part + 1j * imag_part

        # Normalize to unit norm
        state = state / np.linalg.norm(state)

        return state

    def vacuum_state(self) -> np.ndarray:
        """
        Return vacuum state |1⟩ (total charge 1).

        Returns:
            Vacuum state (first basis state): |1⟩ = [1, 0, 0, ...]^T

        Example:
            >>> hilbert = FibonacciHilbertSpace(6)
            >>> vacuum = hilbert.vacuum_state()
            >>> vacuum[0]
            (1+0j)
            >>> np.sum(np.abs(vacuum[1:])) < 1e-14
            True
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def excited_superposition(self, seed: int = None) -> np.ndarray:
        """
        Generate random excited state superposition.

        Avoids vacuum |1⟩ to ensure state space exploration.
        Creates superposition over excited states |τ_k⟩.

        Theory:
            For reservoir computing, initializing in vacuum state can lead to
            insufficient state space exploration, especially for short transient
            dynamics. This method generates a random superposition that explicitly
            excludes the vacuum component:

                |ψ⟩ = Σ_{k=1}^{dim-1} α_k |k⟩

            where α_k are random complex amplitudes with ⟨1|ψ⟩ = 0.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Normalized excited state |ψ⟩ with ⟨1|ψ⟩ ≈ 0

        Example:
            >>> hilbert = FibonacciHilbertSpace(6)
            >>> state = hilbert.excited_superposition(seed=42)
            >>> np.abs(state[0]) < 1e-14  # Vacuum component should be zero
            True
            >>> np.abs(np.linalg.norm(state) - 1.0) < 1e-14  # Normalized
            True
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random complex amplitudes for all states
        state = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)

        # Zero out vacuum component to force excited superposition
        state[0] = 0.0

        # Renormalize
        state = state / np.linalg.norm(state)

        return state

    def basis_state(self, index: int) -> np.ndarray:
        """
        Return computational basis state |i⟩.

        Args:
            index: Basis state index (0 ≤ index < dim)

        Returns:
            Basis state |i⟩ with 1 at position index, 0 elsewhere

        Raises:
            ValueError: If index out of range

        Example:
            >>> hilbert = FibonacciHilbertSpace(4)
            >>> state = hilbert.basis_state(1)  # |τ⟩ for n=4
            >>> state[1]
            (1+0j)
        """
        if not 0 <= index < self.dim:
            raise ValueError(f"Index {index} out of range [0, {self.dim})")

        state = np.zeros(self.dim, dtype=complex)
        state[index] = 1.0
        return state

    def measure_probabilities(self, state: np.ndarray, time_step: int = None,
                              enable_logging: bool = False) -> np.ndarray:
        """
        Compute measurement probabilities in computational basis.

        Args:
            state: Quantum state |ψ⟩
            time_step: Current time step (optional, for logging)
            enable_logging: Enable debug logging to file

        Returns:
            Probability distribution: p_i = |⟨i|ψ⟩|² for i=0,...,dim-1

        Example:
            >>> hilbert = FibonacciHilbertSpace(4)
            >>> state = hilbert.random_state(seed=42)
            >>> probs = hilbert.measure_probabilities(state)
            >>> np.abs(np.sum(probs) - 1.0) < 1e-14
            True
        """
        # Born rule: p_i = |⟨i|ψ⟩|² = |ψ_i|²
        probabilities = np.abs(state)**2

        # Verify normalization
        assert np.abs(np.sum(probabilities) - 1.0) < 1e-12, \
            "Probabilities do not sum to 1!"

        # DEBUG INSTRUMENTATION: Log measurement statistics
        if enable_logging:
            import json
            import os
            from pathlib import Path

            # Compute diagnostic metrics
            epsilon = 1e-15  # Small constant to avoid log(0)
            safe_probs = np.maximum(probabilities, epsilon)
            entropy = -np.sum(probabilities * np.log(safe_probs))
            max_prob = np.max(probabilities)
            vacuum_prob = probabilities[0]

            log_entry = {
                'time_step': time_step,
                'probabilities': probabilities.tolist(),
                'entropy': float(entropy),
                'max_prob': float(max_prob),
                'vacuum_prob': float(vacuum_prob),
                'std_prob': float(np.std(probabilities)),
                'mean_prob': float(np.mean(probabilities))
            }

            # Write to log file
            log_dir = Path('experiments/debug_logs')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / 'measurement_instrumentation.jsonl'

            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        return probabilities

    def __repr__(self) -> str:
        """String representation of Hilbert space."""
        return (f"FibonacciHilbertSpace(n_anyons={self.n_anyons}, "
                f"dim={self.dim}, basis={self.basis_labels})")

    def verify_state_normalization(self, state: np.ndarray) -> bool:
        """
        Verify that state is properly normalized.

        Args:
            state: Quantum state to verify

        Returns:
            True if ||ψ|| = 1, False otherwise

        Example:
            >>> hilbert = FibonacciHilbertSpace(6)
            >>> state = hilbert.random_state()
            >>> hilbert.verify_state_normalization(state)
            True
        """
        norm = np.linalg.norm(state)
        return np.abs(norm - 1.0) < 1e-10
