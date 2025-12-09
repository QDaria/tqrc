"""
TQRC Reservoir Dynamics
=======================

Implements topological quantum reservoir computing dynamics using
Fibonacci anyon braiding operations.

Theory Reference: Section 3.3 (Reservoir Dynamics), Section 3.4 (Fading Memory)
"""

import numpy as np
from typing import Optional, Tuple
from .anyons import FibonacciHilbertSpace
from .braiding import BraidingOperators
from .encoding import InputEncoder


class TQRCReservoir:
    """
    Topological Quantum Reservoir Computing system.

    Implements the quantum reservoir dynamics:

        |ψ(t+1)⟩ = U_res · U_in(u(t)) · |ψ(t)⟩

    where:
        - U_in(u): Input encoding operator (amplitude-encoded braiding)
        - U_res: Fixed reservoir braiding sequence
        - |ψ(t)⟩: Quantum state in Fibonacci anyon Hilbert space

    Attributes:
        n_anyons: Number of Fibonacci anyons
        input_dim: Dimension of input signal
        dim: Hilbert space dimension = F_{n-1}
        braid_length: Length L of random braid word for U_res
        decoherence_rate: γ_eff for fading memory (optional)

    Example:
        >>> reservoir = TQRCReservoir(n_anyons=4, input_dim=1)
        >>> # Single time step
        >>> state = reservoir.hilbert.vacuum_state()
        >>> u = np.array([0.5])
        >>> new_state = reservoir.evolve(state, u)
        >>> new_state.shape
        (2,)
    """

    def __init__(self,
                 n_anyons: int,
                 input_dim: int,
                 braid_length: int = 10,
                 decoherence_rate: float = 0.0,
                 random_seed: Optional[int] = None,
                 initialization: str = 'excited',
                 logger: Optional[object] = None):
        """
        Initialize TQRC reservoir.

        Args:
            n_anyons: Number of Fibonacci anyons
            input_dim: Dimension of input signal
            braid_length: Length L of random braid word for U_res
            decoherence_rate: γ_eff for decoherence (0 = unitary only)
            random_seed: Seed for reproducible reservoir generation
            initialization: Initial state type ('excited', 'random', 'vacuum')
                          - 'excited': Excited superposition (default, avoids vacuum)
                          - 'random': Fully random state (includes vacuum component)
                          - 'vacuum': Pure vacuum state |1⟩
            logger: Optional EvolutionLogger for tracking state evolution

        Example:
            >>> # Mackey-Glass (scalar input) with excited initialization
            >>> reservoir = TQRCReservoir(n_anyons=4, input_dim=1,
            ...                          braid_length=10, random_seed=42,
            ...                          initialization='excited')

            >>> # Lorenz-63 (3D input) with random initialization
            >>> reservoir = TQRCReservoir(n_anyons=4, input_dim=3,
            ...                          braid_length=15, random_seed=42,
            ...                          initialization='random')

            >>> # With evolution logging for diagnostics
            >>> from experiments.debug_logs.evolution_instrumentation import EvolutionLogger
            >>> logger = EvolutionLogger('debug_evolution.json')
            >>> reservoir = TQRCReservoir(n_anyons=4, input_dim=1, logger=logger)
        """
        # Initialize Hilbert space
        self.hilbert = FibonacciHilbertSpace(n_anyons)
        self.n_anyons = n_anyons
        self.dim = self.hilbert.dim

        # Initialize braiding operators
        self.braiding = BraidingOperators(self.hilbert)

        # Initialize input encoder
        self.encoder = InputEncoder(self.braiding, input_dim)
        self.input_dim = input_dim

        # Reservoir parameters
        self.braid_length = braid_length
        self.decoherence_rate = decoherence_rate
        self.random_seed = random_seed
        self.initialization = initialization
        self.logger = logger  # Optional evolution logger

        # Validate initialization type
        if initialization not in ['excited', 'random', 'vacuum']:
            raise ValueError(
                f"Invalid initialization '{initialization}'. "
                f"Must be 'excited', 'random', or 'vacuum'."
            )

        # Generate fixed reservoir unitary U_res
        if random_seed is not None:
            np.random.seed(random_seed)
        self.U_res = self._generate_reservoir_unitary(braid_length)

    def _generate_reservoir_unitary(self, length: int) -> np.ndarray:
        """
        Generate U_res as random braid sequence.

        Theory:
            From Section 3.3.2:
                U_res = B_{σ_L} · B_{σ_{L-1}} · ... · B_{σ_1}

            where σ_j are randomly sampled from {1, 2, ..., n-1}

        Args:
            length: Number of braids in sequence

        Returns:
            Unitary matrix U_res (dim × dim)

        Example:
            >>> reservoir = TQRCReservoir(4, 1, braid_length=5, random_seed=42)
            >>> reservoir.U_res.shape
            (2, 2)
        """
        # Start with identity
        U_res = np.eye(self.dim, dtype=complex)

        # Apply random sequence of braids
        for _ in range(length):
            # Random braid index from {1, ..., n-1}
            sigma = np.random.randint(1, self.n_anyons)

            # Random braiding angle (for diversity)
            theta = np.random.uniform(-np.pi, np.pi)

            # Get braid operator
            B_sigma = self.braiding.fractional_braid(sigma, theta)

            # Compose: U_res ← B_sigma · U_res
            U_res = B_sigma @ U_res

        # Verify unitarity
        assert self.braiding.verify_unitarity(U_res), \
            "Generated U_res not unitary!"

        return U_res

    def evolve(self, state: np.ndarray, u: np.ndarray,
               apply_decoherence: bool = True,
               timestep: Optional[int] = None) -> np.ndarray:
        """
        Evolve quantum state for one time step.

        Theory:
            From Section 3.3.1:
                |ψ(t+1)⟩ = U_res · U_in(u(t)) · |ψ(t)⟩

            With optional decoherence (Section 3.4.2):
                ρ(t+1) = (1-γ)U·ρ(t)·U† + γ·|vacuum⟩⟨vacuum|

        Args:
            state: Current quantum state |ψ(t)⟩
            u: Input signal u(t) (shape: (input_dim,))
            apply_decoherence: Whether to apply decoherence channel
            timestep: Optional timestep for logging

        Returns:
            New quantum state |ψ(t+1)⟩

        Example:
            >>> state = reservoir.hilbert.vacuum_state()
            >>> u = np.array([0.5])
            >>> new_state = reservoir.evolve(state, u)
        """
        # Store state before evolution (for logging)
        if self.logger is not None:
            state_before = state.copy()

        # Encode input into U_in(u)
        U_in = self.encoder.encode(u)

        # Apply unitary evolution: |ψ'⟩ = U_res · U_in · |ψ⟩
        state_new = self.U_res @ (U_in @ state)

        # Apply decoherence if enabled
        if apply_decoherence and self.decoherence_rate > 0.0:
            state_new = self._apply_decoherence(state_new)

        # Normalize (numerical safety)
        state_new = state_new / np.linalg.norm(state_new)

        # Log evolution step if logger is attached
        if self.logger is not None and timestep is not None:
            self.logger.log_evolution_step(
                t=timestep,
                state_before=state_before,
                state_after=state_new,
                u_input=u
            )

        return state_new

    def _apply_decoherence(self, state: np.ndarray) -> np.ndarray:
        """
        Apply decoherence channel for fading memory.

        Theory:
            Simplified decoherence model (Section 3.4.2):

                ρ_out = (1 - γ_eff)·|ψ⟩⟨ψ| + γ_eff·|vacuum⟩⟨vacuum|

            This models amplitude damping toward vacuum state.

        Args:
            state: Pure state |ψ⟩

        Returns:
            Mixed state (represented as pure state approximation)

        Note:
            For full density matrix evolution, use separate decoherence module.
            This is a simplified pure-state approximation.
        """
        gamma = self.decoherence_rate
        vacuum = self.hilbert.vacuum_state()

        # Mix with vacuum state
        # Approximation: interpolate between |ψ⟩ and |vacuum⟩
        state_mixed = np.sqrt(1 - gamma) * state + np.sqrt(gamma) * vacuum

        # Renormalize
        state_mixed = state_mixed / np.linalg.norm(state_mixed)

        return state_mixed

    def run_dynamics(self,
                     input_sequence: np.ndarray,
                     washout: int = 200,
                     initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run full dynamics on input sequence.

        Args:
            input_sequence: Array of shape (T, input_dim)
                           T = total time steps
            washout: Number of initial steps to discard (for ESP)
            initial_state: Initial quantum state (default: vacuum)

        Returns:
            states: Measurement probabilities, shape (T - washout, dim)
                   Each row is |⟨i|ψ(t)⟩|² for i=0,...,dim-1

        Theory:
            From Section 3.5.2:
                - Evolve through washout period (discard states)
                - Collect measurement statistics for training/testing
                - Feature vector: x(t) = [|⟨0|ψ(t)⟩|², ..., |⟨d-1|ψ(t)⟩|²]

        Example:
            >>> # Generate random input
            >>> T = 1000
            >>> input_seq = np.random.uniform(-1, 1, size=(T, 1))
            >>> states = reservoir.run_dynamics(input_seq, washout=200)
            >>> states.shape
            (800, 2)  # (T - washout, dim)
        """
        T = len(input_sequence)

        if washout >= T:
            raise ValueError(f"Washout {washout} >= sequence length {T}")

        # Initialize state
        if initial_state is None:
            if self.initialization == 'excited':
                state = self.hilbert.excited_superposition(seed=self.random_seed)
            elif self.initialization == 'random':
                state = self.hilbert.random_state(seed=self.random_seed)
            elif self.initialization == 'vacuum':
                state = self.hilbert.vacuum_state()
            else:
                # Should never reach here due to validation in __init__
                raise ValueError(f"Invalid initialization: {self.initialization}")
        else:
            state = initial_state.copy()

        # Storage for states (only after washout)
        states_list = []

        # Run dynamics
        for t in range(T):
            u_t = input_sequence[t]

            # Evolve state (with logging if logger attached)
            state = self.evolve(state, u_t, timestep=t)

            # After washout: collect measurement probabilities
            if t >= washout:
                probs = self.hilbert.measure_probabilities(state)
                states_list.append(probs)

        # Convert to array
        states = np.array(states_list)

        return states

    def get_reservoir_state_matrix(self,
                                   input_sequence: np.ndarray,
                                   washout: int = 200) -> np.ndarray:
        """
        Get reservoir state matrix X for training.

        Args:
            input_sequence: Input time series, shape (T, input_dim)
            washout: Washout period

        Returns:
            X: State matrix, shape (dim, T - washout)
               Columns are measurement probability vectors

        Example:
            >>> input_seq = np.random.uniform(-1, 1, (1000, 1))
            >>> X = reservoir.get_reservoir_state_matrix(input_seq, washout=200)
            >>> X.shape
            (2, 800)  # (dim, T - washout)
        """
        states = self.run_dynamics(input_sequence, washout)
        # Transpose to get (dim, T) format for ridge regression
        X = states.T
        return X

    def reset_reservoir(self, new_seed: Optional[int] = None):
        """
        Regenerate reservoir with new random braid sequence.

        Args:
            new_seed: New random seed (optional)

        Example:
            >>> reservoir.reset_reservoir(new_seed=123)
        """
        if new_seed is not None:
            np.random.seed(new_seed)
            self.random_seed = new_seed

        self.U_res = self._generate_reservoir_unitary(self.braid_length)

    def __repr__(self) -> str:
        """String representation."""
        return (f"TQRCReservoir(n_anyons={self.n_anyons}, "
                f"input_dim={self.input_dim}, "
                f"dim={self.dim}, "
                f"braid_length={self.braid_length}, "
                f"decoherence_rate={self.decoherence_rate}, "
                f"initialization='{self.initialization}')")
