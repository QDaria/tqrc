"""
Readout Layer and Ridge Regression Training
============================================

Implements linear readout with ridge regression training for TQRC.

Theory Reference: Section 3.5 (Readout and Training)
"""

import numpy as np
from typing import Optional, Tuple


class RidgeReadout:
    """
    Linear readout with ridge regression training.

    Implements the standard reservoir computing training procedure:

        W_out = Y X^T (XX^T + βI)^{-1}

    where:
        - X: Reservoir state matrix (dim × T)
        - Y: Target output matrix (output_dim × T)
        - β: Ridge regression regularization parameter
        - W_out: Trained readout weights (output_dim × dim)

    Prediction:
        y(t) = W_out · x(t)

    where x(t) is the reservoir state (measurement probabilities).

    Attributes:
        state_dim: Reservoir dimension (F_{n-1} for TQRC)
        output_dim: Output dimension
        beta: Ridge regression regularization
        W_out: Trained output weights (None until trained)

    Example:
        >>> readout = RidgeReadout(state_dim=2, output_dim=1, beta=1e-6)
        >>> # Generate synthetic data
        >>> X = np.random.randn(2, 1000)  # (state_dim, T)
        >>> Y = np.random.randn(1, 1000)  # (output_dim, T)
        >>> readout.train(X, Y)
        >>> # Predict
        >>> x_test = np.random.randn(2)
        >>> y_pred = readout.predict(x_test)
        >>> y_pred.shape
        (1,)
    """

    def __init__(self,
                 state_dim: int,
                 output_dim: int,
                 beta: float = 1e-6):
        """
        Initialize readout layer.

        Args:
            state_dim: Reservoir dimension (dim = F_{n-1} for TQRC)
            output_dim: Output dimension
                       1 for Mackey-Glass
                       3 for Lorenz-63
            beta: Ridge regression regularization parameter
                  Typical values: 1e-8 to 1e-4

        Example:
            >>> # Mackey-Glass (scalar output)
            >>> readout = RidgeReadout(state_dim=5, output_dim=1)

            >>> # Lorenz-63 (3D output)
            >>> readout = RidgeReadout(state_dim=5, output_dim=3)
        """
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.beta = beta

        # Trained weights (None until trained)
        self.W_out = None

        # Training statistics (for diagnostics)
        self.training_error = None
        self.condition_number = None

    def train(self, states: np.ndarray, targets: np.ndarray):
        """
        Train readout using ridge regression.

        Theory:
            From Section 3.5.3:

            Minimize: ||W_out·X - Y||² + β||W_out||²

            Solution: W_out = Y X^T (XX^T + βI)^{-1}

        Args:
            states: Reservoir states X, shape (state_dim, T)
                   Each column is a measurement probability vector
            targets: Target outputs Y, shape (output_dim, T)

        Raises:
            ValueError: If shapes don't match

        Example:
            >>> X = reservoir.get_reservoir_state_matrix(input_seq, washout=200)
            >>> Y = target_seq[:, 200:].T  # (output_dim, T)
            >>> readout.train(X, Y)
        """
        # Validate shapes
        if states.shape[0] != self.state_dim:
            raise ValueError(
                f"States have wrong dimension: expected {self.state_dim}, "
                f"got {states.shape[0]}"
            )

        if targets.shape[0] != self.output_dim:
            raise ValueError(
                f"Targets have wrong dimension: expected {self.output_dim}, "
                f"got {targets.shape[0]}"
            )

        if states.shape[1] != targets.shape[1]:
            raise ValueError(
                f"States and targets have different lengths: "
                f"{states.shape[1]} != {targets.shape[1]}"
            )

        X = states
        Y = targets
        T = X.shape[1]

        # Compute XX^T (state_dim × state_dim)
        XX_T = X @ X.T

        # Add ridge regularization: XX^T + βI
        identity = np.eye(self.state_dim)
        regularized = XX_T + self.beta * identity

        # Compute condition number (for diagnostics)
        self.condition_number = np.linalg.cond(regularized)

        # Compute XY^T (state_dim × output_dim)
        XY_T = X @ Y.T

        # Solve ridge regression: W_out^T = (XX^T + βI)^{-1} X Y^T
        # W_out = Y X^T (XX^T + βI)^{-1}
        try:
            W_out_T = np.linalg.solve(regularized, XY_T)
            self.W_out = W_out_T.T  # (output_dim × state_dim)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            W_out_T = np.linalg.pinv(regularized) @ XY_T
            self.W_out = W_out_T.T

        # Compute training error (NRMSE)
        Y_pred = self.W_out @ X
        self.training_error = self._compute_nrmse(Y, Y_pred)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict output from reservoir state.

        Args:
            state: Single reservoir state (measurement probabilities)
                  Shape: (state_dim,)

        Returns:
            y: Predicted output, shape (output_dim,)

        Raises:
            ValueError: If not trained yet

        Example:
            >>> x_test = np.array([0.6, 0.4])  # Measurement probabilities
            >>> y_pred = readout.predict(x_test)
        """
        if self.W_out is None:
            raise ValueError("Readout not trained yet. Call train() first.")

        state = np.asarray(state)
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape ({self.state_dim},), got {state.shape}"
            )

        # y = W_out · x
        y = self.W_out @ state

        return y

    def predict_sequence(self, states: np.ndarray) -> np.ndarray:
        """
        Predict outputs for sequence of states.

        Args:
            states: State sequence, shape (state_dim, T)

        Returns:
            Y_pred: Predicted outputs, shape (output_dim, T)

        Example:
            >>> X_test = reservoir.get_reservoir_state_matrix(test_seq)
            >>> Y_pred = readout.predict_sequence(X_test)
        """
        if self.W_out is None:
            raise ValueError("Readout not trained yet. Call train() first.")

        # Y = W_out · X
        Y_pred = self.W_out @ states

        return Y_pred

    def evaluate(self,
                 states: np.ndarray,
                 targets: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate readout on test data.

        Args:
            states: Test states X, shape (state_dim, T)
            targets: True outputs Y, shape (output_dim, T)

        Returns:
            Tuple of (NRMSE, NMSE)
                NRMSE: Normalized Root Mean Square Error
                NMSE: Normalized Mean Square Error

        Theory:
            NRMSE = √(⟨(y_pred - y_true)²⟩) / √(⟨y_true²⟩)
            NMSE = ⟨(y_pred - y_true)²⟩ / ⟨y_true²⟩

        Example:
            >>> nrmse, nmse = readout.evaluate(X_test, Y_test)
            >>> print(f"Test NRMSE: {nrmse:.4f}")
        """
        # Predict
        Y_pred = self.predict_sequence(states)

        # Compute errors
        nrmse = self._compute_nrmse(targets, Y_pred)
        nmse = self._compute_nmse(targets, Y_pred)

        return nrmse, nmse

    def _compute_nrmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Normalized Root Mean Square Error.

        Formula:
            NRMSE = √(⟨(y_pred - y_true)²⟩) / √(⟨y_true²⟩)

        Args:
            y_true: True outputs, shape (output_dim, T)
            y_pred: Predicted outputs, shape (output_dim, T)

        Returns:
            NRMSE value (scalar)
        """
        mse = np.mean((y_pred - y_true)**2)
        variance = np.mean(y_true**2)

        # Avoid division by zero
        if variance < 1e-12:
            return 0.0

        nrmse = np.sqrt(mse / variance)
        return nrmse

    def _compute_nmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Normalized Mean Square Error.

        Formula:
            NMSE = ⟨(y_pred - y_true)²⟩ / ⟨y_true²⟩

        Args:
            y_true: True outputs
            y_pred: Predicted outputs

        Returns:
            NMSE value (scalar)
        """
        mse = np.mean((y_pred - y_true)**2)
        variance = np.mean(y_true**2)

        if variance < 1e-12:
            return 0.0

        nmse = mse / variance
        return nmse

    def get_weights(self) -> np.ndarray:
        """
        Get trained readout weights.

        Returns:
            W_out: Weight matrix, shape (output_dim, state_dim)

        Raises:
            ValueError: If not trained yet
        """
        if self.W_out is None:
            raise ValueError("Readout not trained yet.")

        return self.W_out.copy()

    def set_weights(self, W_out: np.ndarray):
        """
        Set readout weights manually (for transfer learning).

        Args:
            W_out: Weight matrix, shape (output_dim, state_dim)

        Raises:
            ValueError: If shape doesn't match
        """
        expected_shape = (self.output_dim, self.state_dim)
        if W_out.shape != expected_shape:
            raise ValueError(
                f"Expected weight shape {expected_shape}, got {W_out.shape}"
            )

        self.W_out = W_out.copy()

    def __repr__(self) -> str:
        """String representation."""
        trained_status = "trained" if self.W_out is not None else "not trained"
        return (f"RidgeReadout(state_dim={self.state_dim}, "
                f"output_dim={self.output_dim}, "
                f"beta={self.beta:.2e}, "
                f"status={trained_status})")


def cross_validate_beta(states: np.ndarray,
                        targets: np.ndarray,
                        beta_values: np.ndarray,
                        validation_fraction: float = 0.2) -> Tuple[float, dict]:
    """
    Cross-validate ridge regression regularization parameter.

    Args:
        states: State matrix, shape (state_dim, T)
        targets: Target matrix, shape (output_dim, T)
        beta_values: Array of beta values to try
        validation_fraction: Fraction of data for validation

    Returns:
        Tuple of (best_beta, results_dict)
            best_beta: Optimal beta value
            results_dict: {beta: validation_nrmse} for all beta values

    Example:
        >>> beta_values = np.logspace(-8, -4, 5)  # [1e-8, ..., 1e-4]
        >>> best_beta, results = cross_validate_beta(X, Y, beta_values)
        >>> print(f"Best beta: {best_beta:.2e}")
    """
    T = states.shape[1]
    T_train = int(T * (1 - validation_fraction))

    # Split data
    X_train = states[:, :T_train]
    Y_train = targets[:, :T_train]
    X_val = states[:, T_train:]
    Y_val = targets[:, T_train:]

    results = {}
    best_nrmse = float('inf')
    best_beta = beta_values[0]

    for beta in beta_values:
        # Train readout
        readout = RidgeReadout(
            state_dim=states.shape[0],
            output_dim=targets.shape[0],
            beta=beta
        )
        readout.train(X_train, Y_train)

        # Validate
        nrmse, _ = readout.evaluate(X_val, Y_val)
        results[beta] = nrmse

        # Track best
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_beta = beta

    return best_beta, results
