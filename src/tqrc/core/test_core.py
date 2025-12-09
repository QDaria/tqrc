"""
Basic Tests for TQRC Core Module
=================================

Simple tests to verify core functionality.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqrc.core import (
    FibonacciHilbertSpace,
    BraidingOperators,
    InputEncoder,
    TQRCReservoir,
    RidgeReadout
)


def test_hilbert_space():
    """Test FibonacciHilbertSpace."""
    print("Testing FibonacciHilbertSpace...")

    # Test n=4 (dim=2)
    hilbert = FibonacciHilbertSpace(4)
    assert hilbert.dim == 2, "Wrong dimension for n=4"

    # Test vacuum state
    vacuum = hilbert.vacuum_state()
    assert np.allclose(vacuum, [1, 0]), "Wrong vacuum state"

    # Test random state normalization
    state = hilbert.random_state(seed=42)
    assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10, "State not normalized"

    # Test measurement probabilities
    probs = hilbert.measure_probabilities(state)
    assert np.abs(np.sum(probs) - 1.0) < 1e-10, "Probabilities don't sum to 1"

    print("✓ FibonacciHilbertSpace tests passed")


def test_braiding_operators():
    """Test BraidingOperators."""
    print("\nTesting BraidingOperators...")

    hilbert = FibonacciHilbertSpace(4)
    braiding = BraidingOperators(hilbert)

    # Test braid matrix unitarity
    B1 = braiding.braid_matrix(1)
    assert B1.shape == (2, 2), "Wrong braid matrix shape"
    assert braiding.verify_unitarity(B1), "Braid matrix not unitary"

    # Test fractional braiding
    theta = np.pi / 4
    B_frac = braiding.fractional_braid(1, theta)
    assert braiding.verify_unitarity(B_frac), "Fractional braid not unitary"

    print("✓ BraidingOperators tests passed")


def test_input_encoder():
    """Test InputEncoder."""
    print("\nTesting InputEncoder...")

    hilbert = FibonacciHilbertSpace(4)
    braiding = BraidingOperators(hilbert)
    encoder = InputEncoder(braiding, input_dim=1)

    # Test scalar encoding
    u = np.array([0.5])
    U_in = encoder.encode(u)
    assert U_in.shape == (2, 2), "Wrong encoded matrix shape"
    assert braiding.verify_unitarity(U_in), "Encoded U_in not unitary"

    # Test normalization functions
    u_mg = encoder.normalize_mackey_glass(1.0)
    assert -1.0 <= u_mg[0] <= 1.0, "Normalized MG value out of range"

    print("✓ InputEncoder tests passed")


def test_reservoir():
    """Test TQRCReservoir."""
    print("\nTesting TQRCReservoir...")

    reservoir = TQRCReservoir(
        n_anyons=4,
        input_dim=1,
        braid_length=5,
        random_seed=42
    )

    # Test single step evolution
    state = reservoir.hilbert.vacuum_state()
    u = np.array([0.5])
    new_state = reservoir.evolve(state, u)
    assert new_state.shape == (2,), "Wrong evolved state shape"
    assert np.abs(np.linalg.norm(new_state) - 1.0) < 1e-10, "Evolved state not normalized"

    # Test full dynamics
    T = 300
    input_seq = np.random.uniform(-1, 1, size=(T, 1))
    states = reservoir.run_dynamics(input_seq, washout=200)
    assert states.shape == (100, 2), f"Wrong states shape: {states.shape}"

    # Verify all states are probability distributions
    for i in range(states.shape[0]):
        assert np.abs(np.sum(states[i]) - 1.0) < 1e-10, f"Row {i} not normalized"
        assert np.all(states[i] >= 0), f"Row {i} has negative probabilities"

    print("✓ TQRCReservoir tests passed")


def test_readout():
    """Test RidgeReadout."""
    print("\nTesting RidgeReadout...")

    # Create synthetic data
    state_dim = 2
    output_dim = 1
    T = 100

    X = np.random.randn(state_dim, T)
    Y = np.random.randn(output_dim, T)

    # Test training
    readout = RidgeReadout(state_dim, output_dim, beta=1e-6)
    readout.train(X, Y)
    assert readout.W_out is not None, "Readout not trained"
    assert readout.W_out.shape == (1, 2), "Wrong weight shape"

    # Test prediction
    x_test = np.random.randn(state_dim)
    y_pred = readout.predict(x_test)
    assert y_pred.shape == (output_dim,), "Wrong prediction shape"

    # Test evaluation
    X_test = np.random.randn(state_dim, 50)
    Y_test = np.random.randn(output_dim, 50)
    nrmse, nmse = readout.evaluate(X_test, Y_test)
    assert nrmse >= 0, "NRMSE should be non-negative"
    assert nmse >= 0, "NMSE should be non-negative"

    print("✓ RidgeReadout tests passed")


def test_full_pipeline():
    """Test complete TQRC pipeline."""
    print("\nTesting full TQRC pipeline...")

    # Create reservoir
    reservoir = TQRCReservoir(
        n_anyons=4,
        input_dim=1,
        braid_length=10,
        random_seed=42
    )

    # Generate synthetic task: delayed input reconstruction
    T_total = 1000
    input_seq = np.random.uniform(-1, 1, size=(T_total, 1))
    delay = 5
    target_seq = np.roll(input_seq, delay, axis=0)  # y(t) = u(t-delay)

    # Run dynamics
    washout = 200
    states = reservoir.run_dynamics(input_seq, washout=washout)
    T_train = 600

    # Prepare training data
    X_train = states[:T_train].T  # (state_dim, T_train)
    Y_train = target_seq[washout:washout+T_train].T  # (output_dim, T_train)

    # Train readout
    readout = RidgeReadout(state_dim=2, output_dim=1, beta=1e-6)
    readout.train(X_train, Y_train)

    # Test
    X_test = states[T_train:].T
    Y_test = target_seq[washout+T_train:].T
    nrmse, nmse = readout.evaluate(X_test, Y_test)

    print(f"  Memory task (delay={delay}): NRMSE = {nrmse:.4f}")
    assert nrmse < 2.0, "NRMSE too high (sanity check)"

    print("✓ Full pipeline test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("TQRC Core Module Tests")
    print("=" * 60)

    test_hilbert_space()
    test_braiding_operators()
    test_input_encoder()
    test_reservoir()
    test_readout()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
