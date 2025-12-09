"""Test script for TQRC benchmark systems.

Verifies that all benchmark implementations produce expected outputs and
follow verified parameters from verification/03_benchmark_systems.md.

Author: [To be filled]
Date: 2025-12-07
"""

import numpy as np
from lorenz import Lorenz63
from mackey_glass import MackeyGlass
from memory_capacity import MemoryCapacityTest


def test_lorenz63():
    """Test Lorenz-63 implementation."""
    print("=" * 70)
    print("TEST 1: Lorenz-63 System")
    print("=" * 70)

    # Initialize with verified parameters
    lorenz = Lorenz63()

    # Verify parameters
    assert lorenz.sigma == 10.0, "σ should be 10.0"
    assert lorenz.rho == 28.0, "ρ should be 28.0"
    assert abs(lorenz.beta - 8.0/3.0) < 1e-10, "β should be 8/3"
    assert abs(lorenz.lambda_1 - 0.9056) < 1e-4, "λ₁ should be 0.9056"
    assert abs(lorenz.lyapunov_time - 1.104) < 1e-3, "T_λ should be ~1.104"

    print("✓ Parameters verified")

    # Generate trajectory
    trajectory = lorenz.generate_trajectory(T=1000, transient=1000)
    assert trajectory.shape == (1000, 3), "Trajectory shape should be (1000, 3)"

    print(f"✓ Generated trajectory: {trajectory.shape}")
    print(f"  Sample state [x,y,z]: [{trajectory[0,0]:.3f}, {trajectory[0,1]:.3f}, {trajectory[0,2]:.3f}]")

    # Test normalization
    normalized, stats = lorenz.normalize_trajectory(trajectory)
    assert normalized.shape == trajectory.shape
    assert normalized.min() >= -1.0 and normalized.max() <= 1.0
    assert 'mean' in stats and 'std' in stats

    print(f"✓ Normalization works: range [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Test training data creation
    train_in, train_out, test_in, test_out, stats = \
        lorenz.create_training_data(T_train=1000, T_test=500)

    assert train_in.shape == (1000, 3)
    assert train_out.shape == (1000, 3)
    assert test_in.shape == (500, 3)
    assert test_out.shape == (500, 3)

    print(f"✓ Training data created:")
    print(f"  Train: {train_in.shape} → {train_out.shape}")
    print(f"  Test:  {test_in.shape} → {test_out.shape}")

    # Get statistics
    attractor_stats = lorenz.get_attractor_statistics(T=5000)
    print(f"✓ Attractor statistics computed")
    print(f"  Mean: [{attractor_stats['mean'][0]:.2f}, {attractor_stats['mean'][1]:.2f}, {attractor_stats['mean'][2]:.2f}]")
    print(f"  Std:  [{attractor_stats['std'][0]:.2f}, {attractor_stats['std'][1]:.2f}, {attractor_stats['std'][2]:.2f}]")

    print("\n✅ Lorenz-63 test PASSED\n")


def test_mackey_glass():
    """Test Mackey-Glass implementation."""
    print("=" * 70)
    print("TEST 2: Mackey-Glass System")
    print("=" * 70)

    # Test mild chaos (τ=17)
    mg_mild = MackeyGlass(tau=17)

    assert mg_mild.tau == 17
    assert mg_mild.a == 0.2
    assert mg_mild.b == 0.1
    assert mg_mild.n == 10
    assert mg_mild.regime == 'mildly_chaotic'

    print("✓ Mild chaos (τ=17) parameters verified")

    # Generate series
    series = mg_mild.generate_series(T=1000, transient=500)
    assert series.shape == (1000,), "Series shape should be (1000,)"
    assert np.all(np.isfinite(series)), "Series should be finite"

    print(f"✓ Generated series: shape {series.shape}")
    print(f"  Mean: {np.mean(series):.3f}, Std: {np.std(series):.3f}")
    print(f"  Range: [{series.min():.3f}, {series.max():.3f}]")

    # Test normalization
    normalized, stats = mg_mild.normalize_series(series)
    assert normalized.min() >= -1.0 and normalized.max() <= 1.0

    print(f"✓ Normalization works: range [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Test prediction task
    train_in, train_out, test_in, test_out, stats = \
        mg_mild.create_prediction_task(T_train=1000, T_test=500)

    assert train_in.shape == (1000, 1)
    assert test_in.shape == (500, 1)

    print(f"✓ Prediction task created:")
    print(f"  Train: {train_in.shape} → {train_out.shape}")
    print(f"  Test:  {test_in.shape} → {test_out.shape}")

    # Test strong chaos (τ=30)
    mg_strong = MackeyGlass(tau=30)
    assert mg_strong.regime == 'strongly_chaotic'

    print("✓ Strong chaos (τ=30) verified")

    # Test multi-step tasks
    multistep = mg_mild.create_multistep_task(
        T_train=1000, T_test=500,
        horizons=[1, 10, 84]
    )

    assert len(multistep) == 3
    assert 1 in multistep and 10 in multistep and 84 in multistep

    print(f"✓ Multi-step tasks created: {list(multistep.keys())}")

    print("\n✅ Mackey-Glass test PASSED\n")


def test_memory_capacity():
    """Test memory capacity measurement."""
    print("=" * 70)
    print("TEST 3: Memory Capacity Test")
    print("=" * 70)

    # Create mock reservoir for testing
    class MockReservoir:
        def __init__(self, N=5):
            self.N = N
            self.state = np.zeros(N)
            self.W = np.random.randn(N, N) * 0.8 / np.sqrt(N)

        def evolve(self, state, u):
            if state is None:
                state = self.state.copy()
            new_state = np.tanh(self.W @ state + u * np.ones(self.N))
            return new_state

    # Create test
    N = 5
    reservoir = MockReservoir(N=N)

    test = MemoryCapacityTest(
        reservoir_evolve_fn=reservoir.evolve,
        readout_train_fn=None,
        reservoir_dim=N
    )

    print(f"✓ Test initialized with mock reservoir (N={N})")

    # Generate random input
    u = test.generate_random_input(T=1000, seed=42)
    assert u.shape == (1000,)
    assert u.min() >= -1.0 and u.max() <= 1.0

    print(f"✓ Random input generated: shape {u.shape}, range [{u.min():.3f}, {u.max():.3f}]")

    # Run memory capacity test
    MC, r_squared = test.measure_capacity(T=2000, k_max=10, T_washout=100)

    assert isinstance(MC, float)
    assert r_squared.shape == (10,)
    assert MC >= 0, "Memory capacity should be non-negative"
    assert MC <= N + 1, f"Memory capacity {MC:.2f} should not greatly exceed dimension {N}"

    print(f"✓ Memory capacity measured: MC = {MC:.2f}")
    print(f"  Efficiency: MC/N = {MC/N:.2%}")
    print(f"  Theoretical bound: MC ≤ {N}")

    # Check r² values
    print(f"✓ Individual r² values (k=1..5):")
    for k in range(min(5, len(r_squared))):
        print(f"    k={k+1}: r² = {r_squared[k]:.4f}")

    assert np.all(r_squared >= 0), "All r² values should be non-negative"
    assert np.all(r_squared <= 1), "All r² values should be ≤ 1"

    print("\n✅ Memory capacity test PASSED\n")


def test_integration():
    """Integration test: All benchmarks together."""
    print("=" * 70)
    print("TEST 4: Integration Test")
    print("=" * 70)

    # Create all benchmark systems
    lorenz = Lorenz63()
    mg = MackeyGlass(tau=17)

    # Generate data from each
    lorenz_data = lorenz.create_training_data(T_train=500, T_test=200)
    mg_data = mg.create_prediction_task(T_train=500, T_test=200)

    print("✓ Generated data from all benchmark systems")
    print(f"  Lorenz-63: Train {lorenz_data[0].shape}, Test {lorenz_data[2].shape}")
    print(f"  Mackey-Glass: Train {mg_data[0].shape}, Test {mg_data[2].shape}")

    # Verify data properties
    assert lorenz_data[0].shape[0] == 500  # T_train
    assert lorenz_data[2].shape[0] == 200  # T_test
    assert mg_data[0].shape[0] == 500
    assert mg_data[2].shape[0] == 200

    # Check normalization consistency
    for data in [lorenz_data[0], lorenz_data[2], mg_data[0], mg_data[2]]:
        assert data.min() >= -1.0 and data.max() <= 1.0, "All data should be in [-1, 1]"

    print("✓ All data properly normalized to [-1, 1]")

    print("\n✅ Integration test PASSED\n")


def run_all_tests():
    """Run all benchmark tests."""
    print("\n" + "=" * 70)
    print("TQRC BENCHMARK SYSTEMS - VERIFICATION TESTS")
    print("=" * 70)
    print()

    try:
        test_lorenz63()
        test_mackey_glass()
        test_memory_capacity()
        test_integration()

        print("=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Lorenz-63 system with verified parameters (σ=10, ρ=28, β=8/3)")
        print("  ✓ Mackey-Glass DDE with τ=17 (mild) and τ=30 (strong)")
        print("  ✓ Memory capacity measurement protocol")
        print("  ✓ Data normalization to [-1, 1] range")
        print("  ✓ Train/test split functionality")
        print()
        print("Ready for TQRC implementation (Phase 2+)")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
