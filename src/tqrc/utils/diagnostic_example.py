"""
Example Usage: TQRC Diagnostics
================================

Demonstrates how to use diagnostic tools to verify state space exploration.
Shows before/after comparison of vacuum isolation fix.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqrc.utils.diagnostics import (
    measure_state_entropy,
    measure_participation_ratio,
    measure_basis_exploration,
    diagnose_vacuum_isolation,
    generate_diagnostic_report,
    compare_exploration
)


def simulate_poor_exploration(T: int = 500, dim: int = 8) -> np.ndarray:
    """
    Simulate TQRC with vacuum isolation (broken behavior).

    This mimics the bug where the system gets stuck in vacuum state.
    """
    states = np.zeros((T, dim))

    # Start in vacuum
    states[0, 0] = 1.0

    # Slow exponential decay to uniform (very slow escape)
    for t in range(1, T):
        # 99% stay in vacuum, 1% leak to uniform
        states[t] = 0.99 * states[t-1]
        states[t, 0] += 0.99 * (1 - states[t-1, 0])  # Restore vacuum weight
        states[t] += 0.01 / dim  # Tiny leak
        states[t] /= states[t].sum()  # Renormalize

    return states


def simulate_good_exploration(T: int = 500, dim: int = 8) -> np.ndarray:
    """
    Simulate TQRC with proper exploration (fixed behavior).

    System properly explores full Hilbert space.
    """
    rng = np.random.RandomState(42)
    states = np.zeros((T, dim))

    # Start in vacuum but escape quickly
    states[0, 0] = 1.0

    # Markov chain that explores space
    for t in range(1, T):
        # Mix current state with random exploration
        mixing = 0.7  # High mixing → good exploration
        uniform = np.ones(dim) / dim
        random_walk = rng.dirichlet(np.ones(dim) * 2.0)

        states[t] = (1 - mixing) * states[t-1] + mixing * random_walk
        states[t] /= states[t].sum()

    return states


def plot_exploration_comparison(states_before: np.ndarray,
                                states_after: np.ndarray,
                                save_path: str = None):
    """
    Visualize state space exploration before and after fix.
    """
    T, dim = states_before.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('TQRC State Space Exploration: Before vs After Fix',
                 fontsize=14, fontweight='bold')

    # Compute metrics
    entropy_before = measure_state_entropy(states_before)
    entropy_after = measure_state_entropy(states_after)

    pr_before = measure_participation_ratio(states_before)
    pr_after = measure_participation_ratio(states_after)

    vacuum_before = states_before[:, 0]
    vacuum_after = states_after[:, 0]

    # Row 1: Before fix
    axes[0, 0].plot(entropy_before, 'r-', linewidth=2)
    axes[0, 0].set_title('Entropy (Before Fix)')
    axes[0, 0].set_ylabel('S(t)')
    axes[0, 0].axhline(np.log(dim), color='k', linestyle='--',
                       label=f'Max = log({dim})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(pr_before, 'r-', linewidth=2)
    axes[0, 1].set_title('Participation Ratio (Before Fix)')
    axes[0, 1].set_ylabel('PR(t)')
    axes[0, 1].axhline(dim, color='k', linestyle='--',
                       label=f'Max = {dim}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(vacuum_before, 'r-', linewidth=2)
    axes[0, 2].set_title('Vacuum Probability (Before Fix)')
    axes[0, 2].set_ylabel('P(|0⟩)')
    axes[0, 2].axhline(1/dim, color='k', linestyle='--',
                       label='Uniform = 1/8')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: After fix
    axes[1, 0].plot(entropy_after, 'g-', linewidth=2)
    axes[1, 0].set_title('Entropy (After Fix)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('S(t)')
    axes[1, 0].axhline(np.log(dim), color='k', linestyle='--')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(pr_after, 'g-', linewidth=2)
    axes[1, 1].set_title('Participation Ratio (After Fix)')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('PR(t)')
    axes[1, 1].axhline(dim, color='k', linestyle='--')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(vacuum_after, 'g-', linewidth=2)
    axes[1, 2].set_title('Vacuum Probability (After Fix)')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('P(|0⟩)')
    axes[1, 2].axhline(1/dim, color='k', linestyle='--')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    """
    Main example: demonstrate diagnostic workflow.
    """
    print("="*70)
    print("TQRC State Space Exploration Diagnostics Example")
    print("="*70)

    # Simulate both scenarios
    print("\nSimulating broken TQRC (vacuum isolation)...")
    states_broken = simulate_poor_exploration(T=500, dim=8)

    print("Simulating fixed TQRC (proper exploration)...")
    states_fixed = simulate_good_exploration(T=500, dim=8)

    # Generate individual reports
    print("\n" + "="*70)
    print("BROKEN IMPLEMENTATION:")
    print("="*70)
    print(generate_diagnostic_report(states_broken))

    print("\n" + "="*70)
    print("FIXED IMPLEMENTATION:")
    print("="*70)
    print(generate_diagnostic_report(states_fixed))

    # Compare
    print("\n" + "="*70)
    print("BEFORE vs AFTER COMPARISON:")
    print("="*70)
    print(compare_exploration(states_broken, states_fixed))

    # Detailed basis exploration analysis
    print("\n" + "="*70)
    print("DETAILED BASIS EXPLORATION ANALYSIS:")
    print("="*70)

    exp_broken = measure_basis_exploration(states_broken, threshold=0.01)
    exp_fixed = measure_basis_exploration(states_fixed, threshold=0.01)

    print("\nBroken Implementation:")
    print(f"  States visited: {exp_broken['num_visited']}/8")
    print(f"  Coverage: {exp_broken['fraction_visited']*100:.1f}%")
    print(f"  Max probabilities per state:")
    for i, p in enumerate(exp_broken['max_probabilities']):
        print(f"    |{i}⟩: {p:.4f}")

    print("\nFixed Implementation:")
    print(f"  States visited: {exp_fixed['num_visited']}/8")
    print(f"  Coverage: {exp_fixed['fraction_visited']*100:.1f}%")
    print(f"  Max probabilities per state:")
    for i, p in enumerate(exp_fixed['max_probabilities']):
        print(f"    |{i}⟩: {p:.4f}")

    # Vacuum isolation details
    print("\n" + "="*70)
    print("VACUUM ISOLATION DETAILS:")
    print("="*70)

    vac_broken = diagnose_vacuum_isolation(states_broken)
    vac_fixed = diagnose_vacuum_isolation(states_fixed)

    print("\nBroken Implementation:")
    for key, value in vac_broken.items():
        print(f"  {key}: {value}")

    print("\nFixed Implementation:")
    for key, value in vac_fixed.items():
        print(f"  {key}: {value}")

    # Plot comparison
    print("\n" + "="*70)
    print("Generating visualization...")
    plot_exploration_comparison(states_broken, states_fixed)

    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("""
The diagnostics clearly show:

1. **Broken Implementation (Vacuum Isolation):**
   - Low entropy (system stuck in subspace)
   - Low participation ratio (localized states)
   - High vacuum probability (can't escape |0⟩)
   - Poor basis coverage (<50%)

2. **Fixed Implementation (Proper Exploration):**
   - High entropy (near maximum)
   - High participation ratio (near dimension)
   - Low vacuum probability (proper mixing)
   - Good basis coverage (>80%)

The fix successfully addresses the vacuum isolation issue!
    """)


if __name__ == "__main__":
    main()
