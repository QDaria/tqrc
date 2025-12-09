"""
TQRC Diagnostics - State Space Exploration Analysis
====================================================

Tools to verify reservoir explores Hilbert space properly.
"""

import numpy as np
from typing import Tuple, Dict


def measure_state_entropy(states: np.ndarray) -> np.ndarray:
    """
    Compute von Neumann entropy for each state in sequence.

    Args:
        states: Measurement probabilities (T, dim)

    Returns:
        entropy: S(t) = -Σ p_i log(p_i) for each time t

    High entropy → good exploration
    Low entropy → stuck in subspace
    """
    eps = 1e-12  # Avoid log(0)
    entropy = -np.sum(states * np.log(states + eps), axis=1)
    return entropy


def measure_participation_ratio(states: np.ndarray) -> np.ndarray:
    """
    Inverse participation ratio: effective number of active states.

    Args:
        states: Measurement probabilities (T, dim)

    Returns:
        PR(t) = 1 / Σ p_i²

    PR ≈ 1 → localized (bad)
    PR ≈ dim → fully delocalized (good)
    """
    pr = 1.0 / np.sum(states**2, axis=1)
    return pr


def measure_basis_exploration(states: np.ndarray,
                              threshold: float = 0.01) -> Dict[str, float]:
    """
    Check if all basis states are visited during dynamics.

    Args:
        states: Measurement probabilities (T, dim)
        threshold: Minimum probability to count as "visited"

    Returns:
        metrics: Dict with exploration statistics
    """
    dim = states.shape[1]

    # Max probability achieved for each basis state
    max_probs = np.max(states, axis=0)

    # Count how many states visited
    visited = np.sum(max_probs > threshold)

    # Average probability across time
    mean_probs = np.mean(states, axis=0)

    return {
        'num_visited': int(visited),
        'fraction_visited': float(visited / dim),
        'max_probabilities': max_probs,
        'mean_probabilities': mean_probs,
        'uniformity': float(np.std(mean_probs))  # Lower = more uniform
    }


def diagnose_vacuum_isolation(states: np.ndarray,
                              vacuum_index: int = 0) -> Dict[str, float]:
    """
    Check if system is stuck in vacuum state.

    Args:
        states: Measurement probabilities (T, dim)
        vacuum_index: Index of vacuum state (usually 0)

    Returns:
        diagnosis: Dict with vacuum metrics
    """
    vacuum_prob = states[:, vacuum_index]

    return {
        'mean_vacuum_prob': float(np.mean(vacuum_prob)),
        'max_vacuum_prob': float(np.max(vacuum_prob)),
        'min_vacuum_prob': float(np.min(vacuum_prob)),
        'vacuum_dominance': float(np.mean(vacuum_prob > 0.9)),  # Fraction > 90%
        'isolated': bool(np.mean(vacuum_prob) > 0.95)  # True if stuck
    }


def generate_diagnostic_report(states: np.ndarray) -> str:
    """
    Generate comprehensive diagnostic report.

    Args:
        states: Measurement probabilities (T, dim)

    Returns:
        report: Human-readable diagnostic report
    """
    entropy = measure_state_entropy(states)
    pr = measure_participation_ratio(states)
    exploration = measure_basis_exploration(states)
    vacuum_diag = diagnose_vacuum_isolation(states)

    dim = states.shape[1]
    good_exploration = exploration['fraction_visited'] > 0.8

    report = f"""
TQRC State Space Exploration Diagnostics
========================================

Entropy Analysis:
  Mean entropy: {np.mean(entropy):.3f}
  Max entropy: {np.max(entropy):.3f}
  Min entropy: {np.min(entropy):.3f}

Participation Ratio:
  Mean PR: {np.mean(pr):.2f} / {dim}
  Max PR: {np.max(pr):.2f}

Basis Exploration:
  States visited: {exploration['num_visited']} / {dim}
  Coverage: {exploration['fraction_visited']*100:.1f}%
  Uniformity (std): {exploration['uniformity']:.4f}

Vacuum Isolation Check:
  Mean vacuum prob: {vacuum_diag['mean_vacuum_prob']:.3f}
  Stuck in vacuum: {vacuum_diag['isolated']}

VERDICT: {"✅ GOOD EXPLORATION" if good_exploration else "❌ POOR EXPLORATION"}
"""
    return report


def compare_exploration(states_before: np.ndarray,
                       states_after: np.ndarray) -> str:
    """
    Compare state space exploration before and after a fix.

    Args:
        states_before: States from original implementation
        states_after: States from fixed implementation

    Returns:
        comparison: Comparison report
    """
    # Compute metrics for both
    entropy_before = measure_state_entropy(states_before)
    entropy_after = measure_state_entropy(states_after)

    pr_before = measure_participation_ratio(states_before)
    pr_after = measure_participation_ratio(states_after)

    exp_before = measure_basis_exploration(states_before)
    exp_after = measure_basis_exploration(states_after)

    vac_before = diagnose_vacuum_isolation(states_before)
    vac_after = diagnose_vacuum_isolation(states_after)

    # Format improvements
    entropy_before_mean = np.mean(entropy_before)
    entropy_after_mean = np.mean(entropy_after)

    # Handle division by zero for entropy
    if entropy_before_mean > 1e-10:
        entropy_improvement = (entropy_after_mean - entropy_before_mean) / entropy_before_mean * 100
    else:
        # If before entropy is ~0, just compare absolute values
        entropy_improvement = (entropy_after_mean - entropy_before_mean) * 100

    pr_improvement = (np.mean(pr_after) - np.mean(pr_before)) / np.mean(pr_before) * 100
    coverage_improvement = (exp_after['fraction_visited'] - exp_before['fraction_visited']) * 100

    report = f"""
State Space Exploration Comparison
===================================

Entropy:
  Before: {np.mean(entropy_before):.3f}
  After:  {np.mean(entropy_after):.3f}
  Change: {entropy_improvement:+.1f}%

Participation Ratio:
  Before: {np.mean(pr_before):.2f}
  After:  {np.mean(pr_after):.2f}
  Change: {pr_improvement:+.1f}%

Basis Coverage:
  Before: {exp_before['fraction_visited']*100:.1f}%
  After:  {exp_after['fraction_visited']*100:.1f}%
  Change: {coverage_improvement:+.1f}pp

Vacuum Isolation:
  Before: {"STUCK" if vac_before['isolated'] else "OK"}
  After:  {"STUCK" if vac_after['isolated'] else "OK"}

Overall: {"✅ IMPROVEMENT" if (entropy_after_mean > entropy_before_mean or pr_improvement > 0 or coverage_improvement > 0) else "❌ NO IMPROVEMENT"}
"""
    return report
