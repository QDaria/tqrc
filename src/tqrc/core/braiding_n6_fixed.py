"""
Fixed Braiding Operators for n=6 Fibonacci Anyons
==================================================

Implements proper F-matrix-based braiding with OFF-DIAGONAL elements
for amplitude mixing between basis states.

This fixes the fundamental limitation of n=4 diagonal operators.

Theory Source:
    - Trebst et al. (2008) Eq. (3.5)
    - BRAIDING_MATRIX_ANALYSIS.md lines 290-309
"""

import numpy as np
from ..constants import R_TT_1, R_TT_TAU, F_00, F_01, F_10, F_11, PHI


def build_n6_braid_matrix_with_mixing(braid_index: int) -> np.ndarray:
    """
    Build proper n=6 braiding operator with amplitude mixing.

    Args:
        braid_index: Which braid (1-5 for n=6)

    Returns:
        5×5 unitary matrix with OFF-DIAGONAL mixing elements

    Theory:
        Unlike n=4 diagonal matrices, n=6 braids have 2×2 blocks
        that MIX amplitudes:

        B_2 = [[R₁,  0,   0,   0,   0  ],
               [0,   a,   b,   0,   0  ],  ← MIXING
               [0,   b*, -a*, 0,   0  ],  ← MIXING
               [0,   0,   0,   c,   d  ],
               [0,   0,   0,   d*, -c* ]]

        where a, b, c, d are computed from F-matrix elements.
    """

    # Compute mixing coefficients from F-matrix
    # Source: Trebst (2008) Eq. (3.5)

    phi_inv = F_00      # φ^{-1} = 0.618...
    phi_inv_half = F_01  # φ^{-1/2} = 0.786...

    # Block 1-2 mixing (intermediate fusion channels)
    a = phi_inv * R_TT_TAU * phi_inv + phi_inv_half * R_TT_1 * phi_inv_half
    b = phi_inv * R_TT_TAU * phi_inv_half - phi_inv_half * R_TT_1 * phi_inv

    # Block 3-4 mixing (different fusion channels)
    c = phi_inv * R_TT_1 * phi_inv + phi_inv_half * R_TT_TAU * phi_inv_half
    d = phi_inv * R_TT_1 * phi_inv_half - phi_inv_half * R_TT_TAU * phi_inv

    # Construct block-diagonal matrix with mixing
    B = np.zeros((5, 5), dtype=complex)

    # Vacuum channel (diagonal)
    B[0, 0] = R_TT_1

    # First mixing block
    B[1, 1] = a
    B[1, 2] = b
    B[2, 1] = np.conj(b)
    B[2, 2] = -np.conj(a)

    # Second mixing block
    B[3, 3] = c
    B[3, 4] = d
    B[4, 3] = np.conj(d)
    B[4, 4] = -np.conj(c)

    return B


def verify_amplitude_mixing(B: np.ndarray, threshold: float = 1e-10) -> dict:
    """
    Verify that matrix has off-diagonal elements (amplitude mixing).

    Args:
        B: Braiding matrix
        threshold: Minimum magnitude for off-diagonal elements

    Returns:
        Dictionary with mixing metrics:
            - has_mixing: True if off-diagonal elements > threshold
            - max_off_diagonal: Largest off-diagonal magnitude
            - is_diagonal: True if effectively diagonal
    """
    dim = B.shape[0]

    # Extract off-diagonal elements
    off_diagonal_elements = []
    for i in range(dim):
        for j in range(dim):
            if i != j:
                off_diagonal_elements.append(np.abs(B[i, j]))

    max_off_diag = np.max(off_diagonal_elements)
    has_mixing = max_off_diag > threshold
    is_diagonal = max_off_diag < threshold

    return {
        'has_mixing': has_mixing,
        'max_off_diagonal': max_off_diag,
        'is_diagonal': is_diagonal,
        'mixing_strength': max_off_diag
    }


def test_n6_vs_n4_comparison():
    """
    Compare n=4 diagonal vs n=6 mixing operators.

    This demonstrates the CRITICAL difference:
    - n=4: Diagonal → phase-only → frozen probabilities
    - n=6: Off-diagonal → amplitude mixing → varying probabilities
    """

    print("=" * 60)
    print("COMPARISON: n=4 Diagonal vs n=6 Mixing")
    print("=" * 60)

    # n=4 diagonal operator
    B_n4 = np.diag([R_TT_1, R_TT_TAU])

    print("\nn=4 Operator (Diagonal):")
    print(f"Shape: {B_n4.shape}")
    mixing_n4 = verify_amplitude_mixing(B_n4)
    print(f"Has mixing: {mixing_n4['has_mixing']}")
    print(f"Max off-diagonal: {mixing_n4['max_off_diagonal']:.6e}")
    print(f"Is diagonal: {mixing_n4['is_diagonal']}")

    # n=6 mixing operator
    B_n6 = build_n6_braid_matrix_with_mixing(braid_index=2)

    print("\nn=6 Operator (With Mixing):")
    print(f"Shape: {B_n6.shape}")
    mixing_n6 = verify_amplitude_mixing(B_n6)
    print(f"Has mixing: {mixing_n6['has_mixing']}")
    print(f"Max off-diagonal: {mixing_n6['max_off_diagonal']:.6f}")
    print(f"Is diagonal: {mixing_n6['is_diagonal']}")

    # Demonstrate amplitude mixing effect
    print("\n" + "=" * 60)
    print("AMPLITUDE MIXING DEMONSTRATION")
    print("=" * 60)

    # Initial state: superposition over first 3 basis states
    psi_n6 = np.zeros(5, dtype=complex)
    psi_n6[0] = 0.6
    psi_n6[1] = 0.5
    psi_n6[2] = 0.5
    psi_n6[3] = 0.3
    psi_n6[4] = 0.2
    psi_n6 = psi_n6 / np.linalg.norm(psi_n6)  # Normalize

    print("\nInitial state probabilities:")
    probs_before = np.abs(psi_n6)**2
    for i, p in enumerate(probs_before):
        print(f"  |{i}⟩: {p:.6f}")

    # Apply n=6 operator
    psi_after = B_n6 @ psi_n6
    probs_after = np.abs(psi_after)**2

    print("\nAfter applying B_n6:")
    for i, p in enumerate(probs_after):
        change = probs_after[i] - probs_before[i]
        print(f"  |{i}⟩: {p:.6f}  (change: {change:+.6f})")

    # Compute probability change magnitude
    prob_change = np.linalg.norm(probs_after - probs_before)
    print(f"\nTotal probability change: {prob_change:.6f}")

    if prob_change > 0.01:
        print("✅ AMPLITUDE MIXING CONFIRMED - Probabilities changed significantly!")
    else:
        print("❌ NO MIXING - Probabilities unchanged (would fail for RC)")

    # Verify unitarity
    unitarity_error = np.linalg.norm(B_n6 @ B_n6.conj().T - np.eye(5), 'fro')
    print(f"\nUnitarity check: ||B†B - I||_F = {unitarity_error:.6e}")

    if unitarity_error < 1e-14:
        print("✅ Unitary")
    else:
        print("❌ Not unitary!")

    print("=" * 60)


if __name__ == "__main__":
    test_n6_vs_n4_comparison()
