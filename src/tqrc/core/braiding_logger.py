"""
Braiding Operations Logger
===========================

Provides diagnostic logging for braiding operations to detect if
fractional braiding is effectively identity (H1 hypothesis).

Usage:
    from tqrc.core.braiding_logger import BraidingLogger

    logger = BraidingLogger("experiments/debug_logs/braiding_instrumentation.json")
    braiding = BraidingOperators(hilbert, logger=logger)

    # Operations are automatically logged
    B = braiding.fractional_braid(1, theta=np.pi*0.5)

    # Analyze effectiveness
    results = analyze_braiding_effectiveness(logger.log_file)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class BraidingLogger:
    """
    Logger for braiding operation diagnostics.

    Logs each fractional_braid() call with:
        - Input parameters (braid_index, theta)
        - Matrix properties (eigenvalues, Frobenius distance from identity)
        - Effectiveness metrics

    Attributes:
        log_file: Path to JSON log file
        enabled: Whether logging is active
    """

    def __init__(self, log_file: str, enabled: bool = True):
        """
        Initialize braiding logger.

        Args:
            log_file: Path to JSON log file (will be created if not exists)
            enabled: Enable/disable logging (default True)
        """
        self.log_file = log_file
        self.enabled = enabled
        self._log_count = 0

        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Initialize empty log file with metadata
        if enabled and not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                metadata = {
                    "metadata": {
                        "created_at": datetime.utcnow().isoformat(),
                        "description": "Braiding operation diagnostics for H1 hypothesis testing",
                        "hypothesis": "Braiding matrices are effectively identity: ||B_i(θ) - I|| ≈ 0"
                    },
                    "operations": []
                }
                json.dump(metadata, f, indent=2)

    def log_braid_operation(
        self,
        braid_index: int,
        theta: float,
        eigenvalues: np.ndarray,
        identity_distance: float,
        trace: complex,
        braid_matrix: np.ndarray
    ) -> None:
        """
        Log a single braiding operation with diagnostic metrics.

        Args:
            braid_index: Which braid operator (1 to n-1)
            theta: Braiding angle in radians
            eigenvalues: Eigenvalues of fractional braid matrix
            identity_distance: ||B(θ) - I||_F (Frobenius norm)
            trace: Trace of braid matrix
            braid_matrix: The full braid matrix B(θ)
        """
        if not self.enabled:
            return

        dim = braid_matrix.shape[0]

        # Compute additional metrics
        trace_diff = abs(trace - dim)  # Distance from identity trace
        max_eigenvalue_magnitude = np.max(np.abs(eigenvalues))
        eigenvalue_phases = np.angle(eigenvalues)

        # Create log entry
        entry = {
            "operation_id": self._log_count,
            "timestamp": datetime.utcnow().isoformat(),
            "braid_index": int(braid_index),
            "theta_rad": float(theta),
            "theta_over_pi": float(theta / np.pi),
            "dim": int(dim),
            "metrics": {
                "frobenius_distance": float(identity_distance),
                "trace_diff": float(trace_diff),
                "trace_real": float(trace.real),
                "trace_imag": float(trace.imag),
                "max_eigenvalue_magnitude": float(max_eigenvalue_magnitude)
            },
            "eigenvalues": {
                "values": [{"real": float(ev.real), "imag": float(ev.imag)} for ev in eigenvalues],
                "magnitudes": [float(abs(ev)) for ev in eigenvalues],
                "phases_rad": [float(phase) for phase in eigenvalue_phases]
            },
            "hypothesis_tests": {
                "effectively_identity": bool(identity_distance < 1e-6),
                "eigenvalues_unit_modulus": bool(np.allclose(np.abs(eigenvalues), 1.0, atol=1e-9)),
                "unitary_verified": bool(np.allclose(
                    braid_matrix @ braid_matrix.conj().T,
                    np.eye(dim),
                    atol=1e-9
                ))
            }
        }

        # Append to log file
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)

            log_data["operations"].append(entry)

            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=self._json_serializer)

            self._log_count += 1

        except Exception as e:
            print(f"Warning: Failed to log braiding operation: {e}")

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complexfloating, np.complex64, np.complex128)):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} not JSON serializable")


def analyze_braiding_effectiveness(log_file: str) -> Dict[str, Any]:
    """
    Analyze logged braiding data to determine if operations are effective.

    Tests H1 hypothesis: Braiding matrices are effectively identity.

    Args:
        log_file: Path to braiding instrumentation JSON log

    Returns:
        Dictionary with analysis results:
            - n_operations: Number of logged operations
            - mean_frobenius_distance: Average ||B-I||_F
            - max_frobenius_distance: Maximum observed distance
            - min_frobenius_distance: Minimum observed distance
            - std_frobenius_distance: Standard deviation
            - effectively_identity: Boolean (True if mean < 1e-6)
            - hypothesis_verdict: "H1_TRUE" or "H1_FALSE"
            - theta_range: Range of theta values tested
            - eigenvalue_consistency: Whether eigenvalues are consistently unit modulus

    Interpretation:
        - If mean_frobenius_distance < 1e-3: Braiding is ineffective (H1 TRUE)
        - If mean_frobenius_distance > 0.1: Braiding is working (H1 FALSE)
        - If 1e-3 < mean < 0.1: Partial effectiveness (needs investigation)
    """
    if not os.path.exists(log_file):
        return {
            "error": f"Log file not found: {log_file}",
            "hypothesis_verdict": "CANNOT_TEST"
        }

    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)

        operations = log_data.get("operations", [])

        if not operations:
            return {
                "error": "No operations logged",
                "hypothesis_verdict": "NO_DATA"
            }

        # Extract metrics
        frobenius_distances = [op["metrics"]["frobenius_distance"] for op in operations]
        trace_diffs = [op["metrics"]["trace_diff"] for op in operations]
        theta_values = [op["theta_rad"] for op in operations]

        # Compute statistics
        mean_frob = np.mean(frobenius_distances)
        max_frob = np.max(frobenius_distances)
        min_frob = np.min(frobenius_distances)
        std_frob = np.std(frobenius_distances)

        # Determine verdict
        if mean_frob < 1e-6:
            verdict = "H1_TRUE_STRONG"
            interpretation = "Braiding is completely ineffective (identity within numerical precision)"
        elif mean_frob < 1e-3:
            verdict = "H1_TRUE_WEAK"
            interpretation = "Braiding is effectively identity (distance < 0.001)"
        elif mean_frob < 0.1:
            verdict = "PARTIAL_EFFECTIVENESS"
            interpretation = "Braiding has weak effect (needs investigation)"
        else:
            verdict = "H1_FALSE"
            interpretation = "Braiding is working correctly (significant non-identity)"

        # Check eigenvalue consistency
        eigenvalue_unit_checks = [op["hypothesis_tests"]["eigenvalues_unit_modulus"] for op in operations]
        eigenvalue_consistency = all(eigenvalue_unit_checks)

        # Build results
        results = {
            "n_operations": len(operations),
            "statistics": {
                "mean_frobenius_distance": float(mean_frob),
                "max_frobenius_distance": float(max_frob),
                "min_frobenius_distance": float(min_frob),
                "std_frobenius_distance": float(std_frob),
                "mean_trace_diff": float(np.mean(trace_diffs))
            },
            "theta_range": {
                "min_rad": float(min(theta_values)),
                "max_rad": float(max(theta_values)),
                "min_over_pi": float(min(theta_values) / np.pi),
                "max_over_pi": float(max(theta_values) / np.pi)
            },
            "hypothesis_tests": {
                "effectively_identity": bool(mean_frob < 1e-6),
                "eigenvalue_consistency": eigenvalue_consistency,
                "all_unitary": all(op["hypothesis_tests"]["unitary_verified"] for op in operations)
            },
            "hypothesis_verdict": verdict,
            "interpretation": interpretation,
            "log_file": log_file,
            "analyzed_at": datetime.utcnow().isoformat()
        }

        return results

    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "hypothesis_verdict": "ERROR"
        }


def print_analysis_report(results: Dict[str, Any]) -> None:
    """
    Pretty-print analysis results.

    Args:
        results: Output from analyze_braiding_effectiveness()
    """
    print("\n" + "="*70)
    print("BRAIDING EFFECTIVENESS ANALYSIS REPORT")
    print("="*70)

    if "error" in results:
        print(f"\n❌ ERROR: {results['error']}")
        print(f"Verdict: {results['hypothesis_verdict']}")
        return

    print(f"\nOperations Analyzed: {results['n_operations']}")
    print(f"Theta Range: [{results['theta_range']['min_over_pi']:.3f}π, {results['theta_range']['max_over_pi']:.3f}π]")

    print("\n" + "-"*70)
    print("FROBENIUS DISTANCE FROM IDENTITY: ||B(θ) - I||_F")
    print("-"*70)
    stats = results['statistics']
    print(f"  Mean:   {stats['mean_frobenius_distance']:.6e}")
    print(f"  Max:    {stats['max_frobenius_distance']:.6e}")
    print(f"  Min:    {stats['min_frobenius_distance']:.6e}")
    print(f"  StdDev: {stats['std_frobenius_distance']:.6e}")

    print("\n" + "-"*70)
    print("HYPOTHESIS TEST RESULTS")
    print("-"*70)
    tests = results['hypothesis_tests']
    print(f"  Effectively Identity:        {'✅ YES' if tests['effectively_identity'] else '❌ NO'}")
    print(f"  Eigenvalues Unit Modulus:    {'✅ YES' if tests['eigenvalue_consistency'] else '❌ NO'}")
    print(f"  All Operations Unitary:      {'✅ YES' if tests['all_unitary'] else '❌ NO'}")

    print("\n" + "="*70)
    print(f"VERDICT: {results['hypothesis_verdict']}")
    print("="*70)
    print(f"\n{results['interpretation']}")
    print("\n")
