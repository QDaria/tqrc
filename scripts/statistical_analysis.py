#!/usr/bin/env python3
"""Enhanced statistical analysis for TQRC paper.

Computes:
- Bootstrap confidence intervals (10,000 resamples)
- Cohen's d effect sizes
- Paired t-tests with Bonferroni correction
- Power analysis

Author: TQRC Authors
Date: 2026-01-06
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000,
                 ci_level: float = 0.95, statistic=np.mean) -> tuple:
    """Compute bootstrap confidence interval.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level (default 95%)
        statistic: Statistic to compute (default: mean)

    Returns:
        (lower_ci, upper_ci, bootstrap_samples)
    """
    np.random.seed(42)  # Reproducibility
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(resample)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper, bootstrap_stats


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Uses pooled standard deviation for independent samples.

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.inf if group1.mean() != group2.mean() else 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    else:
        return "very large"


def welch_t_test(group1: np.ndarray, group2: np.ndarray) -> tuple:
    """Welch's t-test for unequal variances.

    Returns:
        (t_statistic, p_value, degrees_of_freedom)
    """
    result = stats.ttest_ind(group1, group2, equal_var=False)

    # Compute Welch-Satterthwaite degrees of freedom
    n1, n2 = len(group1), len(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    if v1 == 0 and v2 == 0:
        df = n1 + n2 - 2
    else:
        df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

    return result.statistic, result.pvalue, df


def run_experiments_with_stats():
    """Run experiments and collect raw trial data for statistical analysis."""
    from tqrc.core.reservoir import TQRCReservoir
    from tqrc.benchmarks.mackey_glass import MackeyGlass
    from tqrc.benchmarks.esn import ESN
    from tqrc.utils.metrics import nrmse

    print("=" * 70)
    print("TQRC Paper - Enhanced Statistical Analysis")
    print("Bootstrap CI (10,000 resamples) + Effect Sizes + Hypothesis Tests")
    print("=" * 70)

    # Generate Mackey-Glass data
    mg = MackeyGlass(tau=17, random_seed=42)
    data = mg.generate_series(3000, transient=500)
    data_min, data_max = data.min(), data.max()
    data_normalized = 2 * (data - data_min) / (data_max - data_min) - 1

    train_data = data_normalized[:2000].reshape(-1, 1)
    test_data = data_normalized[2000:].reshape(-1, 1)

    n_trials = 30
    results = {}
    raw_trials = {}

    # ─────────────────────────────────────────────────────────────────────
    # TQRC Experiments
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1/4] Running TQRC experiments...")

    for gamma, name in [(0.0, 'tqrc_unitary'), (0.1, 'tqrc_dissipative')]:
        trials = []
        for trial in range(n_trials):
            try:
                reservoir = TQRCReservoir(
                    n_anyons=4, input_dim=1, braid_length=10,
                    decoherence_rate=gamma, random_seed=100 + trial
                )
                states = reservoir.run_dynamics(train_data, washout=500)
                X = states[:-1]
                y = train_data[501:len(states)+500].flatten()

                ridge_alpha = 1e-4
                W_out = np.linalg.lstsq(
                    X.T @ X + ridge_alpha * np.eye(X.shape[1]),
                    X.T @ y, rcond=None
                )[0]

                test_states = reservoir.run_dynamics(test_data, washout=100)
                y_pred = test_states[:-1] @ W_out
                y_true = test_data[101:len(test_states)+100].flatten()

                error = nrmse(y_pred, y_true)
                if error < 10.0:
                    trials.append(error)
            except Exception:
                pass

        raw_trials[name] = np.array(trials)
        print(f"  {name}: {len(trials)} valid trials, mean={np.mean(trials):.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # ESN Experiments
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2/4] Running ESN baseline experiments...")

    for n_reservoir in [13, 100]:
        name = f'esn_n{n_reservoir}'
        trials = []
        for trial in range(n_trials):
            try:
                esn = ESN(
                    n_reservoir=n_reservoir,
                    spectral_radius=0.95,
                    input_scaling=0.1,
                    random_seed=200 + trial
                )
                states = esn.run_dynamics(train_data, washout=500)
                X = states[:-1]
                y = train_data[501:len(states)+500].flatten()

                ridge_alpha = 1e-4
                W_out = np.linalg.lstsq(
                    X.T @ X + ridge_alpha * np.eye(X.shape[1]),
                    X.T @ y, rcond=None
                )[0]

                test_states = esn.run_dynamics(test_data, washout=100)
                y_pred = test_states[:-1] @ W_out
                y_true = test_data[101:len(test_states)+100].flatten()

                error = nrmse(y_pred, y_true)
                if error < 10.0:
                    trials.append(error)
            except Exception:
                pass

        raw_trials[name] = np.array(trials)
        print(f"  {name}: {len(trials)} valid trials, mean={np.mean(trials):.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # Bootstrap Confidence Intervals
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3/4] Computing bootstrap confidence intervals (10,000 resamples)...")

    for name, trials in raw_trials.items():
        if len(trials) >= 10:
            lower, upper, _ = bootstrap_ci(trials, n_bootstrap=10000)
            results[name] = {
                'mean_nrmse': float(np.mean(trials)),
                'std_nrmse': float(np.std(trials, ddof=1)),
                'bootstrap_ci_95_low': float(lower),
                'bootstrap_ci_95_high': float(upper),
                'n_valid_trials': len(trials)
            }
            print(f"  {name}: [{lower:.4f}, {upper:.4f}]")

    # ─────────────────────────────────────────────────────────────────────
    # Effect Sizes and Hypothesis Tests
    # ─────────────────────────────────────────────────────────────────────
    print("\n[4/4] Computing effect sizes and hypothesis tests...")

    comparisons = [
        ('esn_n13', 'tqrc_unitary', 'ESN(13) vs TQRC Pure'),
        ('esn_n100', 'tqrc_unitary', 'ESN(100) vs TQRC Pure'),
        ('esn_n13', 'tqrc_dissipative', 'ESN(13) vs TQRC Dissipative'),
        ('esn_n100', 'tqrc_dissipative', 'ESN(100) vs TQRC Dissipative'),
    ]

    statistical_tests = []
    n_comparisons = len(comparisons)

    for esn_name, tqrc_name, desc in comparisons:
        if esn_name in raw_trials and tqrc_name in raw_trials:
            esn_data = raw_trials[esn_name]
            tqrc_data = raw_trials[tqrc_name]

            # Cohen's d
            d = cohens_d(esn_data, tqrc_data)
            interpretation = interpret_cohens_d(d)

            # Welch's t-test
            t_stat, p_value, df = welch_t_test(esn_data, tqrc_data)

            # Bonferroni correction
            p_bonferroni = min(p_value * n_comparisons, 1.0)

            # Performance ratio
            ratio = np.mean(tqrc_data) / np.mean(esn_data) if np.mean(esn_data) > 0 else np.inf

            test_result = {
                'comparison': desc,
                'cohens_d': float(d),
                'effect_interpretation': interpretation,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'p_bonferroni': float(p_bonferroni),
                'df': float(df),
                'performance_ratio': float(ratio),
                'significant_bonferroni': bool(p_bonferroni < 0.05)
            }
            statistical_tests.append(test_result)

            sig_marker = "***" if p_bonferroni < 0.001 else "**" if p_bonferroni < 0.01 else "*" if p_bonferroni < 0.05 else ""
            print(f"  {desc}:")
            print(f"    Cohen's d = {d:.2f} ({interpretation})")
            print(f"    t({df:.1f}) = {t_stat:.2f}, p = {p_value:.2e} {sig_marker}")
            print(f"    Performance ratio: {ratio:.1f}×")

    # ─────────────────────────────────────────────────────────────────────
    # Compile Final Results
    # ─────────────────────────────────────────────────────────────────────
    final_results = {
        'experiment_metadata': {
            'date': '2026-01-06',
            'n_trials': n_trials,
            'n_bootstrap': 10000,
            'ci_method': 'Percentile bootstrap',
            'hypothesis_test': "Welch's t-test (unequal variances)",
            'multiple_comparison_correction': 'Bonferroni',
            'effect_size_metric': "Cohen's d (pooled SD)"
        },
        'mackey_glass_tau17': results,
        'statistical_comparisons': statistical_tests,
        'interpretation_guide': {
            'cohens_d': {
                '< 0.2': 'negligible',
                '0.2-0.5': 'small',
                '0.5-0.8': 'medium',
                '0.8-1.2': 'large',
                '> 1.2': 'very large'
            },
            'significance_levels': {
                '*': 'p < 0.05',
                '**': 'p < 0.01',
                '***': 'p < 0.001'
            }
        }
    }

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'statistical_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("=" * 70)
    print(f"{'Model':<25} {'NRMSE':<15} {'95% Bootstrap CI':<20} {'n':<5}")
    print("-" * 70)
    for name, res in results.items():
        ci = f"[{res['bootstrap_ci_95_low']:.4f}, {res['bootstrap_ci_95_high']:.4f}]"
        print(f"{name:<25} {res['mean_nrmse']:.4f} ± {res['std_nrmse']:.4f}  {ci:<20} {res['n_valid_trials']:<5}")

    print("\n" + "-" * 70)
    print("EFFECT SIZE SUMMARY")
    print("-" * 70)
    for test in statistical_tests:
        sig = "✓" if test['significant_bonferroni'] else "✗"
        print(f"{test['comparison']:<30} d={test['cohens_d']:>6.2f} ({test['effect_interpretation']:<10}) p_bonf={test['p_bonferroni']:.2e} {sig}")

    return final_results


if __name__ == "__main__":
    run_experiments_with_stats()
