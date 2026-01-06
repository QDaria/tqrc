# Version 2 Changelog

**Paper:** The Fundamental Tension in Topological Quantum Reservoir Computing: Why Unitarity Opposes the Echo State Property

**DOI:** [10.22541/au.176549133.31550916/v2](https://doi.org/10.22541/au.176549133.31550916/v2)

**Author:** Daniel Mo Houshmand (QDaria, Oslo, Norway)

**Date:** January 2026

---

## Summary of Changes

Version 2 represents a comprehensive revision addressing peer review feedback, adding rigorous statistical validation, and improving reproducibility.

---

## Major Additions

### 1. Statistical Rigor (30-Trial Validation)
- All benchmark results now based on **30 independent trials** (up from single runs)
- Added **bootstrap 95% confidence intervals** for all metrics
- Statistical significance testing between TQRC variants and ESN baselines
- Results file: `results/genuine_30trial_results.json`

### 2. New Figure: Quantum State Evolution (Figure 9)
- **First visualization of actual simulated Fibonacci anyon dynamics**
- 5-panel figure showing:
  - (a) Pure unitary TQRC density matrix evolution
  - (b) Dissipative TQRC density matrix evolution
  - (c) Probability trajectories (unitary case)
  - (d) Ground state convergence (dissipative case)
  - (e) Input signal and von Neumann entropy evolution
- Demonstrates the fundamental unitarity-ESP tension visually

### 3. NARMA-10 Benchmark
- Added NARMA-10 nonlinear autoregressive benchmark
- Complements Mackey-Glass chaotic time series
- Further validates the no-go theorem across task types

### 4. Literature Update (2025)
- Added 12 new references from 2024-2025
- Includes Google/Quantinuum Fibonacci anyon demonstrations
- Updated context on quantum reservoir computing advances

---

## Improvements

### 5. Enhanced Figure Captions
- **Figure 4** (Fusion space scaling): Added panel-by-panel description of dimension growth
- **Figure 5** (R/F matrices): Expanded mathematical context for braiding operations
- **Figure 8** (ESP violation): Detailed explanation of Lyapunov exponent analysis

### 6. Colorblind-Safe Visualization
- All figures updated to Wong et al. colorblind-safe palette
- 8% of males have color vision deficiency - now accessible to all readers

### 7. Reproducibility Package
- Added `Dockerfile` for containerized reproduction
- Complete `requirements.txt` with pinned versions
- `scripts/` directory with figure generation code

### 8. Theoretical Framework
- Strengthened mathematical rigor in ESP violation proof
- Added Theorem 2 (Dissipation-Protection Trade-off)
- Explicit connection to Lindblad master equation formalism

---

## Files Changed

| File | Change |
|------|--------|
| `tqrc_ieee.tex` | +622 lines (major revision) |
| `tqrc_ieee.pdf` | Updated (1.3 MB, 14 pages) |
| `figures/fig_quantum_state_evolution.pdf` | NEW |
| `results/genuine_30trial_results.json` | NEW |
| `scripts/generate_state_evolution_figure.py` | NEW |
| `Dockerfile` | NEW |
| Multiple source files | Bug fixes, improved modularity |

---

## Validation

- [x] All 30-trial experiments completed successfully
- [x] LaTeX compiles without errors
- [x] All 17 figures render correctly
- [x] Table values verified against experimental data
- [x] PDF reviewed for formatting issues

---

## Citation

```bibtex
@article{houshmand2025tqrc,
  title={The Fundamental Tension in Topological Quantum Reservoir Computing:
         Why Unitarity Opposes the Echo State Property},
  author={Houshmand, Daniel Mo},
  journal={TechRxiv Preprint},
  year={2025},
  doi={10.22541/au.176549133.31550916/v2},
  note={Version 2}
}
```
