<p align="center">
  <img src="figures/fig_tqrc_architecture_pro.png" width="800" alt="TQRC Architecture">
</p>

<h1 align="center">Topological Quantum Reservoir Computing</h1>

<h3 align="center">A No-Go Theorem: Why Unitarity Opposes the Echo State Property</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2501.XXXXX"><img src="https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.22541/au.176549133.31550916/v2"><img src="https://img.shields.io/badge/TechRxiv-10.22541/au.176549133-blue.svg" alt="TechRxiv"></a>
  <a href="https://doi.org/10.5281/zenodo.17889778"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17889778.svg" alt="Zenodo"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-3776ab.svg?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/QDaria/tqrc/actions"><img src="https://img.shields.io/github/actions/workflow/status/QDaria/tqrc/ci.yml?branch=main&label=CI&logo=github" alt="CI"></a>
  <a href="https://codecov.io/gh/QDaria/tqrc"><img src="https://img.shields.io/codecov/c/github/QDaria/tqrc?logo=codecov" alt="Coverage"></a>
  <a href="https://github.com/QDaria/tqrc/stargazers"><img src="https://img.shields.io/github/stars/QDaria/tqrc?style=social" alt="Stars"></a>
</p>

<p align="center">
  <b>Daniel Mo Houshmand</b><br>
  <a href="https://qdaria.com">QDaria</a> | Oslo, Norway<br>
  <a href="mailto:mo@qdaria.com">mo@qdaria.com</a>
</p>

---

## Overview

This repository contains the complete research artifacts for our paper establishing **fundamental no-go results** for Topological Quantum Reservoir Computing (TQRC) using Fibonacci anyons.

**Key Finding**: The unitary nature of topological quantum evolution mathematically prevents the Echo State Property (ESP) required for reservoir computing. This is not an engineering limitation but a fundamental incompatibility.

| Metric | Pure Unitary TQRC | Classical ESN | Gap |
|--------|-------------------|---------------|-----|
| Mackey-Glass NRMSE | 0.966 | 0.015 | 64x |
| Lorenz-63 NRMSE | 1.01 | 0.005 | 202x |
| NARMA-10 NRMSE | 0.75 | 0.81 | ~1x |

> **Note**: TQRC performs comparably on memory-focused tasks (NARMA-10) but fails catastrophically on chaotic prediction requiring fading memory.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/QDaria/tqrc.git
cd tqrc

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.tqrc import FibonacciAnyon; print('OK')"
```

### Run Experiments

```bash
# Reproduce main results (Table 1-3)
python scripts/run_benchmarks.py

# ESP violation demonstration (Figure 5)
python scripts/esp_violation.py

# Memory capacity analysis (Figure 6)
python scripts/memory_capacity.py
```

### Build Paper

```bash
cd paper/v3
pdflatex -interaction=nonstopmode tqrc_ieee_v3.tex
pdflatex -interaction=nonstopmode tqrc_ieee_v3.tex
pdflatex -interaction=nonstopmode tqrc_ieee_v3.tex
```

---

## Key Results

### Theorem 1: Unitarity-ESP Incompatibility

> **No-Go Theorem**: Let $\mathcal{E}(\rho) = U\rho U^\dagger$ be a unitary quantum channel. Then $\mathcal{E}$ cannot satisfy the quantum Echo State Property.

**Proof**: Unitary evolution preserves trace distance:
$$D_{\text{tr}}(\mathcal{E}(\rho_1), \mathcal{E}(\rho_2)) = D_{\text{tr}}(\rho_1, \rho_2)$$

ESP requires $D_{\text{tr}} \to 0$ as $t \to \infty$, which is impossible when distance is preserved.

### Lemma: Spectral Radius Condition

The quantum ESP is satisfied if and only if the spectral radius of the channel restricted to traceless operators satisfies $\rho(\mathcal{E}|_{\text{traceless}}) < 1$.

For unitary channels, all eigenvalues have magnitude 1, so $\rho = 1$, violating ESP.

---

## Repository Structure

```
tqrc/
├── paper/
│   └── v3/
│       ├── tqrc_ieee_v3.tex      # Main paper (IEEE format)
│       ├── tqrc_ieee_v3.pdf      # Compiled PDF
│       └── figures/              # Symlink to ../figures
├── figures/                      # All 18 publication figures
│   ├── fig01_architecture.pdf
│   ├── fig05_esp_violation.pdf
│   ├── fig06_dissipative_results.pdf
│   ├── fig07_root_cause.pdf
│   └── ...
├── src/
│   └── tqrc/                     # Python package
│       ├── __init__.py
│       ├── fibonacci.py          # Fibonacci anyon implementation
│       ├── reservoir.py          # TQRC reservoir class
│       ├── benchmarks.py         # Mackey-Glass, Lorenz, NARMA
│       └── esn.py                # Classical ESN baseline
├── scripts/
│   ├── run_benchmarks.py         # Reproduce all results
│   ├── esp_violation.py          # ESP demonstration
│   └── memory_capacity.py        # Memory scaling analysis
├── results/                      # Cached experimental results
├── requirements.txt              # Python dependencies
├── CITATION.cff                  # Citation metadata
├── LICENSE                       # CC BY 4.0
└── CLAUDE.md                     # AI assistant instructions
```

---

## Figures

<details>
<summary><b>Figure 5: ESP Violation</b> (click to expand)</summary>
<p align="center">
<img src="figures/fig05_esp_violation.pdf" width="600" alt="ESP Violation">
</p>
Pure unitary evolution (red) maintains constant state distance, violating ESP. Dissipative dynamics (blue) achieve convergence but sacrifice topological protection.
</details>

<details>
<summary><b>Figure 6: Dissipative Results</b></summary>
<p align="center">
<img src="figures/fig06_dissipative_results.pdf" width="600" alt="Dissipative Results">
</p>
NRMSE vs dissipation rate showing optimal performance at Γ ≈ 0.25.
</details>

<details>
<summary><b>Figure 7: Root Cause Analysis</b></summary>
<p align="center">
<img src="figures/fig07_root_cause.pdf" width="600" alt="Root Cause">
</p>
ESN uses 13/13 dimensions with 0.98 input correlation; TQRC uses only 4/13 with near-zero correlation.
</details>

<details>
<summary><b>Figure 11: Protection-ESP Tradeoff</b></summary>
<p align="center">
<img src="figures/fig11_tradeoff.pdf" width="600" alt="Tradeoff">
</p>
Fundamental tradeoff: dissipation enables ESP but destroys topological protection.
</details>

---

## Benchmarks

### Mackey-Glass Time Series

| Model | Dimension | NRMSE | 95% CI |
|-------|-----------|-------|--------|
| Classical ESN | 13 | **0.015** | [0.013, 0.017] |
| Classical ESN | 100 | **0.004** | [0.004, 0.004] |
| Pure Unitary TQRC | 13 | 0.966 | [0.966, 0.966] |
| Dissipative TQRC | 13 | 1.18 | [0.97, 1.45] |

### Lorenz-63 Attractor

| Model | NRMSE | Performance Gap |
|-------|-------|-----------------|
| Classical ESN (13D) | **0.005** | 1x |
| Pure Unitary TQRC | 1.01 | 202x worse |
| Dissipative TQRC | 1.12 | 224x worse |

### NARMA-10 (Memory Task)

| Model | NRMSE | Notes |
|-------|-------|-------|
| Pure Unitary TQRC | **0.75** | Competitive |
| Classical ESN | 0.81 | Similar |

---

## Citation

```bibtex
@article{houshmand2026tqrc,
  title   = {The Fundamental Tension in Topological Quantum Reservoir
             Computing: Why Unitarity Opposes the Echo State Property},
  author  = {Houshmand, Daniel Mo},
  journal = {arXiv preprint arXiv:2501.XXXXX},
  year    = {2026},
  doi     = {10.22541/au.176549133.31550916/v2}
}
```

### CITATION.cff

This repository includes a `CITATION.cff` file for automatic citation in GitHub and Zenodo.

---

## Preprint Links

| Platform | Link | Status |
|----------|------|--------|
| **arXiv** | [arXiv:2501.XXXXX](https://arxiv.org/abs/2501.XXXXX) | Pending |
| **TechRxiv** | [10.22541/au.176549133](https://doi.org/10.22541/au.176549133.31550916/v2) | Published |
| **Zenodo** | [10.5281/zenodo.17889778](https://doi.org/10.5281/zenodo.17889778) | Published |

---

## Related Work

### Foundational Papers

- Jaeger (2001). *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148.
- Nayak et al. (2008). *Non-Abelian anyons and topological quantum computation.* Rev. Mod. Phys. 80, 1083.

### Recent Experimental Advances

- Xu et al. (2024). *Non-Abelian braiding of Fibonacci anyons with a superconducting processor.* Nature Physics 20, 1469.
- Iqbal et al. (2024). *Non-Abelian topological order and anyons on a trapped-ion processor.* Nature 626, 505.

### Quantum Reservoir Computing

- Fujii & Nakajima (2017). *Harnessing disordered-ensemble quantum dynamics for machine learning.* Phys. Rev. Applied 8, 024030.
- Sannia et al. (2024). *Dissipation as a resource for quantum reservoir computing.* Quantum 8, 1291.

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt this material for any purpose with attribution.

Code is licensed under [MIT License](LICENSE).

---

## Acknowledgments

We thank the quantum computing community for valuable discussions on the intersection of topological protection and machine learning dynamics.

---

<p align="center">
  <a href="https://qdaria.com"><img src="https://img.shields.io/badge/QDaria-Quantum%20Computing-purple?style=for-the-badge" alt="QDaria"></a>
</p>
