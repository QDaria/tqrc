# TQRC: Topological Quantum Reservoir Computing

**Fundamental Limitations of Topological Quantum Reservoir Computing: A No-Go Theorem for Fibonacci Anyonic Systems**

## Author

Daniel Mo Houshmand
QDaria, Oslo, Norway
mo@qdaria.com

## Abstract

This paper establishes fundamental no-go results for Topological Quantum Reservoir Computing (TQRC) using Fibonacci anyons. While topological quantum computing offers inherent fault tolerance through non-Abelian anyonic braiding, we demonstrate that this very protection mechanism creates an irreconcilable tension with the Echo State Property (ESP) essential for reservoir computing.

Our rigorous mathematical analysis proves that:

1. **Unitarity-ESP Incompatibility**: The unitary nature of quantum evolution fundamentally prevents asymptotic state convergence
2. **Topological Memory Prevents Fading Memory**: The robust topological protection that enables fault-tolerant computation actively preserves information indefinitely
3. **Lyapunov Stability Analysis**: Computed Lyapunov exponents (λ ≈ 0.007) confirm marginal stability rather than the required contractivity

## Key Results

- **No-Go Theorem**: Rigorous proof that Fibonacci anyonic systems cannot satisfy ESP
- **Memory Capacity Theorem**: Upper bound MC ≤ N log₂(φ) for N-anyon systems
- **Numerical Validation**: Comprehensive simulations confirming theoretical predictions

## Paper Versions

- `tqrc_ieee.tex` - IEEE Transactions format (10 pages)
- `tqrc_acm.tex` - ACM Computing Surveys format (11 pages)

## Building

```bash
# IEEE version
pdflatex tqrc_ieee.tex
bibtex tqrc_ieee
pdflatex tqrc_ieee.tex
pdflatex tqrc_ieee.tex

# ACM version
pdflatex tqrc_acm.tex
bibtex tqrc_acm
pdflatex tqrc_acm.tex
pdflatex tqrc_acm.tex
```

## Citation

```bibtex
@article{houshmand2024tqrc,
  title={Fundamental Limitations of Topological Quantum Reservoir Computing: A No-Go Theorem for Fibonacci Anyonic Systems},
  author={Houshmand, Daniel Mo},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

$\mathbb{Q}|\mathcal{D}\partial\mathfrak{r}\imath\alpha\rangle$ — QDaria
