# CLAUDE.md - TQRC Project

## Project Overview

Topological Quantum Reservoir Computing (TQRC) research paper proving a no-go theorem: unitarity fundamentally opposes the Echo State Property required for reservoir computing.

## Build Commands

### Build Paper (LaTeX) - FULL CYCLE REQUIRED
```bash
cd paper/v3
pdflatex -interaction=nonstopmode tqrc_ieee_v3.tex
pdflatex -interaction=nonstopmode tqrc_ieee_v3.tex
pdflatex -interaction=nonstopmode tqrc_ieee_v3.tex
```

**Note**: Bibliography is inlined (no bibtex needed), but 3 passes required for cross-references.

### Verify Build
```bash
grep "undefined" tqrc_ieee_v3.log | wc -l  # Should be 0
```

## 🚨 MANDATORY: Visual Validation Before Delivery

**CRITICAL**: Before suggesting to publish or deliver ANY PDF:

1. **Build LaTeX properly**: Run pdflatex 3 times minimum
2. **Check for [?] citations**: `grep "undefined" *.log` must return 0
3. **Visually verify with Playwright**:
   - Start local server: `python3 -m http.server 8765`
   - Navigate: `http://localhost:8765/tqrc_ieee_v3.pdf`
   - Take screenshot to confirm citations and figures render correctly
4. **Never say "it should work"** - PROVE IT with visual evidence

## Directory Structure

- `paper/v3/` - Current version (IEEE format)
- `paper/.archive/` - Archived older versions
- `figures/` - Shared figures (symlinked from paper/v3/)
- `archive/` - Backup of root-level files

## Writing Guidelines

**Avoid AI-generative patterns:**
- NO em dashes (---) - use commas, colons, semicolons, or periods
- LIMIT: "comprehensive", "rigorously", "fundamentally", "significantly"
- Keep language neutral and scientific
