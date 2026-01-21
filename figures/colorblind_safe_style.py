#!/usr/bin/env python3
"""Colorblind-safe figure styling for TQRC paper.

This module provides colorblind-safe color palettes and plotting utilities
following Web Content Accessibility Guidelines (WCAG) 2.1 AAA contrast ratios.

References:
- Wong, B. (2011). Points of view: Color blindness. Nature Methods, 8(6), 441.
- Okabe, M. & Ito, K. (2008). Color Universal Design (CUD)

The palette used is the "Wong" palette, verified for deuteranopia, protanopia,
and tritanopia color vision deficiencies.

Author: TQRC Authors
Date: 2026-01-06
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler

# ============================================================================
# WONG COLORBLIND-SAFE PALETTE
# ============================================================================
# From Wong (2011) Nature Methods, verified for all CVD types
# RGB values normalized to 0-1 range

WONG_PALETTE = {
    'black':       '#000000',  # (0, 0, 0)
    'orange':      '#E69F00',  # (230, 159, 0)
    'sky_blue':    '#56B4E9',  # (86, 180, 233)
    'green':       '#009E73',  # (0, 158, 115) - bluish green
    'yellow':      '#F0E442',  # (240, 228, 66)
    'blue':        '#0072B2',  # (0, 114, 178)
    'vermillion':  '#D55E00',  # (213, 94, 0)
    'pink':        '#CC79A7',  # (204, 121, 167) - reddish purple
}

# Primary colors for main figure elements (high contrast pairs)
TQRC_COLORS = {
    'tqrc_pure':       WONG_PALETTE['vermillion'],   # Pure unitary TQRC
    'tqrc_dissipative': WONG_PALETTE['orange'],      # Dissipative TQRC
    'esn':             WONG_PALETTE['blue'],         # Classical ESN baseline
    'theory':          WONG_PALETTE['sky_blue'],     # Theoretical predictions
    'highlight':       WONG_PALETTE['green'],        # Key results
    'background':      WONG_PALETTE['yellow'],       # Background/reference
}

# Ordered list for automatic cycling (maximally distinct pairs first)
COLOR_CYCLE = [
    WONG_PALETTE['blue'],
    WONG_PALETTE['vermillion'],
    WONG_PALETTE['green'],
    WONG_PALETTE['orange'],
    WONG_PALETTE['sky_blue'],
    WONG_PALETTE['pink'],
    WONG_PALETTE['yellow'],
    WONG_PALETTE['black'],
]


# ============================================================================
# HATCHING PATTERNS (for redundant encoding)
# ============================================================================
# Using both color AND pattern ensures accessibility even in grayscale

HATCH_PATTERNS = {
    'tqrc_pure':       '///',
    'tqrc_dissipative': '\\\\\\',
    'esn':             'xxx',
    'theory':          '...',
    'none':            '',
}


# ============================================================================
# MARKER STYLES (for scatter/line plots)
# ============================================================================
# Distinct shapes provide redundant encoding

MARKER_STYLES = {
    'tqrc_pure':       's',  # Square
    'tqrc_dissipative': '^',  # Triangle up
    'esn':             'o',  # Circle
    'theory':          'd',  # Diamond
}


# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================

def setup_colorblind_style():
    """Configure matplotlib for colorblind-safe, publication-quality figures."""

    # Use LaTeX-compatible font
    plt.rcParams.update({
        # Figure size for IEEE column width
        'figure.figsize': (3.5, 2.625),  # Single column width
        'figure.dpi': 300,

        # Font settings (LaTeX-compatible)
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,

        # Color cycle
        'axes.prop_cycle': cycler(color=COLOR_CYCLE),

        # Line properties
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Grid (subtle)
        'grid.alpha': 0.3,
        'grid.linestyle': '--',

        # Axes
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Save settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def get_color(name: str) -> str:
    """Get colorblind-safe color by semantic name.

    Args:
        name: One of 'tqrc_pure', 'tqrc_dissipative', 'esn', 'theory', 'highlight'

    Returns:
        Hex color string
    """
    return TQRC_COLORS.get(name, WONG_PALETTE['black'])


def get_marker(name: str) -> str:
    """Get marker style by semantic name."""
    return MARKER_STYLES.get(name, 'o')


def get_hatch(name: str) -> str:
    """Get hatch pattern by semantic name."""
    return HATCH_PATTERNS.get(name, '')


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def demo_colorblind_palette():
    """Generate demo figure showing the colorblind-safe palette."""
    setup_colorblind_style()

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    # 1. Color palette swatches
    ax = axes[0, 0]
    for i, (name, color) in enumerate(WONG_PALETTE.items()):
        ax.barh(i, 1, color=color, height=0.8, edgecolor='black', linewidth=0.5)
        ax.text(0.5, i, name, ha='center', va='center', fontsize=8,
                color='white' if name in ['black', 'blue', 'vermillion'] else 'black')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(WONG_PALETTE) - 0.5)
    ax.set_title('Wong Colorblind-Safe Palette', fontsize=10)
    ax.axis('off')

    # 2. Line plot example
    ax = axes[0, 1]
    x = np.linspace(0, 2*np.pi, 100)
    for i, (name, color) in enumerate(list(TQRC_COLORS.items())[:4]):
        ax.plot(x, np.sin(x + i*0.5), color=color, label=name,
                marker=list(MARKER_STYLES.values())[i], markevery=15)
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Line Plot Example', fontsize=10)
    ax.legend(loc='upper right', fontsize=7)

    # 3. Bar plot with hatching
    ax = axes[1, 0]
    categories = ['ESN', 'TQRC Pure', 'TQRC Diss.']
    values = [0.02, 0.97, 0.55]
    colors = [TQRC_COLORS['esn'], TQRC_COLORS['tqrc_pure'], TQRC_COLORS['tqrc_dissipative']]
    hatches = [HATCH_PATTERNS['esn'], HATCH_PATTERNS['tqrc_pure'], HATCH_PATTERNS['tqrc_dissipative']]

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_ylabel('NRMSE')
    ax.set_title('Bar Plot with Redundant Encoding', fontsize=10)

    # 4. Scatter plot
    ax = axes[1, 1]
    np.random.seed(42)
    for i, (name, color) in enumerate(list(TQRC_COLORS.items())[:3]):
        x = np.random.randn(20) + i*2
        y = np.random.randn(20)
        ax.scatter(x, y, c=color, marker=list(MARKER_STYLES.values())[i],
                   label=name, s=50, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Scatter Plot Example', fontsize=10)
    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = demo_colorblind_palette()
    fig.savefig('colorblind_palette_demo.pdf')
    fig.savefig('colorblind_palette_demo.png', dpi=150)
    plt.show()
    print("Colorblind-safe palette demo saved to colorblind_palette_demo.pdf")
