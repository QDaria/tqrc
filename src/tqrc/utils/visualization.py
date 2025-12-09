"""Visualization utilities for TQRC experiments.

Plotting functions for benchmark results and analysis.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "matplotlib not installed. Visualization functions will not work. "
        "Install with: pip install matplotlib"
    )


def _check_matplotlib():
    """Raise error if matplotlib not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_lorenz_attractor(
    trajectory: np.ndarray,
    title: str = "Lorenz Attractor",
    figsize: Tuple[int, int] = (10, 8),
    view_angle: Tuple[int, int] = (30, 45),
    show: bool = True
) -> plt.Figure:
    """Plot 3D Lorenz attractor trajectory.

    Args:
        trajectory: Trajectory array, shape (n_steps, 3) for (x, y, z)
        title: Plot title
        figsize: Figure size in inches (width, height)
        view_angle: 3D view angle (elevation, azimuth)
        show: Whether to display plot (True) or just return figure

    Returns:
        Matplotlib figure object

    Example:
        >>> from tqrc.benchmarks import Lorenz63
        >>> lorenz = Lorenz63()
        >>> trajectory = lorenz.generate_trajectory(n_steps=5000)
        >>> plot_lorenz_attractor(trajectory)
    """
    _check_matplotlib()

    if trajectory.shape[1] != 3:
        raise ValueError(
            f"trajectory must have shape (n_steps, 3), got {trajectory.shape}"
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            linewidth=0.5, alpha=0.7)

    # Labels and title
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Grid
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_value: float,
    metric_name: str = "NRMSE",
    n_steps_display: int = 500,
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True
) -> plt.Figure:
    """Plot prediction vs ground truth with error visualization.

    Args:
        y_true: Ground truth time series, shape (n_steps,)
        y_pred: Predicted time series, shape (n_steps,)
        metric_value: Metric value to display
        metric_name: Name of metric (NRMSE, NMSE, VPT, etc.)
        n_steps_display: Number of time steps to display
        figsize: Figure size in inches
        show: Whether to display plot

    Returns:
        Matplotlib figure object

    Example:
        >>> nrmse_val = nrmse(y_pred, y_true)
        >>> plot_prediction(y_true, y_pred, nrmse_val, "NRMSE")
    """
    _check_matplotlib()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
        )

    # Limit display length
    n_display = min(n_steps_display, len(y_true))
    y_true_plot = y_true[:n_display]
    y_pred_plot = y_pred[:n_display]
    time_steps = np.arange(n_display)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Prediction vs ground truth
    ax1.plot(time_steps, y_true_plot, 'b-', linewidth=1.5,
             label='Ground Truth', alpha=0.8)
    ax1.plot(time_steps, y_pred_plot, 'r--', linewidth=1.0,
             label='Prediction', alpha=0.7)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f'Prediction vs Ground Truth ({metric_name} = {metric_value:.4f})',
        fontsize=13
    )

    # Plot 2: Absolute error
    error = np.abs(y_true_plot - y_pred_plot)
    ax2.fill_between(time_steps, 0, error, alpha=0.5, color='orange',
                      label='Absolute Error')
    ax2.plot(time_steps, error, 'r-', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('|Error|', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Show mean error
    mean_error = np.mean(error)
    ax2.axhline(y=mean_error, color='k', linestyle=':', linewidth=1,
                label=f'Mean Error = {mean_error:.4f}')

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_memory_capacity(
    r_squared_values: np.ndarray,
    total_capacity: float,
    theoretical_max: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True
) -> plt.Figure:
    """Plot cumulative memory capacity.

    From verification/03 Section 4:
        MC_k = Σ_{j=1}^k r_j²

    Args:
        r_squared_values: R² values for each delay k, shape (K_max,)
        total_capacity: Total memory capacity (MC = Σ r_k²)
        theoretical_max: Theoretical maximum MC (e.g., reservoir dimension N)
        figsize: Figure size in inches
        show: Whether to display plot

    Returns:
        Matplotlib figure object

    Example:
        >>> mc = memory_capacity(r_squared_values)
        >>> plot_memory_capacity(r_squared_values, mc, theoretical_max=100)
    """
    _check_matplotlib()

    k_values = np.arange(1, len(r_squared_values) + 1)
    cumulative_mc = np.cumsum(r_squared_values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Individual R² values
    ax1.bar(k_values, r_squared_values, alpha=0.6, color='steelblue',
            label='r²(k)')
    ax1.set_ylabel('R²(k)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_title('Memory Capacity Analysis', fontsize=13)

    # Plot 2: Cumulative MC
    ax2.plot(k_values, cumulative_mc, 'b-', linewidth=2, label='Cumulative MC')
    ax2.axhline(y=total_capacity, color='r', linestyle='--', linewidth=1.5,
                label=f'Total MC = {total_capacity:.2f}')

    if theoretical_max is not None:
        ax2.axhline(y=theoretical_max, color='g', linestyle=':', linewidth=1.5,
                    label=f'Theoretical Max = {theoretical_max}')
        # Show efficiency
        efficiency = (total_capacity / theoretical_max) * 100
        ax2.text(0.98, 0.02, f'Efficiency: {efficiency:.1f}%',
                 transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Delay k (time steps)', fontsize=11)
    ax2.set_ylabel('Cumulative MC', fontsize=11)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_phase_space(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coords: Tuple[int, int, int] = (0, 1, 2),
    figsize: Tuple[int, int] = (14, 6),
    show: bool = True
) -> plt.Figure:
    """Plot phase space comparison for 3D systems (e.g., Lorenz).

    Args:
        y_true: Ground truth trajectory, shape (n_steps, n_dim)
        y_pred: Predicted trajectory, shape (n_steps, n_dim)
        coords: Which coordinates to plot (default: first 3)
        figsize: Figure size in inches
        show: Whether to display plot

    Returns:
        Matplotlib figure object

    Example:
        >>> # For Lorenz system
        >>> plot_phase_space(y_true, y_pred)  # Plot (x, y, z)
    """
    _check_matplotlib()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shape {y_true.shape}")

    # Extract coordinates
    x_idx, y_idx, z_idx = coords

    fig = plt.figure(figsize=figsize)

    # Ground truth (left)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(y_true[:, x_idx], y_true[:, y_idx], y_true[:, z_idx],
             linewidth=0.5, alpha=0.7, color='blue')
    ax1.set_xlabel(f'Coord {x_idx}', fontsize=10)
    ax1.set_ylabel(f'Coord {y_idx}', fontsize=10)
    ax1.set_zlabel(f'Coord {z_idx}', fontsize=10)
    ax1.set_title('Ground Truth', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Prediction (right)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(y_pred[:, x_idx], y_pred[:, y_idx], y_pred[:, z_idx],
             linewidth=0.5, alpha=0.7, color='red')
    ax2.set_xlabel(f'Coord {x_idx}', fontsize=10)
    ax2.set_ylabel(f'Coord {y_idx}', fontsize=10)
    ax2.set_zlabel(f'Coord {z_idx}', fontsize=10)
    ax2.set_title('Prediction', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Match viewing angles
    ax2.view_init(elev=ax1.elev, azim=ax1.azim)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_vpt_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lyapunov_time: float,
    dt: float = 0.01,
    threshold: float = 0.4,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True
) -> plt.Figure:
    """Plot Valid Prediction Time analysis with error growth.

    Args:
        y_true: Ground truth time series, shape (n_steps,)
        y_pred: Predicted time series, shape (n_steps,)
        lyapunov_time: Lyapunov time T_λ
        dt: Time step
        threshold: Error threshold for VPT
        figsize: Figure size in inches
        show: Whether to display plot

    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()

    from .metrics import valid_prediction_time, nmse

    n_steps = len(y_true)
    time_array = np.arange(n_steps) * dt

    # Compute cumulative NMSE
    cumsum_squared_error = np.cumsum((y_pred - y_true) ** 2)
    cumsum_true_squared = np.cumsum(y_true ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        nmse_curve = cumsum_squared_error / cumsum_true_squared
    nmse_curve = np.nan_to_num(nmse_curve, nan=0.0)

    # Calculate VPT
    vpt = valid_prediction_time(y_pred, y_true, lyapunov_time, dt, threshold)
    t_star = vpt * lyapunov_time

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot NMSE vs time
    ax.plot(time_array, nmse_curve, 'b-', linewidth=2, label='NMSE(t)')
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold}')
    ax.axvline(x=t_star, color='g', linestyle=':', linewidth=2,
               label=f't* = {t_star:.2f} (VPT = {vpt:.2f} T_λ)')

    # Mark Lyapunov times
    max_time = time_array[-1]
    n_lyap = int(max_time / lyapunov_time) + 1
    for i in range(1, n_lyap):
        t_lyap = i * lyapunov_time
        if t_lyap <= max_time:
            ax.axvline(x=t_lyap, color='gray', linestyle=':', linewidth=0.8,
                       alpha=0.5)

    ax.set_xlabel('Time (units)', fontsize=11)
    ax.set_ylabel('NMSE', fontsize=11)
    ax.set_title(f'Valid Prediction Time Analysis (VPT = {vpt:.2f} Lyapunov times)',
                 fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    if show:
        plt.tight_layout()
        plt.show()

    return fig
