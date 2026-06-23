"""
Figure 3 — The Covariance Matrix Controls the Shape of a 2-D Gaussian
=======================================================================
Visualises how the covariance matrix determines the shape, scale, and
orientation of a two-dimensional Gaussian distribution, using four
representative configurations:

  Spherical      — equal variance in both directions, no correlation.
                   The distribution is circular.
  Diagonal       — unequal variances, no correlation. The distribution
                   is axis-aligned but stretched along one direction.
  Full (positive) — positive off-diagonal entries indicate that x1 and
                   x2 tend to increase together. The distribution is
                   tilted at a positive angle.
  Full (negative) — negative off-diagonal entries indicate that x1 and
                   x2 move in opposite directions. The distribution is
                   tilted at a negative angle.

Each panel shows ellipses at 1 and 2 standard deviations from the mean,
derived from the eigendecomposition of the covariance matrix. The
eigenvalues, printed at the bottom of each panel, determine the length
of the ellipse axes: larger eigenvalue = more spread in that direction.

This figure is entirely self-contained and requires no external variables.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_3 import show
    show()

Copyright © 2026 Robert Lyon. All Rights Reserved.

This notebook and all associated materials are the intellectual property of the author.

Permission is granted solely to read, study, and analyse this material for personal educational purposes. No other rights are granted.

Without the prior written consent of the author, you may not:

* Copy, reproduce, redistribute, publish, transmit, or display this work in whole or in part.
* Modify, adapt, transform, translate, or create derivative works based on this material.
* Incorporate any portion of this work into another project, publication, product, model, dataset, or codebase.
* Use this material for commercial purposes.
* Remove or alter this copyright notice.

All intellectual property rights, including copyright and any derivative rights, remain exclusively vested in the author.

Access to this material does not constitute a transfer of ownership, license, or any other intellectual property rights except as expressly stated above.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ── Covariance configurations ────────────────────────────────────────────────
# Four representative covariance matrices covering the main shape families.
# Each entry is (title, covariance matrix, colour).
CONFIGS = [
    (
        'Spherical\n' + r'($\sigma_x = \sigma_y$, no corr.)',
        np.array([[1.0,  0.0], [0.0,  1.0]]),
        'steelblue',
    ),
    (
        'Diagonal\n' + r'($\sigma_x \neq \sigma_y$, no corr.)',
        np.array([[2.5,  0.0], [0.0,  0.5]]),
        'tomato',
    ),
    (
        'Full\n(positive correlation)',
        np.array([[2.0,  1.5], [1.5,  2.0]]),
        'seagreen',
    ),
    (
        'Full\n(negative correlation)',
        np.array([[2.0, -1.5], [-1.5, 2.0]]),
        'goldenrod',
    ),
]

# The mean is placed at the origin for all panels so the covariance shape
# is the only thing that varies between them.
MU_DEMO = np.array([0.0, 0.0])

# Axis limits: wide enough to show the 2-std ellipse for all configurations
# with a small margin around it.
AXIS_LIM = 4.5

# Colour for the eigenvalue annotation text.
COLOUR_ANNOTATION = '#666666'
COLOUR_GRID       = '#cccccc'


def _draw_gaussian_ellipse(ax, mu, cov, n_std, color, lw, ls='-', label=None):
    """
    Draw an ellipse representing n_std standard deviations of a 2-D Gaussian.

    The ellipse axes and orientation come from the eigendecomposition of the
    covariance matrix. np.linalg.eigh is used rather than np.linalg.eig
    because it is numerically more stable for symmetric matrices and returns
    real-valued eigenvalues directly.

    Parameters
    ----------
    ax : matplotlib Axes
    mu : array-like of shape (2,) — the distribution mean (ellipse centre)
    cov : ndarray of shape (2, 2) — the covariance matrix
    n_std : float — number of standard deviations the ellipse represents
    color, lw, ls : matplotlib style arguments
    label : str or None — legend label
    """
    vals, vecs = np.linalg.eigh(cov)

    # The angle of the major axis (largest eigenvalue) relative to the x-axis.
    # vecs[:, 1] is the eigenvector corresponding to the largest eigenvalue.
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))

    # Ellipse width and height are 2 * n_std * sqrt(eigenvalue) in each
    # principal direction: sqrt converts variance to standard deviation.
    width  = 2 * n_std * np.sqrt(vals[1])
    height = 2 * n_std * np.sqrt(vals[0])

    ell = Ellipse(
        xy=mu, width=width, height=height, angle=angle,
        edgecolor=color, fc='none', lw=lw, linestyle=ls, label=label,
    )
    ax.add_patch(ell)
    ax.scatter(*mu, color=color, s=80, marker='+', lw=2.5, zorder=5)


def show():
    """Render Figure 3: covariance matrix shape and orientation of 2-D Gaussians."""
    plt.close('Notebook8bonus Figure 3')

    # Reduce horizontal space between panels so the figure is compact without
    # the plots being cramped. wspace is the fraction of the average axis
    # width to use as horizontal padding between subplots.
    fig, axes = plt.subplots(
        1, 4, figsize=(12, 5),
        num='Notebook8bonus Figure 3',
    )

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False

    for ax, (title, cov, col) in zip(axes, CONFIGS):

        # 1-std ellipse (solid) and 2-std ellipse (dashed) for each config.
        _draw_gaussian_ellipse(ax, MU_DEMO, cov, n_std=1,
                               color=col, lw=2.5, label='1 std')
        _draw_gaussian_ellipse(ax, MU_DEMO, cov, n_std=2,
                               color=col, lw=1.5, ls='--', label='2 std')

        ax.set_xlim(-AXIS_LIM, AXIS_LIM)
        ax.set_ylim(-AXIS_LIM, AXIS_LIM)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('$x_1$')
        ax.grid(True, color=COLOUR_GRID, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

        # Eigenvalues printed inside each panel: they are the variances along
        # the principal axes of the ellipse, so they directly determine the
        # axis lengths. Larger eigenvalue = longer ellipse axis = more spread.
        vals, _ = np.linalg.eigh(cov)
        ax.text(
            0.04, 0.05,
            f'eigenvalues:\n[{vals[0]:.2f}, {vals[1]:.2f}]',
            transform=ax.transAxes,
            fontsize=8, color=COLOUR_ANNOTATION, va='bottom',
        )

    # Only the leftmost panel needs a y-axis label; remove it from the others
    # to reduce clutter when the panels are close together.
    axes[0].set_ylabel('$x_2$')
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.suptitle(
        'Figure 3: The covariance matrix controls the shape and orientation '
        'of a Gaussian\nellipses show 1-std and 2-std contours',
        fontsize=11,
    )
    # subplots_adjust is used instead of tight_layout because tight_layout
    # emits a warning when axes have set_aspect('equal'), which prevents
    # it from computing the layout reliably.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.12, wspace=0.08)