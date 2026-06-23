"""
Figure 5 — The Limits of Linear Classifiers
============================================

Two side-by-side static panels illustrating why a straight-line boundary
cannot always separate two classes:

Left panel:  Class A (inner cluster) and Class B (outer ring) with the best
             possible straight-line boundary drawn over them — it cannot
             separate the two classes regardless of slope or intercept.

Right panel: The same data with a circular (non-linear) boundary that
             perfectly separates the two classes, motivating the need for
             non-linear classifiers.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_5 import show
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


def show():
    """Render the static Figure 5 linear vs non-linear boundary comparison."""

    plt.close('Notebook2 Figure 5')

    # ── Data generation ───────────────────────────────────────────────────────
    # Fixed seed so every user sees the same clusters.
    rng = np.random.default_rng(7)

    # Class A: a tight central cluster generated in polar coordinates.
    # rng.uniform(low, high, size) draws samples uniformly between low and high.
    a_r   = rng.uniform(0, 1.5, 80)        # Radial distance from the origin
    a_ang = rng.uniform(0, 2 * np.pi, 80)  # Random angle in radians
    ax_pts = a_r * np.cos(a_ang)           # Convert polar → Cartesian x
    ay_pts = a_r * np.sin(a_ang)           # Convert polar → Cartesian y

    # Class B: a ring surrounding Class A — same idea, but larger radii
    b_r   = rng.uniform(2.5, 4.0, 80)
    b_ang = rng.uniform(0, 2 * np.pi, 80)
    bx_pts = b_r * np.cos(b_ang)
    by_pts = b_r * np.sin(b_ang)

    # x-values for drawing the straight-line boundary across the full plot range
    x_line = np.linspace(-4.5, 4.5, 300)

    # 300 angles used to draw the circular boundary as a smooth closed curve
    theta = np.linspace(0, 2 * np.pi, 300)

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook2 Figure 5', figsize=(8, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: linear boundary fails ────────────────────────────────────
    ax = axes[0]

    ax.scatter(ax_pts, ay_pts, color='steelblue', s=50, edgecolors='k',
               linewidth=0.4, label='Class A (inner)', zorder=3)
    ax.scatter(bx_pts, by_pts, color='tomato', s=50, edgecolors='k',
               linewidth=0.4, label='Class B (outer)', zorder=3)

    # A straight line can never wrap around Class A to exclude Class B —
    # no matter the slope or intercept, some points will always be misclassified
    ax.plot(x_line, 0.5 * x_line + 0, 'k--', linewidth=2,
            label='Best linear boundary (still fails)')

    ax.set_title('Linear boundary FAILS on ring-shaped data')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    # set_aspect('equal') ensures circles look round rather than oval
    ax.set_aspect('equal')

    # ── Right panel: circular boundary solves the problem ────────────────────
    ax = axes[1]

    ax.scatter(ax_pts, ay_pts, color='steelblue', s=50, edgecolors='k',
               linewidth=0.4, label='Class A (inner)', zorder=3)
    ax.scatter(bx_pts, by_pts, color='tomato', s=50, edgecolors='k',
               linewidth=0.4, label='Class B (outer)', zorder=3)

    # A circle of radius 2.0 sits in the gap between the two clusters.
    # x = r·cos(θ), y = r·sin(θ) traces out the circle as θ goes from 0 to 2π.
    ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), 'k-', linewidth=2.5,
            label='Ideal boundary (circular — non-linear)')

    ax.set_title('A non-linear boundary SOLVES the problem')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    fig.suptitle('Figure 5: The limits of linear classifiers', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()