"""
Figure 8 & 9 — Error Landscape and Optimal Decision Boundary
=============================================================

Two side-by-side static plots produced by a brute-force grid search over
all (slope, intercept) combinations:

Left panel  (Figure 8): the error landscape — for each slope m, the minimum
              misclassification error achievable by optimising the intercept c.
              A red dashed line marks the globally best slope.

Right panel (Figure 9): the scatter data with the optimal decision boundary
              y = best_m * x + best_c drawn over it.

The grid search evaluates every combination of:
    - 101 slopes   from -200 to +200 in steps of 4
    - 101 intercepts from -50,000 to +50,000 in steps of 1,000

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_8 import show
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
from IPython.display import display


def show():
    """Render Figure 8 and 9 — the error landscape and optimal boundary."""

    plt.close('Notebook2 Figure 8')

    # ── Data generation ───────────────────────────────────────────────────────
    # Same seed as Figure 6 so users see the same two clusters throughout.
    rng = np.random.default_rng(1)
    class_a_x = rng.normal(loc=180, scale=8,    size=50)
    class_a_y = rng.normal(loc=3500, scale=300, size=50)
    class_b_x = rng.normal(loc=210, scale=8,    size=50)
    class_b_y = rng.normal(loc=5000, scale=300, size=50)

    # ── Error rate function ───────────────────────────────────────────────────
    # Counts points on the wrong side of y = mx + c and returns the fraction.
    def error_rate_2d(xa, ya, xb, yb, m, c):
        """Return misclassification rate for the line y = mx + c."""
        total   = len(xa) + len(xb)
        errors  = sum(1 for xi, yi in zip(xa, ya) if yi >= m * xi + c)
        errors += sum(1 for xi, yi in zip(xb, yb) if yi <  m * xi + c)
        return errors / total

    # ── Parameter grid ────────────────────────────────────────────────────────
    # We test every combination of slope and intercept in these ranges.
    # The wide intercept range is needed because body mass sits in the thousands.
    m_values = [m / 10 for m in range(-2000, 2001, 40)]
    c_values = list(range(-50000, 50001, 1000))

    # Track the single best (m, c) pair found across the entire grid
    best_m, best_c, best_err = None, None, float('inf')

    # For the landscape plot: record the best error per slope (minimised over c)
    landscape_m   = []
    landscape_err = []

    # Nested loop — evaluate every (slope, intercept) combination
    for m in m_values:
        # Reset the best error for this slope before scanning all intercepts
        local_best_err = float('inf')
        for c in c_values:
            err = error_rate_2d(class_a_x, class_a_y, class_b_x, class_b_y, m, c)
            if err < local_best_err:
                local_best_err = err      # Best error for this slope so far
            if err < best_err:
                best_err, best_m, best_c = err, m, c   # New global best
        landscape_m.append(m)
        landscape_err.append(local_best_err)

    # Print a summary so users can see the search results in the cell output
    print(f'Grid size: {len(m_values)} slopes x {len(c_values)} intercepts = '
          f'{len(m_values) * len(c_values):,} evaluations')
    print(f'Best slope:     m = {best_m:.1f}')
    print(f'Best intercept: c = {best_c:,}')
    print(f'Min error rate: {best_err:.1%}')
    print(f'Accuracy:       {1 - best_err:.1%}')

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook2 Figure 8', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: error landscape ───────────────────────────────────────────
    ax = axes[0]

    # For each slope, plot the minimum error achievable across all intercepts.
    # The resulting curve shows which slopes are inherently better at separating
    # the two classes, regardless of where the line is positioned vertically.
    ax.plot(landscape_m, landscape_err, color='steelblue', linewidth=2,
            label='Min error over all c (per slope m)')

    # Red dashed line marking the globally optimal slope
    ax.axvline(x=best_m, color='red', linestyle='--', linewidth=1.8,
               label=f'Best m = {best_m:.1f}  (error = {best_err:.1%})')

    ax.set_xlabel('Slope (m)')
    ax.set_ylabel('Minimum misclassification error rate')
    ax.set_title('Figure 8: Error landscape\n'
                 '(best achievable error per slope, optimised over all intercepts)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right panel: scatter + optimal boundary ───────────────────────────────
    ax = axes[1]
    x_plot = np.linspace(155, 240, 300)

    ax.scatter(class_a_x, class_a_y, color='steelblue', s=55, edgecolors='k',
               linewidth=0.4, label='Class A', zorder=3)
    ax.scatter(class_b_x, class_b_y, color='tomato', s=55, edgecolors='k',
               linewidth=0.4, label='Class B', zorder=3)

    # Draw the globally optimal boundary found by the grid search
    ax.plot(x_plot, best_m * x_plot + best_c, 'r--', linewidth=2,
            label=f'Best boundary: y = {best_m:.1f}x + {best_c:,}')

    ax.set_xlabel('Flipper length (mm)')
    ax.set_ylabel('Body mass (g)')
    ax.set_title('Figure 9: Optimal boundary from 2-D grid search')
    ax.set_xlim(155, 240)
    ax.set_ylim(2000, 7000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()