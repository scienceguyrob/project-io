"""
Figure 3 — Loss Landscape: Local vs Global Minima
==================================================

Two side-by-side panels illustrating the concept of local and global minima
using a synthetic multi-valley loss function:

Left panel:  the full loss landscape with both the local minimum (orange dot)
             and the global minimum (red star) marked and labelled.

Right panel: a simulation of gradient descent starting from a point on the
             left side of the landscape, showing how the algorithm descends
             efficiently but gets trapped in the local minimum rather than
             finding the deeper global minimum.

This figure is purely illustrative — the loss function is not tied to any
specific model, but is representative of the kind of complex landscape that
real optimisers must navigate.

Usage
-----
From a Jupyter notebook cell::

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


def show():
    """Render the static Figure 3 local vs global minima illustration."""

    plt.close('Notebook4 Figure 3')

    # ── Construct the synthetic loss landscape ────────────────────────────────
    # np.linspace(-5, 5, 1000) creates 1000 evenly spaced x values from -5 to 5.
    # This gives us a smooth curve when we plot loss against x.
    x_axis = np.linspace(-5, 5, 1000)

    # A synthetic loss function combining a sine wave, a Gaussian envelope,
    # a gentle quadratic bowl, and a sharp negative Gaussian dip.
    # The result is a curve with multiple valleys of different depths.
    loss = (np.sin(2.5 * x_axis) * np.exp(-0.1 * x_axis**2)
            + 0.3 * x_axis**2
            - 0.5 * np.exp(-((x_axis - 1.2)**2) / 0.3))

    # ── Find the global minimum ───────────────────────────────────────────────
    # np.argmin returns the INDEX of the smallest value in the array.
    # We then use that index to look up the corresponding x and loss values.
    global_min_idx = np.argmin(loss)
    global_min_x   = x_axis[global_min_idx]
    global_min_y   = loss[global_min_idx]

    # ── Find a local minimum on the left side of the landscape ───────────────
    # We restrict the search to a sub-range using a boolean mask.
    # (x_axis > -3.5) & (x_axis < -1.0) creates a True/False array that is
    # True only for x values between -3.5 and -1.0.
    local_mask    = (x_axis > -3.5) & (x_axis < -1.0)
    local_min_idx = np.argmin(loss[local_mask])
    local_min_x   = x_axis[local_mask][local_min_idx]
    local_min_y   = loss[local_mask][local_min_idx]

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook4 Figure 3', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: annotated loss landscape ──────────────────────────────────
    ax = axes[0]
    ax.plot(x_axis, loss, color='steelblue', linewidth=2.5, label='Loss function')

    # Orange dot marks the local minimum — a valley, but not the deepest one
    ax.scatter([local_min_x], [local_min_y], color='orange', s=160, zorder=5,
               label=f'Local min  (x = {local_min_x:.2f})',
               edgecolors='k', lw=1.5)

    # Red star marks the global minimum — the deepest valley, the true optimum
    ax.scatter([global_min_x], [global_min_y], color='red', s=220, zorder=5,
               marker='*', label=f'Global min (x = {global_min_x:.2f})',
               edgecolors='k', lw=1.0)

    ax.set_xlabel(r'Parameter value ($\theta$)')
    ax.set_ylabel('Loss')
    ax.set_title('A loss landscape with local and global minima')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Right panel: gradient descent getting trapped ─────────────────────────
    ax = axes[1]
    ax.plot(x_axis, loss, color='steelblue', linewidth=2.5)

    ax.scatter([local_min_x], [local_min_y], color='orange', s=160, zorder=5,
               label='Gets trapped (local min)', edgecolors='k', lw=1.5)
    ax.scatter([global_min_x], [global_min_y], color='red', s=220, zorder=5,
               marker='*', label='True best solution (global min)',
               edgecolors='k', lw=1.0)

    # Simulate 22 gradient descent steps starting from x = -2.5.
    # At each step we estimate the gradient numerically using a tiny increment h,
    # then move a small distance downhill (0.18 is the learning rate).
    px     = -2.5
    path_x = [px]
    for _ in range(22):
        h    = 0.01
        # Numerical gradient: (loss at x+h  minus  loss at x-h) / (2h)
        # np.interp evaluates the loss at any x without needing the formula.
        grad = (np.interp(px + h, x_axis, loss) - np.interp(px - h, x_axis, loss)) / (2 * h)
        px  -= 0.18 * grad   # move downhill by learning_rate × gradient
        path_x.append(px)

    path_y = [float(np.interp(p, x_axis, loss)) for p in path_x]

    # Plot the path taken by gradient descent as a dashed orange line
    ax.plot(path_x, path_y, 'o--', color='orange', markersize=5,
            linewidth=1.5, alpha=0.8, label='Search path (gets trapped)', zorder=4)

    # Annotate the starting point so users can follow the path direction
    ax.annotate('Start here', xy=(path_x[0], path_y[0]), xytext=(-4.5, 0.9),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

    ax.set_xlabel(r'Parameter value ($\theta$)')
    ax.set_ylabel('Loss')
    ax.set_title('How a search can get trapped in a local minimum')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.suptitle(
        'Figure 3: Loss landscape — local vs global minima (left) '
        'and gradient descent getting trapped (right)',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()