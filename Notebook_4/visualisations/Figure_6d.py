"""
Figure 6d — Convex vs Non-Convex Optimisation
==============================================

Two side-by-side panels showing gradient descent paths from multiple
starting points on:

Left panel:  a CONVEX loss function — all paths converge to the same
             global minimum regardless of starting position.

Right panel: a NON-CONVEX loss function — paths from different starting
             points converge to different local minima, showing that the
             final solution depends on where the algorithm begins.

This directly illustrates the key practical difference between convex and
non-convex optimisation in machine learning.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_6d import show
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
    """Render the static Figure 6d convex vs non-convex illustration."""

    plt.close('Notebook4 Figure 6d')

    # ── Loss functions ────────────────────────────────────────────────────────

    def convex_loss(x):
        # A simple bowl — one global minimum, guaranteed convex
        return 0.4 * x**2 + 0.3 * x - 0.5

    def convex_grad(x):
        return 0.8 * x + 0.3

    def nonconvex_loss(x):
        # Multiple valleys — non-convex, several local minima
        return (0.08 * x**4
                - 0.6 * x**2
                + 0.3 * np.sin(3 * x)
                + 0.1 * x)

    def nonconvex_grad(x):
        return (0.32 * x**3
                - 1.2 * x
                + 0.9 * np.cos(3 * x)
                + 0.1)

    # ── Gradient descent helper ───────────────────────────────────────────────
    def run_gd(start, loss_fn, grad_fn, lr=0.05, n_steps=80):
        """Run gradient descent and return the full path (x values and loss values)."""
        x      = start
        path_x = [x]
        for _ in range(n_steps):
            x = x - lr * grad_fn(x)
            # Clip to stay within the plot range
            x = np.clip(x, -4.5, 4.5)
            path_x.append(x)
        path_y = [loss_fn(xi) for xi in path_x]
        return np.array(path_x), np.array(path_y)

    # ── Starting points — same set used for both panels ───────────────────────
    starts = [-4.0, -2.5, -1.0, 0.5, 2.0, 3.5]

    # Colours for each starting point so users can trace individual paths
    colours = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple', 'brown']

    x_curve = np.linspace(-4.5, 4.5, 500)

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook4 Figure 6d', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    for ax, loss_fn, grad_fn, title, subtitle in [
        (axes[0], convex_loss,    convex_grad,
         'Convex loss function',
         'All paths reach the same global minimum'),
        (axes[1], nonconvex_loss, nonconvex_grad,
         'Non-convex loss function',
         'Different starting points → different local minima'),
    ]:
        # Draw the loss curve
        y_curve = loss_fn(x_curve)
        ax.plot(x_curve, y_curve, color='black', linewidth=2.5,
                label='Loss function', zorder=1)

        # Run gradient descent from each starting point and draw the path
        final_positions = []
        for start, colour in zip(starts, colours):
            px, py = run_gd(start, loss_fn, grad_fn)

            # Draw the full path as a faint line
            ax.plot(px, py, '-', color=colour, linewidth=1.2,
                    alpha=0.4, zorder=2)

            # Mark the starting point with a circle
            ax.scatter(px[0], py[0], color=colour, s=80, zorder=5,
                       edgecolors='k', lw=0.8,
                       marker='o')

            # Mark the final position with a star
            ax.scatter(px[-1], py[-1], color=colour, s=180, zorder=6,
                       edgecolors='k', lw=0.8, marker='*')

            final_positions.append(px[-1])

        # Annotate the final positions with vertical dashed lines
        seen = set()
        for fp, colour in zip(final_positions, colours):
            fp_rounded = round(fp, 1)
            if fp_rounded not in seen:
                ax.axvline(fp, color=colour, linewidth=1.0,
                           linestyle=':', alpha=0.6)
                seen.add(fp_rounded)

        ax.set_xlabel('Parameter value θ', fontsize=10)
        ax.set_ylabel('Loss L(θ)', fontsize=10)
        ax.set_title(f'{title}\n{subtitle}', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-4.7, 4.7)

    # Shared legend explanation
    axes[0].scatter([], [], color='gray', s=80,  marker='o',
                    label='Starting point', edgecolors='k', lw=0.8)
    axes[0].scatter([], [], color='gray', s=180, marker='*',
                    label='Final position', edgecolors='k', lw=0.8)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[1].scatter([], [], color='gray', s=80,  marker='o',
                    label='Starting point', edgecolors='k', lw=0.8)
    axes[1].scatter([], [], color='gray', s=180, marker='*',
                    label='Final position (local min)', edgecolors='k', lw=0.8)
    axes[1].legend(fontsize=8, loc='upper right')

    plt.suptitle(
        'Figure 6d: Convex vs non-convex optimisation — '
        'same algorithm, very different outcomes',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()