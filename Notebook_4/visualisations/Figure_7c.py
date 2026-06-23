"""
Figure 7c — Irrelevant Features Dilute the Signal
==================================================

Three side-by-side scatter plots showing the same two informative features
(Feature 1 and Feature 2) as an increasing number of noise dimensions are
added. The scatter plots look identical — the informative features have not
changed — but the within/between distance ratio printed in each title
reveals that a distance-based model sees the classes as increasingly
indistinguishable.

Dimensions used: 2 (no noise), 10 (8 noise dims), 50 (48 noise dims).
The jump to 50 makes the collapse toward ratio = 1.0 much more dramatic
than the original 2/5/10/20 progression.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_7c import show
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
    """Render the static Figure 7c irrelevant features illustration."""

    plt.close('Notebook4 Figure 7c')

    rng = np.random.default_rng(3)
    n   = 80   # samples per class

    # Only x1 and x2 carry genuine information — they separate the two classes.
    # All additional dimensions added later are pure random noise.
    x1_A = rng.normal(2, 0.8, n);  x2_A = rng.normal(2, 0.8, n)
    x1_B = rng.normal(4, 0.8, n);  x2_B = rng.normal(4, 0.8, n)

    # Three dimensionalities chosen to make the contrast stark:
    # 2 dims  = informative features only
    # 10 dims = 8 noise dimensions added
    # 50 dims = 48 noise dimensions added — ratio should be close to 1.0
    total_dims_list = [2, 10, 50]

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, num='Notebook4 Figure 7c', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    for ax, total_dims in zip(axes, total_dims_list):
        extra = total_dims - 2   # number of pure noise dimensions to append

        # Build the full feature matrix for each class.
        # np.column_stack joins arrays as columns — first two are informative,
        # the rest are random noise drawn from N(0, 3).
        A = np.column_stack(
            [x1_A, x2_A] + [rng.normal(0, 3, n) for _ in range(extra)]
        )
        B = np.column_stack(
            [x1_B, x2_B] + [rng.normal(0, 3, n) for _ in range(extra)]
        )

        # Sample 500 random pairwise distances in the FULL d-dimensional space.
        # This is what a distance-based model actually uses — not just features 1 and 2.
        within  = []
        between = []
        for _ in range(500):
            i, j = rng.choice(n, 2, replace=False)
            # np.linalg.norm computes the Euclidean distance between two points
            within.append(np.linalg.norm(A[i] - A[j]))   # two Class-A points
            within.append(np.linalg.norm(B[i] - B[j]))   # two Class-B points
            between.append(
                np.linalg.norm(
                    A[rng.integers(n)] - B[rng.integers(n)]   # one A, one B
                )
            )

        ratio = np.mean(within) / np.mean(between)

        # Colour the title red when the ratio is dangerously close to 1.0
        title_colour = 'red' if ratio > 0.85 else ('orange' if ratio > 0.6 else 'green')

        # Plot only the two informative features so the scatter looks the same
        # in all three panels — reinforcing the point that the visual appearance
        # has not changed even though the distance ratio has collapsed.
        ax.scatter(A[:, 0], A[:, 1], color='steelblue', s=30, alpha=0.6,
                   edgecolors='k', lw=0.3, label='Class A')
        ax.scatter(B[:, 0], B[:, 1], color='tomato', s=30, alpha=0.6,
                   edgecolors='k', lw=0.3, label='Class B')

        noise_label = f'({extra} noise dim{"s" if extra != 1 else ""} added)' \
                      if extra > 0 else '(no noise)'
        ax.set_title(
            f'{total_dims} dimensions  {noise_label}\n'
            f'Within/Between ratio = {ratio:.2f}',
            fontsize=9, color=title_colour, fontweight='bold',
        )
        ax.set_xlabel('Feature 1  (informative)')
        ax.set_ylabel('Feature 2  (informative)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        # Annotate what the ratio means in plain English
        if ratio < 0.6:
            msg = '✓ classes well separated'
        elif ratio < 0.85:
            msg = '⚠ separation weakening'
        else:
            msg = '✗ classes nearly identical\n  to a distance-based model'
        ax.text(0.05, 0.05, msg, transform=ax.transAxes,
                fontsize=7.5, color=title_colour,
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor=title_colour, alpha=0.85))

    plt.suptitle(
        'Figure 7c: The scatter plots look identical — but the distance ratio tells a different story\n'
        'Adding noise dimensions makes classes indistinguishable to distance-based models',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()