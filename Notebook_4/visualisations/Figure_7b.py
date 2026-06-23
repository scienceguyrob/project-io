"""
Figure 7b — Data Sparsity in High Dimensions
=============================================

Two side-by-side panels illustrating how data becomes increasingly sparse
as the number of features (dimensions) grows:

Left panel:  the number of training samples needed to adequately cover the
             feature space grows exponentially with dimensionality. Even with
             5 bins per axis and only 10 required samples per cell, the number
             of samples needed quickly exceeds billions.

Right panel: with a fixed training set of 200 points, the distance to the
             nearest neighbour grows as dimensionality increases — meaning
             that in high dimensions, even the closest training example is
             far away, making distance-based reasoning unreliable.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_7b import show
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
    """Render the static Figure 7b data sparsity illustration."""

    plt.close('Notebook4 Figure 7b')

    # ── Required samples calculation ──────────────────────────────────────────
    # If we discretise each feature axis into k = 5 bins, then in d dimensions
    # we have k^d cells. To have at least 10 data points per cell we need
    # 10 × k^d training samples in total.
    k    = 5
    dims = list(range(1, 15))
    needed = [10 * (k ** d) for d in dims]

    # ── Nearest-neighbour distance experiment ─────────────────────────────────
    # For each dimensionality d: generate 200 random training points and 1 test
    # point, all uniformly distributed in [0,1]^d. Measure how far the nearest
    # training point is from the test point. As d grows, this distance increases
    # because the space becomes increasingly empty relative to the data density.
    rng      = np.random.default_rng(42)
    n_train  = 200
    nn_dists = []

    for d in dims:
        train = rng.uniform(0, 1, (n_train, d))   # 200 training points in d dimensions
        test  = rng.uniform(0, 1, (1, d))          # 1 test point

        # Euclidean distance from the test point to every training point:
        # subtract test from each training point, square each coordinate,
        # sum across coordinates (axis=1), then take the square root.
        dists = np.sqrt(((train - test) ** 2).sum(axis=1))
        nn_dists.append(dists.min())               # keep only the smallest distance

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook4 Figure 7b', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: required samples (log scale) ──────────────────────────────
    ax = axes[0]

    # semilogy uses a logarithmic y-axis — necessary because the values span
    # from tens to billions, which would make a linear scale unreadable.
    ax.semilogy(dims, needed, 'o-', color='tomato', linewidth=2, markersize=7)

    # Horizontal reference lines to give the exponential growth a human scale
    ax.axhline(1e3, color='gray', linestyle=':',  alpha=0.7, label='1,000 samples')
    ax.axhline(1e6, color='gray', linestyle='--', alpha=0.7, label='1,000,000 samples')
    ax.axhline(1e9, color='gray', linestyle='-.', alpha=0.7, label='1 billion samples')

    ax.set_xlabel('Number of features (dimensions)')
    ax.set_ylabel('Training samples needed (log scale)')
    ax.set_title('Samples needed to cover the feature space\n'
                 '(exponential explosion — note the log scale)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right panel: nearest-neighbour distance ───────────────────────────────
    ax = axes[1]

    ax.plot(dims, nn_dists, 's-', color='steelblue', linewidth=2, markersize=7,
            label=f'Distance to nearest neighbour  (n = {n_train})')

    # Shade regions to give intuitive meaning to the distance values
    ax.axhspan(0,    0.25, alpha=0.06, color='green',  label='Close — neighbour is useful')
    ax.axhspan(0.25, 0.5,  alpha=0.06, color='orange', label='Borderline')
    ax.axhspan(0.5,  1.5,  alpha=0.06, color='tomato', label='Far — neighbour is unreliable')

    ax.set_xlabel('Number of features (dimensions)')
    ax.set_ylabel('Distance to nearest neighbour')
    ax.set_title('Nearest neighbour gets further away\n'
                 'as dimensions increase')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Figure 7b: Data sparsity in high dimensions',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # Print a plain-text summary of the nearest-neighbour distances
    print('With 200 training points, distance to nearest neighbour:')
    for d, dist in zip(dims[:8], nn_dists[:8]):
        if dist < 0.25:
            tag = '(close → useful)'
        elif dist > 0.5:
            tag = '(far → unreliable)'
        else:
            tag = ''
        print(f'  d={d:2d}:  {dist:.3f}  {tag}')