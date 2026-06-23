"""
Figure 13 -- Compactness vs Separation: the Two Goals of Clustering Evaluation
===============================================================================
Illustrates the two properties that define a good clustering result:
compactness (points within a cluster sit close to their centroid) and
separation (clusters are well spread apart from one another).

Two datasets are generated using the same cluster centres and the same
random seed, but with different standard deviations:

  Left panel  (good clustering):  cluster_std = 0.5. Clusters are tight
    and well separated. The dashed circles, which mark the RMS radius of
    each cluster, are small and do not overlap.

  Right panel (poor clustering):  cluster_std = 2.5. Clusters spread widely
    and overlap heavily. The dashed circles are large and intersect, making
    it difficult for any algorithm to recover the true groupings.

The dashed circle around each cluster shows its RMS radius: the square root
of the mean squared distance from each point to its centroid. This is a
simple geometric measure of compactness. A good clustering result has small
RMS radii (compact) with centroids far apart (separated). A poor result has
large, overlapping radii regardless of where the centroids sit.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_13 import show
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
from sklearn.datasets import make_blobs

# -- Dataset parameters -------------------------------------------------------
N_SAMPLES    = 200
N_CENTERS    = 3
RANDOM_STATE = 1   # Fixed seed so both panels use identical cluster centres

# Standard deviations chosen to produce a clear contrast between the panels.
STD_GOOD = 0.5   # Tight, well-separated clusters
STD_POOR = 2.5   # Wide, heavily overlapping clusters

# -- Colours for the three clusters -------------------------------------------
CLUSTER_COLOURS = ['steelblue', 'tomato', 'seagreen']


def _draw_panel(ax, X, y, title):
    """
    Draw one clustering panel: scatter points, centroid markers, and a dashed
    RMS-radius circle for each cluster.

    Parameters
    ----------
    ax    : matplotlib Axes to draw on.
    X     : (n_samples, 2) array of feature values.
    y     : (n_samples,) array of integer cluster labels.
    title : string to set as the panel title.
    """
    for k, col in enumerate(CLUSTER_COLOURS):
        mask = y == k
        pts  = X[mask]

        ax.scatter(
            pts[:, 0], pts[:, 1],
            color=col, s=35, alpha=0.7,
            edgecolors='k', lw=0.2,
        )

        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

        # RMS radius: square root of the mean squared distance from each
        # point to the centroid. A small value means a compact cluster.
        rms_r = np.sqrt(((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2).mean())

        circle = plt.Circle(
            (cx, cy), rms_r,
            fill=False, color=col, lw=2, linestyle='--',
        )
        ax.add_patch(circle)

        # Mark the centroid with a cross so it is clearly visible.
        ax.scatter(cx, cy, s=120, color=col, marker='+', lw=2.5, zorder=5)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)


def show():
    """Render Figure 13: compactness vs separation side by side."""
    plt.close('Notebook12 Figure 13')

    X_good, y_good = make_blobs(
        n_samples=N_SAMPLES, centers=N_CENTERS,
        cluster_std=STD_GOOD, random_state=RANDOM_STATE,
    )
    X_poor, y_poor = make_blobs(
        n_samples=N_SAMPLES, centers=N_CENTERS,
        cluster_std=STD_POOR, random_state=RANDOM_STATE,
    )

    fig, axes = plt.subplots(
        1, 2,
        num='Notebook12 Figure 13',
        figsize=(10, 5),
    )
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_visible  = True
    fig.canvas.toolbar_position = 'right'

    _draw_panel(
        axes[0], X_good, y_good,
        'Figure 13a: Good clustering\nCompact and well separated',
    )
    _draw_panel(
        axes[1], X_poor, y_poor,
        'Figure 13b: Poor clustering\nOverlapping — hard to separate',
    )

    fig.suptitle(
        'Figure 13: Compactness vs Separation — the two goals of clustering evaluation',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()