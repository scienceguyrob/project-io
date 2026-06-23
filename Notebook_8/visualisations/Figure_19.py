"""
Figure 19 — DBSCAN vs K-Means: Non-Convex Datasets
====================================================
Compares DBSCAN and K-Means on two datasets where K-Means fails because
it assumes spherical, convex clusters:

  Left column  — K-Means (k=2) — always fails on these shapes
  Right column — DBSCAN (eps=0.30, minPts=5) — recovers true structure

  Top row    — two interleaved crescents (make_moons)
  Bottom row — a ring of points surrounding a central cluster (make_circles)

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_19 import show
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
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons, make_circles


# ── Datasets ──────────────────────────────────────────────────────────────────
X_moons,   _ = make_moons(n_samples=300, noise=0.08, random_state=0)
X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=1)

# ── Clustering parameters ─────────────────────────────────────────────────────
KMEANS_K   = 2
DBSCAN_EPS = 0.30
DBSCAN_MIN = 5

# ── Colour palettes ───────────────────────────────────────────────────────────
KM_PALETTE = ['steelblue', 'tomato']
CMAP_DB    = cm.get_cmap('tab10')
COL_NOISE  = '#444444'


def _plot_kmeans(ax, X, title_base):
    """Fit K-Means and plot the result."""
    km     = KMeans(n_clusters=KMEANS_K, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    for j, col in enumerate(KM_PALETTE):
        mask = labels == j
        ax.scatter(
            X[mask, 0], X[mask, 1],
            color=col, s=25, alpha=0.7,
            edgecolors='k', lw=0.2,
            label=f'Cluster {j}',
            zorder=3,
        )

    ax.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        marker='*', s=280, color='gold',
        edgecolors='black', lw=0.8, zorder=5,
        label='Centroids',
    )

    ax.set_title(
        f'K-Means (k={KMEANS_K})\n'
        f'Fails — assumes convex, spherical clusters',
        fontsize=10,
    )
    ax.legend(fontsize=8, loc='upper right',
              framealpha=1.0, edgecolor='#cccccc')


def _plot_dbscan(ax, X):
    """Fit DBSCAN and plot the result."""
    db     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise    = int((labels == -1).sum())

    for lbl in sorted(set(labels)):
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(
                X[mask, 0], X[mask, 1],
                marker='x', s=50, color=COL_NOISE,
                linewidths=1.5, zorder=2,
                label=f'Noise ({n_noise})',
            )
        else:
            ax.scatter(
                X[mask, 0], X[mask, 1],
                color=CMAP_DB(lbl / 10.0),
                marker='o', s=25, alpha=0.85,
                edgecolors='k', lw=0.2,
                label=f'Cluster {lbl}',
                zorder=3,
            )

    ax.set_title(
        f'DBSCAN  (ε={DBSCAN_EPS}, minPts={DBSCAN_MIN})\n'
        f'{n_clusters} cluster{"s" if n_clusters != 1 else ""}   |   '
        f'{n_noise} noise point{"s" if n_noise != 1 else ""}',
        fontsize=10,
    )
    ax.legend(fontsize=8, loc='upper right',
              framealpha=1.0, edgecolor='#cccccc')


def show():
    """Render Figure 19: DBSCAN vs K-Means on non-convex datasets."""
    plt.close('Notebook8 Figure 19')

    fig, axes = plt.subplots(
        2, 2,
        num='Notebook8 Figure 19',
        figsize=(12, 10),
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    datasets = [
        (X_moons,   'Two-moons'),
        (X_circles, 'Concentric circles'),
    ]

    for row, (X, title_base) in enumerate(datasets):
        _plot_kmeans(axes[row, 0], X, title_base)
        _plot_dbscan(axes[row, 1], X)

        for ax in axes[row]:
            ax.set_xlabel('Feature 1', fontsize=9)
            ax.set_ylabel('Feature 2', fontsize=9)
            ax.grid(True, alpha=0.2)

    # Row labels — placed far enough left that they do not overlap the
    # axis tick labels. x=0.02 and rotation=90 keeps them in the left
    # margin without crowding the plots.
    fig.text(
        0.02, 0.73, 'Two-moons',
        va='center', ha='center',
        fontsize=11, fontweight='bold',
        rotation=90, color='#444444',
    )
    fig.text(
        0.02, 0.27, 'Concentric circles',
        va='center', ha='center',
        fontsize=11, fontweight='bold',
        rotation=90, color='#444444',
    )

    plt.suptitle(
        'Figure 19: DBSCAN vs K-Means on non-convex datasets\n'
        'K-Means imposes straight boundaries; DBSCAN follows density',
        fontsize=12, y=0.98,
    )
    # left margin increased to give the row labels room without
    # overlapping the y-axis tick labels
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.show()