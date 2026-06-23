"""
Figure 17a — Interactive DBSCAN Parameter Explorer
====================================================
A focused single-panel visualisation showing how ε and minPts interact
on a small, hand-crafted dataset designed to make the parameter effects
immediately obvious.

The dataset contains:
  - Two dense clusters (clearly separable)
  - A loose cluster (lower density — sensitive to minPts)
  - A handful of isolated outliers (should always be noise)

Controls
--------
  ε slider    — neighbourhood radius. Small values fragment clusters;
                large values merge them or absorb noise into clusters.
  minPts slider — density threshold. Small values make almost everything
                  a core point; large values shrink clusters to their
                  densest cores.

As either slider moves:
  - Core points are shown as filled circles
  - Border points as diamonds
  - Noise points as grey crosses
  - ε circles drawn around every core point (toggleable)
  - Title updates with cluster count and point type breakdown

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_17a import show
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
import matplotlib.patches as mpatches
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display
from sklearn.cluster import DBSCAN


# ── Dataset ───────────────────────────────────────────────────────────────────
# Hand-crafted to make parameter effects unambiguous:
#   Cluster A : tight, 12 points — always found unless eps is tiny
#   Cluster B : tight, 12 points — always found unless eps is tiny
#   Cluster C : loose, 8 points  — lost when minPts is high or eps is small
#   Outliers  : 5 isolated points — should remain noise in most settings
RNG = np.random.default_rng(7)

cluster_a = RNG.normal(loc=[1.5, 4.0], scale=0.25, size=(12, 2))
cluster_b = RNG.normal(loc=[5.0, 4.0], scale=0.25, size=(12, 2))
cluster_c = RNG.normal(loc=[3.2, 1.5], scale=0.65, size=(8, 2))   # loose
outliers  = np.array([
    [0.2, 1.0], [6.5, 1.2], [3.0, 6.5], [6.0, 6.2], [0.5, 6.0]
])

X_db = np.vstack([cluster_a, cluster_b, cluster_c, outliers])

# ── Visual parameters ─────────────────────────────────────────────────────────
PALETTE   = ['steelblue', 'tomato', 'seagreen', 'goldenrod', 'mediumpurple']
COL_NOISE = '#bbbbbb'
XLIM      = (-0.5, 7.5)
YLIM      = (0.0, 7.5)


def _run_dbscan(eps, min_pts):
    """
    Run DBSCAN and classify every point as core, border, or noise.
    Returns labels, boolean masks for each type, and neighbour counts.
    """
    db     = DBSCAN(eps=eps, min_samples=min_pts).fit(X_db)
    labels = db.labels_

    # Count neighbours within eps for each point, excluding itself
    dists            = np.linalg.norm(X_db[:, None, :] - X_db[None, :, :], axis=2)
    neighbour_counts = (dists < eps).sum(axis=1) - 1

    is_core   = neighbour_counts >= min_pts
    is_noise  = labels == -1
    is_border = (~is_core) & (~is_noise)

    return labels, is_core, is_border, is_noise


def show():
    """Render Figure 17a: interactive DBSCAN parameter explorer."""
    plt.close('Notebook8 Figure 17a')

    fig, ax = plt.subplots(
        num='Notebook8 Figure 17a',
        figsize=(8, 7),
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    def _draw(eps, min_pts, show_circles):
        ax.clear()

        labels, is_core, is_border, is_noise = _run_dbscan(eps, min_pts)
        n_clusters = len(set(labels) - {-1})

        # ── ε circles ─────────────────────────────────────────────────────────
        # Drawn first so they sit behind all scatter points.
        if show_circles:
            for i, pt in enumerate(X_db):
                if is_core[i]:
                    cluster_col = PALETTE[labels[i] % len(PALETTE)]
                    circle = mpatches.Circle(
                        pt, radius=eps,
                        facecolor=cluster_col,
                        edgecolor=cluster_col,
                        alpha=0.08, zorder=1,
                        linewidth=0.6, linestyle='-',
                    )
                    ax.add_patch(circle)

        # ── Scatter points ─────────────────────────────────────────────────────
        for i, pt in enumerate(X_db):
            if is_noise[i]:
                ax.scatter(
                    pt[0], pt[1],
                    marker='x', s=80,
                    color=COL_NOISE, linewidths=2.0,
                    zorder=4,
                )
            else:
                col = PALETTE[labels[i] % len(PALETTE)]
                if is_core[i]:
                    ax.scatter(
                        pt[0], pt[1],
                        marker='o', s=90,
                        color=col, edgecolors='k',
                        lw=0.5, alpha=0.95, zorder=5,
                    )
                else:
                    # Border point — diamond, slightly muted
                    ax.scatter(
                        pt[0], pt[1],
                        marker='D', s=70,
                        color=col, edgecolors='k',
                        lw=0.5, alpha=0.6, zorder=5,
                    )

        # ── Legend ────────────────────────────────────────────────────────────
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='steelblue', markeredgecolor='k',
                       markersize=10, label=f'Core  ({is_core.sum()})'),
            plt.Line2D([0], [0], marker='D', color='w',
                       markerfacecolor='steelblue', markeredgecolor='k',
                       markersize=9, label=f'Border  ({is_border.sum()})'),
            plt.Line2D([0], [0], marker='x', color=COL_NOISE,
                       markersize=10, markeredgewidth=2,
                       label=f'Noise  ({is_noise.sum()})'),
        ]
        if show_circles:
            handles.append(
                mpatches.Patch(
                    facecolor='steelblue', alpha=0.2,
                    label=f'ε neighbourhood  (r = {eps:.2f})',
                )
            )
        ax.legend(handles=handles, fontsize=9, loc='upper right',
                  framealpha=1.0, edgecolor='#cccccc')

        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.set_title(
            f'ε = {eps:.2f}   minPts = {min_pts}   →   '
            f'{n_clusters} cluster{"s" if n_clusters != 1 else ""}   '
            f'|   core: {is_core.sum()}   '
            f'border: {is_border.sum()}   '
            f'noise: {is_noise.sum()}',
            fontsize=10,
        )
        ax.grid(True, alpha=0.2)

        fig.suptitle(
            'Figure 17a: DBSCAN parameter explorer — adjust ε and minPts',
            fontsize=11,
        )
        fig.canvas.draw_idle()

    # ── Widgets ───────────────────────────────────────────────────────────────
    slider_eps = widgets.FloatSlider(
        value=0.60, min=0.05, max=2.50, step=0.05,
        description='ε (radius)',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='420px'),
        readout_format='.2f',
    )
    slider_minpts = widgets.IntSlider(
        value=3, min=1, max=12, step=1,
        description='minPts',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='420px'),
    )
    toggle_circles = widgets.Checkbox(
        value=True,
        description='Show ε circles',
        style={'description_width': '110px'},
    )

    out = interactive_output(
        _draw,
        {
            'eps':          slider_eps,
            'min_pts':      slider_minpts,
            'show_circles': toggle_circles,
        },
    )

    display(
        widgets.VBox([slider_eps, slider_minpts, toggle_circles]),
        out,
    )