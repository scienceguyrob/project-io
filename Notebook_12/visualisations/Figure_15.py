"""
Figure 15 -- Interactive Clustering Metrics: Silhouette, DBI, and Dunn Index
=============================================================================
Provides an interactive demonstration of how three internal clustering
evaluation metrics respond to changes in cluster geometry.

Three clusters of points are arranged on a 2D canvas. Four sliders let
the reader adjust:

  Spread (separation) : moves the three cluster centroids closer together
    or further apart. Low spread = overlapping clusters; high spread =
    well-separated clusters.

  Compactness : controls the standard deviation of points within each
    cluster. Low compactness = loose, scattered clusters; high compactness
    = tight, dense clusters.

As the sliders change, the figure recomputes and displays in real time:

  - Silhouette Score  (higher is better, range -1 to +1)
  - Davies-Bouldin Index  (lower is better, minimum 0)
  - Dunn Index  (higher is better, minimum 0)

For each metric, the annotation panel shows the current value alongside
a brief reminder of what direction is better, so the reader can build
intuition about how geometry drives each number.

The Dunn Index is computed from scratch using scipy cdist, since
scikit-learn does not provide a built-in implementation.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_15 import show
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
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

# -- Fixed cluster parameters -------------------------------------------------
# Three clusters arranged at the vertices of an equilateral triangle.
# Their positions are scaled by the spread slider; their point scatter is
# controlled by the compactness slider.
N_PER_CLUSTER = 30
N_CLUSTERS    = 3
RANDOM_SEED   = 42

# Base centroid positions (unit triangle). Scaled by spread at draw time.
BASE_CENTROIDS = np.array([
    [ 0.0,  1.0],
    [-0.866, -0.5],
    [ 0.866, -0.5],
])

# -- Slider defaults and bounds -----------------------------------------------
DEFAULT_SPREAD      = 3.0
DEFAULT_COMPACTNESS = 0.5   # Inverse of std: high compactness = low std

SPREAD_MIN = 0.5;  SPREAD_MAX = 6.0;  SPREAD_STEP = 0.1
COMP_MIN   = 0.1;  COMP_MAX   = 2.0;  COMP_STEP   = 0.05

# -- Colours ------------------------------------------------------------------
CLUSTER_COLOURS = ['steelblue', 'tomato', 'seagreen']


def _make_clusters(spread, compactness):
    """
    Generate three clusters of N_PER_CLUSTER points each.

    Parameters
    ----------
    spread      : scalar multiplier applied to BASE_CENTROIDS. Large values
                  move centroids apart; small values push them together.
    compactness : inverse standard deviation. High compactness = tight clusters
                  (std = 1/compactness); low compactness = loose clusters.

    Returns
    -------
    X      : (N_PER_CLUSTER * N_CLUSTERS, 2) array of point coordinates.
    labels : (N_PER_CLUSTER * N_CLUSTERS,) integer cluster label array.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    std = 1.0 / compactness   # Convert compactness to standard deviation

    points_list = []
    labels_list = []

    for k, centroid in enumerate(BASE_CENTROIDS * spread):
        pts = rng.normal(loc=centroid, scale=std, size=(N_PER_CLUSTER, 2))
        points_list.append(pts)
        labels_list.append(np.full(N_PER_CLUSTER, k, dtype=int))

    X      = np.vstack(points_list)
    labels = np.concatenate(labels_list)
    return X, labels


def _dunn_index(X, labels):
    """
    Compute the Dunn Index from point coordinates and cluster labels.

    Dunn = min inter-cluster distance / max intra-cluster diameter.
    Higher is better. Returns 0.0 if any cluster has fewer than 2 points.
    """
    unique = np.unique(labels)
    clusters = [X[labels == k] for k in unique]

    # Minimum distance between any point in cluster i and any in cluster j.
    min_inter = float('inf')
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = cdist(clusters[i], clusters[j]).min()
            min_inter = min(min_inter, dist)

    # Maximum diameter: largest intra-cluster point-to-point distance.
    max_intra = 0.0
    for cluster in clusters:
        if len(cluster) > 1:
            max_intra = max(max_intra, cdist(cluster, cluster).max())

    return min_inter / max_intra if max_intra > 0 else 0.0


def _annotation_text(sil, dbi, dunn):
    """Build the metrics panel string."""
    sep = "\u2500" * 33
    return (
        "Clustering metrics\n"
        "{sep}\n\n"
        "Silhouette Score\n"
        "  {sil:+.3f}   (higher is better, max +1)\n\n"
        "{sep}\n\n"
        "Davies-Bouldin Index\n"
        "  {dbi:.3f}    (lower is better, min 0)\n\n"
        "{sep}\n\n"
        "Dunn Index\n"
        "  {dunn:.3f}    (higher is better, min 0)\n\n"
        "{sep}\n\n"
        "Try:\n"
        "  Increase spread \u2192 all metrics improve\n"
        "  Decrease spread \u2192 clusters overlap,\n"
        "    metrics degrade\n"
        "  Decrease compactness \u2192 loose clusters,\n"
        "    metrics degrade even at high spread"
    ).format(sep=sep, sil=sil, dbi=dbi, dunn=dunn)


def show():
    """Render Figure 15: interactive clustering metrics explorer."""
    plt.close("Notebook12 Figure 15")

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num="Notebook12 Figure 15",
        figsize=(10, 5.5),
        gridspec_kw={"width_ratios": [1.3, 1]},
    )
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_visible  = True
    fig.canvas.toolbar_position = "right"

    ax_ann.set_axis_off()

    ann_text = ax_ann.text(
        0.04, 0.97, "",
        transform=ax_ann.transAxes,
        fontsize=9, va="top", ha="left",
        family="monospace",
        linespacing=1.7,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f7f7",
                  edgecolor="#cccccc", alpha=1.0),
    )

    fig.suptitle(
        "Figure 15: How cluster geometry drives silhouette, DBI, and Dunn Index",
        fontsize=11, y=0.99,
    )
    plt.subplots_adjust(wspace=0.05, top=0.91)

    def _draw(spread, compactness):
        ax_plot.clear()

        X, labels = _make_clusters(spread, compactness)

        # Compute all three metrics. Silhouette and DBI require at least
        # two distinct labels, which is always satisfied here (3 clusters).
        sil  = silhouette_score(X, labels)
        dbi  = davies_bouldin_score(X, labels)
        dunn = _dunn_index(X, labels)

        # Draw each cluster with its own colour and centroid marker.
        for k, col in enumerate(CLUSTER_COLOURS):
            mask = labels == k
            ax_plot.scatter(
                X[mask, 0], X[mask, 1],
                color=col, s=45, alpha=0.75,
                edgecolors="k", lw=0.3, zorder=3,
            )
            cx, cy = X[mask, 0].mean(), X[mask, 1].mean()
            ax_plot.scatter(
                cx, cy, s=120, color=col,
                marker="+", lw=2.5, zorder=5,
            )

        ax_plot.set_xlabel("Feature 1", fontsize=10)
        ax_plot.set_ylabel("Feature 2", fontsize=10)
        ax_plot.set_title(
            f"Spread = {spread:.1f}   Compactness = {compactness:.2f}",
            fontsize=9,
        )
        ax_plot.grid(True, alpha=0.2)
        ax_plot.set_aspect("equal")

        ann_text.set_text(_annotation_text(sil, dbi, dunn))
        fig.canvas.draw_idle()

    # -- Sliders --------------------------------------------------------------
    slider_style  = {"description_width": "110px"}
    slider_layout = widgets.Layout(width="360px")

    sl_spread = widgets.FloatSlider(
        value=DEFAULT_SPREAD,
        min=SPREAD_MIN, max=SPREAD_MAX, step=SPREAD_STEP,
        description="Spread",
        style=slider_style, layout=slider_layout,
        readout_format=".1f",
    )
    sl_compact = widgets.FloatSlider(
        value=DEFAULT_COMPACTNESS,
        min=COMP_MIN, max=COMP_MAX, step=COMP_STEP,
        description="Compactness",
        style=slider_style, layout=slider_layout,
        readout_format=".2f",
    )

    out      = interactive_output(_draw, {"spread": sl_spread, "compactness": sl_compact})
    controls = widgets.VBox([sl_spread, sl_compact])
    display(controls, out)