"""
Figure 18 — Interactive DBSCAN Explorer
========================================
Visualises the core DBSCAN concepts interactively on a fixed dataset:

  - Core points   : have at least minPts neighbours within radius ε
  - Border points : within ε of a core point but not core themselves
  - Noise points  : neither core nor within ε of any core point

Two sliders control ε and minPts. As they change:
  - Every point is reclassified as core, border, or noise
  - ε circles are drawn around all core points to show the neighbourhood
  - Points are coloured and shaped by their classification
  - Cluster assignments update live
  - An annotation panel shows the current counts of each point type

A checkbox toggles the ε circles on and off — with many core points
the circles can clutter the plot, so hiding them is useful for seeing
the cluster assignments clearly.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_18 import show
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
import matplotlib.cm as cm
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# ── Dataset ───────────────────────────────────────────────────────────────────
# Two interleaved crescents — a canonical case where K-Means fails and
# DBSCAN succeeds. Standardised so that ε values are interpretable in
# units of standard deviations rather than raw feature units.
X_raw, _ = make_moons(n_samples=150, noise=0.08, random_state=42)
X_db     = StandardScaler().fit_transform(X_raw)

# ── Colour palette for clusters ───────────────────────────────────────────────
# Up to 8 distinct cluster colours. Noise is always shown in grey.
CLUSTER_COLOURS = [
    'steelblue', 'tomato', 'seagreen', 'goldenrod',
    'mediumpurple', 'darkorange', 'deeppink', 'brown',
]
COL_NOISE  = '#bbbbbb'
COL_CORE   = None    # inherits cluster colour
COL_BORDER = None    # inherits cluster colour, different marker
COL_CIRCLE = '#cccccc'

# ── Axis limits ───────────────────────────────────────────────────────────────
PAD  = 0.5
XLIM = (X_db[:, 0].min() - PAD, X_db[:, 0].max() + PAD)
YLIM = (X_db[:, 1].min() - PAD, X_db[:, 1].max() + PAD)


def _classify_points(X, eps, min_pts):
    """
    Classify every point as core, border, or noise and assign cluster labels.

    Returns
    -------
    labels      : array (n,)  — cluster index (-1 = noise)
    is_core     : array (n,)  — boolean, True if core point
    is_border   : array (n,)  — boolean, True if border point
    is_noise    : array (n,)  — boolean, True if noise point
    neighbour_counts : array (n,) — number of points within eps of each point
    """
    db     = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
    labels = db.labels_

    # Count neighbours within eps for each point (excluding the point itself)
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    neighbour_counts = (dists < eps).sum(axis=1) - 1  # exclude self

    is_core   = neighbour_counts >= min_pts
    is_noise  = labels == -1
    is_border = (~is_core) & (~is_noise)

    return labels, is_core, is_border, is_noise, neighbour_counts


def _annotation_text(labels, is_core, is_border, is_noise, eps, min_pts):
    """Build the annotation panel string."""
    n_clusters = len(set(labels) - {-1})
    n_core     = int(is_core.sum())
    n_border   = int(is_border.sum())
    n_noise    = int(is_noise.sum())

    return (
        "DBSCAN classification\n"
        "─────────────────────────────────\n\n"
        f"$\\varepsilon$ (radius)  = {eps:.2f}\n"
        f"minPts            = {min_pts}\n\n"
        "─────────────────────────────────\n\n"
        f"Clusters found:   {n_clusters}\n\n"
        f"Core points:      {n_core}\n"
        f"  (≥ minPts neighbours\n"
        f"   within radius ε)\n\n"
        f"Border points:    {n_border}\n"
        f"  (within ε of a core\n"
        f"   point, but not core)\n\n"
        f"Noise points:     {n_noise}\n"
        f"  (not reachable from\n"
        f"   any core point)\n\n"
        "─────────────────────────────────\n\n"
        "Shapes on plot:\n"
        "  ●  core point\n"
        "  ◆  border point\n"
        "  ✕  noise point\n"
        "  ○  ε neighbourhood circle\n"
        "     (toggle with checkbox)"
    )


def show():
    """Render Figure 18: interactive DBSCAN explorer."""
    plt.close('Notebook8 Figure 18')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 18',
        figsize=(10, 7),
        gridspec_kw={'width_ratios': [1.5, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    ax_ann.set_axis_off()

    # ── Annotation text ───────────────────────────────────────────────────────
    ann_text = ax_ann.text(
        0.04, 0.97, '',
        transform=ax_ann.transAxes,
        fontsize=9, va='top', ha='left',
        linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 18: DBSCAN — core points, border points, and noise\n'
        'Adjust ε and minPts to see how the classification changes',
        fontsize=11,
    )
    plt.subplots_adjust(wspace=0.08, top=0.88)

    # ── Draw function ─────────────────────────────────────────────────────────
    def _draw(eps, min_pts, show_circles):
        ax_plot.clear()

        labels, is_core, is_border, is_noise, n_counts = \
            _classify_points(X_db, eps, min_pts)

        n_clusters = len(set(labels) - {-1})

        # ── ε circles around core points ──────────────────────────────────────
        # Drawn first so they sit behind the scatter points.
        # Each circle shows the neighbourhood that qualifies a point as core.
        if show_circles:
            for i, pt in enumerate(X_db):
                if is_core[i]:
                    circle = mpatches.Circle(
                        pt, radius=eps,
                        facecolor=COL_CIRCLE,
                        edgecolor='#aaaaaa',
                        fill=True,
                        alpha=0.12, zorder=1,
                        linewidth=0.8, linestyle='-',
                    )
                    ax_plot.add_patch(circle)

        # ── Plot points by classification ─────────────────────────────────────
        # Core points: filled circles, coloured by cluster
        # Border points: diamonds, coloured by cluster
        # Noise points: crosses, grey

        for i, pt in enumerate(X_db):
            if is_noise[i]:
                ax_plot.scatter(
                    pt[0], pt[1],
                    marker='x', s=60, color=COL_NOISE,
                    linewidths=1.5, zorder=3,
                )
            else:
                cluster_col = CLUSTER_COLOURS[labels[i] % len(CLUSTER_COLOURS)]
                if is_core[i]:
                    ax_plot.scatter(
                        pt[0], pt[1],
                        marker='o', s=60,
                        color=cluster_col, edgecolors='k',
                        lw=0.4, alpha=0.9, zorder=4,
                    )
                else:
                    # Border point — diamond marker, slightly transparent
                    ax_plot.scatter(
                        pt[0], pt[1],
                        marker='D', s=50,
                        color=cluster_col, edgecolors='k',
                        lw=0.4, alpha=0.6, zorder=4,
                    )

        # ── Legend ────────────────────────────────────────────────────────────
        legend_handles = [
            mpatches.Patch(color='steelblue', label=f'Cluster (example colour)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                       markeredgecolor='k', markersize=9, label='Core point  (●)'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='steelblue',
                       markeredgecolor='k', markersize=8, label='Border point  (◆)'),
            plt.Line2D([0], [0], marker='x', color=COL_NOISE,
                       markersize=9, markeredgewidth=2, label='Noise point  (✕)'),
        ]
        if show_circles:
            legend_handles.append(
                mpatches.Patch(color=COL_CIRCLE, alpha=0.4,
                               label=f'ε neighbourhood  (r = {eps:.2f})')
            )
        ax_plot.legend(
            handles=legend_handles, fontsize=8,
            loc='lower left', framealpha=1.0, edgecolor='#cccccc',
        )

        ax_plot.set_xlim(*XLIM)
        ax_plot.set_ylim(*YLIM)
        ax_plot.set_xlabel('Feature 1 (standardised)', fontsize=10)
        ax_plot.set_ylabel('Feature 2 (standardised)', fontsize=10)
        ax_plot.set_title(
            f'ε = {eps:.2f}   minPts = {min_pts}   '
            f'→   {n_clusters} cluster{"s" if n_clusters != 1 else ""}   '
            f'| core: {is_core.sum()}  '
            f'border: {is_border.sum()}  '
            f'noise: {is_noise.sum()}',
            fontsize=9,
        )
        ax_plot.grid(True, alpha=0.15)

        # Update annotation
        ann_text.set_text(
            _annotation_text(labels, is_core, is_border, is_noise, eps, min_pts)
        )

        fig.canvas.draw_idle()

    # ── Widgets ───────────────────────────────────────────────────────────────
    slider_eps = widgets.FloatSlider(
        value=0.30, min=0.05, max=1.50, step=0.05,
        description='ε (radius)',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='400px'),
        readout_format='.2f',
    )
    slider_minpts = widgets.IntSlider(
        value=5, min=1, max=20, step=1,
        description='minPts',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='400px'),
    )
    toggle_circles = widgets.Checkbox(
        value=True,
        description='Show ε circles',
        style={'description_width': '110px'},
    )

    def _update(eps, min_pts, show_circles):
        _draw(eps, min_pts, show_circles)

    out = interactive_output(
        _update,
        {
            'eps':          slider_eps,
            'min_pts':      slider_minpts,
            'show_circles': toggle_circles,
        },
    )

    controls = widgets.VBox([
        slider_eps,
        slider_minpts,
        toggle_circles,
    ])
    display(controls, out)