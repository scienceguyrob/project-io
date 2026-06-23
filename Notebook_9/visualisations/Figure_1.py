"""
Figure 1 — DBSCAN's Fixed-ε Limitation on Clusters of Varying Density
=======================================================================
Demonstrates the motivating problem for OPTICS: DBSCAN uses a single
fixed neighbourhood radius ε across the whole dataset, but a dataset
can easily contain clusters of very different densities. No single ε
can correctly handle both.

The dataset contains:
  - One tight, dense cluster
  - One loose, sparse cluster
  - A handful of scattered noise points

A single slider controls ε (minPts is fixed). As ε changes:
  - DBSCAN is refit on the data with the chosen ε
  - Points are coloured by their resulting cluster label (grey = noise)
  - An annotation panel reports the outcome for each cluster:
    correctly found, merged with noise, merged with the other cluster,
    or split into fragments

The annotation panel highlights that there is no ε value at which both
clusters are simultaneously well-recovered — increasing ε to capture the
sparse cluster causes the dense cluster to start merging with noise or
with the sparse cluster, while decreasing ε to keep the dense cluster
tight causes the sparse cluster to fragment into noise.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_1 import show
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
from sklearn.cluster import DBSCAN


# ── Fixed analysis parameter ────────────────────────────────────────────────
# minPts is held constant throughout — only ε is varied via the slider.
# This isolates the effect of ε, which is the point this figure makes.
MIN_PTS = 5

# ── Dataset ──────────────────────────────────────────────────────────────────
# Constructed by hand (rather than make_blobs) so that the two clusters have
# deliberately very different densities:
#   - The "dense" cluster is tightly packed in a small region.
#   - The "sparse" cluster covers a much larger area with the same number of
#     points, so its typical point-to-point spacing is much greater.
# A few scattered points are added as genuine noise/outliers.
rng = np.random.default_rng(42)

# Dense cluster: 60 points drawn from a tight Gaussian blob.
# Small standard deviation (0.25) keeps points close together.
dense_cluster = rng.normal(loc=[0.0, 0.0], scale=0.25, size=(60, 2))

# Sparse cluster: 60 points drawn from a much wider Gaussian blob, centred
# far enough away that it does not overlap the dense cluster at small ε.
# Larger standard deviation (1.1) spreads points out, lowering local density.
sparse_cluster = rng.normal(loc=[6.0, 0.0], scale=1.1, size=(60, 2))

# Noise points: a handful of points scattered uniformly across the bounding
# region, representing genuine outliers unrelated to either cluster.
noise_points = rng.uniform(low=[-1.5, -3.0], high=[7.5, 3.0], size=(12, 2))

X = np.vstack([dense_cluster, sparse_cluster, noise_points])

# Ground-truth group membership, used only for annotation/explanation text —
# NOT passed to DBSCAN, which sees only X.
N_DENSE  = len(dense_cluster)
N_SPARSE = len(sparse_cluster)
GROUP = np.array(
    ['dense'] * N_DENSE + ['sparse'] * N_SPARSE + ['true_noise'] * len(noise_points)
)

# ── Axis limits ──────────────────────────────────────────────────────────────
PAD  = 1.0
XLIM = (X[:, 0].min() - PAD, X[:, 0].max() + PAD)
YLIM = (X[:, 1].min() - PAD, X[:, 1].max() + PAD)

COL_NOISE = '#bbbbbb'
COL_DENSE_CLUSTER  = 'steelblue'
COL_SPARSE_CLUSTER = 'tomato'
COL_MERGED         = 'mediumpurple'


def _diagnose(labels):
    """
    Work out, in plain terms, what happened to the dense and sparse
    clusters under the current ε. Returns a dict of diagnostic strings
    used to build the annotation panel.

    The diagnosis is based on majority cluster-label membership within
    each ground-truth group (dense / sparse), since DBSCAN's own label
    numbering is arbitrary and can change between runs.
    """
    dense_labels  = labels[GROUP == 'dense']
    sparse_labels = labels[GROUP == 'sparse']

    def majority_label(group_labels):
        # Most common label within this ground-truth group, ignoring noise
        # unless every point in the group is noise.
        non_noise = group_labels[group_labels != -1]
        if len(non_noise) == 0:
            return -1
        vals, counts = np.unique(non_noise, return_counts=True)
        return int(vals[np.argmax(counts)])

    dense_main  = majority_label(dense_labels)
    sparse_main = majority_label(sparse_labels)

    # Fraction of each ground-truth group labelled as noise (-1)
    dense_noise_frac  = float((dense_labels  == -1).mean())
    sparse_noise_frac = float((sparse_labels == -1).mean())

    # Diagnose the dense cluster
    if dense_main == -1:
        dense_status = 'Lost to noise'
    elif dense_main == sparse_main and sparse_main != -1:
        dense_status = 'Merged with sparse cluster'
    elif dense_noise_frac > 0.15:
        dense_status = 'Fragmenting (partial noise)'
    else:
        dense_status = 'Correctly recovered'

    # Diagnose the sparse cluster
    if sparse_main == -1:
        sparse_status = 'Lost to noise'
    elif sparse_main == dense_main and dense_main != -1:
        sparse_status = 'Merged with dense cluster'
    elif sparse_noise_frac > 0.15:
        sparse_status = 'Fragmenting (partial noise)'
    else:
        sparse_status = 'Correctly recovered'

    return {
        'dense_status':  dense_status,
        'sparse_status': sparse_status,
        'dense_main':    dense_main,
        'sparse_main':   sparse_main,
        'dense_noise_frac':  dense_noise_frac,
        'sparse_noise_frac': sparse_noise_frac,
    }


def _annotation_text(eps, labels, diag):
    """Build the annotation panel string."""
    n_clusters = len(set(labels) - {-1})
    n_noise    = int((labels == -1).sum())

    merged = (
        diag['dense_main'] == diag['sparse_main']
        and diag['dense_main'] != -1
    )

    if merged:
        overall = (
            "Both clusters have collapsed into a\n"
            "single DBSCAN cluster — the gap between\n"
            "them is smaller than ε, so the dense\n"
            "cluster's neighbourhood reaches across\n"
            "into the sparse one."
        )
    elif diag['dense_status'] == 'Correctly recovered' and \
            diag['sparse_status'] == 'Correctly recovered':
        overall = (
            "Both clusters look reasonable at this ε —\n"
            "but check the noise fractions below. Try\n"
            "moving ε slightly in either direction to\n"
            "see how quickly this balance breaks."
        )
    else:
        overall = (
            "No single ε keeps both clusters intact.\n"
            "Tightening ε to clean up the dense cluster\n"
            "starves the sparse cluster of neighbours;\n"
            "loosening ε to rescue the sparse cluster\n"
            "risks merging it with the dense one."
        )

    return (
        "DBSCAN outcome\n"
        "─────────────────────────────────\n\n"
        f"ε (radius)  = {eps:.2f}\n"
        f"minPts      = {MIN_PTS}  (fixed)\n\n"
        "─────────────────────────────────\n\n"
        f"Clusters found:   {n_clusters}\n"
        f"Noise points:     {n_noise}\n\n"
        "─────────────────────────────────\n\n"
        f"Dense cluster:\n"
        f"  status: {diag['dense_status']}\n"
        f"  noise fraction: {diag['dense_noise_frac']:.0%}\n\n"
        f"Sparse cluster:\n"
        f"  status: {diag['sparse_status']}\n"
        f"  noise fraction: {diag['sparse_noise_frac']:.0%}\n\n"
        "─────────────────────────────────\n\n"
        f"{overall}"
    )


def show():
    """Render Figure 1: DBSCAN's fixed-ε limitation on varying-density clusters."""
    plt.close('Notebook9 Figure 1')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook9 Figure 1',
        figsize=(10, 6),
        gridspec_kw={'width_ratios': [1.5, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    ax_ann.set_axis_off()

    ann_text = ax_ann.text(
        0.04, 0.97, '',
        transform=ax_ann.transAxes,
        fontsize=9, va='top', ha='left',
        linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 1: DBSCAN with one ε cannot serve clusters\n'
        'of very different densities at the same time',
        fontsize=11,
        y=0.98,
    )
    plt.subplots_adjust(wspace=0.08, top=0.86)

    def _draw(eps):
        ax_plot.clear()

        # Refit DBSCAN on the raw data with the current ε.
        # Data is not standardised here: the whole point of this figure is
        # to show the effect of varying ε directly in the original feature
        # space, where the two clusters have visibly different spreads.
        model  = DBSCAN(eps=eps, min_samples=MIN_PTS)
        labels = model.fit_predict(X)

        diag = _diagnose(labels)
        merged = (
            diag['dense_main'] == diag['sparse_main']
            and diag['dense_main'] != -1
        )

        # ── Colour each point ───────────────────────────────────────────────
        # If the two ground-truth clusters have merged into one DBSCAN
        # cluster, colour that shared cluster purple so the merge is obvious.
        # Otherwise colour each ground-truth cluster's points by whichever
        # DBSCAN label its majority falls into, using fixed dense/sparse
        # colours so the colours stay stable as eps changes.
        for i, pt in enumerate(X):
            label = labels[i]
            group = GROUP[i]

            if label == -1:
                colour = COL_NOISE
                marker = 'x'
            elif merged:
                colour = COL_MERGED
                marker = 'o'
            elif group == 'dense':
                colour = COL_DENSE_CLUSTER
                marker = 'o'
            elif group == 'sparse':
                colour = COL_SPARSE_CLUSTER
                marker = 'o'
            else:
                # A scattered "true noise" point that DBSCAN has pulled into
                # a cluster — colour it by whichever cluster it joined.
                if label == diag['dense_main']:
                    colour = COL_DENSE_CLUSTER
                elif label == diag['sparse_main']:
                    colour = COL_SPARSE_CLUSTER
                else:
                    colour = COL_MERGED
                marker = 'o'

            if marker == 'x':
                ax_plot.scatter(
                    pt[0], pt[1],
                    marker='x', s=55, color=colour,
                    linewidths=1.5, zorder=3,
                )
            else:
                ax_plot.scatter(
                    pt[0], pt[1],
                    marker='o', s=55, color=colour,
                    edgecolors='k', lw=0.4, alpha=0.85, zorder=4,
                )

        # ── Legend ───────────────────────────────────────────────────────────
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COL_DENSE_CLUSTER, markeredgecolor='k',
                       markersize=9, label='Dense cluster (assigned)'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COL_SPARSE_CLUSTER, markeredgecolor='k',
                       markersize=9, label='Sparse cluster (assigned)'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COL_MERGED, markeredgecolor='k',
                       markersize=9, label='Merged cluster'),
            plt.Line2D([0], [0], marker='x', color=COL_NOISE,
                       markersize=9, markeredgewidth=2, label='Noise (label = -1)'),
        ]
        ax_plot.legend(
            handles=legend_handles, fontsize=8,
            loc='lower left', framealpha=1.0, edgecolor='#cccccc',
        )

        ax_plot.set_xlim(*XLIM)
        ax_plot.set_ylim(*YLIM)
        ax_plot.set_xlabel('Feature 1', fontsize=10)
        ax_plot.set_ylabel('Feature 2', fontsize=10)
        ax_plot.set_title(
            f'ε = {eps:.2f}   minPts = {MIN_PTS}   '
            f'→   dense: {diag["dense_status"]}   |   '
            f'sparse: {diag["sparse_status"]}',
            fontsize=9,
        )
        ax_plot.grid(True, alpha=0.15)

        ann_text.set_text(_annotation_text(eps, labels, diag))

        fig.canvas.draw_idle()

    # ── Widget ───────────────────────────────────────────────────────────────
    slider_eps = widgets.FloatSlider(
        value=0.45, min=0.10, max=2.50, step=0.05,
        description='ε (radius)',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='400px'),
        readout_format='.2f',
    )

    out = interactive_output(_draw, {'eps': slider_eps})

    controls = widgets.VBox([slider_eps])
    display(controls, out)