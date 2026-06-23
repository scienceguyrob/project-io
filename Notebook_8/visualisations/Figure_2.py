"""
Figure 2 — Semi-Supervised Classification via Cluster Pseudo-Labelling
=======================================================================
Demonstrates how cluster structure can be combined with a small set of
known labels to classify an entire dataset without full manual annotation.

The three panels walk through the process step by step:

  Panel 1 — Only 15 points (~5% of the dataset) carry known labels.
             All other points are unlabelled (shown in grey).
  Panel 2 — K-Means clusters the full dataset ignoring labels entirely.
             Gold stars mark the three cluster centroids.
  Panel 3 — Each cluster inherits the majority label of its known points.
             These pseudo-labels are propagated to every unlabelled point,
             and the resulting accuracy is reported in the panel title.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_2 import show
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
from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# ── Reproducibility ───────────────────────────────────────────────────────────
# Two separate RNGs: one for the blob generation (via sklearn random_state),
# one for sampling the 15 known label indices.
RNG = np.random.default_rng(7)

# ── Dataset parameters ────────────────────────────────────────────────────────
N_SAMPLES    = 300
N_CLUSTERS   = 3
CLUSTER_STD  = 0.9    # moderate spread — enough overlap to make the task
                      # non-trivial without making clusters unrecoverable
N_KNOWN      = 15     # ~5% of the dataset carries a true label

# ── Visual parameters ─────────────────────────────────────────────────────────
PALETTE = ['steelblue', 'tomato', 'seagreen']


def _build_data():
    """
    Generate the blob dataset, simulate partial labelling, run K-Means,
    and propagate pseudo-labels. Returns all artefacts needed for plotting.
    """
    X, y = make_blobs(
        n_samples=N_SAMPLES, centers=N_CLUSTERS,
        cluster_std=CLUSTER_STD, random_state=5
    )

    # Simulate having labels for only N_KNOWN points.
    # y_partial uses -1 as a sentinel for "label unknown" — a standard
    # convention in semi-supervised learning libraries (e.g. sklearn).
    known_idx = RNG.choice(len(X), N_KNOWN, replace=False)
    y_partial = -np.ones(len(X), dtype=int)
    y_partial[known_idx] = y[known_idx]

    # K-Means with n_init=10: runs 10 random initialisations and keeps the
    # solution with the lowest within-cluster sum of squares (WCSS). This
    # guards against the algorithm settling in a poor local minimum, which
    # is a real risk with the default single-run initialisation.
    km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    cluster_ids = km.fit_predict(X)

    # For each cluster, find the majority true label among the known points
    # that fall inside it. Counter.most_common(1) returns a [(label, count)]
    # list; we take index [0][0] to extract just the label.
    cluster_to_label = {}
    for c in range(N_CLUSTERS):
        known_in_c = y[(cluster_ids == c) & (y_partial != -1)]
        if len(known_in_c) > 0:
            cluster_to_label[c] = Counter(known_in_c).most_common(1)[0][0]
        else:
            cluster_to_label[c] = -1   # edge case: no known labels in cluster

    # Propagate the majority label to every point in each cluster
    pseudo_labels = np.array([cluster_to_label[c] for c in cluster_ids])
    accuracy = np.mean(pseudo_labels == y)

    return X, y, known_idx, cluster_ids, km, pseudo_labels, accuracy


def show():
    """Render Figure 2: semi-supervised classification via pseudo-labelling."""
    plt.close('Notebook8 Figure 2')

    X, y, known_idx, cluster_ids, km, pseudo_labels, accuracy = _build_data()

    fig, axes = plt.subplots(
        1, 3, num='Notebook8 Figure 2', figsize=(12, 5)
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # ── Panel 1: known labels only ────────────────────────────────────────────
    # All unlabelled points are shown in grey so that the 15 labelled points
    # stand out clearly. Larger marker size and a stronger edge reinforce that
    # these are the only points of known identity.
    ax = axes[0]
    ax.scatter(
        X[:, 0], X[:, 1],
        c='lightgray', s=30, edgecolors='k', lw=0.2, label='Unlabelled'
    )
    ax.scatter(
        X[known_idx, 0], X[known_idx, 1],
        c=[PALETTE[l] for l in y[known_idx]],
        s=90, edgecolors='black', lw=1.2, zorder=5, label='Known labels'
    )
    ax.set_title(f'Step 1: Only {N_KNOWN} labelled points\n(~5% of dataset)', fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 2: K-Means cluster assignments ──────────────────────────────────
    # Colours here reflect cluster membership, not true labels — they may not
    # match panel 1. Gold stars mark the centroids returned by K-Means.
    ax = axes[1]
    for k, col in enumerate(PALETTE):
        ax.scatter(
            X[cluster_ids == k, 0], X[cluster_ids == k, 1],
            color=col, s=30, alpha=0.6, edgecolors='k', lw=0.15
        )
    ax.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        marker='*', s=250, color='gold', edgecolors='black',
        lw=1.2, zorder=5, label='Centroids'
    )
    ax.set_title('Step 2: K-Means clusters\n(gold stars = centroids)', fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 3: pseudo-labels propagated ─────────────────────────────────────
    # Every point now carries the majority label of its cluster. The accuracy
    # in the title is measured against the true labels — in a real scenario
    # those would not be available, so this is an idealised evaluation.
    ax = axes[2]
    for k, col in enumerate(PALETTE):
        ax.scatter(
            X[pseudo_labels == k, 0], X[pseudo_labels == k, 1],
            color=col, s=30, alpha=0.6, edgecolors='k', lw=0.15
        )
    ax.set_title(
        f'Step 3: Pseudo-labels propagated\nAccuracy = {accuracy:.1%}',
        fontsize=10
    )

    # ── Shared axis formatting ────────────────────────────────────────────────
    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.2)

    # ── Figure-level formatting ───────────────────────────────────────────────
    plt.suptitle(
        'Figure 2: Semi-supervised classification via cluster pseudo-labelling',
        fontsize=12, y=0.98
    )
    plt.tight_layout()
    plt.show()

    print(f'Pseudo-label accuracy using only ~5% labelled data: {accuracy:.1%}')
