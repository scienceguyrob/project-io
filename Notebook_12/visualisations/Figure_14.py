"""
Figure 1 -- Within-Cluster Sum of Squares (WCSS) and the Elbow Method
======================================================================
Illustrates the elbow method for choosing the number of clusters k in
k-means, using WCSS (also called inertia) as the evaluation criterion.

WCSS measures compactness: for each cluster, it sums the squared distances
from every point to its assigned centroid, then totals across all clusters.
Lower WCSS means tighter, more compact clusters. As k increases, WCSS
always decreases, but the rate of improvement slows once k exceeds the
true number of natural groupings in the data.

The elbow method exploits this: plot WCSS against k and look for the point
where adding another cluster gives only a small improvement. That kink, or
elbow, is the natural choice of k. The right panel makes the elbow easier
to see by plotting the incremental improvement in WCSS for each additional
cluster directly as a bar chart.

A WCSS value is also computed manually for k=3 and compared against
scikit-learn's inertia_ attribute to confirm the two approaches agree.

Both the left-panel curve and right-panel bar chart highlight k=4, which
is the true number of clusters used to generate the synthetic dataset.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_14 import show
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
from sklearn.cluster  import KMeans

# -- Dataset parameters -------------------------------------------------------
N_SAMPLES    = 300
N_CENTERS    = 4      # True number of clusters; the elbow should appear here
CLUSTER_STD  = 0.8
RANDOM_STATE = 3

# Range of k values to evaluate. Starting at 1 anchors the left end of the
# WCSS curve; stopping at 10 gives enough range to see the plateau clearly.
K_MIN = 1
K_MAX = 10

# k-means is sensitive to random initialisation. Running it N_INIT times and
# keeping the best result (lowest inertia) reduces this sensitivity.
N_INIT = 10

# Verification k: compute WCSS manually for this value and compare to sklearn.
K_VERIFY = 3

# -- Colours ------------------------------------------------------------------
COL_CURVE  = 'steelblue'
COL_BAR    = 'seagreen'
COL_ELBOW  = 'red'


def compute_wcss(X, labels, centers):
    """
    Compute WCSS (within-cluster sum of squares) manually from cluster
    labels and centroid positions.

    For each cluster k, sums the squared Euclidean distance from every
    point assigned to k to the centroid of k, then totals across all clusters.
    This is identical to scikit-learn\'s inertia_ attribute.

    Parameters
    ----------
    X       : (n_samples, n_features) array of data points.
    labels  : (n_samples,) integer array of cluster assignments.
    centers : (n_clusters, n_features) array of centroid positions.
    """
    total = 0.0
    for k, centroid in enumerate(centers):
        mask = labels == k
        if mask.any():
            total += float(np.sum((X[mask] - centroid) ** 2))
    return total


def show():
    """Render Figure 1: WCSS elbow curve and incremental improvement bar chart."""
    plt.close('Notebook12 Figure 1')

    X, _ = make_blobs(
        n_samples=N_SAMPLES, centers=N_CENTERS,
        cluster_std=CLUSTER_STD, random_state=RANDOM_STATE,
    )

    # -- Fit k-means for each k and collect WCSS values ----------------------
    k_range   = range(K_MIN, K_MAX + 1)
    wcss_vals = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=0)
        km.fit(X)
        wcss_vals.append(km.inertia_)

    # -- Verify manual WCSS matches sklearn for K_VERIFY ---------------------
    km_v = KMeans(n_clusters=K_VERIFY, n_init=N_INIT, random_state=0)
    km_v.fit(X)
    wcss_manual  = compute_wcss(X, km_v.labels_, km_v.cluster_centers_)
    wcss_sklearn = km_v.inertia_
    print(f'WCSS for k={K_VERIFY} (manual):  {wcss_manual:.3f}')
    print(f'WCSS for k={K_VERIFY} (sklearn): {wcss_sklearn:.3f}')
    print()

    # -- Incremental improvement: how much WCSS drops from k-1 to k ----------
    # A large value means adding this k is worthwhile; a small value means
    # diminishing returns have set in and we are past the true cluster count.
    improvements = [wcss_vals[i - 1] - wcss_vals[i]
                    for i in range(1, len(wcss_vals))]

    fig, axes = plt.subplots(
        1, 2,
        num='Notebook12 Figure 1',
        figsize=(10, 5),
    )
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_visible  = True
    fig.canvas.toolbar_position = 'right'

    # -- Left panel: raw WCSS vs k -------------------------------------------
    ax = axes[0]
    ax.plot(list(k_range), wcss_vals, 'o-',
            color=COL_CURVE, lw=2, markersize=8)
    ax.axvline(
        N_CENTERS, color=COL_ELBOW, lw=2, linestyle='--',
        label=f'True k = {N_CENTERS} (elbow)',
    )
    ax.set_xlabel('Number of clusters  k')
    ax.set_ylabel('WCSS (inertia)')
    ax.set_title(
        'Figure 14a: Elbow method — WCSS vs k\n'
        'Diminishing returns after the elbow',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Right panel: incremental improvement in WCSS -------------------------
    ax = axes[1]
    ax.bar(
        list(range(K_MIN + 1, K_MAX + 1)), improvements,
        color=COL_BAR, edgecolor='white', lw=0.4, alpha=0.85,
    )
    ax.axvline(
        N_CENTERS, color=COL_ELBOW, lw=2, linestyle='--',
        label=f'Elbow at k = {N_CENTERS}',
    )
    ax.set_xlabel('k')
    ax.set_ylabel('Improvement in WCSS  (k\u22121 \u2192 k)')
    ax.set_title(
        'Figure 14b: Improvement drops sharply after the elbow\n'
        'Choose k where the gain becomes small',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()