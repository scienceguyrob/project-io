"""
Figure 2 — Hard vs Soft Cluster Assignments on Overlapping Data
=======================================================================
Compares hard cluster assignments (K-Means) against soft probabilistic
assignments (Gaussian Mixture Model) on a synthetic two-cluster dataset
with deliberate overlap between the classes.

The figure contains three panels:

  Left   — K-Means hard assignments: every point is assigned to exactly
            one cluster, with no expression of uncertainty.

  Centre — GMM hard labels derived by taking the argmax of the
            per-component responsibilities. The boundary is similar to
            K-Means but the assignment was derived probabilistically.

  Right  — GMM soft assignments: points where neither component
            dominates clearly (max responsibility < 0.80) are
            highlighted in gold, revealing the genuinely ambiguous
            region where a hard label would be most misleading.

This figure is entirely self-contained and requires no external variables.

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
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# ── Dataset parameters ───────────────────────────────────────────────────────
# cluster_std=1.2 on centres separated by 3 units creates a moderate overlap
# zone, enough to produce genuinely uncertain points without making the two
# clusters indistinguishable.
N_SAMPLES      = 300
CLUSTER_CENTRES = [[-1.5, 0], [1.5, 0]]
CLUSTER_STD    = 1.2
RANDOM_STATE   = 7

# Threshold below which a point's maximum component responsibility is
# considered genuinely uncertain. 0.80 means "no component accounts for
# more than 80% of this point's membership".
UNCERTAINTY_THRESHOLD = 0.80

# Colours consistent with the notebook's palette.
PALETTE        = ['steelblue', 'tomato']
COLOUR_UNCERTAIN = 'gold'
COLOUR_GRID    = '#cccccc'


def show():
    """Render Figure 2: hard vs soft cluster assignments on overlapping data."""
    plt.close('Notebook8bonus Figure 2')

    # ── Data and models ───────────────────────────────────────────────────────
    X, _ = make_blobs(
        n_samples=N_SAMPLES,
        centers=CLUSTER_CENTRES,
        cluster_std=CLUSTER_STD,
        random_state=RANDOM_STATE,
    )

    # K-Means: each point receives a single hard cluster label.
    km          = KMeans(n_clusters=2, n_init=10, random_state=0)
    hard_labels = km.fit_predict(X)

    # GMM: each point receives a probability of belonging to each component.
    # predict_proba returns (n_samples, n_components); predict returns argmax.
    gmm        = GaussianMixture(n_components=2, covariance_type='full',
                                 random_state=0, n_init=3)
    gmm.fit(X)
    soft_probs = gmm.predict_proba(X)
    soft_hard  = gmm.predict(X)

    # Points where the maximum responsibility across all components is below
    # the threshold are genuinely uncertain: the model cannot confidently
    # assign them to any single component.
    max_prob  = soft_probs.max(axis=1)
    uncertain = max_prob < UNCERTAINTY_THRESHOLD

    print('Figure 2: hard vs soft cluster assignments')
    print(f'  Total points          : {len(X)}')
    print(f'  Uncertain points      : {uncertain.sum()} '
          f'({100 * uncertain.mean():.1f}% of dataset)')
    print(f'  Uncertainty threshold : max responsibility < {UNCERTAINTY_THRESHOLD}')
    print('  These are the points for which a hard assignment is most misleading.')

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), num='Notebook8bonus Figure 2')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False

    # ── Left panel: K-Means hard assignments ──────────────────────────────────
    ax = axes[0]
    for k, col in enumerate(PALETTE):
        mask = hard_labels == k
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=col, s=30, alpha=0.7,
                   edgecolors='k', lw=0.2, label=f'Cluster {k}')
    ax.set_title('Hard assignments\n(K-Means)', fontsize=10)
    ax.legend(fontsize=9)

    # ── Centre panel: GMM hard labels (argmax of responsibilities) ────────────
    ax = axes[1]
    for k, col in enumerate(PALETTE):
        mask = soft_hard == k
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=col, s=30, alpha=0.7,
                   edgecolors='k', lw=0.2, label=f'Component {k}')
    ax.set_title('GMM hard labels\n(argmax of responsibilities)', fontsize=10)
    ax.legend(fontsize=9)

    # ── Right panel: GMM with uncertain points highlighted ────────────────────
    ax = axes[2]
    for k, col in enumerate(PALETTE):
        # Plot certain points for this component in their component colour,
        # then overlay the uncertain points in gold on top.
        mask = (soft_hard == k) & ~uncertain
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=col, s=30, alpha=0.5,
                   edgecolors='k', lw=0.15, label=f'Component {k} (certain)')
    ax.scatter(
        X[uncertain, 0], X[uncertain, 1],
        color=COLOUR_UNCERTAIN, s=70, edgecolors='black', lw=0.8,
        zorder=5,
        label=f'Uncertain (max prob < {UNCERTAINTY_THRESHOLD})\nn = {uncertain.sum()}',
    )
    ax.set_title('GMM soft assignments\n'
                 f'gold = uncertain (max prob < {UNCERTAINTY_THRESHOLD})',
                 fontsize=10)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, color=COLOUR_GRID, alpha=0.3)

    fig.suptitle(
        'Figure 2: Hard vs soft cluster assignments on overlapping data',
        fontsize=12,
    )
    plt.tight_layout()