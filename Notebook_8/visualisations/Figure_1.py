"""
Figure 1 — The Three Foundational Assumptions of Unsupervised Learning
=======================================================================
Three side-by-side panels, each illustrating one core assumption that
underpins unsupervised learning methods:

  Panel 1 — Cluster assumption   : compact, well-separated Gaussian blobs
  Panel 2 — Smoothness assumption: two-moon crescents with a colour gradient
                                   showing that labels vary smoothly across space
  Panel 3 — Manifold assumption  : a noisy spiral showing that high-dimensional
                                   data can lie on a low-dimensional curved surface

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
from sklearn.datasets import make_blobs, make_moons


# ── Reproducibility ───────────────────────────────────────────────────────────
# Separate RNG for the spiral noise so panel 3 is independent of panels 1 & 2,
# which use sklearn's random_state parameter directly.
RNG = np.random.default_rng(7)

# ── Spiral parameters ─────────────────────────────────────────────────────────
# 300 points wound through 2 full rotations (0 → 4π). The colour encodes
# position along the spiral (θ), making the 1-D manifold structure visible.
N_SPIRAL  = 300
THETA     = np.linspace(0, 4 * np.pi, N_SPIRAL)
NOISE_STD = 0.3   # small perturbation off the 1-D curve — enough to look
                  # realistic without obscuring the manifold structure

# ── Colour palette for cluster blobs ─────────────────────────────────────────
PALETTE = ['steelblue', 'tomato', 'seagreen']


def show():
    """Render Figure 1: the three foundational assumptions of unsupervised learning."""
    plt.close('Notebook8 Figure 1')

    fig, axes = plt.subplots(
        1, 3, num='Notebook8 Figure 1', figsize=(12, 6)
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # ── Panel 1: Cluster assumption ───────────────────────────────────────────
    # make_blobs produces isotropic Gaussian blobs. cluster_std=0.7 keeps the
    # groups compact and visually well-separated without being unrealistically
    # tight. random_state fixes the layout across runs.
    ax = axes[0]
    X_cl, y_cl = make_blobs(
        n_samples=200, centers=3, cluster_std=0.7, random_state=0
    )
    for k, col in enumerate(PALETTE):
        ax.scatter(
            X_cl[y_cl == k, 0], X_cl[y_cl == k, 1],
            color=col, s=35, alpha=0.75, edgecolors='k', lw=0.2
        )
    ax.set_title(
        'Cluster assumption\nData naturally groups into compact, well-separated clouds',
        fontsize=10
    )
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.15)

    # ── Panel 2: Smoothness assumption ────────────────────────────────────────
    # make_moons produces two interleaved crescents. We ignore the class labels
    # and instead colour by x-coordinate: the smooth colour transition shows
    # that a small move in input space produces a small change in output.
    ax = axes[1]
    X_sm, _ = make_moons(n_samples=200, noise=0.08, random_state=1)
    sc = ax.scatter(
        X_sm[:, 0], X_sm[:, 1],
        c=X_sm[:, 0],              # colour encodes x1, not class label
        cmap='coolwarm', s=35, alpha=0.8, edgecolors='k', lw=0.2
    )
    ax.set_title(
        'Smoothness assumption\nNearby points share similar values;\nsmall input change → small output change',
        fontsize=10
    )
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.15)
    plt.colorbar(sc, ax=ax, label='x₁ value')

    # ── Panel 3: Manifold assumption ──────────────────────────────────────────
    # A noisy Archimedean spiral: x = θ·cos(θ),  y = θ·sin(θ), plus Gaussian
    # noise. The data lives in 2-D but the meaningful structure is 1-D (position
    # along the spiral, encoded by θ). The colourbar reveals the true intrinsic
    # dimension — one parameter describes every point.
    ax = axes[2]
    manifold_x = THETA * np.cos(THETA) + RNG.normal(0, NOISE_STD, N_SPIRAL)
    manifold_y = THETA * np.sin(THETA) + RNG.normal(0, NOISE_STD, N_SPIRAL)
    sc2 = ax.scatter(
        manifold_x, manifold_y,
        c=THETA,                   # colour encodes position along the manifold
        cmap='viridis', s=25, alpha=0.8, edgecolors='k', lw=0.1
    )
    ax.set_title(
        'Manifold assumption\nHigh-dimensional data lies on a\nlow-dimensional curved surface',
        fontsize=10
    )
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.15)
    plt.colorbar(sc2, ax=ax, label='Position along manifold')

    # ── Figure-level formatting ───────────────────────────────────────────────
    plt.suptitle(
        'Figure 1: The three foundational assumptions of unsupervised learning',
        fontsize=12, y=0.98
    )
    plt.tight_layout()
    plt.show()