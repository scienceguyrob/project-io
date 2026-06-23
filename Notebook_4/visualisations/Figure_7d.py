"""
Figure 7d — Feature Selection vs Noise: k-NN Accuracy Experiment
=================================================================

A k-nearest neighbours (k=5) classifier is trained on data with two
informative features. Noise dimensions are then progressively added and
accuracy is measured in two ways:

    - Using ALL features (including the noise)
    - Using ONLY the two informative features

The gap between the two lines shows the direct accuracy cost of including
irrelevant features in a distance-based classifier.

Note: the k-NN implementation here uses only NumPy — no scikit-learn.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_7d import show
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


def show():
    """Render the static Figure 7d k-NN accuracy vs noise dimensions plot."""

    plt.close('Notebook4 Figure 7d')

    rng_c           = np.random.default_rng(7)
    n_train, n_test = 120, 40

    # ── Data generation helper ────────────────────────────────────────────────
    def make_data(n, n_noise, rng):
        """
        Generate a two-class dataset with 2 informative and n_noise noise features.
        Class A is centred around (2, 2); Class B around (4, 4).
        Noise features are drawn from N(0, 3) — no relationship to class labels.
        """
        A = np.column_stack(
            [rng.normal(2, 1.0, n), rng.normal(2, 1.0, n)] +
            [rng.normal(0, 3, n) for _ in range(n_noise)]
        )
        B = np.column_stack(
            [rng.normal(4, 1.0, n), rng.normal(4, 1.0, n)] +
            [rng.normal(0, 3, n) for _ in range(n_noise)]
        )
        return np.vstack([A, B]), np.array([0] * n + [1] * n)

    # ── k-NN classifier ───────────────────────────────────────────────────────
    def knn_accuracy(X_tr, y_tr, X_te, y_te, k=5):
        """
        Evaluate k-NN accuracy without scikit-learn.
        For each test point: compute Euclidean distance to all training points,
        find the k closest, take a majority vote on their labels.
        """
        correct = 0
        for xt, yt in zip(X_te, y_te):
            # Distance from this test point to every training point
            dists = np.sqrt(((X_tr - xt) ** 2).sum(axis=1))
            # Labels of the k nearest neighbours (argsort gives indices in distance order)
            nn_labels = y_tr[np.argsort(dists)[:k]]
            # Majority vote: if mean label >= 0.5 predict class 1, else class 0
            pred     = int(nn_labels.mean() >= 0.5)
            correct += (pred == yt)
        return correct / len(y_te)

    # ── Run the experiment across noise levels ────────────────────────────────
    noise_levels = [0, 2, 5, 10, 20, 40]
    acc_all, acc_2 = [], []

    print(f"  {'Noise dims':<12} {'All features':<18} {'First 2 only':<18} {'Total dims'}")
    print('-' * 62)

    for n_noise in noise_levels:
        X_tr, y_tr = make_data(n_train, n_noise, rng_c)
        X_te, y_te = make_data(n_test,  n_noise, rng_c)

        # Accuracy using all features — noise included
        a_all = knn_accuracy(X_tr,        y_tr, X_te,        y_te)
        # Accuracy using only the two informative features (columns 0 and 1)
        a_2   = knn_accuracy(X_tr[:, :2], y_tr, X_te[:, :2], y_te)

        acc_all.append(a_all)
        acc_2.append(a_2)
        print(f"  {n_noise:<12} {a_all:<18.1%} {a_2:<18.1%} {2 + n_noise}")

    total_dims = [2 + n for n in noise_levels]

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook4 Figure 7d', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # Red line: all features — accuracy degrades as noise dimensions are added
    ax.plot(total_dims, acc_all, 'o-', color='tomato', lw=2, ms=7,
            label='All features (including noise)')

    # Blue line: informative features only — accuracy stays high throughout
    ax.plot(total_dims, acc_2, 's-', color='steelblue', lw=2, ms=7,
            label='First 2 features only (informative)')

    # Dotted line at 50% — the accuracy of random guessing on a balanced dataset
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5,
               label='Random-guess baseline (50%)')

    # Shade the gap between the two lines to make the cost of noise visible
    ax.fill_between(total_dims, acc_all, acc_2, alpha=0.08, color='tomato',
                    label='Accuracy lost to noise dimensions')

    ax.set_xlabel('Total number of features (dimensions)')
    ax.set_ylabel('Classification accuracy')
    ax.set_title(
        'Figure 7d: Curse of dimensionality in practice —\n'
        'adding irrelevant features degrades a distance-based classifier',
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()