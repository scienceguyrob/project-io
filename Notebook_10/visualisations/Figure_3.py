"""
Figure 3 — The Two Datasets Used to Demonstrate the No Free Lunch Theorem
=======================================================================
Visualises the two datasets used in the No Free Lunch theorem demonstration,
shown side by side so their structure can be compared directly before any
classifiers are trained on them.

  Dataset 1: a linearly separable dataset, generated with
  make_classification using a single cluster per class. The two classes
  can be separated reasonably well by a straight line.

  Dataset 2: the "two moons" dataset, generated with make_moons. The two
  classes form interlocking crescent shapes that cannot be separated by a
  straight line, the boundary between them is curved.

Both datasets contain 500 points with two features, so they can be plotted
directly without any dimensionality reduction. The point of this figure is
to make the structural difference between the two datasets visible, since
this difference is exactly what causes different classifiers to rank
differently on each one.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_3 import show
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

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons


# ── Datasets ─────────────────────────────────────────────────────────────────
# Dataset 1: a linearly separable dataset. n_clusters_per_class=1 and
# n_redundant=0 keep the structure simple, two clean, well-separated blobs,
# one per class, so that a straight-line decision boundary is sufficient.
X_d1, y_d1 = make_classification(
    n_samples=500, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=1,
)

# Dataset 2: the "two moons" dataset. The two classes form interlocking
# crescent shapes, so no straight line can separate them well, a curved
# decision boundary is required. noise=0.2 adds some scatter so the
# crescents are not perfectly clean.
X_d2, y_d2 = make_moons(n_samples=500, noise=0.2, random_state=1)


def show():
    """Render Figure 3: the two datasets used in the No Free Lunch demonstration."""
    plt.close('Notebook10 Figure 3')

    fig, axes = plt.subplots(1, 2, num='Notebook10 Figure 3', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    for ax, X, y, title in [
        (axes[0], X_d1, y_d1, 'Dataset 1: linearly separable blobs'),
        (axes[1], X_d2, y_d2, 'Dataset 2: two moons (non-linear)'),
    ]:
        # Each class is plotted in a fixed colour, steelblue for class 0 and
        # tomato for class 1, so the two panels are directly comparable and
        # consistent with the colours used later for these datasets.
        for cls, col in [(0, 'steelblue'), (1, 'tomato')]:
            m = y == cls
            ax.scatter(
                X[m, 0], X[m, 1], color=col, s=20, alpha=0.7,
                edgecolors='k', lw=0.2, label=f'Class {cls}',
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        'Figure 3: two datasets with very different structure,\n'
        'a straight line separates one but not the other',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()
    plt.show()