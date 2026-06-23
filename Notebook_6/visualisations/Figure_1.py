"""
Figure 1 — Generative vs Discriminative Classifiers
=====================================================
Shows two synthetic 2-D clusters and illustrates how the two major
philosophies of classification approach the same data differently.

Left panel  (Generative):
    Models the full class-conditional distribution P(X | Y) for each class
    using 1-sigma and 2-sigma ellipses.  The ellipses capture the *shape*
    and *spread* of each class in feature space — the model understands
    where the data came from.

Right panel (Discriminative):
    Fits a logistic regression model (covered in Lab 5) to the same data
    and draws its decision boundary — the line where P(Y=A | X) = 0.5.
    The model ignores the internal structure of each class and focuses
    only on finding the boundary that best separates them.

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
from matplotlib.patches import Ellipse
from sklearn.linear_model import LogisticRegression

# ── Reproducible data generation ────────────────────────────────────────────
# Using a seeded random generator ensures the same clusters appear every time
# the cell is run, which is important for reproducible teaching examples.
rng_gd = np.random.default_rng(7)

# Generate 60 2-D samples for each class from a Gaussian (normal) distribution.
# np.random.default_rng(...).normal(mean, std, size) draws samples where:
#   mean  — the centre of the cluster (one value per feature dimension)
#   std   — how spread out the samples are around that centre
#   size  — (number of samples, number of features)
A_x = rng_gd.normal([2, 2], 0.8, (60, 2))   # Class A: centred at (2, 2)
B_x = rng_gd.normal([5, 5], 0.8, (60, 2))   # Class B: centred at (5, 5)

# ── Fit a logistic regression model for the discriminative panel ─────────────
# We stack both classes into one array X and create a label vector y.
# np.vstack joins arrays row-wise: shape becomes (120, 2).
X_all = np.vstack([A_x, B_x])

# np.array([0]*60 + [1]*60) creates 60 zeros (Class A) followed by 60 ones (Class B).
y_all = np.array([0] * 60 + [1] * 60)

# Fit logistic regression — this is the same model from Lab 5.
# It learns a linear decision boundary directly from the labelled data.
clf = LogisticRegression()
clf.fit(X_all, y_all)

# To draw the decision boundary we evaluate the model across a fine grid
# covering the plot area, then draw the contour where P(Y=1 | X) = 0.5.
# np.linspace(start, stop, n) returns n evenly spaced values between start and stop.
grid_vals = np.linspace(-1, 8, 400)

# np.meshgrid turns two 1-D arrays into two 2-D grids so we can evaluate
# the model at every (x1, x2) coordinate in the plot area.
xx, yy = np.meshgrid(grid_vals, grid_vals)

# np.c_ concatenates column-wise; here it reshapes the grid into a list of
# (x1, x2) coordinate pairs that the classifier can predict on.
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Reshape the predictions back into the same 2-D grid shape for contouring.
Z = Z.reshape(xx.shape)


def show():
    """
    Render Figure 1: Generative vs Discriminative Classifiers.

    Left panel  — generative view: ellipses show the learned class distributions.
    Right panel — discriminative view: logistic regression decision boundary.
    """
    plt.close('Notebook6 Figure 1')
    fig, axes = plt.subplots(1, 2, num='Notebook6 Figure 1', figsize=(10, 5))

    # ── Canvas settings (standard across all Lab 6 figures) ─────────────────
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — Generative model
    # A generative model learns P(X, Y): the joint probability of features
    # and labels.  Equivalently, it models P(X | Y) for each class — what
    # the feature values look like *given* we know the class.
    # The ellipses below are a visual representation of P(X | Y=c):
    #   1-sigma ellipse ≈ region containing ~68% of the class's data
    #   2-sigma ellipse ≈ region containing ~95% of the class's data
    # ════════════════════════════════════════════════════════════════════════
    ax = axes[0]

    # Plot the raw data points for each class.
    # zorder controls the drawing order — higher numbers appear on top.
    ax.scatter(A_x[:, 0], A_x[:, 1], color='steelblue', s=40,
               edgecolors='k', lw=0.3, alpha=0.8, label='Class A', zorder=3)
    ax.scatter(B_x[:, 0], B_x[:, 1], color='tomato', s=40,
               edgecolors='k', lw=0.3, alpha=0.8, label='Class B', zorder=3)

    # Draw 1-sigma and 2-sigma ellipses for each class.
    # Each ellipse is centred at the class mean and sized to represent
    # the spread of the distribution — this is the generative model's
    # internal picture of each class.
    for centre, col, lbl in [
        ([2, 2], 'steelblue', r'$P(\mathbf{X} \mid Y=A)$'),
        ([5, 5], 'tomato',    r'$P(\mathbf{X} \mid Y=B)$'),
    ]:
        for n_std in [1, 2]:
            # Ellipse(xy, width, height) — width and height are the full
            # diameters, so we multiply n_std by 1.8 (≈ 2 × 0.9 radius)
            # to reflect the known standard deviation of 0.8 used above.
            ell = Ellipse(
                xy=centre,
                width=n_std * 1.8,
                height=n_std * 1.8,
                angle=0,
                edgecolor=col,
                fc='none',
                lw=1.5,
                linestyle='--',
                zorder=2,
            )
            ax.add_patch(ell)

        # Add a legend entry for this class distribution.
        # Plotting empty arrays is a common trick to add a custom legend
        # handle without drawing any actual data points.
        ax.scatter([], [], color=col, label=lbl, marker='o', s=80,
                   edgecolors='k', lw=0.5)

    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(
        'Generative model\n'
        r'Learn $P(\mathbf{X}, Y)$ — model the shape of each class'
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — Discriminative model (Logistic Regression)
    # A discriminative model learns P(Y | X) directly — the probability of
    # each class label given the observed features.  It finds a decision
    # boundary that best separates the classes without modelling what each
    # class looks like internally.
    # The boundary below is the line where the logistic regression model
    # outputs P(Y=B | X) = 0.5 — equal probability for either class.
    # ════════════════════════════════════════════════════════════════════════
    ax = axes[1]

    # Plot the same raw data points as the left panel.
    ax.scatter(A_x[:, 0], A_x[:, 1], color='steelblue', s=40,
               edgecolors='k', lw=0.3, alpha=0.8, label='Class A', zorder=3)
    ax.scatter(B_x[:, 0], B_x[:, 1], color='tomato', s=40,
               edgecolors='k', lw=0.3, alpha=0.8, label='Class B', zorder=3)

    # ax.contour draws contour lines of the probability surface Z over the grid.
    # levels=[0.5] draws only the line where P(Y=B | X) = 0.5 — the decision
    # boundary — which is where the model switches its predicted class.
    ax.contour(xx, yy, Z, levels=[0.5], colors='k',
               linewidths=2.5, linestyles='--',
               zorder=4)

    # Add a dummy line to the legend to label the decision boundary.
    ax.plot([], [], 'k--', linewidth=2.5,
            label=r'Decision boundary: $P(Y \mid \mathbf{X}) = 0.5$')

    # ax.contourf fills the regions between contour levels with colour,
    # giving a visual indication of which region the model predicts as each class.
    # alpha controls transparency so the data points remain visible.
    ax.contourf(xx, yy, Z, levels=[0, 0.5], colors=['steelblue'], alpha=0.07,
                zorder=1)
    ax.contourf(xx, yy, Z, levels=[0.5, 1], colors=['tomato'], alpha=0.07,
                zorder=1)

    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(
        'Discriminative model (Logistic Regression)\n'
        r'Learn $P(Y \mid \mathbf{X})$ — model the boundary only'
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Suptitle — use tight_layout with rect to prevent cropping ───────────
    plt.suptitle('Figure 1: Generative vs discriminative classifiers', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.show()