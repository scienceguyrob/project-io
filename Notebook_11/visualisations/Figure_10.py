"""
Figure 10 — The Bias-Variance Trade-off Curve
=======================================================================
Shows how training error and test error evolve as model complexity
increases, using a decision tree's max_depth parameter as the complexity
axis. As depth increases from 1 to 20:

  Training error falls monotonically: a deeper tree can always fit the
  training data more closely, and at sufficient depth it fits it exactly.

  Test error first falls as bias reduces (a very shallow tree is too
  simple to capture the true pattern), then rises as variance dominates
  (a very deep tree overfits to the noise in the training sample and
  generalises poorly).

  The generalisation gap (shaded region between the two curves) widens
  on the right, reflecting the increasing difference between how the
  model performs on training data versus unseen data.

The optimal complexity is the depth at which test error is minimised.
A vertical dashed line marks this point on the plot, and the console
output reports the corresponding training and test error values.

This figure is entirely self-contained and requires no external variables.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_10 import show
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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ── Dataset parameters ───────────────────────────────────────────────────────
# 500 samples with 10 features (5 informative) gives the decision tree
# enough signal to learn from at shallow depths while still showing
# clear overfitting at high depths.
N_SAMPLES    = 500
N_FEATURES   = 10
N_INFORMATIVE = 5
TEST_SIZE    = 0.25
RANDOM_STATE = 0

# Range of max_depth values to sweep. Starting at 1 (a single split,
# maximum bias) and ending at 20 (deep enough to start memorising noise).
DEPTH_MIN = 1
DEPTH_MAX = 20

# Colours consistent with the notebook's palette.
COLOUR_TRAIN = 'steelblue'
COLOUR_TEST  = 'tomato'
COLOUR_GAP   = 'tomato'
COLOUR_GRID  = '#cccccc'

# Annotation positions for the bias and variance labels. These are set in
# data coordinates and may need adjusting if the error curves change shape
# significantly with a different dataset or random state.
BIAS_ANNOTATION_XY     = (2,   0.17)   # arrow tip: a shallow-tree test-error point
BIAS_ANNOTATION_XYTEXT = (2.5, 0.32)   # label position
VAR_ANNOTATION_XY      = (18,  0.17)   # arrow tip: a deep-tree test-error point
VAR_ANNOTATION_XYTEXT  = (13,  0.30)   # label position


def show():
    """Render Figure 10: the bias-variance trade-off curve."""
    plt.close('Notebook11 Figure 10')

    # ── Dataset ───────────────────────────────────────────────────────────────
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        random_state=2,             # fixed separately from RANDOM_STATE so the
    )                               # dataset and splits are independently seeded

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # ── Sweep complexity ──────────────────────────────────────────────────────
    # For each depth, fit a fresh decision tree and record the error rate
    # (1 - accuracy) on both the training set and the test set. Training
    # error is computed on the data the tree was fitted on, which is why it
    # falls monotonically: a deeper tree can always fit training data better.
    depths     = list(range(DEPTH_MIN, DEPTH_MAX + 1))
    train_errs = []
    test_errs  = []

    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        dt.fit(X_tr_s, y_tr)
        train_errs.append(1 - accuracy_score(y_tr, dt.predict(X_tr_s)))
        test_errs.append( 1 - accuracy_score(y_te, dt.predict(X_te_s)))

    train_errs = np.array(train_errs)
    test_errs  = np.array(test_errs)
    best_depth = depths[int(np.argmin(test_errs))]

    print('Figure 10: bias-variance trade-off curve')
    print(f'  Optimal depth (minimum test error) : {best_depth}')
    print(f'  Training error at optimal depth    : {train_errs[best_depth - 1]:.3f}')
    print(f'  Test error at optimal depth        : {test_errs[best_depth - 1]:.3f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), num='Notebook11 Figure 10')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    ax.plot(depths, train_errs, 'o-', color=COLOUR_TRAIN, lw=2, ms=6,
            label='Training error')
    ax.plot(depths, test_errs,  's-', color=COLOUR_TEST,  lw=2, ms=6,
            label='Test error')

    # The shaded region between the two curves is the generalisation gap:
    # how much worse the model performs on unseen data than on training data.
    # It grows on the right as the tree overfits and variance increases.
    ax.fill_between(depths, train_errs, test_errs,
                    alpha=0.12, color=COLOUR_GAP, label='Generalisation gap')

    # Vertical line marking the depth that minimises test error, i.e. the
    # point of optimal bias-variance balance for this dataset.
    ax.axvline(best_depth, color='black', lw=2, linestyle=':',
               label=f'Optimal depth = {best_depth}')

    # Annotations pointing to the underfitting and overfitting regions,
    # so the reader can immediately connect the curve shape to the concepts.
    ax.annotate(
        'HIGH BIAS\n(underfitting)',
        xy=BIAS_ANNOTATION_XY, xytext=BIAS_ANNOTATION_XYTEXT,
        fontsize=9, color=COLOUR_TRAIN,
        arrowprops=dict(arrowstyle='->', color=COLOUR_TRAIN, lw=1.2),
    )
    ax.annotate(
        'HIGH VARIANCE\n(overfitting)',
        xy=VAR_ANNOTATION_XY, xytext=VAR_ANNOTATION_XYTEXT,
        fontsize=9, color=COLOUR_TEST,
        arrowprops=dict(arrowstyle='->', color=COLOUR_TEST, lw=1.2),
    )

    ax.set_xlabel('Model complexity (decision tree max depth)')
    ax.set_ylabel('Error rate (1 - accuracy)')
    ax.set_title(
        'Figure 10: Bias-variance trade-off curve\n'
        'optimal complexity sits at the minimum of the test error curve',
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, color=COLOUR_GRID, alpha=0.4)
    plt.tight_layout()