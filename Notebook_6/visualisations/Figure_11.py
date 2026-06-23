"""
Figure 11 — Overfitting in Decision Trees
==========================================
Investigates the effect of max_depth on training vs test accuracy for a
decision tree classifier trained on the cloud type dataset.

Two panels:
  Left  — training accuracy vs test accuracy as max_depth increases.
           The shaded gap between the two curves is the overfitting gap.
  Right — number of leaf nodes vs max_depth, showing how model complexity
           grows with depth.

The figure is self-contained — it loads data/clouds.csv internally and
does not depend on any variables defined in other notebook cells.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_11 import show
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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Cumulus', 'Stratus', 'Cirrus']
FEAT_NAMES  = ['altitude_km', 'vertical_extent_km',
               'temperature_c', 'optical_thickness']
REFERENCE_DEPTH = 4    # the depth used in the main training cell


def _load_and_split():
    """
    Load data/clouds.csv and return a fixed train/test split.
    Uses the same random_state and stratify settings as the main cell
    so the results are directly comparable.
    """
    df      = pd.read_csv('data/clouds.csv')
    X       = df[FEAT_NAMES].values
    y       = pd.Categorical(df['cloud_type'], categories=CLASS_NAMES).codes
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def show():
    """
    Render Figure 11: Overfitting investigation — train vs test accuracy
    and model complexity across max_depth values 1 to 20.
    """
    plt.close('Notebook6 Figure 11')

    # ── Load data ─────────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = _load_and_split()

    # ── Sweep max_depth from 1 to 20 ─────────────────────────────────────────
    # At each depth we fit a fresh tree on the training set and evaluate it
    # on both the training set and the held-out test set.
    depths    = list(range(1, 21))
    train_acc = []
    test_acc  = []
    n_leaves  = []

    for d in depths:
        # Fit a new tree at this depth — same data, same split, only depth varies.
        t = DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=0)
        t.fit(X_tr, y_tr)

        # Training accuracy — how well the model fits the data it was trained on.
        # A perfect score here does not mean the model generalises well; it may
        # just mean it has memorised the training examples.
        train_acc.append(accuracy_score(y_tr, t.predict(X_tr)))

        # Test accuracy — evaluated on data the model has never seen.
        # This is the honest measure of generalisation performance.
        test_acc.append(accuracy_score(y_te, t.predict(X_te)))

        # Number of leaves — a direct measure of model complexity.
        # Each leaf is one rule; more leaves = more complex = more likely to overfit.
        n_leaves.append(t.get_n_leaves())

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, num='Notebook6 Figure 11', figsize=(10, 5)
    )

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ── Left panel — accuracy vs depth ───────────────────────────────────────
    ax1.plot(depths, train_acc, 'o-', color='steelblue', lw=2,
             label='Training accuracy')
    ax1.plot(depths, test_acc,  'o-', color='tomato',    lw=2,
             label='Test accuracy')

    # Shade the gap between the two curves — this is the overfitting gap.
    # Where the gap is large, the model is fitting noise in the training data
    # that does not generalise to new examples.
    ax1.fill_between(depths, test_acc, train_acc,
                     alpha=0.12, color='tomato',
                     label='Overfitting gap')

    # Mark the reference depth used in the main training cell
    ax1.axvline(REFERENCE_DEPTH, color='seagreen', lw=1.5,
                linestyle='--', alpha=0.8,
                label=f'Reference depth ({REFERENCE_DEPTH})')

    # Mark the best test accuracy
    best_idx   = int(np.argmax(test_acc))
    best_depth = depths[best_idx]
    ax1.axvline(best_depth, color='#f0a500', lw=1.5,
                linestyle=':', alpha=0.9,
                label=f'Best test depth ({best_depth})')

    ax1.set_xlabel('max_depth', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim(0.5, 1.02)
    ax1.set_xticks(depths)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_title(
        'Training vs test accuracy as tree depth increases\n'
        'The gap between the curves is the overfitting gap',
        fontsize=10,
    )

    # ── Right panel — model complexity vs depth ───────────────────────────────
    ax2.plot(depths, n_leaves, 'o-', color='#7b5ea7', lw=2,
             label='Number of leaves')

    ax2.axvline(REFERENCE_DEPTH, color='seagreen', lw=1.5,
                linestyle='--', alpha=0.8,
                label=f'Reference depth ({REFERENCE_DEPTH})')
    ax2.axvline(best_depth, color='#f0a500', lw=1.5,
                linestyle=':', alpha=0.9,
                label=f'Best test depth ({best_depth})')

    ax2.set_xlabel('max_depth', fontsize=11)
    ax2.set_ylabel('Number of leaves', fontsize=11)
    ax2.set_xticks(depths)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_title(
        'Model complexity (number of leaves) vs tree depth\n'
        'More leaves = more rules = higher risk of overfitting',
        fontsize=10,
    )

    plt.suptitle(
        'Figure 11: Overfitting in decision trees — cloud type classification',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f'Best test accuracy : {max(test_acc):.1%}  at max_depth = {best_depth}')
    print(f'Train accuracy     : {train_acc[best_depth-1]:.1%}  at max_depth = {best_depth}')
    print()
    print(f'Train accuracy     : {train_acc[-1]:.1%}  at max_depth = 20')
    print(f'Test  accuracy     : {test_acc[-1]:.1%}  at max_depth = 20')
    print(f'Overfitting gap    : {train_acc[-1] - test_acc[-1]:.1%}  at max_depth = 20')