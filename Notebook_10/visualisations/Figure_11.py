"""
Figure 11 — Choosing k: How k Affects Bias and Variance of the CV Estimate
=======================================================================
Demonstrates how the choice of k in k-fold cross-validation trades off bias
and variance in the resulting accuracy estimate.

  SMALL k (e.g. k=2): each training fold is small, so the model is trained
  on relatively little data and tends to be undertrained. The accuracy
  estimate is therefore biased downward (too pessimistic). However, each
  test fold is large, so the estimate has LOW VARIANCE, it is stable from
  fold to fold.

  LARGE k (e.g. k=20): each training fold is large, so the model is
  well-trained, and the accuracy estimate has LOW BIAS. However, each test
  fold is tiny, so the estimate has HIGHER VARIANCE, small test folds give
  noisier individual fold scores.

k=5 or k=10 is the standard compromise between these two effects. The
extreme case k=n, one fold per sample, is called Leave-One-Out (LOO)
cross-validation.

This figure runs cross-validation with k = 2, 3, 5, 10, and 20 on the same
dataset and model, and plots how the mean and standard deviation of the
resulting CV accuracy change with k.

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

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ── Dataset ──────────────────────────────────────────────────────────────────
# A single dataset is generated and scaled once, up front, so that every
# value of k is evaluated on exactly the same underlying data. Only k
# itself varies across the experiment.
X_kc, y_kc = make_classification(
    n_samples=300, n_features=6, n_informative=3, random_state=11,
)

sc_kc = StandardScaler()
X_kc_s = sc_kc.fit_transform(X_kc)

# The range of k values to compare, spanning the small-k, large-bias /
# low-variance end (k=2) through to the large-k, low-bias / higher-variance
# end (k=20), with the common defaults k=5 and k=10 included for reference.
K_VALUES = [2, 3, 5, 10, 20]


def show():
    """Render Figure 11: the effect of k on the bias and variance of the CV estimate."""
    plt.close('Notebook10 Figure 11')

    # ── Sweep over k ──────────────────────────────────────────────────────────
    k_means = []
    k_stds = []

    for k in K_VALUES:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        scores = cross_val_score(
            DecisionTreeClassifier(max_depth=5, random_state=0),
            X_kc_s, y_kc, cv=cv, scoring='accuracy',
        )
        # scores contains one accuracy value per fold (k values in total).
        # Its mean is the CV accuracy for this k, and its standard
        # deviation captures how much that estimate varies fold to fold.
        k_means.append(scores.mean())
        k_stds.append(scores.std())

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook10 Figure 11', figsize=(10, 4))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # Left panel: mean CV accuracy for each k, with error bars showing
    # plus or minus one standard deviation, illustrating how the central
    # estimate and its spread both shift as k changes.
    ax = axes[0]
    ax.errorbar(
        K_VALUES, k_means, yerr=k_stds,
        fmt='o-', color='steelblue', lw=2, ms=8, capsize=5,
        ecolor='steelblue', elinewidth=1.5,
    )
    ax.set_xlabel('k (number of folds)', fontsize=10)
    ax.set_ylabel('CV accuracy (mean ± std)', fontsize=10)
    ax.set_title(
        'Effect of k on mean accuracy estimate,\n'
        'mean stabilises as k grows',
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    # Right panel: standard deviation of the per-fold accuracy scores for
    # each k, shown as a bar chart, isolating the variance side of the
    # bias-variance trade-off described above.
    ax = axes[1]
    x_pos = range(len(K_VALUES))   # fixed tick positions, set explicitly before set_xticklabels
    ax.bar(x_pos, k_stds, color='tomato', edgecolor='white', lw=0.4, alpha=0.85)

    ax.set_xlabel('k (number of folds)', fontsize=10)
    ax.set_ylabel('Std of CV accuracy', fontsize=10)
    ax.set_title(
        'Effect of k on estimate variance,\n'
        'large k gives higher variance (smaller test folds)',
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, axis='y')

    for i, (k, std) in enumerate(zip(K_VALUES, k_stds)):
        ax.text(i, std + 0.001, f'{std:.4f}', ha='center', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(K_VALUES)

    fig.suptitle(
        'Figure 11: choosing k in k-fold CV, the bias-variance trade-off',
        fontsize=11, y=0.98,
    )
    plt.tight_layout()
    plt.show()