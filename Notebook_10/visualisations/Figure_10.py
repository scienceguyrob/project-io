"""
Figure 10 — Cross-Validation vs Single Split: Stability Comparison
=======================================================================
Compares how much an accuracy estimate changes depending on which random
split is used, for two evaluation approaches:

  (a) A single train/test split, repeated 50 times with different random
  seeds, so each run produces one accuracy estimate from one split.

  (b) 5-fold cross-validation, also repeated 50 times with different random
  seeds, so each run produces one accuracy estimate that is itself the mean
  of 5 internal folds.

The two resulting sets of 50 accuracy estimates are compared by their
spread (standard deviation). A lower standard deviation means the estimate
is more reliable, less dependent on the luck of a particular split.

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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ── Dataset ──────────────────────────────────────────────────────────────────
# A single dataset is generated and scaled once, up front, so that both the
# single-split approach and the cross-validation approach are evaluated on
# exactly the same underlying data. Only the EVALUATION PROCEDURE differs
# between the two, not the data itself.
X_vs, y_vs = make_classification(
    n_samples=400, n_features=8, n_informative=4, random_state=9,
)

sc_vs = StandardScaler()
X_vs_s = sc_vs.fit_transform(X_vs)

# Number of repetitions for each approach. Each repetition uses a different
# random seed, so the spread of results across repetitions reflects how
# sensitive each approach is to which particular split (or splits) it sees.
N_REPS = 50


def show():
    """Render Figure 10: single split vs 5-fold cross-validation, stability comparison."""
    plt.close('Notebook10 Figure 10')

    single_split_accs = []
    cv_accs = []

    for seed in range(N_REPS):
        # (a) Single train/test split with this seed, the same pattern used
        # throughout earlier sections of this notebook.
        X_tr_vs, X_te_vs, y_tr_vs, y_te_vs = train_test_split(
            X_vs_s, y_vs, test_size=0.25, random_state=seed,
        )
        dt_vs = DecisionTreeClassifier(max_depth=5, random_state=0)
        dt_vs.fit(X_tr_vs, y_tr_vs)
        single_split_accs.append(accuracy_score(y_te_vs, dt_vs.predict(X_te_vs)))

        # (b) 5-fold cross-validation with this seed. cross_val_score
        # returns one accuracy per fold, .mean() collapses these 5 values
        # into the single estimate this repetition contributes.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_score = cross_val_score(
            DecisionTreeClassifier(max_depth=5, random_state=0),
            X_vs_s, y_vs, cv=cv, scoring='accuracy',
        ).mean()
        cv_accs.append(cv_score)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook10 Figure 10', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # Left panel: overlapping histograms of the two sets of 50 accuracy
    # estimates. A narrower histogram means the estimates are more tightly
    # clustered, and therefore more reliable.
    ax = axes[0]
    ax.hist(
        single_split_accs, bins=20, color='tomato', alpha=0.65,
        edgecolor='white', lw=0.4,
        label=f'Single split  (std = {np.std(single_split_accs):.3f})',
    )
    ax.hist(
        cv_accs, bins=20, color='steelblue', alpha=0.65,
        edgecolor='white', lw=0.4,
        label=f'5-fold CV      (std = {np.std(cv_accs):.3f})',
    )
    ax.axvline(np.mean(single_split_accs), color='tomato', lw=2.5, linestyle='--')
    ax.axvline(np.mean(cv_accs), color='steelblue', lw=2.5, linestyle='--')

    ax.set_xlabel('Accuracy estimate', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(
        'Single split vs 5-fold CV,\n'
        'CV gives a tighter, more reliable distribution',
        fontsize=10,
    )
    ax.legend(fontsize=9)

    # Right panel: the raw accuracy estimate produced by each of the 50
    # repetitions, for both approaches, plotted in sequence so the
    # run-to-run variability can be seen directly.
    ax = axes[1]
    ax.plot(
        range(N_REPS), single_split_accs, alpha=0.7, color='tomato',
        lw=1.5, label='Single split',
    )
    ax.plot(
        range(N_REPS), cv_accs, alpha=0.7, color='steelblue',
        lw=1.5, label='5-fold CV',
    )
    ax.set_xlabel('Repetition', fontsize=10)
    ax.set_ylabel('Accuracy estimate', fontsize=10)
    ax.set_title(
        'Variability across 50 random seeds,\n'
        'CV is much more stable',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Figure 10: cross-validation vs single split, stability',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # ── Summary printout ─────────────────────────────────────────────────────
    single_std = np.std(single_split_accs)
    cv_std = np.std(cv_accs)

    print(f'Single split: mean = {np.mean(single_split_accs):.3f},  std = {single_std:.4f}')
    print(f'5-fold CV:    mean = {np.mean(cv_accs):.3f},  std = {cv_std:.4f}')
    print(f'CV is {single_std / max(cv_std, 1e-6):.1f}x more stable')