"""
Figure 6 — Train/Test Split Ratio: A Fundamental Trade-Off
=======================================================================
Demonstrates the trade-off involved in choosing how much data to allocate
to training versus testing.

  MORE TRAINING DATA: typically gives a better model, since there is more
  to learn from, but leaves fewer test samples, so the resulting accuracy
  estimate is noisier.

  MORE TEST DATA: gives a more reliable accuracy estimate, but leaves fewer
  training samples, so the model itself may be weaker.

A single Decision Tree configuration is trained across every split ratio
from 10% to 90% training data, and each split is repeated 20 times with
different random seeds to estimate how variable the resulting accuracy is.
The shaded band around the accuracy line represents plus or minus one
standard deviation across those 20 seeds, a wide band means an unreliable
estimate.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_6 import show
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


# ── Dataset ──────────────────────────────────────────────────────────────────
# A single synthetic dataset of 1000 samples is used throughout, so that the
# only thing varying across the experiment is the split ratio and the random
# seed used to create each split, not the underlying data itself.
X_sp, y_sp = make_classification(
    n_samples=1000, n_features=10, n_informative=5, random_state=2,
)

# Test set size is swept from 10% to 90% of the data, in steps of 5
# percentage points. Each split ratio is repeated with 20 different random
# seeds, so that a mean and standard deviation of accuracy can be calculated
# for each ratio rather than relying on a single, potentially lucky or
# unlucky, split.
TEST_FRACTIONS = np.arange(0.10, 0.91, 0.05)
N_SEEDS = 20


def show():
    """Render Figure 6: train/test split ratio as a trade-off between model quality and estimate reliability."""
    plt.close('Notebook10 Figure 6')

    # ── Sweep over split ratios ───────────────────────────────────────────────
    mean_accs, std_accs, n_test_sizes = [], [], []

    for test_frac in TEST_FRACTIONS:
        accs_seed = []

        for seed in range(N_SEEDS):
            X_tr_sp, X_te_sp, y_tr_sp, y_te_sp = train_test_split(
                X_sp, y_sp, test_size=test_frac, random_state=seed,
            )

            sc_sp = StandardScaler()
            X_tr_sp_s = sc_sp.fit_transform(X_tr_sp)   # fit on training data only
            X_te_sp_s = sc_sp.transform(X_te_sp)       # apply same fitted scaler to test data

            clf_sp = DecisionTreeClassifier(max_depth=5, random_state=0)
            clf_sp.fit(X_tr_sp_s, y_tr_sp)
            accs_seed.append(accuracy_score(y_te_sp, clf_sp.predict(X_te_sp_s)))

        # The mean across the 20 seeds estimates the "typical" accuracy at
        # this split ratio, while the standard deviation captures how much
        # that accuracy estimate would bounce around depending on which
        # particular split happened to be drawn.
        mean_accs.append(np.mean(accs_seed))
        std_accs.append(np.std(accs_seed))
        n_test_sizes.append(int(test_frac * 1000))

    mean_accs = np.array(mean_accs)
    std_accs = np.array(std_accs)
    train_pcts = (1 - TEST_FRACTIONS) * 100

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook10 Figure 6', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # Left panel: mean accuracy plus or minus one standard deviation, as a
    # function of training set percentage.
    ax = axes[0]
    ax.plot(train_pcts, mean_accs, 'o-', color='steelblue', lw=2, ms=6)
    ax.fill_between(
        train_pcts, mean_accs - std_accs, mean_accs + std_accs,
        alpha=0.2, color='steelblue', label='±1 std (across 20 seeds)',
    )

    # Two reference lines mark commonly used split ratios, so the trade-off
    # curve can be related back to familiar choices.
    ax.axvline(75, color='tomato',   lw=2, linestyle='--', label='75% train (common choice)')
    ax.axvline(80, color='seagreen', lw=2, linestyle=':',  label='80% train (common choice)')

    ax.set_xlabel('Training set size (% of data)', fontsize=10)
    ax.set_ylabel('Test accuracy (mean ± std)', fontsize=10)
    ax.set_title(
        'More training data improves accuracy,\n'
        'but the estimate becomes noisier',
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right panel: how much the accuracy estimate varies, as a function of
    # the number of samples in the test set.
    ax = axes[1]
    ax.plot(n_test_sizes, std_accs, 's-', color='tomato', lw=2, ms=6)
    ax.set_xlabel('Test set size (number of samples)', fontsize=10)
    ax.set_ylabel('Std of accuracy estimate', fontsize=10)
    ax.set_title(
        'Estimate variance vs test set size,\n'
        'tiny test sets give unstable results',
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Figure 6: train/test split ratio, a fundamental trade-off',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # ── Summary table for common split ratios ──────────────────────────────────
    print('Accuracy at common split ratios (mean ± std over 20 seeds):')
    for pct in [50, 60, 70, 75, 80, 90]:
        # The split ratios swept above do not necessarily land exactly on
        # these round percentages, so the closest available ratio is used
        # for each one.
        idx = np.argmin(np.abs(train_pcts - pct))
        print(
            f'  {pct}% train / {100 - pct}% test:  '
            f'{mean_accs[idx]:.3f} ± {std_accs[idx]:.3f}  '
            f'(test n = {n_test_sizes[idx]})'
        )