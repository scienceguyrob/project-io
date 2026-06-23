"""
Figure 2 — The i.i.d. Assumption: Train on One Distribution, Test on Another
=======================================================================
Demonstrates what happens when the i.i.d. (independently and identically
distributed) assumption is violated by training a model on data from one
population and testing it on data from a different population.

'i.i.d.' has two parts:

  IDENTICALLY DISTRIBUTED: the training and test data must come from the
  same underlying distribution. Training on summer weather data and
  testing on winter data violates this, the test tells you nothing about
  summer performance.

  INDEPENDENT: each data point must be sampled independently. Ten
  measurements from the same person in ten seconds are NOT independent;
  the effective sample size is closer to 1.

This figure simulates a violated i.i.d. assumption:

  - Population A: measurements from one demographic (used for training)
  - Population B: measurements from a different demographic (used for testing)

Same task, but different underlying distributions. A logistic regression
model is trained on Population A and evaluated on both populations, showing
that an accuracy figure that looks excellent on the training distribution
can collapse on a different distribution, even for the same underlying task.

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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ── Populations ──────────────────────────────────────────────────────────────
# Population A and Population B represent two groups that share the same
# general task (separating two classes based on two features), but come
# from different underlying distributions. This is the i.i.d. violation:
# Population A is used for training, Population B for testing, and the
# two are deliberately constructed to differ in both location and
# correlation structure.
rng_iid = np.random.default_rng(7)

# Population A: centred at [2, 2] with positive correlation between the two
# features. This is the training distribution.
X_popA = rng_iid.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 300)
y_popA = (X_popA[:, 0] + X_popA[:, 1] > 4).astype(int)

# Population B: centred at [5, 5] with negative correlation between the two
# features. This is the test distribution, used only for evaluation.
X_popB = rng_iid.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], 300)
y_popB = (X_popB[:, 0] + X_popB[:, 1] > 10).astype(int)


def show():
    """Render Figure 2: i.i.d. violation, train on one distribution, test on another."""
    plt.close('Notebook10 Figure 2')

    # ── Scaling ───────────────────────────────────────────────────────────────
    # The scaler is fitted ONLY on Population A, since in a real i.i.d.
    # violation the test population would not be available during training.
    # The same fitted scaler is then applied to Population B, exactly as it
    # would be applied to genuinely new data. Population B's different
    # location and correlation structure means this scaling will not
    # standardise it as cleanly as it does Population A.
    scaler_iid = StandardScaler()
    X_popA_s = scaler_iid.fit_transform(X_popA)
    X_popB_s = scaler_iid.transform(X_popB)

    # ── Model ─────────────────────────────────────────────────────────────────
    # The classifier is trained only on Population A, then evaluated on both
    # populations. Population A's accuracy shows how the model performs on
    # data drawn from the same distribution it was trained on. Population B's
    # accuracy shows how the same model performs once the i.i.d. assumption
    # no longer holds.
    clf_iid = LogisticRegression(random_state=0)
    clf_iid.fit(X_popA_s, y_popA)

    acc_same_dist = accuracy_score(y_popA, clf_iid.predict(X_popA_s))
    acc_diff_dist = accuracy_score(y_popB, clf_iid.predict(X_popB_s))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook10 Figure 2', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    for ax, X, y, title in [
        (axes[0], X_popA_s, y_popA,
         f'Population A (training distribution)\nAccuracy = {acc_same_dist:.1%}'),
        (axes[1], X_popB_s, y_popB,
         f'Population B (different distribution)\nAccuracy = {acc_diff_dist:.1%}'),
    ]:
        # Each class is plotted in a fixed colour so the two panels are
        # directly comparable: steelblue for class 0, tomato for class 1.
        for cls, col in [(0, 'steelblue'), (1, 'tomato')]:
            m = y == cls
            ax.scatter(
                X[m, 0], X[m, 1], color=col, s=20, alpha=0.6,
                edgecolors='k', lw=0.1, label=f'Class {cls}',
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Feature 1 (standardised)', fontsize=10)
        ax.set_ylabel('Feature 2 (standardised)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        'Figure 2: i.i.d. violation, train on one distribution, test on another',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # ── Summary printout ─────────────────────────────────────────────────────
    print(f'Training distribution accuracy:  {acc_same_dist:.1%}')
    print(f'Different distribution accuracy: {acc_diff_dist:.1%}')
    print(f'Drop in accuracy: {acc_same_dist - acc_diff_dist:.1%}')
    print()
    print('Lesson: a model trained and tested on the same distribution looks great.')
    print('The same model on a different distribution can collapse completely.')
    print('This is why i.i.d. sampling is not optional, it is the foundation.')