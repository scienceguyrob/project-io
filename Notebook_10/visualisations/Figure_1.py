"""
Figure 1 — The Memorisation Problem and the Generalisation Gap
=======================================================================
Demonstrates why evaluating a model on the data it was trained on is
always misleading. A Decision Tree is trained at every depth from 1 to
20, and at each depth we measure:

  - Training accuracy (measured on the data the model learned from)
  - Test accuracy     (measured on data the model has NEVER seen)

As depth increases, the tree becomes more complex and is increasingly
able to memorise the training data, so training accuracy climbs steadily
towards 100%. Test accuracy, however, rises initially as the tree learns
genuinely useful splits, then peaks and falls as the tree starts fitting
noise specific to the training set rather than patterns that generalise.

The gap between the two curves — shaded on the plot — is the
GENERALISATION GAP. A small gap means the model has learned something
that holds up beyond the training data. A large gap means the model has,
to some degree, memorised the training data rather than learning from it.

A vertical dotted line marks the depth at which test accuracy peaks —
this is the depth you would actually want to deploy, even though it is
not the depth with the highest training accuracy.

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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ── Dataset ──────────────────────────────────────────────────────────────────
# A synthetic classification dataset is used so that the underlying signal is
# known and controllable. 10 features are generated in total: 5 of these
# carry genuine information about the class label ("informative"), and 2 are
# redundant — linear combinations of the informative features, included to
# make the problem more realistic (real datasets rarely have purely
# independent features). The remaining 3 features are pure noise.
X_base, y_base = make_classification(
    n_samples=500,       # total number of samples
    n_features=10,       # total number of features
    n_informative=5,     # features that actually carry signal
    n_redundant=2,       # features that are linear combinations of informative ones
    random_state=0,      # fixed seed so the figure is reproducible across runs
)

# A single 75/25 train/test split is used throughout. This is deliberate:
# the point of this figure is to isolate the effect of model COMPLEXITY
# (controlled by max_depth) on the generalisation gap, not the effect of
# the split itself — that is the subject of Task 1 later in this notebook.
X_tr, X_te, y_tr, y_te = train_test_split(
    X_base, y_base, test_size=0.25, random_state=0   # 75/25 split
)


def show():
    """Render Figure 1: the generalisation gap as Decision Tree depth increases."""
    plt.close('Notebook10 Figure 1')

    # ── Train a Decision Tree at every depth from 1 to 20 ───────────────────
    # max_depth controls how many sequential splits the tree is permitted to
    # make. A shallow tree (small max_depth) can only draw a few, coarse
    # decision boundaries. A deep tree (large max_depth) can keep splitting
    # until, in the extreme, every individual training point sits in its own
    # tiny region — memorisation in its purest form.
    depths = list(range(1, 21))
    train_accs = []
    test_accs = []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=0)
        dt.fit(X_tr, y_tr)

        # Accuracy on the data the tree learned from — this is the figure
        # the "exam questions in advance" instinct relies on, and it is
        # always optimistic.
        train_accs.append(accuracy_score(y_tr, dt.predict(X_tr)))

        # Accuracy on data the tree has never seen — this is the figure
        # that actually tells us how the tree will perform in the real world.
        test_accs.append(accuracy_score(y_te, dt.predict(X_te)))

    # The depth that gives the best TEST accuracy is the depth you would
    # actually want to deploy, even though it is not the depth with the
    # highest training accuracy.
    best_test_depth = depths[int(np.argmax(test_accs))]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook10 Figure 1', figsize=(10, 5))
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    ax.plot(
        depths, train_accs, 'o-', color='steelblue', lw=2, ms=6,
        label='Training accuracy  (what the model SAW)',
    )
    ax.plot(
        depths, test_accs, 's-', color='tomato', lw=2, ms=6,
        label='Test accuracy  (what the model must GENERALISE to)',
    )

    # Vertical line marking where the best test accuracy occurs — this is
    # the depth you would actually choose, not the deepest tree available.
    ax.axvline(
        best_test_depth, color='black', lw=1.5, linestyle=':',
        label=f'Best test depth = {best_test_depth}',
    )

    # Shade the region between the two curves — this shaded area IS the
    # generalisation gap, made visible. It widens as depth increases.
    ax.fill_between(
        depths, train_accs, test_accs, alpha=0.12, color='tomato',
        label='Generalisation gap',
    )

    ax.set_xlabel('Decision tree max depth', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title(
        'Figure 1: Overfitting — training accuracy climbs to 100%\n'
        'while test accuracy peaks then falls, revealing memorisation',
        fontsize=11,
    )
    ax.legend(fontsize=9)

    # y-limits fixed to 0.5-1.02 so the climb towards 100% training accuracy
    # and the shape of the test accuracy curve are both clearly visible,
    # without empty space below the lowest accuracy values.
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Summary printout ─────────────────────────────────────────────────────
    print(f'At max_depth={depths[-1]}: train={train_accs[-1]:.1%}, test={test_accs[-1]:.1%}')
    print(f'Generalisation gap = {train_accs[-1] - test_accs[-1]:.1%}')
    print()
    print('Key lesson: training accuracy is ALWAYS optimistic.')
    print('Only test accuracy tells you how the model performs in the real world.')