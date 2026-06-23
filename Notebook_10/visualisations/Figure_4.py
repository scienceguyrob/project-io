"""
Figure 4 — No Free Lunch: Classifier Rankings Differ Across Datasets
=======================================================================
Demonstrates the No Free Lunch (NFL) theorem directly by training six
classifiers on two datasets with very different structures, and comparing
their test accuracy on each.

  Dataset 1: a linearly separable dataset (simple, linear boundary).
  Dataset 2: the two moons dataset (non-linear, curved boundary).

The same six classifiers, the same train/test split ratio, and the same
scaling procedure are used for both datasets, so the only thing that
differs between the two halves of the experiment is the structure of the
data itself. The resulting bar chart shows that the ranking of the six
classifiers is not consistent between the two datasets: a classifier that
performs well on one can perform poorly on the other.

This is the practical illustration of NFL: there is no universally "best"
classifier, the right choice depends on the structure of the data, and the
only way to find out which model suits a given problem is to test several
of them under identical conditions.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_4 import show
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
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ── Datasets ─────────────────────────────────────────────────────────────────
# The same two datasets used in Figure 3: one linearly separable and one
# with a curved, non-linear boundary. The point of reusing them here is that
# their structural difference, made visible in Figure 3, is exactly what
# drives the different classifier rankings shown in this figure.
X_d1, y_d1 = make_classification(
    n_samples=500, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=1,
)
X_d2, y_d2 = make_moons(n_samples=500, noise=0.2, random_state=1)

# ── Classifiers ──────────────────────────────────────────────────────────────
# Six classifiers spanning a range of approaches covered earlier in this
# series, a linear model, a tree-based model, a probabilistic model, an
# instance-based model, a margin-based model, and an ensemble model. All six
# are evaluated identically on both datasets, so any difference in their
# rankings reflects the data, not the evaluation procedure.
classifiers = [
    ('Logistic Reg.',  LogisticRegression(max_iter=1000, random_state=0)),
    ('Decision Tree',  DecisionTreeClassifier(max_depth=4, random_state=0)),
    ('Naive Bayes',    GaussianNB()),
    ('k-NN (k=5)',     KNeighborsClassifier(n_neighbors=5)),
    ('SVM (linear)',   SVC(kernel='linear', C=1.0, random_state=0)),
    ('Random Forest',  RandomForestClassifier(n_estimators=50, random_state=0)),
]


def show():
    """Render Figure 4: classifier rankings on two structurally different datasets."""
    plt.close('Notebook10 Figure 4')

    # ── Evaluation ────────────────────────────────────────────────────────────
    # Each classifier is evaluated on each dataset using an identical
    # pipeline: the same train/test split ratio and random_state (so the
    # split itself is comparable across datasets), and a scaler fitted only
    # on the training data and then applied to the test data, following the
    # same pattern used in Figure 2.
    results = {name: {} for name, _ in classifiers}

    for X_data, y_data, label in [
        (X_d1, y_d1, 'Dataset 1 (linear)'),
        (X_d2, y_d2, 'Dataset 2 (moons)'),
    ]:
        X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
            X_data, y_data, test_size=0.25, random_state=0,   # same split ratio for both datasets
        )

        sc = StandardScaler()
        X_tr_d = sc.fit_transform(X_tr_d)   # fit scaler on training data only
        X_te_d = sc.transform(X_te_d)       # apply same scaler to test data

        for name, clf in classifiers:
            clf.fit(X_tr_d, y_tr_d)
            results[name][label] = accuracy_score(y_te_d, clf.predict(X_te_d))

    # ── Ranking printout ──────────────────────────────────────────────────────
    # Printing the ranking explicitly, rather than just the bar chart, makes
    # it easy to see that the ORDER of classifiers changes between datasets,
    # not just their individual scores.
    for label in ['Dataset 1 (linear)', 'Dataset 2 (moons)']:
        ranked = sorted(results.items(), key=lambda x: x[1][label], reverse=True)
        print(f'Ranking on {label}:')
        for rank, (name, accs) in enumerate(ranked, 1):
            print(f'  {rank}. {name:<20} {accs[label]:.1%}')
        print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    names = [n for n, _ in classifiers]
    x_pos = np.arange(len(classifiers))
    w = 0.35   # bar width, chosen so the two bars per classifier sit side by side without overlapping

    fig, ax = plt.subplots(num='Notebook10 Figure 4', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    ax.bar(
        x_pos - w / 2,
        [results[n]['Dataset 1 (linear)'] for n in names],
        width=w, color='steelblue', edgecolor='white', lw=0.4, label='Dataset 1 (linear)',
    )
    ax.bar(
        x_pos + w / 2,
        [results[n]['Dataset 2 (moons)'] for n in names],
        width=w, color='tomato', edgecolor='white', lw=0.4, label='Dataset 2 (moons)',
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Test accuracy', fontsize=10)
    ax.set_title(
        'Figure 4: No Free Lunch, the best model on one dataset\n'
        'may not be the best on another. Always test multiple models.',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()