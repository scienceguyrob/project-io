"""
Figure 5 — The Experimental Pipeline: Keep Test Conditions Identical
=======================================================================
Demonstrates the single most important rule in classifier evaluation:
every model must be evaluated under exactly the same conditions. This means
the same train/test split, the same preprocessing (a scaler fitted on the
training data and then applied to the test data), and the same test set for
every model in the comparison.

This figure compares two pipelines applied to the same dataset and the same
five classifiers:

  CORRECT pattern: one split, one fitted scaler, all models evaluated on
  the same test set.

  WRONG pattern: a different random split, and therefore a different test
  set, for each model.

If a different test set is used for each model, any difference in scores
could be due to one model simply having been tested on easier examples,
rather than a genuine difference in model quality. The bar charts show how
the apparent ranking of the five models can shift once this rule is broken.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_5 import show
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ── Dataset ──────────────────────────────────────────────────────────────────
# A single synthetic dataset is used for both the correct and the wrong
# pipeline, so that the dataset itself is never the source of any difference
# between them. Only the evaluation procedure differs.
X_pipe, y_pipe = make_classification(
    n_samples=600, n_features=8, n_informative=4, random_state=3,
)

# The same five classifiers are used in both pipelines, with the same
# hyperparameters and random states, so that any difference in their scores
# can only come from how the data was split and scaled.
MODEL_LIST = [
    ('Logistic Reg.',  LogisticRegression(max_iter=1000, random_state=0)),
    ('Decision Tree',  DecisionTreeClassifier(max_depth=4, random_state=0)),
    ('Naive Bayes',    GaussianNB()),
    ('k-NN (k=5)',     KNeighborsClassifier(n_neighbors=5)),
    ('Random Forest',  RandomForestClassifier(n_estimators=50, random_state=0)),
]


def show():
    """Render Figure 5: correct vs wrong evaluation pipeline."""
    plt.close('Notebook10 Figure 5')

    # ── CORRECT pattern ───────────────────────────────────────────────────────
    # One split, one fitted scaler, and every model is evaluated on the same
    # test set. random_state=42 is fixed deliberately and should not be
    # changed, since the whole point is that this single split is shared by
    # every model.
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_pipe, y_pipe, test_size=0.25, random_state=42,
    )

    sc_c = StandardScaler()
    X_tr_c_s = sc_c.fit_transform(X_tr_c)   # fit on training data only
    X_te_c_s = sc_c.transform(X_te_c)       # apply the same fitted scaler to test data

    correct_results = {}
    for name, clf in MODEL_LIST:
        clf.fit(X_tr_c_s, y_tr_c)
        correct_results[name] = accuracy_score(y_te_c, clf.predict(X_te_c_s))

    # ── WRONG pattern ─────────────────────────────────────────────────────────
    # A different random_state is used for the split on every iteration of
    # the loop, so each model ends up trained and tested on a different
    # split of the data, and therefore a different test set. This is the
    # mistake the figure is warning against.
    wrong_results = {}
    for seed, (name, clf) in enumerate(MODEL_LIST):
        X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(
            X_pipe, y_pipe, test_size=0.25, random_state=seed,
        )

        sc_w = StandardScaler()
        clf.fit(sc_w.fit_transform(X_tr_w), y_tr_w)
        wrong_results[name] = accuracy_score(y_te_w, clf.predict(sc_w.transform(X_te_w)))

    # ── Comparison printout ───────────────────────────────────────────────────
    print('Comparison: correct pipeline vs wrong pipeline (different splits)')
    print(f'  {"Model":<18}  {"CORRECT":>10}  {"WRONG":>10}  {"Diff":>8}')
    print('-' * 52)
    for name in correct_results:
        c = correct_results[name]
        w = wrong_results[name]
        print(f'  {name:<18}  {c:>10.1%}  {w:>10.1%}  {w - c:>+8.1%}')
    print()
    print('The rankings change when different test sets are used.')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook10 Figure 5', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    names_p = list(correct_results.keys())
    x_pos = range(len(names_p))   # fixed tick positions, set explicitly to avoid a tick/label mismatch warning

    for ax, res, title, col in [
        (axes[0], correct_results, 'CORRECT: same split for all models', 'steelblue'),
        (axes[1], wrong_results,   'WRONG: different split per model',   'tomato'),
    ]:
        vals = [res[n] for n in names_p]
        bars = ax.bar(x_pos, vals, color=col, edgecolor='white', lw=0.4, alpha=0.85)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Test accuracy', fontsize=10)
        ax.set_ylim(0.5, 1.05)

        # set_xticks is called first, with the same fixed positions used for
        # the bars, so that set_xticklabels assigns labels to a known,
        # fixed set of ticks rather than triggering a UserWarning about an
        # unset or variable number of ticks.
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names_p, rotation=18, ha='right', fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')

        # Label each bar with its exact accuracy, placed just above the bar.
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.005,
                f'{v:.1%}', ha='center', fontsize=8,
            )

    fig.suptitle(
        'Figure 5: always use the same test set when comparing models',
        fontsize=11, y=0.98,
    )
    plt.tight_layout()
    plt.show()