"""
Figure 9 -- Precision-Recall Curves: ROC vs PR on Imbalanced and Balanced Data
===============================================================================
Compares ROC and PR curves side by side to show when each is more informative.

Three panels are shown:

  Panel 1 (ROC, imbalanced): ROC curves for two synthetic models on a 95/5
    negative/positive dataset. Both curves look acceptable, giving little
    indication of how differently the models actually behave.

  Panel 2 (PR, imbalanced): PR curves for the same two models on the same
    dataset. The gap between the models is now clearly visible. A model that
    catches most positives (high recall) at reasonable precision stands apart
    from one that achieves low FPR by simply predicting negative most of the
    time.

  Panel 3 (PR, balanced): PR curves for three classifiers on a synthetic
    balanced dataset, provided for contrast. When classes are roughly equal
    in size, a good classifier keeps precision high across the full recall
    range, and the curve stays well above the random baseline.

All data is generated internally with fixed random seeds. The figure is
fully self-contained and requires no inputs from the calling notebook.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_9 import show
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
from sklearn.datasets        import make_classification
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

# -- Imbalanced dataset parameters --------------------------------------------
# Mirrors the 95/5 split used in Figure 7 so the comparison is consistent.
N_NEG_IMB = 950
N_POS_IMB = 50

# -- Balanced dataset parameters ----------------------------------------------
# A more tractable synthetic problem used in Panel 3 to show what a good
# PR curve looks like when classes are roughly equal in size.
N_BALANCED    = 2000
N_FEATURES    = 8
N_INFORMATIVE = 4

# -- Colours ------------------------------------------------------------------
COL_A  = "steelblue"
COL_B  = "tomato"
COL_LR = "steelblue"
COL_RF = "seagreen"
COL_DT = "tomato"


def _make_imbalanced():
    """
    Generate the same 95/5 imbalanced dataset and synthetic score distributions
    used in Figure 7, so Panels 1 and 2 are directly comparable to that figure.
    """
    rng = np.random.default_rng(17)

    y = np.array([0] * N_NEG_IMB + [1] * N_POS_IMB)

    sc_A = np.concatenate([
        rng.beta(2, 5, N_NEG_IMB),   # Negatives: medium-low scores
        rng.beta(5, 2, N_POS_IMB),   # Positives: higher scores
    ])
    sc_B = np.concatenate([
        rng.beta(1, 8, N_NEG_IMB),   # Negatives: very low scores
        rng.beta(3, 3, N_POS_IMB),   # Positives: medium scores, poor recall
    ])

    return y, sc_A, sc_B


def _make_balanced():
    """
    Generate a balanced synthetic classification dataset and fit three
    classifiers on it, returning their test labels and predicted scores.
    A fixed random seed ensures the panel is identical on every run.
    """
    X, y = make_classification(
        n_samples=N_BALANCED,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=2,
        n_classes=2,
        flip_y=0.03,
        random_state=42,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    classifiers = [
        ("Logistic Reg.",  LogisticRegression(max_iter=1000, random_state=0)),
        ("Random Forest",  RandomForestClassifier(n_estimators=100, random_state=0)),
        ("Decision Tree",  DecisionTreeClassifier(max_depth=4, random_state=0)),
    ]
    colours = [COL_LR, COL_RF, COL_DT]

    results = []
    for (name, clf), col in zip(classifiers, colours):
        clf.fit(X_tr_s, y_tr)
        scores = clf.predict_proba(X_te_s)[:, 1]
        results.append((name, scores, col))

    return y_te, results


def show():
    """Render Figure 9: three-panel ROC vs PR comparison."""
    plt.close("Notebook12 Figure 9")

    y_imb, sc_A, sc_B     = _make_imbalanced()
    y_bal, bal_results     = _make_balanced()

    pos_rate_imb = N_POS_IMB / (N_NEG_IMB + N_POS_IMB)
    pos_rate_bal = y_bal.mean()

    fig, axes = plt.subplots(
        1, 3,
        num="Notebook12 Figure 9",
        figsize=(13, 5),
    )
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_visible  = True
    fig.canvas.toolbar_position = "right"

    # -- Panel 1: ROC curves on the imbalanced dataset -----------------------
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4, label="Random (AUC = 0.50)")
    for sc_m, name, col in [(sc_A, "Model A", COL_A), (sc_B, "Model B", COL_B)]:
        fpr, tpr, _ = roc_curve(y_imb, sc_m)
        auc = roc_auc_score(y_imb, sc_m)
        ax.plot(fpr, tpr, color=col, lw=2.5, label=f"{name} (AUC = {auc:.3f})")
    ax.set_xlabel("FPR  (False Positive Rate)")
    ax.set_ylabel("TPR  (True Positive Rate)")
    ax.set_title(
        "Figure 9: ROC curves (95/5 imbalanced)\n"
        "Both models look acceptable here",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)

    # -- Panel 2: PR curves on the imbalanced dataset ------------------------
    ax = axes[1]
    # The random baseline for a PR curve is a horizontal line at the
    # positive class prevalence: a random classifier achieves precision
    # equal to the fraction of positives in the dataset at any recall level.
    ax.axhline(
        pos_rate_imb, color="black", lw=1.5, linestyle=":",
        label=f"Random baseline (precision = {pos_rate_imb:.2f})",
    )
    for sc_m, name, col in [(sc_A, "Model A", COL_A), (sc_B, "Model B", COL_B)]:
        prec, rec, _ = precision_recall_curve(y_imb, sc_m)
        ap = average_precision_score(y_imb, sc_m)
        ax.plot(rec, prec, color=col, lw=2.5, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Figure 10: PR curves (95/5 imbalanced)\n"
        "The real performance gap is now visible",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)

    # -- Panel 3: PR curves on the balanced synthetic dataset ----------------
    ax = axes[2]
    ax.axhline(
        pos_rate_bal, color="black", lw=1.2, linestyle=":",
        label=f"Random baseline (precision = {pos_rate_bal:.2f})",
    )
    for name, scores, col in bal_results:
        prec, rec, _ = precision_recall_curve(y_bal, scores)
        ap = average_precision_score(y_bal, scores)
        ax.plot(rec, prec, color=col, lw=2.2, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Figure 11: PR curves (balanced dataset)\n"
        "Curves stay high = good precision at high recall",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()