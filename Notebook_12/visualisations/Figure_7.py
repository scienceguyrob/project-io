"""
Figure 7 -- ROC Curve Weaknesses: AUC on Imbalanced Data and Crossing Curves
=============================================================================
Demonstrates two situations in which AUC can be misleading:

  1. Class imbalance. A 95/5 negative/positive split is used. Two synthetic
     models (A and B) are constructed with similar global AUC values, but
     their recall for the rare positive class differs substantially at the
     default threshold of 0.5. The ROC curve does not make this difference
     visible; the PR curve (Section 8) does.

  2. Crossing curves. When the two ROC curves cross, global AUC ranking
     reverses depending on which operating region you care about. The model
     with the lower global AUC may outperform the other in the low-FPR
     region that high-specificity applications require.

The left panel shows the full ROC plot for both models. The right panel
zooms into the low-FPR region (0 to 0.10) and adds a vertical line at
FPR = 0.05 to highlight the crossing point and the ranking reversal.

A confusion matrix summary at threshold = 0.5 is printed to stdout when
show() is called, so the recall difference is visible numerically alongside
the plot.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_7 import show
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
from sklearn.metrics import roc_curve, roc_auc_score

# -- Dataset parameters -------------------------------------------------------
# 95/5 class split to simulate a realistic imbalanced detection problem.
N_NEG = 950
N_POS = 50

# Threshold at which confusion matrix counts are evaluated.
EVAL_THRESHOLD = 0.5

# Zoom region for the right panel.
ZOOM_FPR_MAX = 0.10
ZOOM_TPR_MAX = 0.80

# FPR reference line drawn on the zoomed panel to mark the crossing point.
MAX_FPR_REFERENCE = 0.05

# -- Colours ------------------------------------------------------------------
COL_A     = "steelblue"
COL_B     = "tomato"
COL_VLINE = "black"


def _make_data():
    """
    Generate synthetic score distributions for Model A and Model B.

    Model A: reasonable separation. Negatives score medium-low (beta(2,5));
             positives score higher (beta(5,2)). Decent recall but some FP.

    Model B: negatives score very low (beta(1,8)), making FPR small at most
             thresholds. Positives score only medium (beta(3,3)), so recall
             is poor at the default threshold despite the low FPR.

    Both models produce similar global AUC values, which is the point:
    the ROC curve cannot distinguish them in overall terms, but their
    behaviour at any fixed threshold is meaningfully different.
    """
    rng = np.random.default_rng(17)

    y = np.array([0] * N_NEG + [1] * N_POS)

    sc_A = np.concatenate([
        rng.beta(2, 5, N_NEG),   # Negatives: medium-low scores
        rng.beta(5, 2, N_POS),   # Positives: higher scores
    ])
    sc_B = np.concatenate([
        rng.beta(1, 8, N_NEG),   # Negatives: very low scores
        rng.beta(3, 3, N_POS),   # Positives: medium scores, poor separation
    ])

    return y, sc_A, sc_B


def _confusion_summary(y, scores, name):
    """
    Compute and return a formatted confusion matrix summary string for a
    single model at the fixed evaluation threshold.
    """
    yp = (scores >= EVAL_THRESHOLD).astype(int)

    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    auc    = roc_auc_score(y, scores)

    return f"  {name:<10}  {auc:>6.3f}  {tp:>4}  {fp:>4}  {fn:>4}  {tn:>4}  {recall:>8.2f}"


def show():
    """Render Figure 7: ROC weaknesses on imbalanced data with crossing curves."""
    plt.close("Notebook12 Figure 7")

    y, sc_A, sc_B = _make_data()

    # Print confusion matrix summary to stdout so the recall difference is
    # visible numerically alongside the plot.
    header = f"  {'Model':<10}  {'AUC':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  {'Recall':>8}"
    print(header)
    print("  " + "-" * 55)
    print(_confusion_summary(y, sc_A, "Model A"))
    print(_confusion_summary(y, sc_B, "Model B"))

    fpr_A, tpr_A, _ = roc_curve(y, sc_A)
    fpr_B, tpr_B, _ = roc_curve(y, sc_B)
    auc_A = roc_auc_score(y, sc_A)
    auc_B = roc_auc_score(y, sc_B)

    fig, axes = plt.subplots(
        1, 2,
        num="Notebook12 Figure 7",
        figsize=(10, 5),
    )
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_visible  = True
    fig.canvas.toolbar_position = "right"

    # -- Left panel: full ROC plot --------------------------------------------
    ax = axes[0]
    ax.plot(fpr_A, tpr_A, color=COL_A, lw=2.5, label=f"Model A (AUC = {auc_A:.3f})")
    ax.plot(fpr_B, tpr_B, color=COL_B, lw=2.5, label=f"Model B (AUC = {auc_B:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4, label="Random (AUC = 0.50)")
    ax.set_xlabel("FPR  (False Positive Rate)")
    ax.set_ylabel("TPR  (True Positive Rate)")
    ax.set_title(
        "Figure 7: Both models have similar AUC\n"
        "but recall at threshold = 0.5 differs substantially",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)

    # -- Right panel: zoomed into the low-FPR region --------------------------
    ax = axes[1]
    ax.plot(fpr_A, tpr_A, color=COL_A, lw=2.5, label=f"Model A (AUC = {auc_A:.3f})")
    ax.plot(fpr_B, tpr_B, color=COL_B, lw=2.5, label=f"Model B (AUC = {auc_B:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4)

    # Vertical reference line marks the maximum acceptable FPR for a
    # high-specificity application, to show where the crossing matters.
    ax.axvline(
        MAX_FPR_REFERENCE, color=COL_VLINE, lw=1.5, linestyle=":",
        label=f"Max FPR = {MAX_FPR_REFERENCE}",
    )
    ax.set_xlim(0, ZOOM_FPR_MAX)
    ax.set_ylim(0, ZOOM_TPR_MAX)
    ax.set_xlabel("FPR (zoomed)")
    ax.set_ylabel("TPR  (True Positive Rate)")
    ax.set_title(
        "Figure 8: Zoomed — curves cross at low FPR\n"
        "Model with lower global AUC wins in this region",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Figures 7-8: ROC weaknesses — similar AUC can hide very different behaviour",
        fontsize=11, y=0.98,
    )
    plt.tight_layout()