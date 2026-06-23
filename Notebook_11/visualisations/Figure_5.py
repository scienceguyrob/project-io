"""
Figure 5 — Recall, Specificity, and FPR vs Decision Threshold, and the ROC Curve
=======================================================================
A two-panel figure illustrating how recall (TPR), specificity (TNR), and
the false positive rate (FPR) change as the decision threshold is swept,
and how the relationship between FPR and TPR traces out the ROC curve.

This figure does not build its own classifier. It receives the already-
fitted classifier and test data from the notebook cell in Section 4.1.
Keeping the classifier outside the figure ensures that all rates and the
ROC curve shown here are derived from exactly the same model and test set
discussed in the surrounding notebook text.

show() expects three arguments:

  clf     - a fitted scikit-learn classifier that supports predict_proba,
            i.e. the clf object from Section 4.1
  X_te_s  - the standardised test feature matrix, i.e. X_te_s from Section 4.1
  y_te    - the true test labels, i.e. y_te from Section 4.1

The figure contains two panels:

  Left panel  — Recall (TPR), Specificity (TNR), and FPR plotted against
                the decision threshold. Shows that TPR and FPR move
                together as the threshold changes: lowering the threshold
                raises recall but also raises the false positive rate, and
                raising it does the reverse. Specificity and FPR are
                mirror images of each other (they always sum to 1.0).

  Right panel — The ROC curve, plotting TPR against FPR across all
                thresholds. A random classifier produces the diagonal
                dashed line; a perfect classifier would reach the top-left
                corner (FPR=0, TPR=1). The ROC curve makes the FPR-TPR
                trade-off visible as a single curve rather than two
                separate threshold plots.

Usage
-----
In a Jupyter notebook cell (clf, X_te_s, and y_te must already exist from
the Section 4.1 cell):

    %matplotlib widget
    from visualisations.Figure_5 import show
    show(clf, X_te_s, y_te)

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
from sklearn.metrics import confusion_matrix


# Number of evenly spaced thresholds to evaluate across [0.01, 0.99].
# 200 points gives smooth curves without being computationally heavy.
N_THRESHOLDS = 200

# Colours consistent with the notebook's palette throughout.
COLOUR_TPR  = 'steelblue'
COLOUR_TNR  = 'seagreen'
COLOUR_FPR  = 'tomato'
COLOUR_GRID = '#cccccc'


def _rates_at_threshold(y_true, y_pred):
    """
    Compute TPR, TNR, and FPR from binary predictions.

    Returns (tpr, tnr, fpr), each in [0, 1]. Returns 0 for any rate whose
    denominator is zero, which can occur at extreme threshold values where
    the classifier predicts only one class for every instance.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, tnr, fpr


def show(clf, X_te_s, y_te):
    """
    Render Figure 5: TPR, TNR, FPR vs threshold and the ROC curve.

    Parameters
    ----------
    clf : fitted scikit-learn classifier
        Must support predict_proba. Use the clf trained in Section 4.1.
    X_te_s : ndarray of shape (n_test, n_features)
        Standardised test feature matrix (X_te_s from Section 4.1).
    y_te : ndarray of shape (n_test,)
        True binary test labels (y_te from Section 4.1).
    """
    plt.close('Notebook11 Figure 5')

    # predict_proba returns (n_samples, n_classes); column 1 is P(positive).
    probs = clf.predict_proba(X_te_s)[:, 1]

    # Sweep the threshold and record TPR, TNR, FPR at each step.
    thresholds = np.linspace(0.01, 0.99, N_THRESHOLDS)
    tprs, tnrs, fprs = [], [], []

    for t in thresholds:
        y_pred_t = (probs >= t).astype(int)
        tpr, tnr, fpr = _rates_at_threshold(y_te, y_pred_t)
        tprs.append(tpr)
        tnrs.append(tnr)
        fprs.append(fpr)

    # ── Metrics at the default threshold ─────────────────────────────────────
    # Reported to console so the reader can cross-check with the Section 4.1
    # output before inspecting the figure.
    y_pred_default = (probs >= 0.5).astype(int)
    tpr_d, tnr_d, fpr_d = _rates_at_threshold(y_te, y_pred_default)

    print('Figure 5: specificity and FPR (Section 4.1 classifier at default threshold)')
    print(f'  Recall (TPR)        = {tpr_d:.4f}')
    print(f'  Specificity (TNR)   = {tnr_d:.4f}')
    print(f'  FPR                 = {fpr_d:.4f}')
    print(f'  TNR + FPR (= 1.0)   = {tnr_d + fpr_d:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), num='Notebook11 Figure 5')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # ── Left panel: all three rates vs threshold ──────────────────────────────
    # Plotting TPR, TNR, and FPR together on the same axes makes it easy to
    # see that TNR and FPR are mirror images (they always sum to 1.0), and
    # that TPR and FPR tend to move in the same direction as the threshold
    # changes, which is the core tension the ROC curve captures.
    ax = axes[0]
    ax.plot(thresholds, tprs, color=COLOUR_TPR, lw=2,   label='Recall (TPR)')
    ax.plot(thresholds, tnrs, color=COLOUR_TNR, lw=2,   label='Specificity (TNR)')
    ax.plot(thresholds, fprs, color=COLOUR_FPR, lw=2,   label='FPR', linestyle='--')
    ax.axvline(0.5, color='#555555', lw=1.5, linestyle='--',
               label='Default threshold (0.5)')
    ax.set_xlabel('Decision threshold (probability cutoff)')
    ax.set_ylabel('Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        'How each rate changes with the decision threshold\n'
        'TNR and FPR are mirror images: they always sum to 1.0',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, color=COLOUR_GRID, alpha=0.5)

    # ── Right panel: ROC curve ────────────────────────────────────────────────
    # The ROC curve is simply the (FPR, TPR) pairs from the left panel
    # replotted against each other, with the threshold now implicit rather
    # than shown on an axis. The diagonal dashed line is the expected
    # performance of a classifier that ignores the features entirely and
    # assigns labels at random: any useful classifier should sit above it.
    ax = axes[1]
    ax.plot(fprs, tprs, color=COLOUR_TPR, lw=2.5, label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random classifier')
    ax.set_xlabel('FPR (1 − Specificity)')
    ax.set_ylabel('TPR (Recall)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        'ROC curve: FPR vs TPR across all thresholds\n'
        'upper-left corner is ideal (TPR=1, FPR=0)',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, color=COLOUR_GRID, alpha=0.5)

    fig.suptitle(
        'Figure 5: Recall, Specificity, and FPR vs decision threshold',
        fontsize=11,
    )
    plt.tight_layout()