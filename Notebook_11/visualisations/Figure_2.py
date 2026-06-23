"""
Figure 2 — Recall and the Precision–Recall Trade-off vs Decision Threshold
=======================================================================
Shows how recall (sensitivity / true positive rate) and precision change
as the decision threshold of a logistic regression classifier is swept
from near 0 to near 1.

This figure does not build its own classifier. It receives the already-fitted
classifier and test data from the notebook cell in Section 4.1, which trains
a logistic regression model on a synthetic imbalanced dataset (10% positive
class, 2,000 samples). Keeping the classifier outside the figure ensures that
the curves shown here are derived from exactly the same model and predictions
discussed in the surrounding notebook text, rather than a silently different
one trained internally.

show() expects three arguments:

  clf     - a fitted scikit-learn classifier that supports predict_proba,
            i.e. the clf object from Section 4.1
  X_te_s  - the standardised test feature matrix, i.e. X_te_s from Section 4.1
  y_te    - the true test labels, i.e. y_te from Section 4.1

The figure contains two panels:

  Left panel  — Recall vs decision threshold: shows that recall falls as
                the threshold rises, i.e. as the classifier becomes more
                conservative about predicting the positive class. A dot and
                annotation mark the recall value at the default threshold
                of 0.5.

  Right panel — Recall and Precision vs decision threshold together: shows
                the classic trade-off where increasing one metric tends to
                decrease the other, with a vertical dashed line marking the
                default threshold of 0.5.

Usage
-----
In a Jupyter notebook cell (clf, X_te_s, and y_te must already exist from
the Section 4.1 cell):

    %matplotlib widget
    from visualisations.Figure_2 import show
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
from sklearn.metrics import recall_score, precision_score


# Default decision threshold used by most classifiers out of the box.
DEFAULT_THRESHOLD = 0.5

# Number of evenly spaced thresholds to evaluate across [0.01, 0.99].
# 100 points gives smooth curves without being computationally heavy.
N_THRESHOLDS = 100

# Plot colours, kept consistent with the notebook's palette.
COLOUR_RECALL          = 'steelblue'
COLOUR_PRECISION       = 'tomato'
COLOUR_THRESHOLD_LINE  = '#555555'
COLOUR_GRID            = '#cccccc'


def show(clf, X_te_s, y_te):
    """
    Render Figure 2: recall and the precision-recall trade-off vs decision threshold.

    Parameters
    ----------
    clf : fitted scikit-learn classifier
        Must support predict_proba. Use the clf trained in Section 4.1.
    X_te_s : ndarray of shape (n_test, n_features)
        Standardised test feature matrix (X_te_s from Section 4.1).
    y_te : ndarray of shape (n_test,)
        True binary test labels (y_te from Section 4.1).
    """
    plt.close('Notebook11 Figure 2')

    # predict_proba returns (n_samples, n_classes); column 1 is P(positive).
    # Using the same clf and X_te_s as the rest of the notebook ensures the
    # curves are derived from the identical model discussed in the text.
    probs = clf.predict_proba(X_te_s)[:, 1]

    # Sweep the threshold from just above 0 to just below 1, converting
    # probability scores into hard labels at each step and computing recall
    # and precision so the trade-off curve becomes visible.
    thresholds = np.linspace(0.01, 0.99, N_THRESHOLDS)
    recalls    = []
    precisions = []

    for t in thresholds:
        y_pred_t = (probs >= t).astype(int)
        # zero_division=0 suppresses the warning that arises when no positive
        # predictions are made (all instances fall below the threshold),
        # returning 0 rather than raising a ZeroDivisionError.
        recalls.append(recall_score(y_te, y_pred_t, zero_division=0))
        precisions.append(precision_score(y_te, y_pred_t, zero_division=0))

    # ── Metrics at the default threshold ─────────────────────────────────────
    # Computed separately so they can be annotated on the plots with a dot
    # and label, giving the reader an anchor point at the threshold most
    # classifiers use by default.
    y_pred_default        = (probs >= DEFAULT_THRESHOLD).astype(int)
    recall_at_default     = recall_score(y_te, y_pred_default, zero_division=0)
    precision_at_default  = precision_score(y_te, y_pred_default, zero_division=0)

    # Derive test set counts from y_te so the console output always matches
    # what was actually passed in, rather than repeating constants.
    n_test     = len(y_te)
    n_pos_test = int(y_te.sum())

    print('Figure 2: test set used for threshold sweep')
    print(f'  Total test samples : {n_test:,}')
    print(f'  Actual positives   : {n_pos_test:,}  ({n_pos_test / n_test:.1%} of test set)')
    print(f'  Actual negatives   : {n_test - n_pos_test:,}')
    print(f'  Recall at default threshold ({DEFAULT_THRESHOLD})    : {recall_at_default:.4f}')
    print(f'  Precision at default threshold ({DEFAULT_THRESHOLD}) : {precision_at_default:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), num='Notebook11 Figure 2')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # ── Left panel: recall alone ──────────────────────────────────────────────
    # Isolating recall first makes the direction of the threshold effect clear
    # before the trade-off with precision is introduced in the right panel.
    ax = axes[0]
    ax.plot(thresholds, recalls, color=COLOUR_RECALL, lw=2.5, label='Recall')
    ax.axvline(
        DEFAULT_THRESHOLD, color=COLOUR_THRESHOLD_LINE, lw=1.5,
        linestyle='--', label=f'Default threshold ({DEFAULT_THRESHOLD})',
    )
    ax.scatter(
        [DEFAULT_THRESHOLD], [recall_at_default],
        color=COLOUR_RECALL, s=60, zorder=5,
    )
    ax.annotate(
        f'  Recall = {recall_at_default:.2f}\n  at threshold = {DEFAULT_THRESHOLD}',
        xy=(DEFAULT_THRESHOLD, recall_at_default),
        xytext=(DEFAULT_THRESHOLD + 0.05, recall_at_default),
        fontsize=9, color=COLOUR_RECALL,
    )
    ax.set_xlabel('Decision threshold (probability cutoff)')
    ax.set_ylabel('Recall')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        'Recall vs decision threshold\n'
        'lower threshold → more positives predicted → higher recall',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, color=COLOUR_GRID, alpha=0.5)

    # ── Right panel: recall and precision together ────────────────────────────
    # Showing both on the same axes makes the trade-off visible: the two
    # curves tend to move in opposite directions as the threshold changes.
    ax = axes[1]
    ax.plot(thresholds, recalls,    color=COLOUR_RECALL,    lw=2.5, label='Recall')
    ax.plot(thresholds, precisions, color=COLOUR_PRECISION, lw=2.5, label='Precision')
    ax.axvline(
        DEFAULT_THRESHOLD, color=COLOUR_THRESHOLD_LINE, lw=1.5,
        linestyle='--', label=f'Default threshold ({DEFAULT_THRESHOLD})',
    )
    ax.scatter(
        [DEFAULT_THRESHOLD, DEFAULT_THRESHOLD],
        [recall_at_default, precision_at_default],
        color=[COLOUR_RECALL, COLOUR_PRECISION], s=60, zorder=5,
    )
    ax.set_xlabel('Decision threshold (probability cutoff)')
    ax.set_ylabel('Score')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        'Recall vs Precision: the classic trade-off\n'
        'increasing one tends to decrease the other',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, color=COLOUR_GRID, alpha=0.5)

    fig.suptitle(
        'Figure 2: Recall and how it depends on the decision threshold',
        fontsize=12,
    )
    plt.tight_layout()