"""
Figure 2 -- Interactive ROC Curve with Threshold Slider
=======================================================
Demonstrates how sweeping the decision threshold traces the ROC curve.

A synthetic dataset of 10,000 examples is generated with a realistic
score distribution: positive-class scores drawn from a higher-mean
Gaussian, negative-class scores from a lower-mean Gaussian. This gives
a smooth, well-shaped ROC curve that a logistic regression model on
a real dataset would typically produce.

The user controls a single threshold slider. As it moves:
  - The full ROC curve is drawn from all threshold values across the
    score range. This is the real curve, not an approximation.
  - A red dot marks the current threshold's operating point on the curve.
  - The annotation panel shows the confusion matrix counts (TP, FP, TN, FN)
    and derived metrics (TPR, FPR, precision, accuracy, AUC) for the
    current threshold, assuming the fixed 10,000-example dataset.

The AUC is computed once from the full score distribution and does not
change as the threshold moves, because AUC is threshold-free.

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
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display
from sklearn.metrics import roc_curve, roc_auc_score

# -- Dataset parameters -------------------------------------------------------
# 10,000 examples split evenly between positive and negative classes.
# Scores for each class are drawn from Gaussians with different means,
# producing a realistic overlap region where threshold choice matters.
N_TOTAL        = 10_000
N_POS          = 5_000
N_NEG          = 5_000

SCORE_MEAN_POS = 0.70   # Positive-class scores cluster around 0.7
SCORE_MEAN_NEG = 0.30   # Negative-class scores cluster around 0.3
SCORE_STD      = 0.15   # Standard deviation; controls overlap between classes

DEFAULT_THRESHOLD = 0.50

# -- Generate synthetic scores once at module load ----------------------------
# Fixed seed so the curve is identical every time the figure is opened.
rng = np.random.default_rng(42)

scores_pos = np.clip(rng.normal(loc=SCORE_MEAN_POS, scale=SCORE_STD, size=N_POS), 0.0, 1.0)
scores_neg = np.clip(rng.normal(loc=SCORE_MEAN_NEG, scale=SCORE_STD, size=N_NEG), 0.0, 1.0)

SCORES = np.concatenate([scores_pos, scores_neg])
LABELS = np.concatenate([np.ones(N_POS), np.zeros(N_NEG)])

# -- Pre-compute the full ROC curve and AUC -----------------------------------
# roc_curve() sweeps every unique score value as a threshold and returns
# the (FPR, TPR) pair at each one. Computed once; the slider only moves
# the operating point marker, it does not recompute the curve.
FPR_CURVE, TPR_CURVE, THRESHOLDS = roc_curve(LABELS, SCORES)
AUC = roc_auc_score(LABELS, SCORES)


def _metrics_at_threshold(threshold):
    """
    Compute confusion matrix counts and derived metrics for a given threshold
    applied to the fixed 10,000-example synthetic dataset.
    """
    pred_pos = SCORES >= threshold

    tp = int(( pred_pos & (LABELS == 1)).sum())
    fp = int(( pred_pos & (LABELS == 0)).sum())
    tn = int((~pred_pos & (LABELS == 0)).sum())
    fn = int((~pred_pos & (LABELS == 1)).sum())

    # Guard against zero denominators at extreme thresholds where one
    # class may have zero predictions.
    tpr       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy  = (tp + tn) / N_TOTAL

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'tpr': tpr, 'fpr': fpr,
        'precision': precision, 'accuracy': accuracy,
    }


def _annotation_text(threshold, m):
    """Build the metrics panel string for the current threshold."""
    sep = "\u2500" * 33
    return (
        "Threshold = {:.2f}\n"
        "{}\n\n"
        "Confusion matrix  (n = 10,000)\n\n"
        "  TP = {:<6,}  FN = {:<6,}\n"
        "  FP = {:<6,}  TN = {:<6,}\n\n"
        "{}\n\n"
        "Derived metrics\n\n"
        "  TPR (Recall)  = {:.3f}\n"
        "  FPR           = {:.3f}\n"
        "  Precision     = {:.3f}\n"
        "  Accuracy      = {:.3f}\n\n"
        "{}\n\n"
        "  AUC           = {:.3f}\n\n"
        "(AUC does not change with\n"
        " threshold -- it summarises\n"
        " the full curve.)"
    ).format(
        threshold, sep,
        m['tp'], m['fn'],
        m['fp'], m['tn'],
        sep,
        m['tpr'], m['fpr'], m['precision'], m['accuracy'],
        sep,
        AUC,
    )


def show():
    """Render Figure 2: interactive ROC curve with threshold slider."""
    plt.close('Notebook12 Figure 2')

    fig, (ax_roc, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook12 Figure 2',
        figsize=(10, 5.5),
        gridspec_kw={'width_ratios': [1.2, 1]},
    )
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_visible  = True
    fig.canvas.toolbar_position = 'right'

    ax_ann.set_axis_off()

    ann_text = ax_ann.text(
        0.04, 0.97, '',
        transform=ax_ann.transAxes,
        fontsize=9, va='top', ha='left',
        family='monospace',
        linespacing=1.7,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    # Draw the full ROC curve once. It does not change with the threshold;
    # only the operating point marker moves.
    ax_roc.plot(FPR_CURVE, TPR_CURVE, color='steelblue', lw=2.5,
                label=f'ROC curve (AUC = {AUC:.3f})')
    ax_roc.fill_between(FPR_CURVE, TPR_CURVE, alpha=0.10, color='steelblue')
    ax_roc.plot([0, 1], [0, 1], color='#888888', lw=1.4, ls='--', alpha=0.6,
                label='Random classifier (AUC = 0.50)')

    ax_roc.set_xlabel('FPR  (False Positive Rate)  =  1 − Specificity', fontsize=10)
    ax_roc.set_ylabel('TPR  (True Positive Rate)  =  Recall', fontsize=10)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(True, alpha=0.25)

    fig.suptitle(
        'Figure 2: Moving the threshold moves the operating point along the ROC curve',
        fontsize=11, y=0.99,
    )
    plt.subplots_adjust(wspace=0.05, top=0.91)

    # The operating point is created as a line artist and updated in place,
    # avoiding a full ax.clear() on every slider change.
    op_point, = ax_roc.plot([], [], 'o', color='tomato', markersize=10,
                             markeredgecolor='k', markeredgewidth=0.8,
                             zorder=6, label='Current threshold')
    ax_roc.legend(fontsize=8, loc='lower right')

    def _draw(threshold):
        m = _metrics_at_threshold(threshold)
        op_point.set_data([m['fpr']], [m['tpr']])
        ann_text.set_text(_annotation_text(threshold, m))
        fig.canvas.draw_idle()

    slider_thresh = widgets.FloatSlider(
        value=DEFAULT_THRESHOLD,
        min=0.01, max=0.99, step=0.01,
        description='Threshold',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='400px'),
        readout_format='.2f',
    )

    out      = interactive_output(_draw, {'threshold': slider_thresh})
    controls = widgets.VBox([slider_thresh])
    display(controls, out)