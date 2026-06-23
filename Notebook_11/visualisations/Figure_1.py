"""
Figure 1 — Accuracy, Precision, and Recall on an Imbalanced Dataset
=======================================================================
Provides an interactive confusion matrix for a binary classification
problem. Four sliders control the dataset and the classifier's behaviour:

  n              - the total number of samples in the dataset (1 to
                   1,000,000, on a log scale)
  % positive     - the proportion of the dataset belonging to the positive
                   (minority) class, controlling how imbalanced the
                   dataset is
  Recall (TP rate)  - the proportion of actual positives that the
                   classifier correctly identifies, directly controlling TP
                   and FN
  FP rate        - the proportion of actual negatives that the classifier
                   incorrectly flags as positive, directly controlling FP
                   and TN

From these, the four confusion matrix counts are derived as:

  n_pos = round(% positive * n)
  n_neg = n - n_pos
  TP    = round(Recall * n_pos),   FN = n_pos - TP
  FP    = round(FP rate * n_neg),  TN = n_neg - FP

The confusion matrix on the left updates accordingly, and the annotation
panel on the right recalculates and displays:

  Accuracy  = (TP + TN) / n
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)

This is intended to make the recap point from Section 4 concrete: setting
% positive low and Recall/FP rate to 0 reproduces a model that always
predicts the majority class, which can show a very high accuracy while its
recall is 0% and its precision is undefined, because it never identifies a
single positive case. Moving the Recall and FP rate sliders away from these
extremes shows how a more "useful" classifier trades off these metrics, and
increasing n shows that the same patterns hold from a handful of samples up
to a million.

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
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display


# ── Slider ranges ───────────────────────────────────────────────────────────
# n is controlled on a log scale so that the same slider can sensibly cover
# everything from a tiny illustrative dataset (n = 1) up to a dataset large
# enough to feel "realistic" (n = 1,000,000). N_LOG_MIN/MAX are exponents of
# 10, i.e. the slider ranges over 10**N_LOG_MIN to 10**N_LOG_MAX.
N_LOG_MIN = 0
N_LOG_MAX = 6
N_DEFAULT = 1_000

# Proportion of the dataset belonging to the positive (minority) class.
# The default of 0.1 represents a moderately imbalanced dataset, similar to
# the kind of class imbalance discussed in the recap.
POSITIVE_FRACTION_MIN = 0.01
POSITIVE_FRACTION_MAX = 0.50
POSITIVE_FRACTION_DEFAULT = 0.10

# Recall (TP rate): the fraction of actual positives the classifier catches.
# Setting this to 0 means the classifier never predicts positive for a true
# positive case, contributing to the "always predict majority" scenario.
RECALL_MIN = 0.0
RECALL_MAX = 1.0
RECALL_DEFAULT = 0.80

# FP rate: the fraction of actual negatives the classifier incorrectly
# flags as positive. Setting this to 0 means the classifier never raises a
# false alarm, the other half of the "always predict majority" scenario.
FPR_MIN = 0.0
FPR_MAX = 1.0
FPR_DEFAULT = 0.05

# Cell text is drawn in white once a cell's value exceeds this fraction of
# the matrix's largest value, so labels stay readable on both dark
# (high-count) and light (low-count) cells, same approach as Notebook 10
# Figure 12.
HIGH_VALUE_TEXT_THRESHOLD = 0.5


def _safe_divide(numerator, denominator):
    """Return numerator / denominator, or None if the denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def show():
    """Render Figure 1: an interactive confusion matrix driven by sample size, class balance, recall, and false positive rate."""
    plt.close('Notebook11 Figure 1')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook11 Figure 1',
        figsize=(10, 8),
        gridspec_kw={'width_ratios': [1.2, 1]},
    )

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    ax_ann.set_axis_off()
    ann_text = ax_ann.text(
        0.04, 0.97, '',
        transform=ax_ann.transAxes,
        fontsize=10, va='top', ha='left',
        linespacing=1.8,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 1: explore how class imbalance, recall, and false positive\n'
        'rate combine to drive accuracy, precision, and recall',
        fontsize=11, y=0.99,
    )
    plt.subplots_adjust(wspace=0.15, top=0.84)

    def _draw(n, positive_fraction, recall_rate, fpr):
        ax_plot.clear()

        n = int(round(n))
        # Guard against the slider's lower end rounding down to zero, since
        # a dataset of size zero has no meaningful confusion matrix.
        n = max(n, 1)

        # positive_fraction splits the dataset into actual positives
        # (minority class) and actual negatives (majority class).
        n_pos = int(round(positive_fraction * n))
        n_pos = min(n_pos, n)
        n_neg = n - n_pos

        # recall_rate controls how many of the actual positives are
        # correctly caught (TP), with the remainder being missed (FN).
        tp = int(round(recall_rate * n_pos))
        tp = min(tp, n_pos)
        fn = n_pos - tp

        # fpr controls how many of the actual negatives are incorrectly
        # flagged as positive (FP), with the remainder correctly left alone
        # (TN).
        fp = int(round(fpr * n_neg))
        fp = min(fp, n_neg)
        tn = n_neg - fp

        # Same row/column orientation as Notebook 10 Figure 12: rows are
        # the ACTUAL class, columns are the PREDICTED class.
        #
        #                 Predicted Pos   Predicted Neg
        #   Actual Pos         TP              FN
        #   Actual Neg         FP              TN
        matrix = np.array([[tp, fn], [fp, tn]])

        # interpolation='nearest' avoids a faint grid-like antialiasing
        # artifact that imshow's default interpolation can introduce on
        # small (2x2) arrays, which otherwise makes each cell look like it
        # is split into four smaller boxes.
        ax_plot.imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.max(), 1), interpolation='nearest')
        ax_plot.grid(False)

        labels = np.array([['TP', 'FN'], ['FP', 'TN']])
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                text_colour = 'white' if value > matrix.max() * HIGH_VALUE_TEXT_THRESHOLD else 'black'
                ax_plot.text(
                    j, i, f'{labels[i, j]}\n{value:,}',
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color=text_colour,
                )

        ax_plot.set_xticks([0, 1])
        ax_plot.set_yticks([0, 1])
        ax_plot.set_xticklabels(['Predicted Positive', 'Predicted Negative'], fontsize=9)
        ax_plot.set_yticklabels(['Actual Positive\n(minority class)', 'Actual Negative\n(majority class)'], fontsize=9)
        ax_plot.set_title(f'Confusion matrix (n = {n:,})', fontsize=10)

        # ── Metrics ───────────────────────────────────────────────────────────
        accuracy = _safe_divide(tp + tn, n)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)

        def fmt(value):
            # Precision is undefined whenever TP + FP = 0 (the classifier
            # never predicts positive at all), and recall is undefined
            # whenever TP + FN = 0 (there are no actual positives in the
            # dataset). _safe_divide returns None in either case, and this
            # formats that explicitly as "undefined" rather than showing a
            # misleading 0.0% or raising a ZeroDivisionError.
            return f'{value:.1%}' if value is not None else 'undefined (denom = 0)'

        ann_text.set_text(
            'Dataset\n'
            '─────────────────────────────\n\n'
            f'Total samples (n)   = {n:,}\n'
            f'Actual positives    = {n_pos:,}\n'
            f'Actual negatives    = {n_neg:,}\n\n'
            '─────────────────────────────\n\n'
            'Confusion matrix counts\n\n'
            f'TP = {tp:,}\n'
            f'FN = {fn:,}\n'
            f'FP = {fp:,}\n'
            f'TN = {tn:,}\n\n'
            '─────────────────────────────\n\n'
            'Metrics\n\n'
            f'Accuracy  = (TP+TN)/n\n'
            f'          = {fmt(accuracy)}\n\n'
            f'Precision = TP/(TP+FP)\n'
            f'          = {fmt(precision)}\n\n'
            f'Recall    = TP/(TP+FN)\n'
            f'          = {fmt(recall)}'
        )

        fig.canvas.draw_idle()

    # ── Widgets ──────────────────────────────────────────────────────────────
    # FloatLogSlider is used for n so that the slider position is linear in
    # log10(n), giving useful resolution across the full 1 to 1,000,000
    # range, while the value passed to _draw is the actual sample count.
    slider_layout = widgets.Layout(width='320px')
    slider_style = {'description_width': '110px'}

    slider_n = widgets.FloatLogSlider(
        value=N_DEFAULT, base=10, min=N_LOG_MIN, max=N_LOG_MAX, step=0.01,
        description='n (samples)', readout_format='.0f',
        style=slider_style, layout=slider_layout,
    )
    slider_positive_fraction = widgets.FloatSlider(
        value=POSITIVE_FRACTION_DEFAULT, min=POSITIVE_FRACTION_MIN, max=POSITIVE_FRACTION_MAX, step=0.01,
        description='% positive', readout_format='.0%',
        style=slider_style, layout=slider_layout,
    )
    slider_recall = widgets.FloatSlider(
        value=RECALL_DEFAULT, min=RECALL_MIN, max=RECALL_MAX, step=0.01,
        description='Recall (TP rate)', readout_format='.0%',
        style=slider_style, layout=slider_layout,
    )
    slider_fpr = widgets.FloatSlider(
        value=FPR_DEFAULT, min=FPR_MIN, max=FPR_MAX, step=0.01,
        description='FP rate', readout_format='.0%',
        style=slider_style, layout=slider_layout,
    )

    out = interactive_output(
        _draw,
        {
            'n': slider_n,
            'positive_fraction': slider_positive_fraction,
            'recall_rate': slider_recall,
            'fpr': slider_fpr,
        },
    )

    controls = widgets.VBox([slider_n, slider_positive_fraction, slider_recall, slider_fpr])

    display(controls, out)