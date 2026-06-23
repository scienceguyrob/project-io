"""
Figure 12 — Interactive Confusion Matrix Explorer
=======================================================================
Provides an interactive confusion matrix for a binary classification
problem, with four sliders controlling the four cells of the matrix:

  TP (true positives)  - actual positive, predicted positive
  FN (false negatives) - actual positive, predicted negative
  FP (false positives) - actual negative, predicted positive
  TN (true negatives)  - actual negative, predicted negative

As the four counts are changed, the confusion matrix on the left updates,
and the annotation panel on the right recalculates and displays:

  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)

This is intended as a hands-on way to build intuition for how these
metrics respond to changes in the underlying counts, for example, seeing
how precision and recall can move in opposite directions, or how accuracy
can stay high even when one class is poorly handled.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_12 import show
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


# ── Slider range ─────────────────────────────────────────────────────────────
# All four counts share the same range and step, so that any combination
# from "all zero" up to a maximum of 100 per cell can be explored. Defaults
# are chosen to represent a moderately imbalanced problem with a reasonable
# number of errors in both directions, a useful starting point for
# exploration rather than a "perfect" classifier.
MAX_COUNT = 100
DEFAULTS = {'tp': 50, 'fn': 10, 'fp': 10, 'tn': 30}


def _safe_divide(numerator, denominator):
    """Return numerator / denominator, or None if the denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def show():
    """Render Figure 12: an interactive confusion matrix with live accuracy, precision, and recall."""
    plt.close('Notebook10 Figure 12')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook10 Figure 12',
        figsize=(10, 6.5),
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
        'Figure 12: explore how the confusion matrix drives\n'
        'accuracy, precision, and recall',
        fontsize=11, y=0.99,
    )
    plt.subplots_adjust(wspace=0.15, top=0.84)

    def _draw(tp, fn, fp, tn):
        ax_plot.clear()

        total = tp + fn + fp + tn

        # The matrix is laid out with rows as the ACTUAL class and columns
        # as the PREDICTED class, the standard orientation:
        #
        #                 Predicted Pos   Predicted Neg
        #   Actual Pos         TP              FN
        #   Actual Neg         FP              TN
        matrix = np.array([[tp, fn], [fp, tn]])

        # imshow's colour scale is based on the matrix values directly, so
        # larger counts are shown as darker cells, giving an immediate
        # visual sense of where most of the predictions fall.
        ax_plot.imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.max(), 1))

        labels = np.array([['TP', 'FN'], ['FP', 'TN']])
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                # Text colour is chosen based on the cell's value relative
                # to the current maximum, so labels remain readable on both
                # dark (high-value) and light (low-value) cells.
                text_colour = 'white' if value > matrix.max() * 0.5 else 'black'
                ax_plot.text(
                    j, i, f'{labels[i, j]}\n{value}',
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color=text_colour,
                )

        ax_plot.set_xticks([0, 1])
        ax_plot.set_yticks([0, 1])
        ax_plot.set_xticklabels(['Predicted Positive', 'Predicted Negative'], fontsize=9)
        ax_plot.set_yticklabels(['Actual Positive', 'Actual Negative'], fontsize=9)
        ax_plot.set_title(f'Confusion matrix (total n = {total})', fontsize=10)

        # ── Metrics ───────────────────────────────────────────────────────────
        accuracy = _safe_divide(tp + tn, total)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)

        def fmt(value):
            # Each metric's denominator can be zero, for example, precision
            # is undefined if there are no positive predictions at all
            # (TP = FP = 0). Rather than letting this raise a
            # ZeroDivisionError, _safe_divide returns None, and this
            # formats that case explicitly as "undefined" rather than
            # crashing or silently showing a misleading 0.0%.
            return f'{value:.1%}' if value is not None else 'undefined (denominator = 0)'

        ann_text.set_text(
            'Confusion matrix counts\n'
            '─────────────────────────────\n\n'
            f'TP (true positive)  = {tp}\n'
            f'FN (false negative) = {fn}\n'
            f'FP (false positive) = {fp}\n'
            f'TN (true negative)  = {tn}\n\n'
            f'Total n = {total}\n\n'
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
    # One slider per confusion matrix cell. description_width is set wide
    # enough that the four-character labels (TP, FN, FP, TN) line up neatly.
    slider_style = {'description_width': '40px'}
    slider_layout = widgets.Layout(width='300px')

    slider_tp = widgets.IntSlider(
        value=DEFAULTS['tp'], min=0, max=MAX_COUNT, step=1,
        description='TP', style=slider_style, layout=slider_layout,
    )
    slider_fn = widgets.IntSlider(
        value=DEFAULTS['fn'], min=0, max=MAX_COUNT, step=1,
        description='FN', style=slider_style, layout=slider_layout,
    )
    slider_fp = widgets.IntSlider(
        value=DEFAULTS['fp'], min=0, max=MAX_COUNT, step=1,
        description='FP', style=slider_style, layout=slider_layout,
    )
    slider_tn = widgets.IntSlider(
        value=DEFAULTS['tn'], min=0, max=MAX_COUNT, step=1,
        description='TN', style=slider_style, layout=slider_layout,
    )

    out = interactive_output(
        _draw,
        {'tp': slider_tp, 'fn': slider_fn, 'fp': slider_fp, 'tn': slider_tn},
    )

    # Sliders are grouped two-by-two, mirroring the layout of the confusion
    # matrix itself (TP/FN on one row, FP/TN on the next), so the controls
    # visually correspond to the cells they affect.
    controls = widgets.VBox([
        widgets.HBox([slider_tp, slider_fn]),
        widgets.HBox([slider_fp, slider_tn]),
    ])

    display(controls, out)