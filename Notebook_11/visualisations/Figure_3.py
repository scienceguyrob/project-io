"""
Figure 3 — Visualising the Precision–Recall Trade-off
=======================================================================
A two-panel figure that builds intuition for how precision and recall
relate to each other through the confusion matrix.

This figure does not build its own classifier. It receives the four
confusion matrix counts extracted from the classifier trained in Section
4.1 of the notebook. Keeping these values outside the figure ensures that
the numbers shown here are identical to those discussed in the surrounding
notebook text.

show() expects four arguments:

  TP - true positives  (correctly predicted positive)
  FP - false positives (predicted positive, was actually negative)
  FN - false negatives (predicted negative, was actually positive)
  TN - true negatives  (correctly predicted negative)

These are the TP, FP, FN, TN variables extracted from the confusion matrix
in Section 4.1.

The figure contains two panels:

  Left panel  — Venn diagram showing the relationship between the two
                denominators: recall's denominator is TP + FN (all actual
                positives, the blue circle), and precision's denominator is
                TP + FP (all predicted positives, the red circle). The
                overlap is TP. Annotated with the actual counts from the
                classifier so the diagram is grounded in real numbers rather
                than being purely schematic.

  Right panel — Three labelled operating points in precision-recall space,
                illustrating that choosing between high recall and high
                precision is a decision, not a property of the model itself.
                The point derived from the actual classifier at the default
                threshold is plotted alongside two schematic extremes.

Usage
-----
In a Jupyter notebook cell (TP, FP, FN, TN must already exist from the
Section 4.1 cell):

    %matplotlib widget
    from visualisations.Figure_3 import show
    show(TP, FP, FN, TN)

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
import matplotlib.patches as mpatches
from matplotlib.patches import Circle


# Venn diagram geometry: the two circles are offset horizontally so that
# their overlap region is wide enough to label clearly. The overlap size is
# purely cosmetic; it does not encode the actual TP count.
VENN_LEFT_CENTRE  = (0.35, 0.52)
VENN_RIGHT_CENTRE = (0.60, 0.52)
VENN_LEFT_RADIUS  = 0.26
VENN_RIGHT_RADIUS = 0.23

# Colours consistent with the notebook's palette throughout.
COLOUR_RECALL    = 'steelblue'
COLOUR_PRECISION = 'tomato'
COLOUR_GRID      = '#cccccc'

# Alpha for the Venn circle fills: light enough that the overlap region
# is visually distinct from either circle alone.
VENN_ALPHA = 0.35

# Schematic operating points for the right panel. These are illustrative
# extremes, not real model outputs; the actual classifier point is added
# dynamically from the passed-in counts.
# Each entry is (label, recall, precision, colour, ha) where ha controls
# which side of the dot the annotation anchors to: 'left' means the text
# starts to the right of the dot, 'right' means it ends to the left.
# The high-recall point sits near the right edge of the plot so 'right'
# keeps it inside; the high-precision point has plenty of room to the right.
OPERATING_POINTS = [
    ('High recall\nlow precision',  0.92, 0.30, COLOUR_RECALL,    'right'),
    ('High precision\nlow recall',  0.28, 0.95, COLOUR_PRECISION, 'left'),
]

# Pixel gap between a dot and its annotation text in the x direction.
ANNOTATION_X_PAD = 8


def _safe_divide(numerator, denominator):
    """Return numerator / denominator, or 0.0 if the denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def show(TP, FP, FN, TN):
    """
    Render Figure 3: precision-recall Venn diagram and operating points.

    Parameters
    ----------
    TP : int
        True positives from the Section 4.1 confusion matrix.
    FP : int
        False positives from the Section 4.1 confusion matrix.
    FN : int
        False negatives from the Section 4.1 confusion matrix.
    TN : int
        True negatives from the Section 4.1 confusion matrix.
    """
    plt.close('Notebook11 Figure 3')

    precision = _safe_divide(TP, TP + FP)
    recall    = _safe_divide(TP, TP + FN)

    print('Figure 3: precision and recall at the default threshold (Section 4.1 classifier)')
    print(f'  TP = {TP},  FP = {FP},  FN = {FN},  TN = {TN}')
    print(f'  Precision = TP / (TP + FP) = {TP} / {TP + FP} = {precision:.4f}')
    print(f'  Recall    = TP / (TP + FN) = {TP} / {TP + FN} = {recall:.4f}')

    # Extra bottom margin to accommodate the figure-level legend placed
    # below both panels, keeping the Venn diagram completely clear.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), num='Notebook11 Figure 3')
    fig.subplots_adjust(bottom=0.18)

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # ── Left panel: Venn diagram ──────────────────────────────────────────────
    # The blue circle represents all actual positives (TP + FN); its left
    # region contains the FN (missed positives) and its overlap with the red
    # circle contains the TP. The red circle represents all predicted positives
    # (TP + FP); its right region contains the FP (false alarms).
    ax = axes[0]

    patch_recall    = mpatches.Patch(color=COLOUR_RECALL,    alpha=0.5, label='Actual positives  (TP + FN)')
    patch_precision = mpatches.Patch(color=COLOUR_PRECISION, alpha=0.5, label='Predicted positives (TP + FP)')

    ax.add_patch(Circle(VENN_LEFT_CENTRE,  VENN_LEFT_RADIUS,  color=COLOUR_RECALL,    alpha=VENN_ALPHA))
    ax.add_patch(Circle(VENN_RIGHT_CENTRE, VENN_RIGHT_RADIUS, color=COLOUR_PRECISION, alpha=VENN_ALPHA))

    # Region labels: placed at the left of the blue circle (FN), in the
    # overlap (TP), and at the right of the red circle (FP).
    ax.text(0.20, 0.52, f'FN\n(missed)\nn={FN}',
            ha='center', va='center', fontsize=9, color='#1a5276')
    ax.text(0.48, 0.52, f'TP\n(caught)\nn={TP}',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(0.74, 0.52, f'FP\n(false alarm)\nn={FP}',
            ha='center', va='center', fontsize=9, color='#922b21')

    # Recall bracket: spans the full width of the blue circle (TP + FN).
    ax.annotate('', xy=(0.09, 0.22), xytext=(0.61, 0.22),
                arrowprops=dict(arrowstyle='<->', color=COLOUR_RECALL, lw=2))
    ax.text(0.35, 0.15,
            f'Recall = TP/(TP+FN) = {TP}/({TP}+{FN}) = {recall:.2f}',
            ha='center', fontsize=8, color=COLOUR_RECALL)

    # Precision bracket: spans the full width of the red circle (TP + FP).
    ax.annotate('', xy=(0.37, 0.84), xytext=(0.83, 0.84),
                arrowprops=dict(arrowstyle='<->', color=COLOUR_PRECISION, lw=2))
    ax.text(0.60, 0.90,
            f'Precision = TP/(TP+FP) = {TP}/({TP}+{FP}) = {precision:.2f}',
            ha='center', fontsize=8, color=COLOUR_PRECISION)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Recall vs Precision: what each denominator counts', fontsize=10)

    # ── Right panel: operating points in precision-recall space ───────────────
    # Two schematic extremes illustrate the ends of the trade-off, and the
    # actual classifier result at the default threshold is plotted as a third
    # point so the reader can see where this model sits in that space.
    ax = axes[1]

    for label, rec, prec, col, ha in OPERATING_POINTS:
        ax.scatter(rec, prec, s=180, color=col, edgecolors='k', lw=0.8, zorder=4)
        # When ha='right' the text is right-aligned and the x offset is
        # negative, placing the label to the left of the dot so it stays
        # inside the plot box. When ha='left' the text starts to the right
        # of the dot as normal.
        x_off = -ANNOTATION_X_PAD if ha == 'right' else ANNOTATION_X_PAD
        ax.annotate(label, (rec, prec), textcoords='offset points',
                    xytext=(x_off, 6), fontsize=9, color=col, ha=ha)

    # The actual classifier point is placed to the right of the dot. For a
    # heavily imbalanced dataset at the default threshold, recall is typically
    # very low, so the point sits near the left edge where there is room.
    ax.scatter(recall, precision, s=200, color='#555555', edgecolors='k',
               lw=0.8, zorder=5, marker='D')
    ax.annotate(
        f'This classifier\n(threshold=0.5)\nR={recall:.2f}, P={precision:.2f}',
        (recall, precision),
        textcoords='offset points', xytext=(ANNOTATION_X_PAD, -38),
        fontsize=9, color='#333333', ha='left',
    )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Three operating points in Precision–Recall space', fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, color=COLOUR_GRID, alpha=0.5)

    # Legend placed in the figure footer below both panels, so it does not
    # compete for space with any content inside either axes.
    fig.legend(
        handles=[patch_recall, patch_precision],
        loc='lower center',
        ncol=2,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.28, 0.01),
    )

    fig.suptitle(
        'Figure 3: Visualising the Precision–Recall trade-off',
        fontsize=12,
    )