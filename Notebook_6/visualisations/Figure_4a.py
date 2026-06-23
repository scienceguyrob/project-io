"""
Figure 4a — Decision Stump Structure Diagram
============================================
A static diagram showing the anatomy of a decision stump:
  - A single root node asking a threshold question on one feature
  - Two branches: Yes (>= theta) and No (< theta)
  - Two leaf nodes showing the predicted class for each branch

This figure is purely illustrative — it uses the fruit classification
example (apple vs orange by diameter) to label the nodes concretely,
but does not plot any data.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_4a import show
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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colours ──────────────────────────────────────────────────────────────────
COL_NODE   = '#dce8f5'
COL_BORDER = '#4a6fa5'
COL_APPLE  = '#cce5ff'
COL_ORANGE = '#ffe0cc'
COL_TEXT   = '#1a1a2e'
COL_BRANCH = '#555555'


def _box(ax, x, y, w, h, fc, ec, text_lines, fontsizes):
    """
    Draw a rounded rectangle at (x, y) with width w and height h,
    then place one text string per line centred inside it.

    Parameters
    ----------
    text_lines : list of str
    fontsizes  : list of float, one per line
    """
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=fc, edgecolor=ec,
        linewidth=2.0, zorder=3,
    )
    ax.add_patch(patch)

    # Distribute lines evenly inside the box
    n = len(text_lines)
    for i, (line, fs) in enumerate(zip(text_lines, fontsizes)):
        # offset each line symmetrically around the centre
        offset = (i - (n - 1) / 2) * (h / (n + 0.5))
        ax.text(x, y - offset, line,
                ha='center', va='center',
                fontsize=fs, color=COL_TEXT,
                fontweight='bold' if i == 0 else 'normal',
                zorder=4)


def _arrow(ax, x0, y0, x1, y1, label, label_side='left'):
    """
    Draw an arrow from (x0,y0) to (x1,y1) with a branch label.
    label_side controls whether the label sits left or right of the arrow.
    """
    ax.annotate(
        '', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle='->', color=COL_BRANCH,
            lw=2.0, connectionstyle='arc3,rad=0.0',
        ),
        zorder=2,
    )
    # Place label at 30% along the arrow, offset sideways so it
    # sits beside the line rather than directly on top of it.
    mx = x0 + 0.30 * (x1 - x0)
    my = y0 + 0.30 * (y1 - y0)
    x_offset = -0.55 if label_side == "left" else 0.55
    ax.text(mx + x_offset, my, label,
            ha="center", va="center",
            fontsize=10, color=COL_BRANCH,
            style="italic", zorder=4)


def show():
    """
    Render Figure 4a: Decision Stump Structure Diagram.
    """
    plt.close('Notebook6 Figure 4a')
    fig, ax = plt.subplots(num='Notebook6 Figure 4a', figsize=(10, 5))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # ── Root node ────────────────────────────────────────────────────────────
    # The root node holds the single threshold question.
    _box(ax, x=5.0, y=4.8, w=3.6, h=1.2,
         fc=COL_NODE, ec=COL_BORDER,
         text_lines=['Feature: Diameter (cm)',
                     'Diameter >= θ ?'],
         fontsizes=[10, 11])

    # ── Branches ─────────────────────────────────────────────────────────────
    # Left branch — condition is False (diameter < theta) → predict Apple
    _arrow(ax, x0=3.8, y0=4.2, x1=2.5, y1=2.8,
           label='No  (< \u03b8)', label_side='left')

    # Right branch — condition is True (diameter >= theta) → predict Orange
    _arrow(ax, x0=6.2, y0=4.2, x1=7.5, y1=2.8,
           label='Yes  (\u2265 \u03b8)', label_side='right')

    # ── Left leaf — predict Apple ─────────────────────────────────────────────
    _box(ax, x=2.2, y=2.2, w=3.2, h=1.4,
         fc=COL_APPLE, ec=COL_BORDER,
         text_lines=['Predict: Apple',
                     'diameter < theta',
                     'most points here are apples'],
         fontsizes=[11, 9, 8])

    # ── Right leaf — predict Orange ───────────────────────────────────────────
    _box(ax, x=7.8, y=2.2, w=3.2, h=1.4,
         fc=COL_ORANGE, ec=COL_BORDER,
         text_lines=['Predict: Orange',
                     'diameter >= theta',
                     'most points here are oranges'],
         fontsizes=[11, 9, 8])

    # ── Annotation labels explaining each part ────────────────────────────────
    # Root label
    ax.text(5.0, 5.75,
            'Root node — one threshold question on one feature',
            ha='center', va='center', fontsize=9,
            color='#4a6fa5', style='italic', zorder=4)

    # Leaf labels
    ax.text(2.2, 0.9,
            'Leaf node — final prediction',
            ha='center', va='center', fontsize=8.5,
            color='#4a6fa5', style='italic', zorder=4)
    ax.text(7.8, 0.9,
            'Leaf node — final prediction',
            ha='center', va='center', fontsize=8.5,
            color='#4a6fa5', style='italic', zorder=4)

    # theta annotation below the root
    ax.text(5.0, 3.55,
            r'$\theta$ = the learned threshold value (e.g. 7.5 cm)',
            ha='center', va='center', fontsize=9,
            color=COL_BRANCH, zorder=4)

    ax.set_title(
        'Figure 4a: Anatomy of a Decision Stump\n'
        'A single threshold question on one feature produces two leaf predictions',
        fontsize=11,
    )

    plt.tight_layout()
    plt.show()