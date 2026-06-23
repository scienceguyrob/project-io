"""
Figure 7 — Interactive Jaccard Similarity Explorer
===================================================
Visualises Jaccard similarity between two binary feature vectors using
a Venn-diagram-style set display and a feature grid.

Two binary vectors A and B are shown as coloured grids — each cell
represents one feature, filled if present and empty if absent. Below the
grids, a Venn diagram shows the three regions that define the Jaccard
calculation:

  Intersection (A ∩ B)  — features present in both
  A only               — features present in A but not B
  B only               — features present in B but not A
  Neither              — features absent from both (not shown — this is
                         the key point: Jaccard ignores shared absences)

A slider controls the number of features each vector contains (i.e. the
number of 1s), and a second slider controls the overlap between them.
The annotation panel on the right shows the full Jaccard calculation with
current numeric values substituted at each step.

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display


# ── Fixed parameters ──────────────────────────────────────────────────────────
N_FEATURES = 20       # total number of binary features in each vector
GRID_COLS  = 10       # features displayed per row in the grid
GRID_ROWS  = N_FEATURES // GRID_COLS

# ── Colours ───────────────────────────────────────────────────────────────────
COL_A        = 'steelblue'
COL_B        = 'tomato'
COL_BOTH     = 'mediumpurple'   # intersection
COL_ABSENT   = '#e8e8e8'        # feature absent in this vector
COL_NEITHER  = '#f0f0f0'        # absent in both — deliberately muted


def _make_vectors(n_a, n_b, n_overlap):
    """
    Build two binary vectors of length N_FEATURES with controlled overlap.

    The first n_overlap features are set in both A and B (intersection).
    The next (n_a - n_overlap) features are set only in A.
    The next (n_b - n_overlap) features are set only in B.
    All remaining features are 0 in both vectors.

    This layout keeps the set membership unambiguous and directly readable
    from the grid display.
    """
    a = np.zeros(N_FEATURES, dtype=int)
    b = np.zeros(N_FEATURES, dtype=int)

    # Shared features (intersection)
    a[:n_overlap] = 1
    b[:n_overlap] = 1

    # A-only features
    a[n_overlap: n_a] = 1

    # B-only features
    b[n_a: n_a + (n_b - n_overlap)] = 1

    return a, b


def _jaccard(a, b):
    """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    intersection = int(np.sum((a == 1) & (b == 1)))
    union        = int(np.sum((a == 1) | (b == 1)))
    return intersection, union, (intersection / union if union > 0 else 0.0)


def _annotation(a, b):
    """Step-by-step Jaccard calculation string for the annotation panel."""
    intersection, union, j = _jaccard(a, b)
    a_only   = int(np.sum((a == 1) & (b == 0)))
    b_only   = int(np.sum((a == 0) & (b == 1)))
    neither  = int(np.sum((a == 0) & (b == 0)))

    return (
        "Jaccard similarity\n"
        "─────────────────────────────────\n\n"
        r"$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$"
        "\n\n"
        f"Features present in A:      {int(a.sum())}\n"
        f"Features present in B:      {int(b.sum())}\n\n"
        f"$|A \\cap B|$ (both):         {intersection}\n"
        f"A only:                     {a_only}\n"
        f"B only:                     {b_only}\n\n"
        f"$|A \\cup B|$ = {intersection} + {a_only} + {b_only} = {union}\n\n"
        f"Shared absences (neither):  {neither}\n"
        f"(not counted in $|A \\cup B|$)\n\n"
        "─────────────────────────────────\n\n"
        f"$J = \\frac{{{intersection}}}{{{union}}}$"
        f" $= \\mathbf{{{j:.3f}}}$\n\n"
        f"Jaccard distance\n"
        f"$d_J = 1 - J = \\mathbf{{{1 - j:.3f}}}$"
    )


def _draw_grid(ax, a, b):
    """
    Draw the feature grid on ax.
    Each cell is coloured by membership:
      purple  — present in both (intersection)
      blue    — present in A only
      red     — present in B only
      light   — absent from both
    """
    ax.clear()
    ax.set_axis_off()

    cell_w = 1.0
    cell_h = 0.8
    pad    = 0.06

    for idx in range(N_FEATURES):
        row = idx // GRID_COLS
        col = idx  % GRID_COLS
        x   = col * (cell_w + pad)
        y   = -(row * (cell_h + pad))

        in_a = bool(a[idx])
        in_b = bool(b[idx])

        if in_a and in_b:
            colour = COL_BOTH
            label  = 'A∩B'
        elif in_a:
            colour = COL_A
            label  = 'A'
        elif in_b:
            colour = COL_B
            label  = 'B'
        else:
            colour = COL_NEITHER
            label  = ''

        rect = FancyBboxPatch(
            (x, y), cell_w, cell_h,
            boxstyle='round,pad=0.05',
            facecolor=colour, edgecolor='white', linewidth=1.5,
        )
        ax.add_patch(rect)

        if label:
            ax.text(
                x + cell_w / 2, y + cell_h / 2, label,
                ha='center', va='center', fontsize=7,
                color='white', fontweight='bold',
            )

    # Feature index labels along the top
    for col in range(GRID_COLS):
        x = col * (cell_w + pad) + cell_w / 2
        ax.text(x, cell_h + 0.15, str(col + 1),
                ha='center', va='bottom', fontsize=7, color='#888')

    total_w = GRID_COLS  * (cell_w + pad)
    total_h = GRID_ROWS  * (cell_h + pad)
    ax.set_xlim(-0.2, total_w)
    ax.set_ylim(-total_h, cell_h + 0.4)

    # Legend
    handles = [
        mpatches.Patch(color=COL_BOTH,    label='Both (A ∩ B)'),
        mpatches.Patch(color=COL_A,       label='A only'),
        mpatches.Patch(color=COL_B,       label='B only'),
        mpatches.Patch(color=COL_NEITHER, label='Neither (ignored)'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower center',
              ncol=4, bbox_to_anchor=(0.5, -0.18),
              framealpha=1.0, edgecolor='#cccccc')


def show():
    """Render Figure 7: interactive Jaccard similarity explorer."""
    plt.close('Notebook8 Figure 7')

    fig, (ax_grid, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 7',
        figsize=(10, 8),
        gridspec_kw={'width_ratios': [1.6, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # ── Annotation panel (static axes — only text updates) ───────────────────
    ax_ann.set_axis_off()
    a_init, b_init = _make_vectors(10, 10, 5)
    ann_text = ax_ann.text(
        0.04, 0.97,
        _annotation(a_init, b_init),
        transform=ax_ann.transAxes,
        fontsize=9.5, va='top', ha='left',
        linespacing=1.7,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    _, _, j_init = _jaccard(a_init, b_init)
    fig_title = fig.suptitle(
        f'Figure 7: Jaccard similarity — J = {j_init:.3f}   '
        f'Jaccard distance = {1 - j_init:.3f}',
        fontsize=11,
    )

    _draw_grid(ax_grid, a_init, b_init)
    plt.subplots_adjust(wspace=0.12, bottom=0.18)

    # ── Widgets ───────────────────────────────────────────────────────────────
    # n_a and n_b: how many features are set in each vector
    # n_overlap:   how many of those are shared (capped dynamically)
    slider_a = widgets.IntSlider(
        value=10, min=1, max=N_FEATURES, step=1,
        description='Features in A',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
    )
    slider_b = widgets.IntSlider(
        value=10, min=1, max=N_FEATURES, step=1,
        description='Features in B',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
    )
    slider_overlap = widgets.IntSlider(
        value=5, min=0, max=10, step=1,
        description='Overlap (A ∩ B)',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
    )

    def _update(n_a, n_b, n_overlap):
        # Cap overlap so it never exceeds either vector's feature count or the
        # total number of features available for both together
        max_overlap = min(n_a, n_b, N_FEATURES - max(n_a, n_b) + min(n_a, n_b))
        max_overlap = max(max_overlap, 0)
        n_overlap_safe = min(n_overlap, max_overlap)

        a, b = _make_vectors(n_a, n_b, n_overlap_safe)
        _, _, j = _jaccard(a, b)

        _draw_grid(ax_grid, a, b)
        ann_text.set_text(_annotation(a, b))
        fig_title.set_text(
            f'Figure 7: Jaccard similarity — J = {j:.3f}   '
            f'Jaccard distance = {1 - j:.3f}'
        )
        fig.canvas.draw_idle()

    out = interactive_output(
        _update,
        {'n_a': slider_a, 'n_b': slider_b, 'n_overlap': slider_overlap},
    )

    controls = widgets.VBox([slider_a, slider_b, slider_overlap])
    display(controls, out)