"""
Figure 4 — F1 Score as a Function of Precision and Recall
=======================================================================
A two-panel figure that illustrates the F1 score and why it behaves
differently from the arithmetic mean of precision and recall.

This figure does not build its own classifier. It receives the precision
and recall values computed from the classifier trained in Section 4.1 of
the notebook, so the "Our model" marker on the heatmap corresponds exactly
to the model discussed in the surrounding text.

show() expects two arguments:

  precision - precision of the Section 4.1 classifier at the default
              threshold, i.e. TP / (TP + FP)
  recall    - recall of the Section 4.1 classifier at the default
              threshold, i.e. TP / (TP + FN)

The figure contains two panels:

  Left panel  — A comparison table of five precision/recall combinations,
                showing how the arithmetic mean and F1 (harmonic mean)
                diverge. The actual model row is highlighted so the reader
                can locate it in the table before looking at the heatmap.

  Right panel — A filled contour heatmap of F1 as a function of precision
                and recall across the full [0, 1] x [0, 1] space, with
                iso-F1 contour lines at 0.5, 0.7, 0.8, and 0.9, and a
                marker showing where the actual classifier sits.

Usage
-----
In a Jupyter notebook cell (precision and recall must already be computed
from the Section 4.1 classifier):

    %matplotlib widget
    from visualisations.Figure_4 import show
    show(precision, recall)

where precision = TP / (TP + FP) and recall = TP / (TP + FN), using the
TP, FP, FN variables from Section 4.1.

Copyright © 2026 Robert Lyon. All Rights Reserved.

This notebook and all associated materials are the intellectual property of the author.

Permission is granted solely to read, study, and analyse this material for personal educational purposes. No other rights are granted.

Without the prior written consent of the author, you may not:

* Copy, reproduce, redistribute, publish, transmit, or distribute this work in whole or in part.
* Modify, adapt, transform, translate, or create derivative works based on this material.
* Incorporate any portion of this work into another project, publication, product, model, dataset, or codebase.
* Use this material for commercial purposes.
* Remove or alter this copyright notice.

All intellectual property rights, including copyright and any derivative rights, remain exclusively vested in the author.

Access to this material does not constitute a transfer of ownership, license, or any other intellectual property rights except as expressly stated above.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Colours consistent with the notebook's palette.
COLOUR_GRID        = '#cccccc'
COLOUR_MODEL       = 'white'
COLOUR_HIGHLIGHT   = '#fff9c4'   # pale yellow to highlight the "Our model" table row

# Contour levels drawn as dashed black iso-lines over the heatmap, giving
# the reader explicit reference points for common F1 thresholds.
ISO_F1_LEVELS = [0.5, 0.7, 0.8, 0.9]

# Resolution of the precision/recall grid used to compute the F1 surface.
# 200 points gives smooth contours without being computationally heavy.
GRID_RESOLUTION = 200

# Schematic precision/recall combinations for the comparison table. These
# are illustrative; the actual model row is appended dynamically in show().
EXAMPLE_SCENARIOS = [
    ('Perfect balance',   0.80, 0.80),
    ('Slight imbalance',  0.90, 0.70),
    ('High P, low R',     0.99, 0.01),
    ('Low P, high R',     0.50, 0.99),
]


def _f1(p, r):
    """Compute F1 from precision p and recall r, returning 0.0 if both are zero."""
    denom = p + r
    if denom == 0:
        return 0.0
    return 2 * p * r / denom


def show(precision, recall):
    """
    Render Figure 4: F1 comparison table and heatmap.

    Parameters
    ----------
    precision : float
        Precision of the Section 4.1 classifier at the default threshold.
    recall : float
        Recall of the Section 4.1 classifier at the default threshold.
    """
    plt.close('Notebook11 Figure 4')

    f1_model = _f1(precision, recall)
    arith_model = (precision + recall) / 2

    print('Figure 4: F1 score (Section 4.1 classifier at default threshold)')
    print(f'  Precision        = {precision:.4f}')
    print(f'  Recall           = {recall:.4f}')
    print(f'  Arithmetic mean  = {arith_model:.4f}')
    print(f'  F1 (harmonic)    = {f1_model:.4f}')
    print()
    print(f'  {"Scenario":<22} {"P":>6} {"R":>6} {"Arith":>8} {"F1":>8}')
    print('  ' + '-' * 54)
    for name, p, r in EXAMPLE_SCENARIOS:
        print(f'  {name:<22} {p:>6.3f} {r:>6.3f} {(p+r)/2:>8.3f} {_f1(p,r):>8.3f}')
    print(f'  {"Our model":<22} {precision:>6.3f} {recall:>6.3f} {arith_model:>8.3f} {f1_model:>8.3f}')

    # ── F1 surface ────────────────────────────────────────────────────────────
    # Computed over the full [0.01, 1] x [0.01, 1] grid so that contours
    # reach the edges cleanly without a divide-by-zero at the origin.
    p_grid = np.linspace(0.01, 1.0, GRID_RESOLUTION)
    r_grid = np.linspace(0.01, 1.0, GRID_RESOLUTION)
    P_g, R_g = np.meshgrid(p_grid, r_grid)
    F1_g = 2 * P_g * R_g / (P_g + R_g)

    fig = plt.figure(num='Notebook11 Figure 4', figsize=(10, 5.5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # Two panels with unequal widths: the table is narrower than the heatmap.
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], figure=fig)
    ax_table = fig.add_subplot(gs[0])
    ax_heat  = fig.add_subplot(gs[1])

    # ── Left panel: comparison table ─────────────────────────────────────────
    ax_table.axis('off')
    ax_table.set_title(
        'Arithmetic mean vs F1 across\ndifferent precision/recall combinations',
        fontsize=10,
    )

    col_headers = ['Scenario', 'P', 'R', 'Arith', 'F1']
    col_x       = [0.02, 0.50, 0.62, 0.74, 0.88]
    col_align   = ['left', 'center', 'center', 'center', 'center']

    # Header row
    for x, h, ha in zip(col_x, col_headers, col_align):
        ax_table.text(x, 0.92, h, transform=ax_table.transAxes,
                      fontsize=9, fontweight='bold', ha=ha, va='top',
                      family='monospace')

    # Draw a separator line below the header using a simple horizontal line
    # in axes coordinates via ax_table.plot, since axhline does not accept
    # a transform argument.
    ax_table.plot([0.01, 0.99], [0.88, 0.88], color='#aaaaaa', lw=0.8,
                  transform=ax_table.transAxes, clip_on=False)

    all_rows = EXAMPLE_SCENARIOS + [('Our model', precision, recall)]
    n_rows   = len(all_rows)

    for i, (name, p, r) in enumerate(all_rows):
        # Rows are spaced evenly between y=0.82 (top) and y=0.10 (bottom).
        y   = 0.82 - i * (0.72 / (n_rows - 1))
        am  = (p + r) / 2
        hm  = _f1(p, r)
        is_model = (name == 'Our model')

        # Highlight the actual model row with a pale background rectangle so
        # the reader can locate it easily before looking at the heatmap.
        if is_model:
            ax_table.axhspan(y - 0.06, y + 0.06, color=COLOUR_HIGHLIGHT,
                             alpha=0.8, transform=ax_table.transAxes)

        values = [name, f'{p:.3f}', f'{r:.3f}', f'{am:.3f}', f'{hm:.3f}']
        weight = 'bold' if is_model else 'normal'
        for x, val, ha in zip(col_x, values, col_align):
            ax_table.text(x, y, val, transform=ax_table.transAxes,
                          fontsize=9, ha=ha, va='center',
                          family='monospace', fontweight=weight)

    # ── Right panel: F1 heatmap ───────────────────────────────────────────────
    cs = ax_heat.contourf(p_grid, r_grid, F1_g, levels=20, cmap='RdYlGn')
    plt.colorbar(cs, ax=ax_heat, label='F1 score')

    # Dashed iso-F1 contour lines give explicit reference points for common
    # F1 thresholds, making it easy to see which (P, R) combinations are
    # needed to reach each level.
    cl = ax_heat.contour(p_grid, r_grid, F1_g,
                         levels=ISO_F1_LEVELS,
                         colors='black', linewidths=0.8,
                         linestyles='--', alpha=0.6)
    ax_heat.clabel(cl, fmt='F1=%.1f', fontsize=8, inline=True)

    # Marker for the actual classifier: white dot with black edge so it is
    # visible against both the light and dark regions of the heatmap.
    ax_heat.scatter(
        [precision], [recall],
        s=160, color=COLOUR_MODEL, edgecolors='black', lw=1.5,
        zorder=5, label=f'Our model  (F1 = {f1_model:.2f})',
    )
    ax_heat.set_xlabel('Precision')
    ax_heat.set_ylabel('Recall')
    ax_heat.set_title(
        'F1 score across all (Precision, Recall) combinations\n'
        'high P or R alone is not enough: both must be high',
        fontsize=10,
    )
    ax_heat.legend(fontsize=9, loc='upper left')

    fig.suptitle(
        'Figure 4: F1 score — the harmonic mean of Precision and Recall',
        fontsize=12,
    )
    plt.tight_layout()