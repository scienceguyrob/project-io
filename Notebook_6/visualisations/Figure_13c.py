"""
Figure 13c — Poor Boundaries vs Maximum Margin
===============================================
Static two-panel figure illustrating why the maximum margin boundary
is better than tight boundaries that merely separate the classes.

Left panel  — Two tight boundaries (Line 1 and Line 2) each hugging one
              class. Points correctly classified by each boundary are shown.
              Two star points are misclassified.

Right panel — The maximum margin boundary (Line 3) centred between Line 1
              and Line 2. The same two star points are now correctly
              classified.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_13c import show
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
import matplotlib.lines as mlines
from matplotlib.patches import Patch

# ── Fixed boundary parameters ─────────────────────────────────────────────────
# All lines share slope = -1.  y = -x + c
M   = -1.0
C1  = 3.0    # Line 1 — hugs Class -1  (lower)
C2  = 7.0    # Line 2 — hugs Class +1  (upper)
C3  = 5.0    # Line 3 — maximum margin boundary (centred)

# ── Reproducible samples ──────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# Class -1 (red) — one point ON Line 1, rest BELOW
on_neg_x  = 1.5
on_neg_y  = M * on_neg_x + C1          # exactly on Line 1
below_neg = rng.uniform(0, 0.9, (12, 2))
# Shift so they sit below Line 1: y < -x + C1
# Generate in unit square then map: x in [0,4], y in [0, -x+C1 - 0.3]
neg_pts = []
for _ in range(60):
    x = rng.uniform(-1.0, 2.7)   # extended to -1 on the left
    hi = M * x + C1 - 0.15
    if hi <= 0.05:
        continue
    y = rng.uniform(0.05, hi)
    neg_pts.append([x, y])
    if len(neg_pts) == 14:
        break
neg_pts = np.array(neg_pts)

# Class +1 (blue) — one point ON Line 2, rest ABOVE
on_pos_x  = 3.5
on_pos_y  = M * on_pos_x + C2          # exactly on Line 2
pos_pts = []
for _ in range(20):
    x = rng.uniform(0.5, 6.5)
    y = rng.uniform(M * x + C2 + 0.15, 8.0)
    if y <= 8.0:
        pos_pts.append([x, y])
pos_pts = np.array(pos_pts[:14])

# ── Star test points (misclassified on left, correct on right) ────────────────
STAR_NEG = np.array([1.0, 3.0])   # true class -1
STAR_POS = np.array([3.0, 3.0])   # true class +1

# ── Plot range ────────────────────────────────────────────────────────────────
XLO, XHI = -1.0, 6.0
YLO, YHI = 0.0, 8.0
xs = np.linspace(XLO, XHI, 400)

COL_POS = 'steelblue'
COL_NEG = 'tomato'


def show():
    plt.close('Notebook6 Figure 13c')

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, num='Notebook6 Figure 13c', figsize=(12, 6),
    )

    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT PANEL
    # ═════════════════════════════════════════════════════════════════════════
    ax = ax_l

    # Shading: below Line 1 → red, above Line 2 → blue
    y1 = M * xs + C1
    y2 = M * xs + C2
    ax.fill_between(xs, YLO, y1, alpha=0.10, color=COL_NEG, zorder=1)
    ax.fill_between(xs, y2,  YHI, alpha=0.10, color=COL_POS, zorder=1)

    # Boundary lines
    ax.plot(xs, y1, color=COL_NEG, lw=2.0, ls='--', zorder=3,
            label='Boundary A  (hugs Class −1)')
    ax.plot(xs, y2, color=COL_POS, lw=2.0, ls='--', zorder=3,
            label='Boundary B  (hugs Class +1)')

    # Class -1 samples: one on Line 1, rest below
    ax.scatter(on_neg_x, on_neg_y, color=COL_NEG, s=60,
               edgecolors='k', lw=0.4, zorder=4)
    ax.scatter(neg_pts[:, 0], neg_pts[:, 1], color=COL_NEG, s=55,
               edgecolors='k', lw=0.4, zorder=4, label='Class −1')

    # Class +1 samples: one on Line 2, rest above
    ax.scatter(on_pos_x, on_pos_y, color=COL_POS, s=60,
               edgecolors='k', lw=0.4, zorder=4)
    ax.scatter(pos_pts[:, 0], pos_pts[:, 1], color=COL_POS, s=55,
               edgecolors='k', lw=0.4, zorder=4, label='Class +1')

    # Misclassified star points
    ax.scatter(*STAR_NEG, marker='*', s=280, color=COL_NEG,
               edgecolors='k', lw=0.8, zorder=6,
               label='Misclassified −1 point')
    ax.scatter(*STAR_POS, marker='*', s=280, color=COL_POS,
               edgecolors='k', lw=0.8, zorder=6,
               label='Misclassified +1 point')

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel(r'Feature $x_1$', fontsize=11)
    ax.set_ylabel(r'Feature $x_2$', fontsize=11)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8.5, loc='upper right',
              framealpha=1.0, edgecolor='#ccc')
    ax.set_title(
        'Tight boundaries — both separate the classes\n'
        'but the star points are misclassified',
        fontsize=10,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL
    # ═════════════════════════════════════════════════════════════════════════
    ax = ax_r

    y1 = M * xs + C1
    y2 = M * xs + C2
    y3 = M * xs + C3

    # Shading: below Line 1 → red, between Line 1 and Line 2 → grey margin,
    # above Line 2 → blue
    ax.fill_between(xs, YLO, y1,  alpha=0.10, color=COL_NEG, zorder=1)
    ax.fill_between(xs, y1,  y2,  alpha=0.12, color='#888',  zorder=1)
    ax.fill_between(xs, y2,  YHI, alpha=0.10, color=COL_POS, zorder=1)

    # Grey dashed margin boundary lines (Line 1 and Line 2)
    ax.plot(xs, y1, color='#999', lw=1.5, ls='--', zorder=3,
            label='Margin boundaries')
    ax.plot(xs, y2, color='#999', lw=1.5, ls='--', zorder=3)

    # Maximum margin decision boundary (Line 3)
    ax.plot(xs, y3, color='#111', lw=2.5, ls='-', zorder=4,
            label='Maximum margin boundary')

    # Class samples — same as left panel
    ax.scatter(on_neg_x, on_neg_y, color=COL_NEG, s=60,
               edgecolors='k', lw=0.4, zorder=5)
    ax.scatter(neg_pts[:, 0], neg_pts[:, 1], color=COL_NEG, s=55,
               edgecolors='k', lw=0.4, zorder=5, label='Class −1')

    ax.scatter(on_pos_x, on_pos_y, color=COL_POS, s=60,
               edgecolors='k', lw=0.4, zorder=5)
    ax.scatter(pos_pts[:, 0], pos_pts[:, 1], color=COL_POS, s=55,
               edgecolors='k', lw=0.4, zorder=5, label='Class +1')

    # Correctly classified star points
    ax.scatter(*STAR_NEG, marker='*', s=280, color=COL_NEG,
               edgecolors='k', lw=0.8, zorder=6,
               label='Correctly classified −1')
    ax.scatter(*STAR_POS, marker='*', s=280, color=COL_POS,
               edgecolors='k', lw=0.8, zorder=6,
               label='Correctly classified +1')

    # Margin annotation — double-headed arrow from (3,0) to (5,2)
    ax.annotate('', xy=(5.0, 2.0), xytext=(3.0, 0.0),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1.8))
    ax.text(5.15, 1.0, '',
            fontsize=9, color='#333', rotation=0, ha='left', va='center')

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel(r'Feature $x_1$', fontsize=11)
    ax.set_ylabel(r'Feature $x_2$', fontsize=11)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8.5, loc='upper right',
              framealpha=1.0, edgecolor='#ccc')
    ax.set_title(
        'Maximum margin boundary — centred between classes\n'
        'the same star points are now correctly classified',
        fontsize=10,
    )

    plt.suptitle(
        'Figure 13c: A boundary with a larger margin is more tolerant '
        'of variation in new data',
        fontsize=10,
    )
    plt.subplots_adjust(top=0.88, wspace=0.15)
    plt.show()