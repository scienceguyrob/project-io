"""
Figure 13e — Interactive Hinge Loss Explorer
=============================================
Shows how the hinge loss changes as a single point moves relative to
the fixed decision boundary and margin.

The boundary and margin are fixed. A single slider controls the position
of a query point along the axis perpendicular to the boundary. The hinge
loss is computed and displayed live, along with a visual showing which
region the point is in:

  - Outside the margin, correct side   → loss = 0
  - Inside the margin, correct side    → loss > 0 (partial violation)
  - On the wrong side of the boundary  → loss > 0 (full violation)

The right-hand panel shows the hinge loss curve across all possible
positions, with a marker tracking the current point.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_13e import show
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

# ── Fixed boundary: same as Figure 13c ───────────────────────────────────────
# y = -x + 5  (slope -1, intercept 5)
# Class +1 is above the boundary, Class -1 is below
M_BD  = -1.0
C_BD  =  5.0
# Margin lines: y = -x + 3  and  y = -x + 7
C_M1  =  3.0   # lower margin boundary (Class -1 support vector line)
C_M2  =  7.0   # upper margin boundary (Class +1 support vector line)

XLO, XHI = -1.0, 6.0
YLO, YHI =  0.0, 8.0
xs = np.linspace(XLO, XHI, 400)

# ── Training data (same seed as Figure 13c) ───────────────────────────────────
rng = np.random.default_rng(42)

M = M_BD
C1, C2 = C_M1, C_M2

on_neg_x = 1.5
on_neg_y = M * on_neg_x + C1

neg_pts = []
for _ in range(60):
    x  = rng.uniform(-1.0, 2.7)
    hi = M * x + C1 - 0.15
    if hi <= 0.05:
        continue
    y = rng.uniform(0.05, hi)
    neg_pts.append([x, y])
    if len(neg_pts) == 12:
        break
neg_pts = np.array(neg_pts)

on_pos_x = 3.5
on_pos_y = M * on_pos_x + C2

pos_pts = []
for _ in range(20):
    x = rng.uniform(0.5, 6.5)
    y = rng.uniform(M * x + C2 + 0.15, 8.0)
    if y <= 8.0:
        pos_pts.append([x, y])
pos_pts = np.array(pos_pts[:12])

COL_POS  = 'steelblue'
COL_NEG  = 'tomato'

# ── Query point moves along a horizontal scan line at y = 3 ──────────────────
# True label of query point is +1 (blue)
# x controls position: far left = wrong side, far right = correct side
Y_QUERY  = 3.5
TRUE_LABEL = 1   # +1

# Score normalised so that:
#   score = +1 at the +1 margin boundary (x1 + Y_QUERY = C_M2)
#   score =  0 at the decision boundary  (x1 + Y_QUERY = C_BD)
#   score = -1 at the -1 margin boundary (x1 + Y_QUERY = C_M1)
# The margin half-width in x1 space is (C_BD - C_M1) = 2
# So the normalised score = (x1 + Y_QUERY - C_BD) / (C_BD - C_M1)

MARGIN_HALF = C_BD - C_M1   # = 2.0

def score(x1):
    """
    Normalised score for a point at (x1, Y_QUERY).
    Returns +1 at the +1 margin boundary, 0 at the decision boundary,
    -1 at the -1 margin boundary — matching the SVM convention exactly.
    """
    return (x1 + Y_QUERY - C_BD) / MARGIN_HALF

def hinge_loss(x1, y_true):
    """Hinge loss: max(0, 1 - y_i * score)."""
    s = score(x1)
    return max(0.0, 1.0 - y_true * s)

# Pre-compute the hinge loss curve across x1 values
X1_RANGE = np.linspace(XLO, XHI, 400)
HINGE_CURVE = np.array([hinge_loss(x, TRUE_LABEL) for x in X1_RANGE])

# Where is the boundary, margin, etc at y = Y_QUERY?
# Boundary: x1 + Y_QUERY - 5 = 0 → x1 = 5 - Y_QUERY
X_BD_AT_Y   = C_BD  - Y_QUERY   # boundary crosses scan line here
X_M1_AT_Y   = C_M1  - Y_QUERY   # lower margin boundary
X_M2_AT_Y   = C_M2  - Y_QUERY   # upper margin boundary (score = +1 for +1 class)

X1_INIT = 1.5   # start on wrong side


def _region_label(x1, y_true):
    """Describe which region the point is in."""
    s = score(x1)
    hl = hinge_loss(x1, y_true)
    if y_true * s >= 1:
        return 'Outside margin — correct side\nLoss = 0', 'seagreen'
    elif y_true * s >= 0:
        return 'Inside margin — correct side\nLoss > 0  (partial violation)', '#e67e22'
    else:
        return 'Wrong side of boundary\nLoss > 0  (full violation)', 'tomato'


def _eq_string(x1, y_true):
    s   = score(x1)
    hl  = hinge_loss(x1, y_true)
    lbl = '+1' if y_true == 1 else '-1'
    return (
        r'$\ell = \max(0,\ 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$'
        + '\n\n'
        + rf'$y_i = {lbl}$'
        + '\n'
        + rf'$\mathrm{{score}} = {s:.3f}$'
        + '\n'
        + rf'$y_i \times \mathrm{{score}} = {y_true * s:.3f}$'
        + '\n'
        + rf'$1 - {y_true * s:.3f} = {1 - y_true * s:.3f}$'
        + '\n\n'
        + rf'$\ell = \max(0,\ {1 - y_true * s:.3f}) = \mathbf{{{hl:.3f}}}$'
    )


def show():
    """Render Figure 13e: Interactive Hinge Loss Explorer."""
    plt.close('Notebook6 Figure 13e')

    fig, (ax_scene, ax_curve) = plt.subplots(
        1, 2, num='Notebook6 Figure 13e', figsize=(14, 6),
        gridspec_kw={'width_ratios': [1.1, 1]},
    )

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — scene with boundary, margin, data, query point
    # ════════════════════════════════════════════════════════════════════════

    y1 = M * xs + C1
    y2 = M * xs + C2
    y3 = M * xs + C_BD

    # Shading
    ax_scene.fill_between(xs, YLO, y1,  alpha=0.08, color=COL_NEG, zorder=1)
    ax_scene.fill_between(xs, y1,  y2,  alpha=0.10, color='#888',  zorder=1)
    ax_scene.fill_between(xs, y2,  YHI, alpha=0.08, color=COL_POS, zorder=1)

    # Margin boundary lines
    ax_scene.plot(xs, y1, color='#999', lw=1.4, ls='--', zorder=3,
                  label='Margin boundaries')
    ax_scene.plot(xs, y2, color='#999', lw=1.4, ls='--', zorder=3)

    # Decision boundary
    ax_scene.plot(xs, y3, color='#111', lw=2.5, ls='-', zorder=4,
                  label='Decision boundary')

    # Training data
    ax_scene.scatter(on_neg_x, on_neg_y, color=COL_NEG, s=55,
                     edgecolors='k', lw=0.4, zorder=5)
    ax_scene.scatter(neg_pts[:, 0], neg_pts[:, 1], color=COL_NEG, s=45,
                     edgecolors='k', lw=0.4, zorder=5, label='Class −1')
    ax_scene.scatter(on_pos_x, on_pos_y, color=COL_POS, s=55,
                     edgecolors='k', lw=0.4, zorder=5)
    ax_scene.scatter(pos_pts[:, 0], pos_pts[:, 1], color=COL_POS, s=45,
                     edgecolors='k', lw=0.4, zorder=5, label='Class +1')

    # Horizontal scan line showing where query point travels
    ax_scene.axhline(Y_QUERY, color='#bbb', lw=1.0, ls=':', zorder=2,
                     label=f'Query point path  (y = {Y_QUERY})')

    # Query point (star) — dynamic
    hl_init  = hinge_loss(X1_INIT, TRUE_LABEL)
    lbl_init, col_init = _region_label(X1_INIT, TRUE_LABEL)

    (query_pt,) = ax_scene.plot(
        X1_INIT, Y_QUERY, '*',
        color=col_init, ms=18,
        markeredgecolor='k', markeredgewidth=0.8,
        zorder=8, label='Query point (Class +1)',
    )

    # Equation annotation
    eq_text = ax_scene.text(
        0.02, 0.97, _eq_string(X1_INIT, TRUE_LABEL),
        transform=ax_scene.transAxes,
        fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#ccc', alpha=1.0),
        zorder=9, linespacing=1.6, family='monospace',
    )

    ax_scene.set_xlim(XLO, XHI)
    ax_scene.set_ylim(YLO, YHI)
    ax_scene.set_xlabel(r'Feature $x_1$', fontsize=11)
    ax_scene.set_ylabel(r'Feature $x_2$', fontsize=11)
    ax_scene.grid(True, alpha=0.15)
    ax_scene.legend(fontsize=8, loc='upper right',
                    framealpha=1.0, edgecolor='#ccc')
    scene_title = ax_scene.set_title(
        f'Query point:  score = {score(X1_INIT):.3f}  |  '
        f'hinge loss = {hl_init:.3f}\n{lbl_init.split(chr(10))[0]}',
        fontsize=10,
    )

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — hinge loss curve
    # ════════════════════════════════════════════════════════════════════════

    # Full hinge loss curve
    ax_curve.plot(X1_RANGE, HINGE_CURVE, color='#7b5ea7', lw=2.5, zorder=3,
                  label='Hinge loss')

    # Region shading on the loss curve
    ax_curve.fill_between(X1_RANGE, 0, HINGE_CURVE,
                          alpha=0.12, color='#7b5ea7', zorder=2)

    # Vertical markers for boundary and margin positions
    ax_curve.axvline(X_BD_AT_Y, color='#111', lw=1.5, ls='-', zorder=3,
                     alpha=0.6, label='Decision boundary')
    ax_curve.axvline(X_M2_AT_Y, color='#999', lw=1.2, ls='--', zorder=3,
                     alpha=0.8, label='Margin boundary (+1 side)')
    ax_curve.axvline(X_M1_AT_Y, color='#999', lw=1.2, ls=':', zorder=3,
                     alpha=0.8, label='Margin boundary (−1 side)')

    # Region labels
    ax_curve.text(X_M2_AT_Y + 0.15, max(HINGE_CURVE) * 0.85,
                  'Loss = 0\n(correct,\noutside margin)',
                  fontsize=7.5, color='seagreen', va='top')
    ax_curve.text(X_BD_AT_Y + 0.08, max(HINGE_CURVE) * 0.5,
                  'Partial\nviolation',
                  fontsize=7.5, color='#e67e22', va='top')
    ax_curve.text(XLO + 0.1, max(HINGE_CURVE) * 0.85,
                  'Full\nviolation',
                  fontsize=7.5, color='tomato', va='top')

    # Tracking marker on the loss curve
    (loss_dot,) = ax_curve.plot(
        X1_INIT, hinge_loss(X1_INIT, TRUE_LABEL),
        'o', color='#333', ms=10, zorder=6,
        label='Current loss',
    )
    (loss_drop,) = ax_curve.plot(
        [X1_INIT, X1_INIT], [0, hinge_loss(X1_INIT, TRUE_LABEL)],
        color='#333', lw=1.2, ls=':', zorder=5, alpha=0.7,
    )

    ax_curve.set_xlim(XLO, XHI)
    ax_curve.set_ylim(-0.1, max(HINGE_CURVE) * 1.15)
    ax_curve.set_xlabel(r'Query point position  $x_1$', fontsize=11)
    ax_curve.set_ylabel(r'Hinge loss  $\ell$', fontsize=11)
    ax_curve.grid(True, alpha=0.15)
    ax_curve.legend(fontsize=8, loc='upper right',
                    framealpha=1.0, edgecolor='#ccc')
    curve_title = ax_curve.set_title(
        rf'Hinge loss = {hl_init:.3f}',
        fontsize=10,
    )

    plt.suptitle(
        'Figure 13e: Hinge loss — how the penalty changes as a point '
        'moves relative to the decision boundary',
        fontsize=10,
    )
    plt.subplots_adjust(top=0.88, wspace=0.22)

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════
    x1_slider = widgets.FloatSlider(
        value=X1_INIT, min=XLO, max=XHI, step=0.05,
        description=r'x₁ (position)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='460px'),
        readout=True, readout_format='.2f',
    )
    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_reset(_):
        x1_slider.value = X1_INIT

    reset_btn.on_click(on_reset)
    controls = widgets.HBox([x1_slider, widgets.Label(' '), reset_btn])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # ════════════════════════════════════════════════════════════════════════
    def update(x1):
        hl          = hinge_loss(x1, TRUE_LABEL)
        lbl, col    = _region_label(x1, TRUE_LABEL)

        # Move query point and update colour
        query_pt.set_xdata([x1])
        query_pt.set_color(col)

        # Update equation panel
        eq_text.set_text(_eq_string(x1, TRUE_LABEL))

        # Update scene title
        scene_title.set_text(
            f'Query point:  score = {score(x1):.3f}  |  '
            f'hinge loss = {hl:.3f}\n{lbl.split(chr(10))[0]}'
        )

        # Update loss curve marker and drop line
        loss_dot.set_xdata([x1])
        loss_dot.set_ydata([hl])
        loss_drop.set_xdata([x1, x1])
        loss_drop.set_ydata([0, hl])

        # Update curve panel title
        curve_title.set_text(rf'Hinge loss = {hl:.3f}')

        fig.canvas.draw_idle()

    out = interactive_output(update, {'x1': x1_slider})
    display(controls, out)