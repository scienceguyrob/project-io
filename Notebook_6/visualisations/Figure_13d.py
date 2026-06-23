"""
Figure 13d — The Inverse Relationship Between ||w|| and the Margin
==================================================================
Shows interactively how the margin width changes as ||w|| changes.

The boundary orientation is fixed — dragging the slider scales ||w||
without rotating the boundary. As ||w|| grows, the margin shrinks.
As ||w|| shrinks, the margin widens.

The live equation panel shows:
  - Current ||w||
  - Margin = 2 / ||w||
  - The formula with numbers substituted in

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_13d import show
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

# ── Fixed setup ───────────────────────────────────────────────────────────────
# The boundary direction is fixed: a diagonal line y = -x + 5
# w is always in direction [1, 1] (perpendicular to the boundary)
# The slider scales ||w|| which changes the margin but not the boundary position

# Unit direction of w (always [1,1]/sqrt(2))
W_DIR   = np.array([1.0, 1.0]) / np.sqrt(2.0)
B_FIXED = -5.0 / np.sqrt(2.0)   # keeps the boundary at y = -x + 5

# Boundary line: w · x + b = 0 → x1 + x2 - 5 = 0 → x2 = -x1 + 5
XLO, XHI = -1.0, 7.0
YLO, YHI = -1.0, 7.0
xs = np.linspace(XLO, XHI, 400)
y_boundary = -xs + 5.0

# ── Reproducible training data either side of the boundary ────────────────────
rng = np.random.default_rng(42)

pos_pts = []
for _ in range(200):
    x = rng.uniform(1.5, 6.0)
    y_bd = -x + 5.0
    y = rng.uniform(y_bd + 0.3, min(y_bd + 2.5, YHI - 0.1))
    if YLO < y < YHI:
        pos_pts.append([x, y])
    if len(pos_pts) == 14:
        break
pos_pts = np.array(pos_pts)

neg_pts = []
for _ in range(80):
    x = rng.uniform(XLO, 5.5)
    y_bd = -x + 5.0
    y = rng.uniform(YLO, y_bd - 0.3)
    if y < y_bd - 0.3 and y > YLO:
        neg_pts.append([x, y])
    if len(neg_pts) == 14:
        break
neg_pts = np.array(neg_pts)

COL_POS = 'steelblue'
COL_NEG = 'tomato'

W_INIT = 1.0
W_MIN  = 0.3
W_MAX  = 4.0


def _margin(w_norm):
    return 2.0 / w_norm


def _margin_lines(w_norm):
    """
    Compute the two margin boundary lines.
    For boundary w·x + b = 0 with ||w|| = w_norm,
    margin boundaries are at perpendicular distance 1/w_norm from the boundary.
    Since w direction is [1,1]/sqrt(2), shifting the intercept by w_norm gives
    the margin lines: x2 = -x1 + 5 ± sqrt(2)/w_norm
    """
    offset = np.sqrt(2.0) / w_norm
    y_top  = -xs + 5.0 + offset
    y_bot  = -xs + 5.0 - offset
    return y_top, y_bot


def _eq_string(w_norm):
    margin = _margin(w_norm)
    return (
        rf'$\|\mathbf{{w}}\| = {w_norm:.3f}$' + '\n\n'
        rf'$\text{{Margin}} = \dfrac{{2}}{{\|\mathbf{{w}}\|}}$' + '\n\n'
        rf'$= \dfrac{{2}}{{{w_norm:.3f}}}$' + '\n\n'
        rf'$= {margin:.3f}$'
    )


def show():
    """Render Figure 13d: Interactive margin vs ||w|| explorer."""
    plt.close('Notebook6 Figure 13d')
    fig, ax = plt.subplots(num='Notebook6 Figure 13d', figsize=(9, 7))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ── Static elements ───────────────────────────────────────────────────────
    # Training data points — never change
    ax.scatter(pos_pts[:, 0], pos_pts[:, 1], color=COL_POS, s=50,
               edgecolors='k', lw=0.4, zorder=4, label='Class +1')
    ax.scatter(neg_pts[:, 0], neg_pts[:, 1], color=COL_NEG, s=50,
               edgecolors='k', lw=0.4, zorder=4, label='Class −1')

    # Decision boundary — fixed, never moves
    ax.plot(xs, y_boundary, color='#111', lw=2.5, zorder=3,
            label='Decision boundary')

    # ── Dynamic elements ──────────────────────────────────────────────────────
    y_top_init, y_bot_init = _margin_lines(W_INIT)
    margin_init = _margin(W_INIT)

    # Margin boundary lines
    (line_top,) = ax.plot(xs, y_top_init, color='#777', lw=1.5,
                          ls='--', zorder=3, label='Margin boundaries')
    (line_bot,) = ax.plot(xs, y_bot_init, color='#777', lw=1.5,
                          ls='--', zorder=3)

    # Shaded margin region
    fill_margin = [ax.fill_between(xs, y_bot_init, y_top_init,
                                   alpha=0.15, color='#888', zorder=2,
                                   label='Margin region')]

    # ── Equation annotation ───────────────────────────────────────────────────
    eq_text = ax.text(
        0.02, 0.97, _eq_string(W_INIT),
        transform=ax.transAxes,
        fontsize=10, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#ccc', alpha=1.0),
        zorder=7, linespacing=1.7,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    title = ax.set_title(
        rf'$\|\mathbf{{w}}\| = {W_INIT:.3f}$  →  '
        rf'Margin $= \frac{{2}}{{\|\mathbf{{w}}\|}} = {margin_init:.3f}$  '
        rf'— drag the slider to see the inverse relationship',
        fontsize=10,
    )

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel(r'Feature $x_1$', fontsize=11)
    ax.set_ylabel(r'Feature $x_2$', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=9, loc='lower left', framealpha=1.0, edgecolor='#ccc')
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════
    w_slider = widgets.FloatSlider(
        value=W_INIT, min=W_MIN, max=W_MAX, step=0.05,
        description=r'||w||',
        style={'description_width': '50px'},
        layout=widgets.Layout(width='460px'),
        readout=True, readout_format='.3f',
    )
    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_reset(_):
        w_slider.value = W_INIT

    reset_btn.on_click(on_reset)

    controls = widgets.HBox([w_slider, widgets.Label(' '), reset_btn])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # ════════════════════════════════════════════════════════════════════════
    def update(w_norm):
        margin = _margin(w_norm)
        y_top, y_bot = _margin_lines(w_norm)

        # Update margin lines
        line_top.set_ydata(y_top)
        line_bot.set_ydata(y_bot)

        # Redraw margin fill
        fill_margin[0].remove()
        fill_margin[0] = ax.fill_between(xs, y_bot, y_top,
                                          alpha=0.15, color='#888', zorder=2)

        # Update equation and title
        eq_text.set_text(_eq_string(w_norm))
        title.set_text(
            rf'$\|\mathbf{{w}}\| = {w_norm:.3f}$  →  '
            rf'Margin $= \frac{{2}}{{\|\mathbf{{w}}\|}} = {margin:.3f}$  '
            rf'— drag the slider to see the inverse relationship'
        )

        fig.canvas.draw_idle()

    out = interactive_output(update, {'w_norm': w_slider})
    display(controls, out)