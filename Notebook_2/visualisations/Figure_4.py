"""
Figure 4 — Interactive Three-Class Decision Boundary Explorer
=============================================================

Three clusters of penguin data (Adelie, Chinstrap, Gentoo) are plotted using
flipper length (x-axis) and body mass (y-axis). Two straight-line boundaries
divide the plot into three regions — one predicted class per region.

The user rotates and shifts each boundary line independently using angle
and offset sliders. Accuracy updates live, both overall and per class.

Classification rule:
    Above Line 1                    → Chinstrap
    Below Line 1 AND above Line 2   → Adelie
    Below Line 2                    → Gentoo

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_4 import show
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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox
from IPython.display import display


def show():
    """Render the interactive Figure 4 three-class boundary explorer."""

    plt.close('Notebook2 Figure 4')

    # ── Data generation ───────────────────────────────────────────────────────
    # Fixed seed so every user sees the same three clusters.
    rng = np.random.default_rng(42)

    # Three species clusters: Adelie (short flippers, low mass),
    # Chinstrap (medium flippers, high mass), Gentoo (long flippers, medium mass)
    cx = [rng.normal(180, 7, 50), rng.normal(205, 7, 50), rng.normal(225, 7, 50)]
    cy = [rng.normal(3400, 280, 50), rng.normal(4800, 280, 50), rng.normal(3600, 280, 50)]
    colours = ['steelblue', 'tomato', 'seagreen']
    labels  = ['Adelie', 'Chinstrap', 'Gentoo']

    # ── True labels ───────────────────────────────────────────────────────────
    all_x = np.concatenate(cx)
    all_y = np.concatenate(cy)
    true_labels = np.array([0]*50 + [1]*50 + [2]*50)

    # ── Compute default boundaries ────────────────────────────────────────────
    # The perpendicular bisector between two centroids is the natural starting
    # boundary — it sits equidistant from both class centres.
    def perp_bisector(xa, ya, xb, yb):
        """Return slope and intercept of the perpendicular bisector between two points."""
        mid  = np.array([(xa+xb)/2, (ya+yb)/2])
        diff = np.array([xb-xa, yb-ya])
        m    = -diff[0] / diff[1]
        c    = mid[1] - m * mid[0]
        return m, c

    means_x = [arr.mean() for arr in cx]
    means_y = [arr.mean() for arr in cy]

    # Line 1 separates Adelie from Chinstrap; Line 2 separates Chinstrap from Gentoo
    m1_def, c1_def = perp_bisector(means_x[0], means_y[0], means_x[1], means_y[1])
    m2_def, c2_def = perp_bisector(means_x[1], means_y[1], means_x[2], means_y[2])

    ANG1_DEF = round(np.degrees(np.arctan(m1_def)), 1)
    ANG2_DEF = round(np.degrees(np.arctan(m2_def)), 1)

    # Pivot points used when rotating the lines — each line rotates around the
    # midpoint between its two adjacent centroids so it stays roughly in place
    pivot1 = np.array([(means_x[0]+means_x[1])/2, (means_y[0]+means_y[1])/2])
    pivot2 = np.array([(means_x[1]+means_x[2])/2, (means_y[1]+means_y[2])/2])

    # ── Fixed axis limits and x sample points ─────────────────────────────────
    XLIM   = (160, 245)
    YLIM   = (2200, 6800)
    x_plot = np.linspace(XLIM[0], XLIM[1], 400)

    # ── Classification rule ───────────────────────────────────────────────────
    # Points are assigned to a class based on which region they fall in,
    # defined by whether they are above or below each boundary line.
    def classify(m1, c1, m2, c2):
        above1 = all_y > m1 * all_x + c1
        above2 = all_y > m2 * all_x + c2
        return np.where(above1, 1, np.where(above2, 0, 2))

    # ── Helper: draw the three shaded regions ─────────────────────────────────
    # fill_between() has no mutation API so the fills must be removed and
    # redrawn on every update — this helper keeps that logic in one place.
    def make_fills(y1, y2):
        y1c = np.clip(y1, *YLIM)
        y2c = np.clip(y2, *YLIM)
        fa = ax.fill_between(x_plot, y2c, y1c,    alpha=0.07, color='steelblue', zorder=1)
        fb = ax.fill_between(x_plot, y1c, YLIM[1], alpha=0.07, color='tomato',   zorder=1)
        fc = ax.fill_between(x_plot, YLIM[0], y2c, alpha=0.07, color='seagreen', zorder=1)
        return [fa, fb, fc]

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook2 Figure 4', figsize=(8, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # Static scatter points — one group per species
    for xs, ys, col, lab in zip(cx, cy, colours, labels):
        ax.scatter(xs, ys, color=col, s=60, edgecolors='k',
                   linewidth=0.4, label=lab, zorder=3)

    # The two boundary lines — stored for set_ydata() updates
    y1_init = m1_def * x_plot + c1_def
    y2_init = m2_def * x_plot + c2_def
    (line1,) = ax.plot(x_plot, y1_init, 'k--', linewidth=2.5, label='Line 1', zorder=4)
    (line2,) = ax.plot(x_plot, y2_init, 'k:',  linewidth=2.5, label='Line 2', zorder=4)

    # Initial shaded regions — stored in a list so update() can clear them
    fills = make_fills(y1_init, y2_init)

    ax.set_xlabel('Flipper length (mm)')
    ax.set_ylabel('Body mass (g)')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Figure 4: Two linear boundaries separating three penguin species')

    # Accuracy readout in the bottom-right corner
    acc_text = ax.text(
        0.98, 0.05, '', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
    )

    # m and c readout above the accuracy box
    mc_text = ax.text(
        0.98, 0.28, '', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9, color='#444',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'),
    )

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever any slider changes.
    def update(ang1, off1, ang2, off2):
        # Convert angles to slopes; pivot each line around its centroid midpoint
        m1 = np.tan(np.radians(ang1))
        c1 = (pivot1[1] - m1 * pivot1[0]) + off1
        m2 = np.tan(np.radians(ang2))
        c2 = (pivot2[1] - m2 * pivot2[0]) + off2

        y1 = m1 * x_plot + c1
        y2 = m2 * x_plot + c2

        # Update boundary lines in place
        line1.set_ydata(y1)
        line2.set_ydata(y2)

        # Remove old fills and redraw
        for f in fills:
            f.remove()
        fills.clear()
        fills.extend(make_fills(y1, y2))

        # Compute and display accuracy
        predicted = classify(m1, c1, m2, c2)
        correct   = np.sum(predicted == true_labels)
        accuracy  = correct / len(true_labels) * 100
        per_class = [np.sum((predicted == i) & (true_labels == i)) for i in range(3)]

        acc_text.set_text(
            f'Overall accuracy: {correct}/{len(true_labels)} ({accuracy:.1f}%)\n'
            f'Adelie: {per_class[0]}/50   '
            f'Chinstrap: {per_class[1]}/50   '
            f'Gentoo: {per_class[2]}/50'
        )
        mc_text.set_text(
            f'Line 1:  m = {m1:.3f},  c = {c1:.0f}\n'
            f'Line 2:  m = {m2:.3f},  c = {c2:.0f}'
        )

        ax.legend(fontsize=9, loc='upper left')
        fig.canvas.draw_idle()

    # ── Sliders and text boxes ────────────────────────────────────────────────
    # Helper functions keep the repetitive slider/box construction concise
    def make_slider(label, val, lo, hi, step):
        return FloatSlider(
            value=val, min=lo, max=hi, step=step,
            description=label, style={'description_width': '110px'},
            layout=widgets.Layout(width='380px'), continuous_update=True,
        )

    def make_box(val, lo, hi, step):
        return widgets.BoundedFloatText(
            value=val, min=lo, max=hi, step=step,
            description='', layout=widgets.Layout(width='100px'),
        )

    ang1_s = make_slider('Line 1 angle (°)', ANG1_DEF, -89, 89, 0.5)
    off1_s = make_slider('Line 1 offset',    0, -1500, 1500, 10)
    ang2_s = make_slider('Line 2 angle (°)', ANG2_DEF, -89, 89, 0.5)
    off2_s = make_slider('Line 2 offset',    0, -1500, 1500, 10)

    ang1_b = make_box(ANG1_DEF, -89, 89, 0.5)
    off1_b = make_box(0, -1500, 1500, 10)
    ang2_b = make_box(ANG2_DEF, -89, 89, 0.5)
    off2_b = make_box(0, -1500, 1500, 10)

    for s, b in [(ang1_s, ang1_b), (off1_s, off1_b), (ang2_s, ang2_b), (off2_s, off2_b)]:
        widgets.jslink((s, 'value'), (b, 'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(
        description='Reset',
        button_style='info',
        layout=widgets.Layout(width='150px'),
    )

    def on_reset(b):
        ang1_s.value = ANG1_DEF
        off1_s.value = 0
        ang2_s.value = ANG2_DEF
        off2_s.value = 0

    reset_btn.on_click(on_reset)

    # ── Wire sliders to update and display ────────────────────────────────────
    # A horizontal rule visually separates the two sets of controls
    sep = widgets.HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HBox([ang1_s, ang1_b]),
        HBox([off1_s, off1_b]),
        sep,
        HBox([ang2_s, ang2_b]),
        HBox([off2_s, off2_b]),
        reset_btn,
    ])

    out = interactive_output(update, {
        'ang1': ang1_s, 'off1': off1_s,
        'ang2': ang2_s, 'off2': off2_s,
    })

    display(controls, out)