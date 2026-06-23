"""
Figure 13a — Interactive Weight Vector and Hyperplane Explorer
==============================================================
Shows the relationship between the weight vector w and the decision boundary.

Sliders control w1, w2, b and the query point x1, x2.
The prediction for the query point is shown in an HTML box BELOW the plot.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_13a import show
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

LIM    = 5.0
X_VALS = np.linspace(-LIM, LIM, 400)


def _boundary_y(w1, w2, b):
    if abs(w2) < 1e-6:
        return None
    return -(w1 * X_VALS + b) / w2


def _score_grid(w1, w2, b):
    xs = np.linspace(-LIM, LIM, 200)
    xx, yy = np.meshgrid(xs, xs)
    return xx, yy, w1 * xx + w2 * yy + b


def _pred_html(w1, w2, b, x1, x2):
    score = w1 * x1 + w2 * x2 + b
    pred  = '+1' if score >= 0 else '-1'
    col   = '#2471a3' if score >= 0 else '#c0392b'
    return (
        f'<div style="display:inline-block;padding:10px 20px;'
        f'border:2px solid {col};border-radius:8px;background:#fff;'
        f'font-family:monospace;font-size:13px;line-height:2.0;color:{col};">'
        f'<b>Query point prediction</b><br>'
        f'x = [{x1:.2f}, {x2:.2f}]<br>'
        f'Score = {w1:.2f} &times; {x1:.2f} + {w2:.2f} &times; {x2:.2f} + ({b:.2f}) '
        f'= <b>{score:.4f}</b><br>'
        f'Prediction: <b style="font-size:16px">{pred}</b>'
        f'</div>'
    )


def _make_eq(w1, w2, b):
    w_norm = np.sqrt(w1**2 + w2**2)
    angle  = np.degrees(np.arctan2(w2, w1))
    return (
        r'$\mathbf{w}^\top \mathbf{x} + b = 0$' + '\n\n'
        rf'${w1:.2f} \cdot x_1 + {w2:.2f} \cdot x_2 + ({b:.2f}) = 0$' + '\n\n'
        rf'$\|\mathbf{{w}}\| = {w_norm:.3f}$' + '\n\n'
        rf'angle $= {angle:.1f}°$'
    )


def show():
    plt.close('Notebook6 Figure 13a')
    fig, ax = plt.subplots(num='Notebook6 Figure 13a', figsize=(9, 7))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    W1, W2, B   = 1.0, 1.0, 0.0
    QX1, QX2    = 1.0, 1.0

    # ── Boundary line ─────────────────────────────────────────────────────────
    y_bd = _boundary_y(W1, W2, B)
    (bd_line,) = ax.plot(X_VALS, y_bd if y_bd is not None else X_VALS * 0,
                         color='#222', lw=2.5, zorder=4,
                         label=r'Decision boundary $\mathbf{w}^\top\mathbf{x}+b=0$')

    # ── Shaded half-planes ────────────────────────────────────────────────────
    xx, yy, Z = _score_grid(W1, W2, B)
    fill_pos = [ax.contourf(xx, yy, Z, levels=[0, Z.max()+1],
                            colors=['steelblue'], alpha=0.08, zorder=1)]
    fill_neg = [ax.contourf(xx, yy, Z, levels=[Z.min()-1, 0],
                            colors=['tomato'], alpha=0.08, zorder=1)]

    # ── Weight vector arrow ───────────────────────────────────────────────────
    w_norm  = np.sqrt(W1**2 + W2**2)
    w_disp  = np.array([W1, W2]) / max(w_norm, 1e-6) * 2.0
    arrow   = ax.annotate('', xy=(w_disp[0], w_disp[1]), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='#c0392b',
                                          lw=2.5, mutation_scale=18), zorder=6)
    w_lbl   = ax.text(w_disp[0]+0.15, w_disp[1]+0.15,
                      r'$\mathbf{w}$', fontsize=11,
                      color='#c0392b', fontweight='bold', zorder=7)

    # ── Region labels ─────────────────────────────────────────────────────────
    pos_lbl = ax.text(3.5, 3.5, 'Predict +1', ha='center', va='center',
                      fontsize=9, color='steelblue', fontweight='bold', zorder=5)
    neg_lbl = ax.text(-3.5, -3.5, 'Predict -1', ha='center', va='center',
                      fontsize=9, color='tomato', fontweight='bold', zorder=5)

    # ── Query point dot on the plot ───────────────────────────────────────────
    init_score = W1*QX1 + W2*QX2 + B
    init_col   = 'steelblue' if init_score >= 0 else 'tomato'
    (query_dot,) = ax.plot(QX1, QX2, '*', color=init_col, ms=14,
                           markeredgecolor='black', markeredgewidth=0.8,
                           zorder=8, label='Query point')

    # ── Right-angle marker container ──────────────────────────────────────────
    perp_line = [None]

    def draw_perp(w1, w2):
        if perp_line[0] is not None:
            try:
                perp_line[0].remove()
            except Exception:
                pass
            perp_line[0] = None
        wn = np.sqrt(w1**2 + w2**2)
        if wn < 1e-6 or abs(w2) < 1e-6:
            return
        uw = np.array([w1, w2]) / wn
        ub = np.array([-w2, w1]) / wn
        s  = 0.35
        corners = np.array([[0,0], s*ub, s*ub+s*uw, s*uw, [0,0]])
        p, = ax.plot(corners[:,0], corners[:,1], color='#888', lw=1.0, zorder=5)
        perp_line[0] = p

    draw_perp(W1, W2)

    # ── Equation annotation (top-left of plot) ────────────────────────────────
    eq_text = ax.text(
        0.02, 0.98, _make_eq(W1, W2, B),
        transform=ax.transAxes, fontsize=9.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#ccc', alpha=0.95),
        zorder=7, linespacing=1.6,
    )

    title = ax.set_title(
        rf'$w_1={W1:.2f},\ w_2={W2:.2f},\ b={B:.2f}$  |  '
        rf'$\|\mathbf{{w}}\|={np.sqrt(W1**2+W2**2):.3f}$', fontsize=10)

    ax.axhline(0, color='#ccc', lw=0.8, zorder=0)
    ax.axvline(0, color='#ccc', lw=0.8, zorder=0)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r'Feature 1  ($x_1$)', fontsize=11)
    ax.set_ylabel(r'Feature 2  ($x_2$)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════
    def sl(label, val, lo=-5, hi=5, step=0.1, w='260px'):
        s = widgets.FloatSlider(value=val, min=lo, max=hi, step=step,
                                description=label,
                                style={'description_width': '40px'},
                                layout=widgets.Layout(width=w),
                                readout=True, readout_format='.2f')
        return s

    w1_sl = sl('w₁', W1)
    w2_sl = sl('w₂', W2)
    b_sl  = sl('b',  B)
    x1_sl = sl('x₁', QX1)
    x2_sl = sl('x₂', QX2)

    reset_btn = widgets.Button(description='Reset',
                               layout=widgets.Layout(width='80px'))

    # Prediction HTML box — lives entirely outside the matplotlib canvas
    pred_box = widgets.HTML(
        value=_pred_html(W1, W2, B, QX1, QX2),
        layout=widgets.Layout(margin='6px 0 0 0'),
    )

    def on_reset(_):
        w1_sl.value = W1
        w2_sl.value = W2
        b_sl.value  = B
        x1_sl.value = QX1
        x2_sl.value = QX2

    reset_btn.on_click(on_reset)

    row1 = widgets.HBox([
        widgets.Label('Hyperplane:', layout=widgets.Layout(width='85px')),
        w1_sl, w2_sl, b_sl, widgets.Label(' '), reset_btn,
    ])
    row2 = widgets.HBox([
        widgets.Label('Query point:', layout=widgets.Layout(width='85px')),
        x1_sl, x2_sl,
    ])
    controls = widgets.VBox([row1, row2])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE
    # ════════════════════════════════════════════════════════════════════════
    def update(w1, w2, b, x1, x2):
        # Boundary
        y_bd = _boundary_y(w1, w2, b)
        if y_bd is not None:
            bd_line.set_visible(True)
            bd_line.set_ydata(y_bd)
        else:
            bd_line.set_visible(False)

        # Shading
        fill_pos[0].remove()
        fill_neg[0].remove()
        xx, yy, Z = _score_grid(w1, w2, b)
        fill_pos[0] = ax.contourf(xx, yy, Z, levels=[0, max(Z.max(),0.01)+1],
                                  colors=['steelblue'], alpha=0.08, zorder=1)
        fill_neg[0] = ax.contourf(xx, yy, Z, levels=[min(Z.min(),-0.01)-1, 0],
                                  colors=['tomato'], alpha=0.08, zorder=1)

        # Weight vector arrow
        wn = np.sqrt(w1**2 + w2**2)
        if wn > 1e-6:
            wd = np.array([w1, w2]) / wn * 2.0
            arrow.set_visible(True)
            arrow.xy     = (wd[0], wd[1])
            arrow.xytext = (0, 0)
            w_lbl.set_position((wd[0]+0.15, wd[1]+0.15))
            unit_w = np.array([w1, w2]) / wn
            pos_lbl.set_position(tuple(unit_w * 3.0))
            neg_lbl.set_position(tuple(-unit_w * 3.0))
        else:
            arrow.set_visible(False)

        draw_perp(w1, w2)

        # Query dot colour only — dot moves with x1/x2 sliders
        score = w1*x1 + w2*x2 + b
        col   = 'steelblue' if score >= 0 else 'tomato'
        query_dot.set_xdata([x1])
        query_dot.set_ydata([x2])
        query_dot.set_color(col)

        # Prediction box (HTML widget, outside the canvas)
        pred_box.value = _pred_html(w1, w2, b, x1, x2)

        # Equation and title
        eq_text.set_text(_make_eq(w1, w2, b))
        title.set_text(
            rf'$w_1={w1:.2f},\ w_2={w2:.2f},\ b={b:.2f}$  |  '
            rf'$\|\mathbf{{w}}\|={np.sqrt(w1**2+w2**2):.3f}$'
        )
        fig.canvas.draw_idle()

    out = interactive_output(
        update,
        {'w1': w1_sl, 'w2': w2_sl, 'b': b_sl, 'x1': x1_sl, 'x2': x2_sl},
    )

    # Display order: controls → prediction HTML box → matplotlib canvas (via out)
    display(controls, pred_box, out)