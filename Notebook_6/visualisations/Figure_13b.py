"""
Figure 13b — The Norm of the Weight Vector: Pythagoras in Action
================================================================
Visualises ||w|| = sqrt(w1^2 + w2^2) as a right-angled triangle,
connecting the geometric length of the weight vector to Pythagoras' theorem.

Shows:
  - The weight vector w as an arrow from the origin to [w1, w2]
  - A right-angled triangle with sides w1 (horizontal), w2 (vertical),
    and hypotenuse ||w|| (the norm)
  - Labelled side lengths updating live as the sliders change
  - The norm formula with numbers substituted in

Sliders control w1 and w2. The triangle and formula update live.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_13b import show
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


def show():
    """Render Figure 13b: Interactive weight vector norm visualisation."""
    plt.close('Notebook6 Figure 13b')
    fig, ax = plt.subplots(num='Notebook6 Figure 13b', figsize=(8, 7))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    W1_INIT, W2_INIT = 3.0, 2.0
    LIM = 5.5

    # ── Colours ───────────────────────────────────────────────────────────────
    COL_W    = '#c0392b'    # weight vector arrow
    COL_W1   = '#2471a3'    # horizontal side (w1)
    COL_W2   = '#27ae60'    # vertical side (w2)
    COL_NORM = '#8e44ad'    # hypotenuse (||w||)
    COL_TRI  = '#f0e6ff'    # triangle fill

    def _norm(w1, w2):
        return np.sqrt(w1**2 + w2**2)

    # ── Triangle fill ─────────────────────────────────────────────────────────
    # Filled triangle with vertices at origin, (w1, 0), (w1, w2)
    tri_fill = [ax.fill(
        [0, W1_INIT, W1_INIT, 0],
        [0, 0,       W2_INIT, 0],
        color=COL_TRI, alpha=0.5, zorder=1,
    )]

    # ── Horizontal side — w1 ─────────────────────────────────────────────────
    (line_w1,) = ax.plot([0, W1_INIT], [0, 0],
                         color=COL_W1, lw=3.0, zorder=3,
                         solid_capstyle='round')

    # ── Vertical side — w2 ───────────────────────────────────────────────────
    (line_w2,) = ax.plot([W1_INIT, W1_INIT], [0, W2_INIT],
                         color=COL_W2, lw=3.0, zorder=3,
                         solid_capstyle='round')

    # ── Hypotenuse — ||w|| ───────────────────────────────────────────────────
    (line_hyp,) = ax.plot([0, W1_INIT], [0, W2_INIT],
                          color=COL_NORM, lw=2.5, zorder=3,
                          linestyle='--')

    # ── Weight vector arrow ───────────────────────────────────────────────────
    arrow = ax.annotate(
        '', xy=(W1_INIT, W2_INIT), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=COL_W,
                        lw=2.5, mutation_scale=18),
        zorder=5,
    )

    # ── Right-angle marker at (w1, 0) ─────────────────────────────────────────
    s = 0.18
    (perp_h,) = ax.plot([W1_INIT - s, W1_INIT - s, W1_INIT],
                        [0,           s,            s],
                        color='#555', lw=1.2, zorder=4)

    # ── Origin dot ────────────────────────────────────────────────────────────
    ax.plot(0, 0, 'o', color='#333', ms=6, zorder=6)

    # ── Side length labels ────────────────────────────────────────────────────
    norm_init = _norm(W1_INIT, W2_INIT)

    # w1 label — below the horizontal side
    lbl_w1 = ax.text(W1_INIT / 2, -0.35,
                     rf'$w_1 = {W1_INIT:.2f}$',
                     ha='center', va='top', fontsize=11,
                     color=COL_W1, fontweight='bold', zorder=6)

    # w2 label — to the right of the vertical side
    lbl_w2 = ax.text(W1_INIT + 0.25, W2_INIT / 2,
                     rf'$w_2 = {W2_INIT:.2f}$',
                     ha='left', va='center', fontsize=11,
                     color=COL_W2, fontweight='bold', zorder=6)

    # ||w|| label — along the hypotenuse, slightly offset
    mid_x = W1_INIT / 2 - 0.4
    mid_y = W2_INIT / 2 + 0.3
    lbl_norm = ax.text(mid_x, mid_y,
                       rf'$\|\mathbf{{w}}\| = {norm_init:.3f}$',
                       ha='center', va='center', fontsize=11,
                       color=COL_NORM, fontweight='bold', zorder=6,
                       rotation=np.degrees(np.arctan2(W2_INIT, W1_INIT)))

    # w label on the arrow tip
    lbl_w = ax.text(W1_INIT + 0.15, W2_INIT + 0.15,
                    r'$\mathbf{w} = [w_1, w_2]$',
                    ha='left', va='bottom', fontsize=10,
                    color=COL_W, fontweight='bold', zorder=6)

    # ── Formula annotation ────────────────────────────────────────────────────
    eq_text = ax.text(
        0.03, 0.97,
        _make_eq(W1_INIT, W2_INIT),
        transform=ax.transAxes,
        fontsize=10,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#ccc', alpha=0.95),
        zorder=7,
        linespacing=1.7,
    )

    # ── Legend patches ────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=COL_W1,   label=r'$w_1$ — horizontal side'),
        mpatches.Patch(color=COL_W2,   label=r'$w_2$ — vertical side'),
        mpatches.Patch(color=COL_NORM, label=r'$\|\mathbf{w}\|$ — hypotenuse (the norm)'),
        mpatches.Patch(color=COL_W,    label=r'$\mathbf{w}$ — weight vector'),
    ], fontsize=9, loc='lower right')

    # ── Title ─────────────────────────────────────────────────────────────────
    title = ax.set_title(
        rf'$\|\mathbf{{w}}\| = \sqrt{{w_1^2 + w_2^2}} = \sqrt{{{W1_INIT:.2f}^2 + {W2_INIT:.2f}^2}} = {norm_init:.3f}$',
        fontsize=11,
    )

    ax.axhline(0, color='#ccc', lw=0.8, zorder=0)
    ax.axvline(0, color='#ccc', lw=0.8, zorder=0)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r'$w_1$', fontsize=12)
    ax.set_ylabel(r'$w_2$', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.12)
    plt.tight_layout()

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════
    w1_slider = widgets.FloatSlider(
        value=W1_INIT, min=-5.0, max=5.0, step=0.1,
        description='w₁', style={'description_width': '30px'},
        layout=widgets.Layout(width='340px'),
        readout=True, readout_format='.2f',
    )
    w2_slider = widgets.FloatSlider(
        value=W2_INIT, min=-5.0, max=5.0, step=0.1,
        description='w₂', style={'description_width': '30px'},
        layout=widgets.Layout(width='340px'),
        readout=True, readout_format='.2f',
    )
    reset_btn = widgets.Button(description='Reset',
                               layout=widgets.Layout(width='80px'))

    def on_reset(_):
        w1_slider.value = W1_INIT
        w2_slider.value = W2_INIT

    reset_btn.on_click(on_reset)

    controls = widgets.HBox([w1_slider, w2_slider,
                             widgets.Label('  '), reset_btn])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # ════════════════════════════════════════════════════════════════════════
    def update(w1, w2):
        norm = _norm(w1, w2)
        angle_deg = np.degrees(np.arctan2(w2, w1)) if norm > 1e-6 else 0

        # Triangle fill — remove and redraw
        tri_fill[0][0].remove()
        tri_fill[0] = ax.fill(
            [0, w1, w1, 0],
            [0, 0,  w2, 0],
            color=COL_TRI, alpha=0.5, zorder=1,
        )

        # Sides
        line_w1.set_xdata([0, w1])
        line_w1.set_ydata([0, 0])
        line_w2.set_xdata([w1, w1])
        line_w2.set_ydata([0, w2])
        line_hyp.set_xdata([0, w1])
        line_hyp.set_ydata([0, w2])

        # Arrow
        arrow.xy     = (w1, w2)
        arrow.xytext = (0, 0)

        # Right-angle marker
        perp_h.set_xdata([w1 - s, w1 - s, w1])
        perp_h.set_ydata([0,      s,       s])

        # Labels
        lbl_w1.set_position((w1 / 2, -0.35))
        lbl_w1.set_text(rf'$w_1 = {w1:.2f}$')

        lbl_w2.set_position((w1 + 0.25, w2 / 2))
        lbl_w2.set_text(rf'$w_2 = {w2:.2f}$')

        # Hypotenuse label — rotate to match angle
        mx = w1 / 2 - 0.4 * np.sin(np.radians(angle_deg))
        my = w2 / 2 + 0.3 * np.cos(np.radians(angle_deg))
        lbl_norm.set_position((mx, my))
        lbl_norm.set_text(rf'$\|\mathbf{{w}}\| = {norm:.3f}$')
        lbl_norm.set_rotation(angle_deg)

        lbl_w.set_position((w1 + 0.15, w2 + 0.15))

        # Formula and title
        eq_text.set_text(_make_eq(w1, w2))
        title.set_text(
            rf'$\|\mathbf{{w}}\| = \sqrt{{w_1^2 + w_2^2}} = '
            rf'\sqrt{{{w1:.2f}^2 + {w2:.2f}^2}} = {norm:.3f}$'
        )

        fig.canvas.draw_idle()

    out = interactive_output(update, {'w1': w1_slider, 'w2': w2_slider})
    display(controls, out)


def _make_eq(w1, w2):
    """Build the live formula string."""
    norm = np.sqrt(w1**2 + w2**2)
    return (
        r'$\|\mathbf{w}\| = \sqrt{w_1^2 + w_2^2}$' + '\n\n'
        rf'$= \sqrt{{({w1:.2f})^2 + ({w2:.2f})^2}}$' + '\n\n'
        rf'$= \sqrt{{{w1**2:.3f} + {w2**2:.3f}}}$' + '\n\n'
        rf'$= \sqrt{{{w1**2 + w2**2:.3f}}}$' + '\n\n'
        rf'$= {norm:.4f}$'
    )