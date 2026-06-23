"""
Figure 6b — Interactive Gradient Derivation Explorer
=====================================================

Shows how the gradient formula 2θ arises from making a tiny nudge h
and watching what happens as h shrinks toward zero.

Two side-by-side panels:

Left panel:  The loss curve L(θ) = θ². The user positions a point θ
             on the curve and a nudge h is shown. A secant line (the line
             joining the two points) is drawn — its slope is the gradient
             approximation (L(θ+h) - L(θ)) / h. As h shrinks, the secant
             line rotates to become the tangent line, whose slope is
             exactly 2θ.

Right panel: Shows the gradient approximation (2θ + h) plotted as a
             horizontal bar, alongside the exact gradient (2θ), so
             user can see how close the approximation is for
             different values of h, and that as h → 0 they become equal.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_6b import show
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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the interactive Figure 6b gradient derivation explorer."""

    plt.close('Notebook4 Figure 6b')

    # ── Loss function ─────────────────────────────────────────────────────────
    def L(x):
        return x ** 2

    theta_curve = np.linspace(-4, 4, 400)
    loss_curve  = L(theta_curve)

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_THETA = 2.0
    DEFAULT_H     = 2.0

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, num='Notebook4 Figure 6b', figsize=(10, 5),
        gridspec_kw={'width_ratios': [3, 2]},
    )
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: loss curve with nudge illustration ────────────────────────
    ax_left.plot(theta_curve, loss_curve, color='steelblue', linewidth=2.5,
                 label='L(θ) = θ²', zorder=1)

    # Mutable artists
    # Point at θ
    dot_theta  = ax_left.scatter([], [], color='black', s=100, zorder=6,
                                 label='Current θ')
    # Point at θ + h
    dot_nudge  = ax_left.scatter([], [], color='tomato', s=100, zorder=6,
                                 label='θ + h  (nudged point)')
    # Secant line joining the two points — its slope is the approximation
    secant_line, = ax_left.plot([], [], color='tomato', linewidth=2,
                                linestyle='--', zorder=4,
                                label='Secant line (slope ≈ gradient)')
    # Tangent line — slope is exactly 2θ
    tangent_line, = ax_left.plot([], [], color='green', linewidth=2,
                                 linestyle='-', zorder=3,
                                 label='Tangent line (slope = 2θ, exact gradient)')

    # Vertical and horizontal dashed lines showing the rise and run
    rise_line, = ax_left.plot([], [], color='grey', linewidth=1.2,
                              linestyle=':', zorder=2)
    run_line,  = ax_left.plot([], [], color='grey', linewidth=1.2,
                              linestyle=':', zorder=2)

    ax_left.set_xlabel('θ', fontsize=11)
    ax_left.set_ylabel('L(θ) = θ²', fontsize=11)
    ax_left.set_xlim(-4.2, 4.2)
    ax_left.set_ylim(-0.5, 18)
    ax_left.grid(True, alpha=0.2)
    ax_left.legend(fontsize=8, loc='upper center')

    # ── Right panel: approximation vs exact gradient ──────────────────────────
    ax_right.axis('off')
    formula_text = ax_right.text(
        0.05, 0.95, '',
        transform=ax_right.transAxes,
        ha='left', va='top',
        fontsize=10, family='monospace',
        linespacing=1.8,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f9f9e8',
                  edgecolor='#bbb', linewidth=1.2),
    )

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    def update(theta, h):
        L_theta   = L(theta)
        L_nudged  = L(theta + h)

        # Gradient approximation: (L(θ+h) - L(θ)) / h  =  2θ + h
        approx_gradient = (L_nudged - L_theta) / h
        exact_gradient  = 2 * theta
        error           = abs(approx_gradient - exact_gradient)

        # ── Update left panel ─────────────────────────────────────────────────

        # The two points on the curve
        dot_theta.set_offsets([[theta,     L_theta]])
        dot_nudge.set_offsets([[theta + h, L_nudged]])

        # Secant line: extends slightly beyond both points
        margin  = 0.6
        x_sec   = np.array([theta - margin, theta + h + margin])
        # Secant slope = rise / run = (L(θ+h) - L(θ)) / h
        y_sec   = L_theta + approx_gradient * (x_sec - theta)
        secant_line.set_data(x_sec, y_sec)

        # Tangent line at θ: slope = exact gradient = 2θ
        x_tan = np.array([theta - margin * 2, theta + margin * 2])
        y_tan = L_theta + exact_gradient * (x_tan - theta)
        tangent_line.set_data(x_tan, y_tan)

        # Dotted lines showing the rise (vertical) and run (horizontal)
        rise_line.set_data([theta + h, theta + h], [L_theta, L_nudged])
        run_line.set_data([theta, theta + h], [L_theta, L_theta])

        ax_left.set_title(
            f'θ = {theta:.2f}   h = {h:.3f}\n'
            f'Secant slope = {approx_gradient:.4f}   |   '
            f'Exact gradient (2θ) = {exact_gradient:.4f}',
            fontsize=10,
        )

        # ── Update right panel formula readout ────────────────────────────────
        formula_text.set_text(
            f' Gradient approximation\n'
            f' formula: (2θ + h)\n'
            f' ──────────────────────\n'
            f' θ          = {theta:.3f}\n'
            f' h          = {h:.4f}\n'
            f'\n'
            f' 2θ         = {exact_gradient:.4f}\n'
            f' 2θ + h     = {approx_gradient:.4f}\n'
            f'\n'
            f' Difference = {error:.4f}\n'
            f'\n'
            f' {"✓ h is tiny — approximation" if h < 0.1 else "← shrink h to see"}\n'
            f' {"  is nearly exact" if h < 0.1 else "   approximation improve"}\n'
            f'\n'
            f' As h → 0:\n'
            f'   2θ + h → 2θ\n'
            f'   {approx_gradient:.4f} → {exact_gradient:.4f}'
        )

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    theta_s = FloatSlider(
        value=DEFAULT_THETA, min=-3.5, max=3.5, step=0.1,
        description='θ (position)',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )
    h_s = FloatSlider(
        value=DEFAULT_H, min=0.001, max=3.0, step=0.001,
        description='h (nudge size)',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
        readout_format='.3f',
    )

    theta_b = widgets.BoundedFloatText(value=DEFAULT_THETA, min=-3.5, max=3.5,
                                        step=0.1, description='',
                                        layout=widgets.Layout(width='90px'))
    h_b     = widgets.BoundedFloatText(value=DEFAULT_H, min=0.001, max=3.0,
                                        step=0.001, description='',
                                        layout=widgets.Layout(width='90px'))

    widgets.jslink((theta_s, 'value'), (theta_b, 'value'))
    widgets.jslink((h_s,     'value'), (h_b,     'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        theta_s.value = DEFAULT_THETA
        h_s.value     = DEFAULT_H

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Controls</b>'),
        HTML('<span style="font-size:0.9em;color:#555">'
             'Start with a large h and slowly drag it toward zero.<br>'
             'Watch the red secant line rotate to match the green tangent line.</span>'),
        sep,
        HBox([theta_s, theta_b]),
        HBox([h_s, h_b]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {'theta': theta_s, 'h': h_s})

    display(controls, out)