"""
Figure 6 — Interactive Gradient Explorer
=========================================

Shows the loss curve and allows the user to place a point anywhere on it
by dragging a slider. At that point the figure draws:

    - The tangent line — the straight line that just touches the curve at
      that point, whose slope IS the gradient dL/dθ
    - An arrow showing the direction gradient descent would step
    - A live readout of the gradient value, its sign interpretation,
      and the next θ value for a given learning rate

This makes dL/dθ concrete and visual before users see it in equations.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_6 import show
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
    """Render the interactive Figure 6 gradient explorer."""

    plt.close('Notebook4 Figure 6')

    # ── Loss function and its gradient ────────────────────────────────────────
    def loss_fn(x):
        return 0.3 * x**2 + 0.5 * np.sin(2 * x) + 0.1 * x

    def gradient_fn(x):
        # Analytical gradient — the exact slope of loss_fn at any point x
        return 0.6 * x + 1.0 * np.cos(2 * x) + 0.1

    x_curve = np.linspace(-4, 4, 500)
    y_curve  = loss_fn(x_curve)

    true_min_x = x_curve[np.argmin(y_curve)]
    true_min_y = loss_fn(true_min_x)

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_THETA = -2.0
    DEFAULT_LR    = 0.2

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook4 Figure 6', figsize=(10, 6))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # Static loss curve
    ax.plot(x_curve, y_curve, color='steelblue', linewidth=2.5,
            label='Loss function L(θ)', zorder=1)
    ax.scatter([true_min_x], [true_min_y], color='red', s=200, marker='*',
               zorder=5, label=f'True minimum (θ = {true_min_x:.2f})',
               edgecolors='k', lw=0.8)

    # ── Mutable artists — all updated on every slider change ──────────────────

    # The current point on the curve
    current_dot = ax.scatter([], [], color='black', s=120, zorder=6,
                             edgecolors='white', lw=1.5, label='Current θ')

    # Tangent line — a short straight line with slope = gradient at current θ
    tangent_line, = ax.plot([], [], color='tomato', linewidth=2.5,
                            linestyle='-', zorder=4,
                            label='Tangent line (slope = gradient)')

    # Arrow showing the direction of the next step
    step_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle='->', color='green',
                                            lw=2.5))

    # Next position marker
    next_dot = ax.scatter([], [], color='green', s=120, zorder=6,
                          edgecolors='k', lw=1.0, marker='D',
                          label='Next θ after one step')

    # Vertical dashed line dropping from the current point to the x-axis
    vline, = ax.plot([], [], color='grey', linewidth=1.0,
                     linestyle=':', zorder=2)

    ax.set_xlabel('Parameter value (θ)', fontsize=11)
    ax.set_ylabel('Loss L(θ)', fontsize=11)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(y_curve.min() - 0.5, y_curve.max() + 0.5)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()

    # ── Readout widget ────────────────────────────────────────────────────────
    readout = widgets.Output()

    # ── Update function ───────────────────────────────────────────────────────
    def update(theta, lr):
        L      = loss_fn(theta)
        grad   = gradient_fn(theta)
        next_theta = theta - lr * grad
        next_L     = loss_fn(next_theta)

        # ── Update current point ──────────────────────────────────────────────
        current_dot.set_offsets([[theta, L]])

        # ── Draw the tangent line ─────────────────────────────────────────────
        # A tangent line touches the curve at one point and has slope = gradient.
        # We draw it as a short segment centred on the current point.
        half_width = 0.8
        tx = np.array([theta - half_width, theta + half_width])
        ty = L + grad * (tx - theta)   # y = L + gradient × (x - theta)
        tangent_line.set_data(tx, ty)

        # ── Draw the step arrow ───────────────────────────────────────────────
        # The arrow shows the direction and distance of the next update.
        # It runs from (theta, L) to (next_theta, L) horizontally,
        # so the user can see the step size on the x-axis directly.
        step_arrow.set_position((next_theta, L))
        step_arrow.xy = (next_theta, L)
        step_arrow.xytext = (theta, L)
        step_arrow.arrowprops = dict(arrowstyle='->', color='green', lw=2.5)

        # ── Update next position marker ───────────────────────────────────────
        next_dot.set_offsets([[next_theta, next_L]])

        # ── Vertical dashed line from current point to x-axis ─────────────────
        vline.set_data([theta, theta], [0, L])

        # ── Gradient sign interpretation ──────────────────────────────────────
        if abs(grad) < 0.01:
            sign_text  = '≈ 0  →  curve is flat here — at or near a minimum'
            step_dir   = 'No meaningful step taken'
            arrow_col  = 'gray'
        elif grad > 0:
            sign_text  = f'> 0  →  curve slopes UPWARD to the right'
            step_dir   = f'Step LEFT  (θ decreases by {lr * grad:.4f})'
            arrow_col  = 'green'
        else:
            sign_text  = f'< 0  →  curve slopes DOWNWARD to the right'
            step_dir   = f'Step RIGHT  (θ increases by {abs(lr * grad):.4f})'
            arrow_col  = 'green'

        ax.set_title(
            f'Current θ = {theta:.3f}  |  '
            f'Loss L(θ) = {L:.4f}  |  '
            f'Gradient dL/dθ = {grad:.4f}',
            fontsize=11,
        )

        # ── Print the live readout ────────────────────────────────────────────
        readout.clear_output(wait=True)
        with readout:
            print('=' * 60)
            print(f'  Current position')
            print(f'    θ               = {theta:.4f}')
            print(f'    Loss  L(θ)      = {L:.4f}')
            print()
            print(f'  Gradient  dL/dθ   = {grad:.4f}')
            print(f'  Sign interpretation: {sign_text}')
            print()
            print(f'  Update rule:  θ_new = θ - α × dL/dθ')
            print(f'              = {theta:.4f} - {lr} × ({grad:.4f})')
            print(f'              = {theta:.4f} - ({lr * grad:.4f})')
            print(f'              = {next_theta:.4f}')
            print()
            print(f'  Direction:    {step_dir}')
            print(f'  New loss:     L(θ_new) = {next_L:.4f}')
            improvement = L - next_L
            print(f'  Improvement:  {improvement:+.4f}  '
                  f'({"loss reduced ✓" if improvement > 0 else "loss increased — learning rate too large ✗" if improvement < -0.001 else "no change"})')
            print('=' * 60)

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    theta_s = FloatSlider(
        value=DEFAULT_THETA, min=-4.0, max=4.0, step=0.05,
        description='θ (position)',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )
    lr_s = FloatSlider(
        value=DEFAULT_LR, min=0.01, max=0.9, step=0.01,
        description='Learning rate α',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )

    theta_b = widgets.BoundedFloatText(value=DEFAULT_THETA, min=-4.0, max=4.0,
                                        step=0.05, description='',
                                        layout=widgets.Layout(width='90px'))
    lr_b    = widgets.BoundedFloatText(value=DEFAULT_LR, min=0.01, max=0.9,
                                        step=0.01, description='',
                                        layout=widgets.Layout(width='90px'))

    widgets.jslink((theta_s, 'value'), (theta_b, 'value'))
    widgets.jslink((lr_s,    'value'), (lr_b,    'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        theta_s.value = DEFAULT_THETA
        lr_s.value    = DEFAULT_LR

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Gradient explorer controls</b>'),
        HBox([theta_s, theta_b]),
        HBox([lr_s,    lr_b]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        'theta': theta_s,
        'lr':    lr_s,
    })

    display(controls, out)
    display(readout)