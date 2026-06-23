"""
Figure 5 — Interactive Gradient Descent Explorer
=================================================

A simple 1D loss function is shown as a curve. The user controls:

    - Starting position: where on the curve the algorithm begins
    - Learning rate:     how large each step is
    - Number of steps:   how many iterations to run

The figure shows the path taken by gradient descent as a dotted orange line.
A printed step-by-step trace below the plot shows the parameter value, loss,
and gradient at each iteration.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_5 import show
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
from ipywidgets import interactive_output, FloatSlider, IntSlider, VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the interactive Figure 5 gradient descent explorer."""

    plt.close('Notebook4 Figure 5')

    # ── Define a simple loss function ─────────────────────────────────────────
    def loss_fn(x):
        return 0.3 * x**2 + 0.5 * np.sin(2 * x) + 0.1 * x

    def gradient_fn(x):
        # The gradient is the slope of the loss curve at position x.
        # A positive gradient means the curve is sloping upward to the right;
        # a negative gradient means it is sloping downward.
        return 0.6 * x + 1.0 * np.cos(2 * x) + 0.1

    x_curve = np.linspace(-4, 4, 500)
    y_curve  = loss_fn(x_curve)

    true_min_x = x_curve[np.argmin(y_curve)]
    true_min_y = loss_fn(true_min_x)

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_START = -3.0
    DEFAULT_LR    = 0.1
    DEFAULT_STEPS = 15

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook4 Figure 5', figsize=(10, 6))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    ax.plot(x_curve, y_curve, color='steelblue', linewidth=2.5,
            label='Loss function', zorder=1)
    ax.scatter([true_min_x], [true_min_y], color='red', s=200, marker='*',
               zorder=5, label=f'True minimum (θ = {true_min_x:.2f})',
               edgecolors='k', lw=0.8)

    path_line, = ax.plot([], [], 'o--', color='orange', markersize=7,
                         linewidth=1.5, alpha=0.85, zorder=3, label='Descent path')
    start_dot  = ax.scatter([], [], color='green', s=150, zorder=6,
                            edgecolors='k', lw=1.2, label='Starting point')
    end_dot    = ax.scatter([], [], color='black', s=150, zorder=6,
                            marker='D', edgecolors='k', lw=1.2, label='Final position')

    ax.set_xlabel('Parameter value (θ)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(y_curve.min() - 0.3, y_curve.max() + 0.3)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()

    # ── Output widget for the printed trace ───────────────────────────────────
    trace_out = widgets.Output()

    # ── Update function ───────────────────────────────────────────────────────
    def update(start, lr, n_steps):
        path = [start]
        x    = start
        for _ in range(n_steps):
            grad = gradient_fn(x)
            x    = x - lr * grad
            path.append(x)

        path   = np.array(path)
        path_y = loss_fn(path)

        path_line.set_data(path, path_y)
        start_dot.set_offsets([[path[0],  path_y[0]]])
        end_dot.set_offsets([[path[-1], path_y[-1]]])

        ax.set_title(
            f'Start = {start:.2f}  |  Learning rate = {lr}  |  Steps = {n_steps}\n'
            f'Final: θ = {path[-1]:.4f}  |  Loss = {path_y[-1]:.4f}  |  '
            f'True minimum: θ = {true_min_x:.4f}',
            fontsize=10,
        )

        # Print the step-by-step trace into the Output widget
        trace_out.clear_output(wait=True)
        with trace_out:
            print(f"{'Step':<6} {'θ (parameter)':>16} {'Loss':>10} {'Gradient':>12} {'Action'}")
            print('-' * 65)
            for i, (xi, yi) in enumerate(zip(path, path_y)):
                grad_i = gradient_fn(xi)
                if abs(grad_i) < 0.001:
                    action = 'at minimum — stop'
                elif grad_i > 0:
                    action = 'slope upward → step left'
                else:
                    action = 'slope downward → step right'
                print(f'{i:<6} {xi:>16.4f} {yi:>10.4f} {grad_i:>12.4f}   {action}')

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    start_s = FloatSlider(
        value=DEFAULT_START, min=-4.0, max=4.0, step=0.1,
        description='Start (θ₀)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='350px'),
        continuous_update=True,
    )
    lr_s = FloatSlider(
        value=DEFAULT_LR, min=0.01, max=0.8, step=0.01,
        description='Learning rate',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='350px'),
        continuous_update=True,
    )
    steps_s = IntSlider(
        value=DEFAULT_STEPS, min=1, max=50, step=1,
        description='Steps',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='350px'),
        continuous_update=True,
    )

    start_b = widgets.BoundedFloatText(value=DEFAULT_START, min=-4.0, max=4.0,
                                        step=0.1, description='',
                                        layout=widgets.Layout(width='90px'))
    lr_b    = widgets.BoundedFloatText(value=DEFAULT_LR, min=0.01, max=0.8,
                                        step=0.01, description='',
                                        layout=widgets.Layout(width='90px'))
    steps_b = widgets.BoundedIntText(value=DEFAULT_STEPS, min=1, max=50,
                                      step=1, description='',
                                      layout=widgets.Layout(width='90px'))

    widgets.jslink((start_s, 'value'), (start_b, 'value'))
    widgets.jslink((lr_s,    'value'), (lr_b,    'value'))
    widgets.jslink((steps_s, 'value'), (steps_b, 'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        start_s.value = DEFAULT_START
        lr_s.value    = DEFAULT_LR
        steps_s.value = DEFAULT_STEPS

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Gradient descent controls</b>'),
        HBox([start_s,  start_b]),
        HBox([lr_s,     lr_b]),
        HBox([steps_s,  steps_b]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        'start':   start_s,
        'lr':      lr_s,
        'n_steps': steps_s,
    })

    display(controls, out)
    display(trace_out)