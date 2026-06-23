"""
Figure 3 — The Effect of Slope (m) and Intercept (c) on a Linear Function
==========================================================================

Left subplot  (Figure 3a): four reference lines with different slopes, plus
              one interactive red dashed line whose slope the user adjusts.
              Intercept is fixed at c = 1 throughout.

Right subplot (Figure 3b): four reference lines with different intercepts,
              plus one interactive red dashed line whose intercept the user
              adjusts. Slope is fixed at m = 1 throughout.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_3 import show
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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox, BoundedFloatText, Button
from IPython.display import display


def show():
    """Render the interactive Figure 3 plots inline in a Jupyter notebook."""

    DEFAULT_M = 1.0
    DEFAULT_C = 0.0

    x = np.linspace(-5, 5, 200)

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.canvas.resizable = True
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = False
    fig.canvas.toolbar_position = 'right'

    ax_left  = axes[0]
    ax_right = axes[1]

    # ── Figure 3a: four static lines + one interactive red line ───────────────
    slope_examples = [
        ( 2,   'm = 2   (steep positive)'),
        ( 0.5, 'm = 0.5 (gentle positive)'),
        ( 0,   'm = 0   (horizontal)'),
        (-1,   'm = -1  (negative)'),
    ]
    for m_val, label in slope_examples:
        ax_left.plot(x, m_val * x + 1, linewidth=2, label=label)

    (line_left,) = ax_left.plot(
        x, DEFAULT_M * x + 1,
        color='red', linewidth=2.5, linestyle='--',
        label=f'm = {DEFAULT_M:.1f} (adjust me)', zorder=5,
    )

    ax_left.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax_left.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax_left.set_title('Figure 3a: Varying slope m  (c = 1 fixed)')
    ax_left.set_xlabel('x')
    ax_left.set_ylabel('y')
    ax_left.legend(fontsize=8)
    ax_left.set_ylim(-8, 12)
    ax_left.grid(True, alpha=0.3)

    # ── Figure 3b: four static lines + one interactive red line ───────────────
    intercept_examples = [
        ( 3,  'c = 3  (shifted up)'),
        ( 1,  'c = 1'),
        ( 0,  'c = 0  (through origin)'),
        (-2,  'c = -2 (shifted down)'),
    ]
    for c_val, label in intercept_examples:
        ax_right.plot(x, x + c_val, linewidth=2, label=label)

    (line_right,) = ax_right.plot(
        x, x + DEFAULT_C,
        color='red', linewidth=2.5, linestyle='--',
        label=f'c = {DEFAULT_C:.1f} (adjust me)', zorder=5,
    )

    ax_right.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax_right.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax_right.set_title('Figure 3b: Varying intercept c  (m = 1 fixed)')
    ax_right.set_xlabel('x')
    ax_right.set_ylabel('y')
    ax_right.legend(fontsize=8)
    ax_right.set_ylim(-8, 12)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever either slider changes.
    # Only mutates existing artists — never creates new ones.
    def update(m, c):
        line_left.set_ydata(m * x + 1)
        line_left.set_label(f'm = {m:.2f} (adjust me)')
        ax_left.legend(fontsize=8)

        line_right.set_ydata(x + c)
        line_right.set_label(f'c = {c:.2f} (adjust me)')
        ax_right.legend(fontsize=8)

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    m_slider = FloatSlider(
        value=DEFAULT_M, min=-5.0, max=5.0, step=0.1,
        description='m (slope)',
        style={'description_width': '90px'},
        continuous_update=True,
    )
    c_slider = FloatSlider(
        value=DEFAULT_C, min=-5.0, max=5.0, step=0.1,
        description='c (intercept)',
        style={'description_width': '90px'},
        continuous_update=True,
    )

    # Paired text boxes so users can type values directly
    text_m = BoundedFloatText(
        value=DEFAULT_M, min=-5.0, max=5.0, step=0.1,
        description='', layout=widgets.Layout(width='100px'),
    )
    text_c = BoundedFloatText(
        value=DEFAULT_C, min=-5.0, max=5.0, step=0.1,
        description='', layout=widgets.Layout(width='100px'),
    )

    # jslink keeps slider and text box in sync in the browser without a Python round-trip
    widgets.jslink((m_slider, 'value'), (text_m, 'value'))
    widgets.jslink((c_slider, 'value'), (text_c, 'value'))

    # Reset button returns both sliders to their defaults
    btn_reset = Button(
        description='Reset', button_style='warning',
        layout=widgets.Layout(width='100px'),
    )

    def on_reset(btn):
        m_slider.value = DEFAULT_M
        c_slider.value = DEFAULT_C

    btn_reset.on_click(on_reset)

    # ── Wire sliders to update function ───────────────────────────────────────
    # interactive_output handles rendering the figure canvas automatically,
    # matching the pattern used in Figure 2.
    controls = VBox([
        HBox([m_slider, text_m]),
        HBox([c_slider, text_c]),
        btn_reset,
    ])
    out = interactive_output(update, {'m': m_slider, 'c': c_slider})

    display(controls, out)