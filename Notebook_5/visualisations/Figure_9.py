"""
Figure 9 — Interactive Sigmoid Function Explorer
=================================================

Two side-by-side panels:

Left panel  (Figure 9a): the sigmoid curve with a movable point the user
             can drag along it. The current z value, sigmoid output, and
             plain-English interpretation are shown live. Shaded regions
             show where the model predicts class 0 vs class 1.

Right panel (Figure 9b): a live bar showing the predicted probability at
             the current z, together with a clear visual of the 0.5 decision
             threshold and which class would be predicted.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_9 import show
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
    """Render the interactive Figure 9 sigmoid explorer."""

    plt.close('Notebook5 Figure 9')

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z_curve = np.linspace(-8, 8, 400)
    s_curve = sigmoid(z_curve)

    DEFAULT_Z = 0.0

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, num='Notebook5 Figure 9', figsize=(10, 5),
        gridspec_kw={'width_ratios': [3, 2]},
    )
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'bottom'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: sigmoid curve ─────────────────────────────────────────────
    ax_left.plot(z_curve, s_curve, color='steelblue', linewidth=2.5,
                 label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', zorder=2)

    # Shade class 0 and class 1 regions
    ax_left.fill_between(z_curve, s_curve, 0.5,
                          where=(s_curve <= 0.5),
                          alpha=0.12, color='steelblue', label='Predict class 0')
    ax_left.fill_between(z_curve, s_curve, 0.5,
                          where=(s_curve > 0.5),
                          alpha=0.12, color='tomato', label='Predict class 1')

    # Reference lines
    ax_left.axhline(0.5, color='red',   linewidth=1.5, linestyle='--',
                    label='Decision threshold = 0.5')
    ax_left.axhline(0,   color='gray',  linewidth=0.8, linestyle=':')
    ax_left.axhline(1,   color='gray',  linewidth=0.8, linestyle=':')
    ax_left.axvline(0,   color='black', linewidth=0.8, linestyle=':')

    # Mutable: moving dot on the curve
    moving_dot = ax_left.scatter([], [], color='black', s=120, zorder=6,
                                  edgecolors='white', lw=1.5)

    # Mutable: vertical dashed line from x-axis to the dot
    v_line, = ax_left.plot([], [], color='black', linewidth=1.2,
                            linestyle=':', zorder=3)

    # Mutable: horizontal dashed line from dot to y-axis
    h_line, = ax_left.plot([], [], color='black', linewidth=1.2,
                            linestyle=':', zorder=3)

    ax_left.set_xlabel('$z$  (linear combination of features)', fontsize=10)
    ax_left.set_ylabel(r'$\sigma(z)$ — predicted probability', fontsize=10)
    ax_left.set_xlim(-8.2, 8.2)
    ax_left.set_ylim(-0.05, 1.1)
    ax_left.legend(fontsize=8, loc='upper left')
    ax_left.grid(True, alpha=0.2)

    # ── Right panel: probability gauge ───────────────────────────────────────
    ax_right.axis('off')
    gauge_text = ax_right.text(
        0.5, 0.5, '',
        transform=ax_right.transAxes,
        ha='center', va='center',
        fontsize=12, family='monospace',
        linespacing=2.0,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f9f9e8',
                  edgecolor='#bbb', linewidth=1.2),
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    # ── Update function ───────────────────────────────────────────────────────
    def update(z):
        s = sigmoid(z)

        # Update moving dot and cross-hair lines
        moving_dot.set_offsets([[z, s]])
        v_line.set_data([z, z], [0, s])
        h_line.set_data([-8.2, z], [s, s])

        # Determine prediction
        pred       = 1 if s > 0.5 else 0
        confidence = s if pred == 1 else 1 - s
        if confidence > 0.9:
            cert = 'very confident'
        elif confidence > 0.7:
            cert = 'fairly confident'
        else:
            cert = 'uncertain'

        ax_left.set_title(
            f'Figure 9a: The sigmoid function\n'
            f'z = {z:.2f}  →  σ(z) = {s:.4f}  →  predict class {pred}',
            fontsize=10,
        )

        # Colour the dot by prediction
        moving_dot.set_facecolor('tomato' if pred == 1 else 'steelblue')

        # Update the gauge panel
        bar_filled = int(s * 20)
        bar_empty  = 20 - bar_filled
        bar_str    = '█' * bar_filled + '░' * bar_empty

        gauge_text.set_text(
            f' z = {z:.3f}\n'
            f'\n'
            f' σ(z) = 1 / (1 + e^{-z:.2f})\n'
            f'      = 1 / (1 + {np.exp(-z):.3f})\n'
            f'      = {s:.4f}\n'
            f'\n'
            f' [{bar_str}]\n'
            f'   0        0.5        1\n'
            f'\n'
            f' Threshold: σ(z) > 0.5 → class 1\n'
            f' Prediction: class {pred}  ({cert})\n'
            f' Probability of class 1: {s:.1%}'
        )

        gauge_text.get_bbox_patch().set_facecolor(
            '#fff0f0' if pred == 1 else '#f0f0ff'
        )

        fig.canvas.draw_idle()

    # ── Slider ────────────────────────────────────────────────────────────────
    z_s = FloatSlider(
        value=DEFAULT_Z, min=-8.0, max=8.0, step=0.05,
        description='z value:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='400px'),
        continuous_update=True,
        readout_format='.2f',
    )
    z_b = widgets.BoundedFloatText(
        value=DEFAULT_Z, min=-8.0, max=8.0, step=0.05,
        description='', layout=widgets.Layout(width='90px'),
    )
    widgets.jslink((z_s, 'value'), (z_b, 'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        z_s.value = DEFAULT_Z

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Drag z to see how the sigmoid maps any number to a probability:</b>'),
        HBox([z_s, z_b, reset_btn]),
        sep,
    ])

    out = interactive_output(update, {'z': z_s})
    display(controls, out)