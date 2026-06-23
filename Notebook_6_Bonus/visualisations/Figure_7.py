"""
Figure 7 — Activation Functions: Sigmoid, Tanh, and ReLU
===========================================================

Shows the three activation functions discussed in Section 10.3 — sigmoid,
tanh, and ReLU — plotted side by side, each as a smooth curve over
z from -5 to 5. A single slider controls a shared input value z, and a
vertical marker on all three plots simultaneously shows where that value
of z falls on each curve, together with the resulting output value of
each function. This lets the reader directly compare how the same
weighted sum is treated differently by each activation function, rather
than reading the three curves separately.

Each panel shows the function's curve, its output range, dashed
reference lines at z=0 and output=0, and a live marker plus readout
giving the current output value to four decimal places. ReLU's panel
also notes that the function is not differentiable at z=0.

No "fires / does not fire" label is shown, since sigmoid, tanh, and ReLU
do not share a common notion of a firing threshold the way the Heaviside
function in Section 5.3 does. Sigmoid's natural midpoint is 0.5, tanh's
is 0, and ReLU has no bounded output at all, so imposing a single
threshold across all three would be misleading. Instead, the figure
simply reports each function's exact output value for the current z,
leaving the reader to compare the numbers directly.

A single slider, rather than three independent ones, is used
deliberately: the comparison only works if all three panels are showing
the response to the same input. Moving the slider sweeps the same z
value across sigmoid, tanh, and ReLU simultaneously.

The key teaching point: although sigmoid, tanh, and ReLU all turn a
weighted sum into a useful activation, they behave very differently for
the same input, particularly for large positive or negative z, where
sigmoid and tanh flatten out (the vanishing gradient problem) while ReLU
continues to grow linearly.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_7 import show
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
    """Render the interactive Figure 7 activation function comparison."""

    plt.close('Notebook6bonus Figure 7')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_CURVE  = '#4C78A8'
    COL_MARKER = '#b23b3b'
    COL_TEXT   = '#222222'
    COL_MUTED  = '#888888'

    DEFAULT_Z = 0.0

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def tanh(z):
        return np.tanh(z)

    def relu(z):
        return np.maximum(0.0, z)

    funcs = [
        {'name': 'Sigmoid', 'fn': sigmoid, 'range': '(0, 1)', 'ylim': (-0.25, 1.25)},
        {'name': 'Tanh', 'fn': tanh, 'range': '(-1, 1)', 'ylim': (-1.25, 1.25)},
        {'name': 'ReLU', 'fn': relu, 'range': '[0, infinity)', 'ylim': (-0.6, 5.3)},
    ]

    z_plot = np.linspace(-5, 5, 400)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, num='Notebook6bonus Figure 7', figsize=(13.5, 4.8),
                              constrained_layout=True)

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    fig.suptitle('Figure 7: Sigmoid, tanh, and ReLU — the same input z, '
                 'three different outputs', fontsize=12.5, color=COL_TEXT)

    markers = []
    readouts = []
    value_texts = []

    for ax, spec in zip(axes, funcs):
        y_plot = spec['fn'](z_plot)
        ax.plot(z_plot, y_plot, color=COL_CURVE, lw=2.4, zorder=3)
        ax.axhline(0, color='#bbbbbb', lw=0.9, linestyle='--', zorder=1)
        ax.axvline(0, color='#bbbbbb', lw=0.9, linestyle='--', zorder=1)

        ax.set_xlim(-5, 5)
        ax.set_ylim(*spec['ylim'])
        ax.set_xlabel('z (weighted sum)', fontsize=10)
        ax.set_ylabel('Activation output', fontsize=10)
        ax.set_title(spec['name'], fontsize=12, color=COL_TEXT)
        ax.grid(True, alpha=0.2)

        ax.text(0.03, 0.95, f"Output range: {spec['range']}", transform=ax.transAxes,
                fontsize=8.5, color=COL_MUTED, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#dddddd', alpha=0.9))

        if spec['name'] == 'ReLU':
            ax.annotate('Not differentiable\nat z = 0', xy=(0, 0), xytext=(0.04, 0.30),
                        textcoords='axes fraction', fontsize=8, color=COL_MUTED,
                        ha='left',
                        arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.8,
                                         connectionstyle='arc3,rad=0.2'))

        # vertical marker line tracking the shared z slider
        marker_line = ax.axvline(DEFAULT_Z, color=COL_MARKER, lw=1.4,
                                  linestyle=':', zorder=2)
        marker_dot, = ax.plot([DEFAULT_Z], [spec['fn'](DEFAULT_Z)], 'o',
                               color=COL_MARKER, markersize=9, zorder=5,
                               markeredgecolor='white', markeredgewidth=1.2)
        value_txt = ax.text(0.97, 0.05, '', transform=ax.transAxes, fontsize=9.5,
                             ha='right', va='bottom', color=COL_MARKER,
                             fontweight='bold')

        markers.append(marker_line)
        readouts.append(marker_dot)
        value_texts.append(value_txt)

    # ── Update function ───────────────────────────────────────────────────────
    def update(z):
        for spec, marker_line, marker_dot, value_txt in zip(
                funcs, markers, readouts, value_texts):
            output = float(spec['fn'](np.array(z)))
            marker_line.set_xdata([z, z])
            marker_dot.set_data([z], [output])
            value_txt.set_text(f'z = {z:.2f}\noutput = {output:.4f}')
        fig.canvas.draw_idle()

    # ── Slider ───────────────────────────────────────────────────────────────
    z_s = FloatSlider(value=DEFAULT_Z, min=-5.0, max=5.0, step=0.1,
                       description='z', style={'description_width': '30px'},
                       layout=widgets.Layout(width='400px'),
                       continuous_update=True)

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                                layout=widgets.Layout(width='100px'))

    def on_reset(b):
        z_s.value = DEFAULT_Z

    reset_btn.on_click(on_reset)

    controls = VBox([
        HTML('<b>Move z and compare how each activation function responds</b>'),
        HBox([z_s, reset_btn]),
    ])

    out = interactive_output(update, {'z': z_s})

    display(controls, out)