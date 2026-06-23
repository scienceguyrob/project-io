"""
Figure 1 — Interactive 1D Decision Boundary Explorer
=====================================================

Two species of penguin are represented by their flipper lengths, plotted as
a 1D scatter (all points sit on y = 0). The user drags a vertical threshold
line left and right to find the value that best separates the two species.

An accuracy readout updates live, showing how many points are correctly
classified and how many from each species are misclassified.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_1 import show
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
from ipywidgets import interactive_output, FloatSlider, HBox
from IPython.display import display


def show():
    """Render the interactive Figure 1 decision boundary explorer."""

    plt.close('Notebook2 Figure 1')

    # ── Data generation ───────────────────────────────────────────────────────
    # Fixed seed (0) means every user sees identical data points.
    rng = np.random.default_rng(0)

    # Species A: shorter flippers centred around 180 mm
    species_a = np.array(rng.normal(loc=180, scale=8, size=40))
    # Species B: longer flippers centred around 210 mm
    species_b = np.array(rng.normal(loc=210, scale=8, size=40))

    # Combine both species for vectorised accuracy calculation
    all_x = np.concatenate([species_a, species_b])
    # True labels: 0 = Species A (first 40), 1 = Species B (last 40)
    true_labels = np.array([0] * 40 + [1] * 40)

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook2 Figure 1', figsize=(8, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # Plot both species as 1D scatter points — all sitting on y = 0
    ax.scatter(species_a, [0] * len(species_a),
               color='steelblue', s=60, alpha=0.7, label='Species A',
               edgecolors='k', linewidth=0.4, zorder=3)
    ax.scatter(species_b, [0] * len(species_b),
               color='tomato', s=60, alpha=0.7, label='Species B',
               edgecolors='k', linewidth=0.4, zorder=3)

    # Vertical boundary line — stored so set_xdata() can move it later
    boundary = ax.axvline(x=195, color='black', linewidth=2, linestyle='--',
                          label='Boundary: x = 195')

    ax.set_xlabel('Flipper length (mm)')
    ax.set_yticks([])   # Hide y-axis ticks — the y position carries no meaning here
    ax.set_xlim(145, 245)
    ax.grid(True, alpha=0.2, axis='x')
    ax.legend(loc='upper left')

    # Accuracy annotation in the top-right corner — updated in place via set_text()
    acc_text = ax.text(
        0.98, 0.95, '', transform=ax.transAxes,
        ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
    )

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever the threshold slider changes.
    # Mutates existing artists only — no new plot objects are created.
    def update(threshold):
        # Move the vertical boundary line to the new threshold position
        boundary.set_xdata([threshold, threshold])
        boundary.set_label(f'Boundary: x = {threshold:.1f}')

        # Points left of the threshold are predicted Species A (0),
        # points on or right of it are predicted Species B (1)
        predicted = (all_x >= threshold).astype(int)

        correct  = np.sum(predicted == true_labels)
        accuracy = correct / len(true_labels) * 100
        a_wrong  = np.sum(predicted[:40] != 0)   # Species A predicted as B
        b_wrong  = np.sum(predicted[40:] != 1)   # Species B predicted as A

        acc_text.set_text(
            f'Accuracy: {correct}/{len(true_labels)} ({accuracy:.1f}%)\n'
            f'Species A misclassified: {a_wrong}   |   Species B misclassified: {b_wrong}'
        )

        ax.set_title(f'Figure 1: Single threshold x = {threshold:.1f} mm separates the two species')
        ax.legend(loc='upper left')
        fig.canvas.draw_idle()
        plt.tight_layout()

    # ── Slider and text box ───────────────────────────────────────────────────
    t_slider = FloatSlider(
        value=195, min=145.0, max=245.0, step=0.5,
        description='Threshold (mm)',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px'),
        continuous_update=True,
    )

    t_box = widgets.BoundedFloatText(
        value=195, min=145.0, max=245.0, step=0.5,
        description='',
        layout=widgets.Layout(width='100px'),
    )

    # jslink keeps the slider and text box in sync in the browser
    widgets.jslink((t_slider, 'value'), (t_box, 'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(
        description='Reset to x = 195',
        button_style='info',
        layout=widgets.Layout(width='160px'),
    )

    def on_reset(b):
        t_slider.value = 195

    reset_btn.on_click(on_reset)

    # ── Wire slider to update and display ─────────────────────────────────────
    controls = HBox([t_slider, t_box, reset_btn])
    out = interactive_output(update, {'threshold': t_slider})

    display(controls, out)