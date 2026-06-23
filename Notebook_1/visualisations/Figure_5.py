"""
Figure 5 — Interactive Decision Boundary Explorer
==================================================

Two clusters of points (Group A and Group B) are plotted on a scatter chart.
The user adjusts the slope m and intercept c of a straight-line decision
boundary y = mx + c to separate the two groups as accurately as possible.

An accuracy readout updates live, showing how many points are correctly
classified and how many from each group are misclassified.

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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox
from IPython.display import display


def show():
    """Render the interactive Figure 5 decision boundary explorer."""

    # ── Generate the two point clusters (fixed seed so they never change) ─────
    # np.random.default_rng(42) creates a reproducible random number generator.
    # Using a fixed seed means every user sees the same data points.
    rng = np.random.default_rng(42)

    # Group A: centred around (2.5, 1.0) — represents one class, e.g. Species 1
    group_a_x = rng.normal(loc=2.5, scale=0.4, size=30)
    group_a_y = rng.normal(loc=1.0, scale=0.3, size=30)

    # Group B: centred around (4.5, 2.0) — represents the other class
    group_b_x = rng.normal(loc=4.5, scale=0.4, size=30)
    group_b_y = rng.normal(loc=2.0, scale=0.3, size=30)

    # Combine both groups into single arrays for vectorised accuracy calculation
    all_x = np.concatenate([group_a_x, group_b_x])
    all_y = np.concatenate([group_a_y, group_b_y])

    # True labels: 0 = Group A (first 30 points), 1 = Group B (last 30 points)
    true_labels = np.array([0] * 30 + [1] * 30)

    # x-values for drawing the boundary line across the full plot range
    x_line = np.linspace(-1, 8, 200)

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.header_visible = False
    fig.canvas.resizable = True
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # Draw the two scatter groups — these never move so we don't need to store them
    ax.scatter(group_a_x, group_a_y,
               color='steelblue', label='Group A (e.g. Species 1)',
               s=60, edgecolors='k', linewidth=0.5, zorder=3)
    ax.scatter(group_b_x, group_b_y,
               color='tomato', label='Group B (e.g. Species 2)',
               s=60, edgecolors='k', linewidth=0.5, zorder=3)

    # The decision boundary line — stored so set_ydata() can update it later
    (boundary_line,) = ax.plot(
        x_line, 0.5 * x_line + (-0.2),
        'k--', linewidth=2,
        label='Boundary: y = 0.5x + -0.2',
    )

    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.set_title('Figure 5: Decision boundary separating two classes')
    ax.set_xlim(-1, 8)
    ax.set_ylim(-2, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # Text box in the corner showing live accuracy stats.
    # ax.text() returns a Text artist we can update with set_text() later.
    accuracy_text = ax.text(
        0.98, 0.05, '', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
    )

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever either slider changes.
    # Mutates existing artists only — no new plot objects are created.
    def update(m, c):
        # Move the boundary line to its new position
        boundary_line.set_ydata(m * x_line + c)
        boundary_line.set_label(f'Boundary: y = {m:.3f}x + {c:.3f}')

        # A point is predicted Group B (1) if it lies above the boundary line,
        # i.e. if its y-value exceeds m*x + c at that point's x position.
        predicted = (all_y > m * all_x + c).astype(int)

        # Count correct predictions and compute percentage accuracy
        correct = np.sum(predicted == true_labels)
        accuracy = correct / len(true_labels) * 100

        # Count misclassified points per group for the detailed readout
        a_wrong = np.sum(predicted[:30] != 0)
        b_wrong = np.sum(predicted[30:] != 1)

        accuracy_text.set_text(
            f'Accuracy: {correct}/{len(true_labels)} correct ({accuracy:.1f}%)\n'
            f'Group A misclassified: {a_wrong}   |   Group B misclassified: {b_wrong}'
        )

        ax.set_title(f'Figure 5: Decision boundary y = {m:.3f}x + {c:.3f} separating two classes')
        ax.legend(loc='upper left')
        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    m_slider = FloatSlider(
        value=0.5, min=-20.0, max=50.0, step=0.05,
        description='m (slope)',
        style={'description_width': '100px'},
        continuous_update=True,
    )
    c_slider = FloatSlider(
        value=-0.2, min=-50.0, max=50.0, step=0.05,
        description='c (intercept)',
        style={'description_width': '100px'},
        continuous_update=True,
    )

    # Paired text boxes so users can type precise values directly
    m_box = widgets.BoundedFloatText(
        value=0.5, min=-20.0, max=50.0, step=0.05,
        description='', layout=widgets.Layout(width='100px'),
    )
    c_box = widgets.BoundedFloatText(
        value=-0.2, min=-50.0, max=50.0, step=0.05,
        description='', layout=widgets.Layout(width='100px'),
    )

    # jslink keeps each slider and its text box in sync entirely in the browser
    widgets.jslink((m_slider, 'value'), (m_box, 'value'))
    widgets.jslink((c_slider, 'value'), (c_box, 'value'))

    # ── Wire sliders to update and display ────────────────────────────────────
    controls = VBox([
        HBox([m_slider, m_box]),
        HBox([c_slider, c_box]),
    ])
    out = interactive_output(update, {'m': m_slider, 'c': c_slider})

    display(controls, out)