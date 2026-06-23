"""
Figure 3 — Interactive 2D Decision Boundary Explorer
=====================================================

Two classes of penguin are plotted using two features: flipper length (x-axis)
and body mass (y-axis). The user rotates and shifts a straight-line decision
boundary to separate the two classes as accurately as possible.

The boundary angle and offset are controlled independently — m and c are
computed automatically from these and displayed as a readout. Shaded regions
show which side of the boundary is predicted as each class.

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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox
from IPython.display import display


def show():
    """Render the interactive Figure 3 2D decision boundary explorer."""

    plt.close('Notebook2 Figure 3')

    # ── Data generation ───────────────────────────────────────────────────────
    # Fixed seed so every user sees the same point clouds.
    rng = np.random.default_rng(1)

    class_a_x = rng.normal(loc=180, scale=8,    size=50)
    class_a_y = rng.normal(loc=3500, scale=300, size=50)
    class_b_x = rng.normal(loc=210, scale=8,    size=50)
    class_b_y = rng.normal(loc=5000, scale=300, size=50)

    # ── Compute the default boundary ──────────────────────────────────────────
    # The best starting boundary is the perpendicular bisector of the line
    # joining the two class centroids — it sits exactly between the two clouds.
    mean_a = np.array([class_a_x.mean(), class_a_y.mean()])
    mean_b = np.array([class_b_x.mean(), class_b_y.mean()])
    diff   = mean_b - mean_a
    mid    = (mean_a + mean_b) / 2

    # Perpendicular slope: if the centroid direction has slope dy/dx,
    # the perpendicular has slope -dx/dy
    M_DEFAULT     = -diff[0] / diff[1]
    C_DEFAULT     = mid[1] - M_DEFAULT * mid[0]
    ANGLE_DEFAULT = round(np.degrees(np.arctan(M_DEFAULT)), 1)

    # ── Fixed axis limits ─────────────────────────────────────────────────────
    XLIM = (155, 240)
    YLIM = (2000, 7000)

    # ── Precompute true labels ────────────────────────────────────────────────
    all_x = np.concatenate([class_a_x, class_b_x])
    all_y = np.concatenate([class_a_y, class_b_y])
    true_labels = np.array([0] * 50 + [1] * 50)

    # x-values for drawing the boundary line across the full plot range
    x_line = np.linspace(XLIM[0], XLIM[1], 300)

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook2 Figure 3', figsize=(8, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # Static scatter points — never move so we don't need to store them
    ax.scatter(class_a_x, class_a_y, color='steelblue', s=60, edgecolors='k',
               linewidth=0.4, label='Class A', zorder=3)
    ax.scatter(class_b_x, class_b_y, color='tomato', s=60, edgecolors='k',
               linewidth=0.4, label='Class B', zorder=3)

    # The boundary line — stored so set_ydata() can update it later
    y_init = M_DEFAULT * x_line + C_DEFAULT
    (boundary_line,) = ax.plot(
        x_line, y_init, 'k--', linewidth=2,
        label=f'Boundary: y = {M_DEFAULT:.1f}x + {C_DEFAULT:.0f}',
    )

    # Shaded regions showing predicted class on each side of the boundary.
    # These are stored in a list so they can be removed and redrawn on update —
    # fill_between() doesn't support set_ydata() like Line2D does.
    fills = [
        ax.fill_between(x_line, np.clip(y_init, *YLIM), YLIM[1],
                        alpha=0.08, color='tomato',    label='Predicted Class B'),
        ax.fill_between(x_line, YLIM[0], np.clip(y_init, *YLIM),
                        alpha=0.08, color='steelblue', label='Predicted Class A'),
    ]

    ax.set_xlabel('Flipper length (mm)')
    ax.set_ylabel('Body mass (g)')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(f'Figure 3: Decision boundary  |  angle = {ANGLE_DEFAULT}°  |  offset = 0')

    # Accuracy readout in the bottom-right corner
    acc_text = ax.text(
        0.98, 0.05, '', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
    )

    # m and c readout above the accuracy box
    mc_text = ax.text(
        0.98, 0.25, '', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, color='#444',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'),
    )

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever either slider changes.
    # fill_between artists cannot be mutated so they are removed and redrawn.
    def update(angle, offset):
        # Convert angle in degrees to slope m using trigonometry
        m = np.tan(np.radians(angle))
        # Pivot the line around the midpoint between centroids, then shift by offset
        c = (mid[1] - m * mid[0]) + offset

        y_line = m * x_line + c

        # Update the boundary line in place
        boundary_line.set_ydata(y_line)

        # Remove old fill regions and redraw — fill_between has no mutation API
        for f in fills:
            f.remove()
        fills.clear()
        fills.append(ax.fill_between(x_line, np.clip(y_line, *YLIM), YLIM[1],
                                     alpha=0.08, color='tomato',    label='Predicted Class B'))
        fills.append(ax.fill_between(x_line, YLIM[0], np.clip(y_line, *YLIM),
                                     alpha=0.08, color='steelblue', label='Predicted Class A'))

        # A point is predicted Class B if it lies above the boundary line
        predicted = (all_y > m * all_x + c).astype(int)
        correct   = np.sum(predicted == true_labels)
        accuracy  = correct / len(true_labels) * 100
        a_wrong   = np.sum(predicted[:50] != 0)
        b_wrong   = np.sum(predicted[50:] != 1)

        acc_text.set_text(
            f'Accuracy: {correct}/{len(true_labels)} ({accuracy:.1f}%)\n'
            f'Class A misclassified: {a_wrong}   |   Class B misclassified: {b_wrong}'
        )
        mc_text.set_text(f'm = {m:.3f}   |   c = {c:.0f}')

        ax.set_title(f'Figure 3: Decision boundary  |  angle = {angle:.3f}°  |  offset = {offset:.0f}')
        ax.legend(fontsize=8, loc='upper left')
        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    angle_slider = FloatSlider(
        value=ANGLE_DEFAULT, min=-89.0, max=89.0, step=0.5,
        description='Angle (°)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )
    offset_slider = FloatSlider(
        value=0, min=-2000, max=2000, step=10,
        description='Offset',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )

    # Paired text boxes so users can type precise values directly
    angle_box = widgets.BoundedFloatText(
        value=ANGLE_DEFAULT, min=-89.0, max=89.0, step=0.5,
        description='', layout=widgets.Layout(width='100px'),
    )
    offset_box = widgets.BoundedFloatText(
        value=0, min=-2000, max=2000, step=10,
        description='', layout=widgets.Layout(width='100px'),
    )

    # jslink keeps each slider and its text box in sync in the browser
    widgets.jslink((angle_slider,  'value'), (angle_box,  'value'))
    widgets.jslink((offset_slider, 'value'), (offset_box, 'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(
        description='Reset',
        button_style='info',
        layout=widgets.Layout(width='150px'),
    )

    def on_reset(b):
        angle_slider.value  = ANGLE_DEFAULT
        offset_slider.value = 0

    reset_btn.on_click(on_reset)

    # ── Wire sliders to update and display ────────────────────────────────────
    controls = VBox([
        HBox([angle_slider,  angle_box]),
        HBox([offset_slider, offset_box]),
        reset_btn,
    ])
    out = interactive_output(update, {'angle': angle_slider, 'offset': offset_slider})

    display(controls, out)