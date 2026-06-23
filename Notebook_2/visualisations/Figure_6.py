"""
Figure 6 & 7 — Interactive Error Minimisation Explorer
=======================================================

Two side-by-side panels that update together:

Left panel  (Figure 6): the misclassification error rate plotted against slope m
              for the current intercept c. A red dashed line marks the slope that
              minimises error; a black dot tracks the users current slope.

Right panel (Figure 7): the scatter data with the users current decision
              boundary drawn over it.

The user adjusts the boundary using slope and offset sliders. Both panels
update live so users can see how moving the boundary changes the error curve.

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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox
from IPython.display import display


def show():
    """Render the interactive Figure 6 & 7 error minimisation explorer."""

    plt.close('Notebook2 Figure 6')

    # ── Data generation ───────────────────────────────────────────────────────
    rng = np.random.default_rng(1)
    class_a_x = rng.normal(loc=180, scale=8,    size=50)
    class_a_y = rng.normal(loc=3500, scale=300, size=50)
    class_b_x = rng.normal(loc=210, scale=8,    size=50)
    class_b_y = rng.normal(loc=5000, scale=300, size=50)

    # ── Error rate function ───────────────────────────────────────────────────
    # Counts how many points are on the wrong side of y = mx + c,
    # then divides by the total number of points to get a rate between 0 and 1.
    def error_rate_2d(xa, ya, xb, yb, m, c):
        total   = len(xa) + len(xb)
        errors  = sum(1 for xi, yi in zip(xa, ya) if yi >= m * xi + c)
        errors += sum(1 for xi, yi in zip(xb, yb) if yi <  m * xi + c)
        return errors / total

    # ── Pre-compute the initial error curve ───────────────────────────────────
    # We sample 400 slope values and compute the error rate at each one.
    # This gives the full error-vs-slope curve shown in the left panel.
    slopes    = [m / 10 for m in range(-990, 1000, 5)]
    fixed_c   = -3500
    error_list = [error_rate_2d(class_a_x, class_a_y, class_b_x, class_b_y, m, fixed_c)
                  for m in slopes]
    best_m   = slopes[error_list.index(min(error_list))]
    best_err = min(error_list)

    # ── Pivot point ───────────────────────────────────────────────────────────
    # The boundary rotates around the midpoint between the two cluster centres,
    # so it stays roughly centred between the clouds as the angle changes.
    pivot_x = (class_a_x.mean() + class_b_x.mean()) / 2
    pivot_y = (class_a_y.mean() + class_b_y.mean()) / 2

    # Default slider values — start at the best slope, zero offset
    SLOPE_DEFAULT  = round(best_m, 2)
    OFF_DEFAULT    = 0.0

    XLIM   = (155, 240)
    YLIM   = (2000, 7000)
    x_plot = np.linspace(XLIM[0], XLIM[1], 300)

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook2 Figure 6', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: error rate curve ──────────────────────────────────────────
    ax_err = axes[0]

    # The error curve — updated via set_ydata() when c changes
    (error_curve,) = ax_err.plot(slopes, error_list, color='steelblue', linewidth=2,
                                 label=f'Error rate (c = {fixed_c})')

    # Red dashed line marking the slope that gives minimum error
    best_vline = ax_err.axvline(x=best_m, color='red', linestyle='--', linewidth=1.2,
                                label=f'Best slope m = {best_m:.2f}  (error = {best_err:.1%})')

    # Black dot tracking the users current slope on the error curve
    (current_dot,) = ax_err.plot([best_m], [best_err], 'o', color='black',
                                  markersize=9, zorder=5, label='Current line')

    # Dotted vertical line tracking the users current slope
    current_vline = ax_err.axvline(x=best_m, color='black', linestyle=':',
                                   linewidth=1.2)

    ax_err.set_xlabel('Slope (m)')
    ax_err.set_ylabel('Misclassification error rate')
    ax_err.set_title(f'Error rate vs slope (current c = {fixed_c})')
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(min(slopes), max(slopes))
    ax_err.set_ylim(0, 0.55)

    # ── Right panel: scatter + boundary line ──────────────────────────────────
    ax_data = axes[1]

    ax_data.scatter(class_a_x, class_a_y, color='steelblue', s=55, edgecolors='k',
                    linewidth=0.4, label='Class A', zorder=3)
    ax_data.scatter(class_b_x, class_b_y, color='tomato', s=55, edgecolors='k',
                    linewidth=0.4, label='Class B', zorder=3)

    y_init = best_m * x_plot + fixed_c
    (boundary_line,) = ax_data.plot(x_plot, y_init, 'k--', linewidth=2,
                                    label=f'y = {best_m:.2f}x + {fixed_c}', zorder=4)

    ax_data.set_xlabel('Flipper length (mm)')
    ax_data.set_ylabel('Body mass (g)')
    ax_data.set_xlim(*XLIM)
    ax_data.set_ylim(*YLIM)
    ax_data.grid(True, alpha=0.2)
    ax_data.legend(fontsize=8, loc='upper left')

    # Info box showing current m, c, error and accuracy
    info_text = ax_data.text(
        0.98, 0.05, '', transform=ax_data.transAxes,
        ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
    )

    fig.suptitle('Figure 6 & 7: Error landscape and current boundary', fontsize=12)
    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever either slider changes.
    def update(slope, offset):
        # Use slope directly — no angle conversion needed
        m = slope
        c = (pivot_y - m * pivot_x) + offset

        # Recompute the full error curve at the new c value
        current_errors   = [error_rate_2d(class_a_x, class_a_y, class_b_x, class_b_y, s, c)
                            for s in slopes]
        current_best_m   = slopes[current_errors.index(min(current_errors))]
        current_best_err = min(current_errors)

        # Update error curve and best-slope marker
        error_curve.set_ydata(current_errors)
        error_curve.set_label(f'Error rate (c = {c:.0f})')
        best_vline.set_xdata([current_best_m, current_best_m])
        best_vline.set_label(f'Best slope m = {current_best_m:.2f}  (error = {current_best_err:.1%})')

        # Update current-slope dot and tracking line
        current_err = error_rate_2d(class_a_x, class_a_y, class_b_x, class_b_y, m, c)
        current_dot.set_data([m], [current_err])
        current_vline.set_xdata([m, m])

        # Update boundary line on right panel
        boundary_line.set_ydata(m * x_plot + c)
        boundary_line.set_label(f'y = {m:.2f}x + {c:.0f}')

        info_text.set_text(
            f'm = {m:.2f}   c = {c:.0f}\n'
            f'Error: {current_err:.1%}   Accuracy: {1 - current_err:.1%}'
        )

        ax_err.set_title(f'Error rate vs slope (current c = {c:.0f})')
        ax_data.set_title(f'Boundary: y = {m:.2f}x + {c:.0f}')
        ax_err.legend(fontsize=8)
        ax_data.legend(fontsize=8, loc='upper left')
        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    # Slope slider controls m directly — more intuitive than angle for this figure
    slope_s = FloatSlider(
        value=SLOPE_DEFAULT, min=-99.0, max=99.0, step=0.05,
        description='Slope (m)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )
    offset_s = FloatSlider(
        value=OFF_DEFAULT, min=-2000, max=2000, step=10,
        description='Offset',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )

    slope_b = widgets.BoundedFloatText(
        value=SLOPE_DEFAULT, min=-99.0, max=99.0, step=0.05,
        description='', layout=widgets.Layout(width='100px'),
    )
    offset_b = widgets.BoundedFloatText(
        value=OFF_DEFAULT, min=-2000, max=2000, step=10,
        description='', layout=widgets.Layout(width='100px'),
    )

    widgets.jslink((slope_s,  'value'), (slope_b,  'value'))
    widgets.jslink((offset_s, 'value'), (offset_b, 'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(
        description='Reset',
        button_style='info',
        layout=widgets.Layout(width='150px'),
    )

    def on_reset(b):
        slope_s.value  = SLOPE_DEFAULT
        offset_s.value = OFF_DEFAULT

    reset_btn.on_click(on_reset)

    # ── Wire sliders to update and display ────────────────────────────────────
    controls = VBox([
        HBox([slope_s,  slope_b]),
        HBox([offset_s, offset_b]),
        reset_btn,
    ])

    out = interactive_output(update, {'slope': slope_s, 'offset': offset_s})
    display(controls, out)