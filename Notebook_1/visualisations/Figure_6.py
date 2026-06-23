"""
Figure 6 — Interactive Residuals & MAE Explorer
================================================

A small dataset of 7 (x, y) pairs is plotted as scatter points. The user
adjusts the slope m and intercept c of a fitted line to see how the residuals
(vertical red lines from each data point to the line) change, and how the
Mean Absolute Error (MAE) responds.

A reset button restores the least-squares best-fit line so users can
compare their manual fit against the optimal one.

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
    """Render the interactive Figure 6 residuals explorer."""

    plt.close('Figure 6')

    # ── Dataset ───────────────────────────────────────────────────────────────
    # A small fixed set of (x, y) pairs representing e.g. study hours vs score.
    x_data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    y_data = np.array([3.3, 5.8, 7.1, 9.7, 11.2, 13.4, 15.8])

    # ── Compute the least-squares best-fit line ───────────────────────────────
    # np.polyfit(x, y, 1) returns [slope, intercept] that minimise the sum of
    # squared residuals — this is the mathematically optimal straight-line fit.
    m_best, c_best = np.polyfit(x_data, y_data, 1)

    # x-values for drawing the fitted line across the full plot range
    x_line = np.linspace(0.5, 7.5, 200)

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Figure 6', figsize=(8, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # Static scatter points — these never move so we don't store them
    ax.scatter(x_data, y_data, color='steelblue', s=80, zorder=4, label='Data points')

    # The fitted line — stored so set_ydata() can update it later
    (fit_line,) = ax.plot(
        x_line, m_best * x_line + c_best,
        'k-', linewidth=2,
        label=f'Line: y = {m_best:.2f}x + {c_best:.2f}',
        zorder=2,
    )

    # One vertical residual line per data point — stored in a list so the
    # update function can move each one individually
    residual_lines = [
        ax.plot([xi, xi], [yi, yi], color='red', linewidth=1.5, zorder=3)[0]
        for xi, yi in zip(x_data, y_data)
    ]

    # A dummy plot just to add a residuals entry to the legend
    ax.plot([], [], color='red', linewidth=1.5, label='Residuals (errors)')

    ax.set_xlabel('x (e.g. hours of study)')
    ax.set_ylabel('y (e.g. exam score)')
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3)

    # MAE annotation in the top-left corner — updated in place via set_text()
    mae_text = ax.text(
        0.03, 0.95, '', transform=ax.transAxes,
        ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'),
    )

    ax.legend(loc='lower right')
    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever either slider changes.
    # Mutates existing artists only — no new plot objects are created.
    def update(m, c):
        # Compute predicted y-values and residuals for the current m and c
        y_pred = m * x_data + c
        errors = y_data - y_pred
        mae = np.mean(np.abs(errors))

        # MAE of the best-fit line — used as a reference for comparison
        mae_best = np.mean(np.abs(y_data - (m_best * x_data + c_best)))

        # Move the fitted line to its new position
        fit_line.set_ydata(m * x_line + c)
        fit_line.set_label(f'Line: y = {m:.2f}x + {c:.2f}')

        # Stretch each residual line from the actual data point to the
        # predicted point directly above or below it on the fitted line
        for rl, xi, yi, yp in zip(residual_lines, x_data, y_data, y_pred):
            rl.set_ydata([yi, yp])

        # Update the MAE readout — positive difference means worse than best fit
        mae_text.set_text(
            f'MAE (current):   {mae:.3f}\n'
            f'MAE (best fit):  {mae_best:.3f}\n'
            f'Difference:      +{mae - mae_best:.3f}'
        )

        ax.set_title(f'Figure 6: Residuals for y = {m:.3f}x + {c:.3f}')
        ax.legend(loc='lower right')
        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    m_slider = FloatSlider(
        value=m_best, min=-5.0, max=10.0, step=0.05,
        description='m (slope)',
        style={'description_width': '100px'},
        continuous_update=True,
    )
    c_slider = FloatSlider(
        value=c_best, min=-10.0, max=10.0, step=0.05,
        description='c (intercept)',
        style={'description_width': '100px'},
        continuous_update=True,
    )

    # Paired text boxes so users can type precise values directly
    m_box = widgets.BoundedFloatText(
        value=m_best, min=-5.0, max=10.0, step=0.05,
        description='', layout=widgets.Layout(width='100px'),
    )
    c_box = widgets.BoundedFloatText(
        value=c_best, min=-10.0, max=10.0, step=0.05,
        description='', layout=widgets.Layout(width='100px'),
    )

    # jslink keeps each slider and its text box in sync in the browser
    widgets.jslink((m_slider, 'value'), (m_box, 'value'))
    widgets.jslink((c_slider, 'value'), (c_box, 'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    # Setting the slider values triggers interactive_output automatically,
    # so no manual redraw is needed here.
    reset_btn = widgets.Button(
        description='Reset to Best Fit',
        button_style='info',
        layout=widgets.Layout(width='160px'),
    )

    def on_reset(b):
        m_slider.value = m_best
        c_slider.value = c_best

    reset_btn.on_click(on_reset)

    # ── Wire sliders to update and display ────────────────────────────────────
    controls = VBox([
        HBox([m_slider, m_box]),
        HBox([c_slider, c_box]),
        reset_btn,
    ])
    out = interactive_output(update, {'m': m_slider, 'c': c_slider})

    display(controls, out)