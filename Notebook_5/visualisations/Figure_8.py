"""
Figure 8 — Why Linear Regression Cannot Reliably Classify
==========================================================

Two side-by-side panels demonstrating the two fundamental problems with
applying linear regression to binary classification:

Figure 8a (left):  linear regression fit to binary labels — predictions
                   can fall outside [0, 1] and have no probabilistic meaning.

Figure 8b (right): adding a single extreme data point forces a complete
                   refit of the line and shifts the decision boundary,
                   demonstrating instability.

The user can drag a slider to move the extreme point further out and
watch how dramatically the boundary shifts in response.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_8 import show
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
from sklearn.linear_model import LinearRegression


def show():
    """Render the interactive Figure 8 linear regression classification demo."""

    plt.close('Notebook5 Figure 8')

    # ── Generate the two-class dataset ────────────────────────────────────────
    rng3  = np.random.default_rng(5)
    n_cls = 80

    x_cls = np.concatenate([
        rng3.normal(3, 1, n_cls // 2),   # Class 0 centred at x = 3
        rng3.normal(7, 1, n_cls // 2),   # Class 1 centred at x = 7
    ])
    y_cls = np.array([0] * (n_cls // 2) + [1] * (n_cls // 2))

    # Fit the original linear regression (no extreme point)
    lr_orig  = LinearRegression().fit(x_cls.reshape(-1, 1), y_cls)
    x_plot   = np.linspace(-1, 22, 400)
    y_orig   = lr_orig.predict(x_plot.reshape(-1, 1))

    # Boundary position for the original fit (where ŷ = 0.5)
    b_orig = (0.5 - lr_orig.intercept_) / lr_orig.coef_[0]

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_EXTREME = 15.0

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, num='Notebook5 Figure 8', figsize=(10, 5),
    )
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'bottom'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Figure 8a: left panel — original fit ──────────────────────────────────
    ax_left.scatter(x_cls[y_cls == 0], y_cls[y_cls == 0],
                    color='steelblue', s=60, edgecolors='k', lw=0.4,
                    zorder=3, label='Class 0')
    ax_left.scatter(x_cls[y_cls == 1], y_cls[y_cls == 1],
                    color='tomato', s=60, edgecolors='k', lw=0.4,
                    zorder=3, label='Class 1')
    ax_left.plot(x_plot, y_orig, 'k-', linewidth=2,
                 label='Linear regression fit')
    ax_left.axhline(0,   color='gray',  linewidth=0.8, linestyle=':')
    ax_left.axhline(1,   color='gray',  linewidth=0.8, linestyle=':')
    ax_left.axhline(0.5, color='green', linewidth=1.5, linestyle='--',
                    label='Threshold = 0.5')

    # Shade the region outside [0, 1] to highlight invalid predictions
    ax_left.fill_between(x_plot, y_orig, 1, where=(y_orig > 1),
                          color='orange', alpha=0.25, label='Prediction > 1 (invalid)')
    ax_left.fill_between(x_plot, y_orig, 0, where=(y_orig < 0),
                          color='orange', alpha=0.25, label='Prediction < 0 (invalid)')

    ax_left.set_xlabel('Feature $x$', fontsize=10)
    ax_left.set_ylabel('Label $y$  (0 or 1)', fontsize=10)
    ax_left.set_title(
        'Figure 8a: Linear regression on binary labels\n'
        'Predictions can fall outside [0, 1]',
        fontsize=10,
    )
    ax_left.set_xlim(-1, 12)
    ax_left.set_ylim(-0.3, 1.4)
    ax_left.legend(fontsize=7, loc='upper left')
    ax_left.grid(True, alpha=0.2)

    # ── Figure 8b: right panel — mutable (extreme point + refitted line) ─────
    ax_right.scatter(x_cls[y_cls == 0], y_cls[y_cls == 0],
                     color='steelblue', s=55, edgecolors='k', lw=0.4,
                     zorder=3, label='Original data (class 0)')
    ax_right.scatter(x_cls[y_cls == 1], y_cls[y_cls == 1],
                     color='tomato', s=55, edgecolors='k', lw=0.4, zorder=3,
                     label='Original data (class 1)')

    # Original fit shown as a solid line for comparison
    ax_right.plot(x_plot, y_orig, 'k-', linewidth=1.5, alpha=0.4,
                  label=f'Original boundary: x = {b_orig:.2f}')
    ax_right.axhline(0.5, color='green', linewidth=1.2, linestyle='--')

    # Mutable artists — updated when slider moves
    extreme_dot  = ax_right.scatter([], [], color='purple', s=150, zorder=6,
                                     marker='*', edgecolors='k', lw=0.8,
                                     label='Extreme point')
    new_line,    = ax_right.plot([], [], 'k--', linewidth=2,
                                  label='Refitted line')
    new_boundary = ax_right.axvline(x=b_orig, color='red', linewidth=1.8,
                                     linestyle=':', alpha=0.0,
                                     label='New boundary (shifted)')

    ax_right.set_xlabel('Feature $x$', fontsize=10)
    ax_right.set_ylabel('Label $y$', fontsize=10)
    ax_right.set_xlim(-1, 22)
    ax_right.set_ylim(-0.3, 1.4)
    ax_right.legend(fontsize=7, loc='lower right')
    ax_right.grid(True, alpha=0.2)

    plt.tight_layout()

    # ── Readout ───────────────────────────────────────────────────────────────
    readout = widgets.Output()

    # ── Update function ───────────────────────────────────────────────────────
    def update(extreme_x):
        # Add the extreme point (always class 1) to the dataset
        x_new = np.append(x_cls, extreme_x)
        y_new = np.append(y_cls, 1)

        # Refit the linear regression with the new point included
        lr_new   = LinearRegression().fit(x_new.reshape(-1, 1), y_new)
        y_new_line = lr_new.predict(x_plot.reshape(-1, 1))

        # New decision boundary (where ŷ = 0.5)
        b_new = (0.5 - lr_new.intercept_) / lr_new.coef_[0]
        shift = b_new - b_orig

        # Update the mutable artists
        extreme_dot.set_offsets([[extreme_x, 1]])
        new_line.set_data(x_plot, y_new_line)
        new_boundary.set_xdata([b_new, b_new])
        new_boundary.set_alpha(0.85)

        ax_right.set_title(
            f'Figure 8b: Extreme point at x = {extreme_x:.1f}\n'
            f'Original boundary: {b_orig:.2f}  →  New boundary: {b_new:.2f}'
            f'  (shift = {shift:+.2f})',
            fontsize=9,
        )
        ax_right.legend(fontsize=7, loc='lower right')

        readout.clear_output(wait=True)
        with readout:
            print(f'Extreme point position:   x = {extreme_x:.1f}')
            print(f'Original boundary:        x = {b_orig:.2f}')
            print(f'New boundary:             x = {b_new:.2f}')
            print(f'Boundary shift:           {shift:+.2f}  '
                  f'({"→ large shift!" if abs(shift) > 1 else "→ small shift"})')

        fig.canvas.draw_idle()

    # ── Slider ────────────────────────────────────────────────────────────────
    extreme_s = FloatSlider(
        value=DEFAULT_EXTREME, min=8.0, max=20.0, step=0.1,
        description='Extreme point x:',
        style={'description_width': '130px'},
        layout=widgets.Layout(width='400px'),
        continuous_update=True,
    )
    extreme_b = widgets.BoundedFloatText(
        value=DEFAULT_EXTREME, min=8.0, max=20.0, step=0.1,
        description='', layout=widgets.Layout(width='90px'),
    )
    widgets.jslink((extreme_s, 'value'), (extreme_b, 'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        extreme_s.value = DEFAULT_EXTREME

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Drag the extreme point to see how the decision boundary shifts:</b>'),
        HBox([extreme_s, extreme_b, reset_btn]),
        sep,
    ])

    out = interactive_output(update, {'extreme_x': extreme_s})

    display(controls, out)
    display(readout)