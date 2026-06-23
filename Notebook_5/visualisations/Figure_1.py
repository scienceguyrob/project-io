"""
Figure 1 — Simple Linear Regression: Fit and Predict
=====================================================

Shows a synthetic house-price dataset with a fitted regression line.
The fitted line is annotated with the values of β₀ and β₁.

The user can type a floor area into the text box and click Predict
to see the predicted price drawn on the plot — a vertical line up to
the fitted line, then a horizontal line across to the y-axis, with
the predicted value labelled clearly.

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
from ipywidgets import VBox, HBox, HTML
from IPython.display import display


def show():
    """Render Figure 1 — regression fit and prediction."""

    plt.close('Notebook5 Figure 1')

    # ── Generate synthetic data ───────────────────────────────────────────────
    rng    = np.random.default_rng(42)
    n      = 60
    x_data = rng.uniform(40, 160, n)
    y_data = 30 + 2.2 * x_data + rng.normal(0, 18, n)

    # ── Fit the line ──────────────────────────────────────────────────────────
    beta1, beta0 = np.polyfit(x_data, y_data, 1)

    x_line = np.linspace(30, 175, 300)
    y_line = beta0 + beta1 * x_line

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook5 Figure 1', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    ax.scatter(x_data, y_data, color='steelblue', s=55, alpha=0.75,
               edgecolors='k', lw=0.4, zorder=3, label='Training data')

    ax.plot(x_line, y_line, color='black', linewidth=2.5, zorder=2,
            label=f'Fitted line: price = {beta0:.1f} + {beta1:.2f} × area')

    ax.annotate(f'β₀ = {beta0:.1f}  (intercept)',
                xy=(30, beta0), xytext=(35, beta0 + 30),
                fontsize=9, color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    ax.annotate(f'β₁ = {beta1:.2f}  (slope: £{beta1:.2f}k per m²)',
                xy=(130, beta0 + beta1 * 130),
                xytext=(95, beta0 + beta1 * 130 + 45),
                fontsize=9, color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

    # Mutable prediction artists — invisible until user clicks Predict
    vline, = ax.plot([], [], color='seagreen', linewidth=2,
                     linestyle='--', zorder=4)
    hline, = ax.plot([], [], color='seagreen', linewidth=2,
                     linestyle='--', zorder=4)
    pred_dot = ax.scatter([], [], color='seagreen', s=180, zorder=6,
                          edgecolors='k', lw=1.2, marker='*')
    pred_label = ax.text(0, 0, '', fontsize=10, color='seagreen',
                         fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   edgecolor='seagreen', alpha=0.9))

    ax.set_xlabel('Floor area (m²)', fontsize=11)
    ax.set_ylabel('Sale price (£k)', fontsize=11)
    ax.set_xlim(25, 180)
    ax.set_ylim(50, 430)
    ax.grid(True, alpha=0.2)
    ax.set_title(
        f'Figure 1. Simple linear regression:  price = {beta0:.1f} + {beta1:.2f} × area\n'
        f'β₀ = {beta0:.1f}  |  β₁ = {beta1:.2f}',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()

    # ── Controls ──────────────────────────────────────────────────────────────
    area_box = widgets.BoundedFloatText(
        value=100, min=40, max=160, step=1,
        description='Floor area (m²):',
        style={'description_width': '130px'},
        layout=widgets.Layout(width='280px'),
    )
    predict_btn = widgets.Button(
        description='Predict price',
        button_style='success',
        layout=widgets.Layout(width='130px'),
    )
    clear_btn = widgets.Button(
        description='Clear',
        button_style='warning',
        layout=widgets.Layout(width='90px'),
    )

    def do_predict(b=None):
        area      = area_box.value
        predicted = beta0 + beta1 * area
        vline.set_data([area, area], [50, predicted])
        hline.set_data([25, area], [predicted, predicted])
        pred_dot.set_offsets([[area, predicted]])
        pred_label.set_position((27, predicted + 5))
        pred_label.set_text(f'ŷ = £{predicted:.1f}k')
        fig.canvas.draw_idle()

    def do_clear(b=None):
        vline.set_data([], [])
        hline.set_data([], [])
        pred_dot.set_offsets(np.empty((0, 2)))
        pred_label.set_text('')
        fig.canvas.draw_idle()

    predict_btn.on_click(do_predict)
    clear_btn.on_click(do_clear)

    controls = VBox([
        HTML('<b>Enter a floor area to see the predicted sale price:</b>'),
        HBox([area_box, predict_btn, clear_btn]),
    ])

    display(controls)