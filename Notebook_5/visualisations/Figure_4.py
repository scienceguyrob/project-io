"""
Figure 4 — Interactive Least-Squares Loss Explorer
===================================================

Two side-by-side panels:

Left panel  (Figure 4a): the scatter data with the current regression line
             overlaid. Residuals are drawn as vertical red lines. The line
             updates live as the user adjusts β₀ and β₁.

Right panel (Figure 4b): the SSE loss surface as a filled contour plot
             with β₀ on the x-axis and β₁ on the y-axis. A red dot marks
             the current (β₀, β₁) position; a yellow star marks the OLS
             optimal solution. As the user moves the sliders, the dot
             moves across the surface, showing how the loss changes.

This directly connects the idea of "searching for parameters that minimise
the loss" from Lab 4 to the specific case of linear regression.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_4 import show
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
    """Render the interactive Figure 4 least-squares loss explorer."""

    plt.close('Notebook5 Figure 4')

    # ── Generate the same synthetic dataset as Figure 1 ───────────────────────
    rng    = np.random.default_rng(42)
    n      = 60
    area   = rng.uniform(40, 160, n)
    price  = 1.8 * area + 50 + rng.normal(0, 18, n)

    # ── OLS optimal solution ──────────────────────────────────────────────────
    x_mean  = area.mean()
    y_mean  = price.mean()
    beta1_opt = np.sum((area - x_mean) * (price - y_mean)) / np.sum((area - x_mean)**2)
    beta0_opt = y_mean - beta1_opt * x_mean

    # ── Pre-compute the loss surface ──────────────────────────────────────────
    # Evaluate SSE on a grid of (β₀, β₁) values so we can draw the contour plot.
    # We keep this fixed — it does not depend on the sliders.
    b0_grid  = np.linspace(0,   120, 150)
    b1_grid  = np.linspace(0.5, 3.5, 150)
    B0, B1   = np.meshgrid(b0_grid, b1_grid)

    # SSE at every (β₀, β₁) combination on the grid
    # price[:, None] broadcasts price to shape (60, 1) for vectorised computation
    SSE_grid = np.sum(
        (price[:, None, None] - B0[None, :, :] - B1[None, :, :] * area[:, None, None]) ** 2,
        axis=0,
    )

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_B0 = 60.0
    DEFAULT_B1 = 1.5

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, (ax_data, ax_loss) = plt.subplots(
        1, 2, num='Notebook5 Figure 4', figsize=(10, 5),
    )
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Figure 4a: data + line + residuals ────────────────────────────────────
    x_line = np.linspace(35, 165, 300)

    ax_data.scatter(area, price, color='steelblue', s=45,
                    edgecolors='k', lw=0.4, alpha=0.8, zorder=4,
                    label='Training data')

    # Mutable: fitted line
    fit_line, = ax_data.plot([], [], 'r-', linewidth=2.5, zorder=3,
                             label='Current line')

    # Mutable: residual lines — one per point, stored for update
    res_lines = [ax_data.plot([xi, xi], [yi, yi], color='tomato',
                               linewidth=0.9, alpha=0.65, zorder=2)[0]
                 for xi, yi in zip(area, price)]

    ax_data.plot([], [], color='tomato', linewidth=1.5,
                 label='Residuals')

    ax_data.set_xlabel('Floor area (m²)', fontsize=10)
    ax_data.set_ylabel('Sale price (£k)', fontsize=10)
    ax_data.set_xlim(30, 170)
    ax_data.set_ylim(50, 430)
    ax_data.grid(True, alpha=0.2)
    ax_data.legend(fontsize=8, loc='upper left')

    # ── Figure 4b: SSE loss surface ───────────────────────────────────────────
    ax_loss.contourf(B0, B1, SSE_grid, levels=60, cmap='Blues_r', alpha=0.9)
    ax_loss.contour(B0,  B1, SSE_grid, levels=20, colors='white',
                    alpha=0.2, linewidths=0.5)

    # Yellow star at the OLS optimum
    ax_loss.scatter(beta0_opt, beta1_opt, color='yellow', s=250,
                    marker='*', zorder=6, edgecolors='k', lw=0.8,
                    label=f'OLS optimum\nβ₀={beta0_opt:.1f}, β₁={beta1_opt:.2f}')

    # Mutable: red dot showing the current (β₀, β₁) position
    current_dot = ax_loss.scatter([], [], color='tomato', s=120,
                                   zorder=7, edgecolors='k', lw=1.0,
                                   marker='o', label='Current parameters')

    ax_loss.set_xlabel('β₀  (intercept)', fontsize=10)
    ax_loss.set_ylabel('β₁  (slope)', fontsize=10)
    ax_loss.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    # ── Update function ───────────────────────────────────────────────────────
    def update(b0, b1):
        y_hat     = b0 + b1 * area
        residuals = price - y_hat
        sse       = np.sum(residuals ** 2)
        opt_sse   = np.sum((price - beta0_opt - beta1_opt * area) ** 2)

        # Update fitted line
        fit_line.set_data(x_line, b0 + b1 * x_line)
        fit_line.set_label(f'ŷ = {b0:.1f} + {b1:.2f}x')

        # Update residual lines
        y_hat_pts = b0 + b1 * area
        for rl, xi, yi, yhi in zip(res_lines, area, price, y_hat_pts):
            rl.set_data([xi, xi], [yi, yhi])

        ax_data.set_title(
            f'Figure 4a: β₀ = {b0:.1f},  β₁ = {b1:.2f}\n'
            f'SSE = {sse:,.0f}  |  OLS optimal SSE = {opt_sse:,.0f}',
            fontsize=9,
        )
        ax_data.legend(fontsize=8, loc='upper left')

        # Update current position on the loss surface
        current_dot.set_offsets([[b0, b1]])

        ax_loss.set_title(
            f'Figure 4b: Loss surface  |  current SSE = {sse:,.0f}\n'
            f'(darker = lower loss — move toward the yellow star)',
            fontsize=9,
        )

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    b0_s = FloatSlider(
        value=DEFAULT_B0, min=0.0, max=120.0, step=0.5,
        description='β₀ (intercept)',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )
    b1_s = FloatSlider(
        value=DEFAULT_B1, min=0.5, max=3.5, step=0.02,
        description='β₁ (slope)',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='380px'),
        continuous_update=True,
    )

    b0_b = widgets.BoundedFloatText(value=DEFAULT_B0, min=0.0, max=120.0,
                                     step=0.5, description='',
                                     layout=widgets.Layout(width='90px'))
    b1_b = widgets.BoundedFloatText(value=DEFAULT_B1, min=0.5, max=3.5,
                                     step=0.02, description='',
                                     layout=widgets.Layout(width='90px'))

    widgets.jslink((b0_s, 'value'), (b0_b, 'value'))
    widgets.jslink((b1_s, 'value'), (b1_b, 'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))
    opt_btn   = widgets.Button(description='Show OLS optimum',
                               button_style='success',
                               layout=widgets.Layout(width='160px'))

    def on_reset(b):
        b0_s.value = DEFAULT_B0
        b1_s.value = DEFAULT_B1

    def on_optimal(b):
        b0_s.value = round(beta0_opt, 1)
        b1_s.value = round(beta1_opt, 2)

    reset_btn.on_click(on_reset)
    opt_btn.on_click(on_optimal)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Adjust the regression line parameters:</b>'),
        HBox([b0_s, b0_b]),
        HBox([b1_s, b1_b]),
        sep,
        HBox([reset_btn, opt_btn]),
    ])

    out = interactive_output(update, {'b0': b0_s, 'b1': b1_s})

    display(controls, out)