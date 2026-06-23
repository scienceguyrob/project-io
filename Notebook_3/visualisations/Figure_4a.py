"""
Figure 4a — Interactive Pearson Correlation Explorer
=====================================================

A single scatter plot that lets users explore how the Pearson correlation
coefficient r changes as they alter the dataset parameters. The right panel
shows the formula for r with actual numbers substituted in at each step,
so users can follow the calculation from raw data to final result.

The user controls:
    - Slope:      how steeply y increases or decreases with x
                  (positive slope → positive r, negative → negative r)
    - Noise:      how much random scatter is added around the trend line
                  (more noise → r closer to 0)
    - Relationship type: linear, quadratic, or sine wave — letting users
                  see that r = 0 does not always mean "no relationship"

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_4a import show
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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox, HTML, Dropdown
from IPython.display import display


def show():
    """Render the interactive Figure 4a Pearson correlation explorer."""

    plt.close('Notebook3 Figure 4a')

    # ── Fixed x values — same for every update ────────────────────────────────
    rng_x = np.random.default_rng(42)
    N     = 150
    x     = np.sort(rng_x.uniform(0, 10, N))

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_SLOPE = 2.0
    DEFAULT_NOISE = 2.0
    DEFAULT_REL   = 'Linear'

    # ── Build the figure: left = scatter, right = formula panel ───────────────
    fig, (ax_scatter, ax_formula) = plt.subplots(
        1, 2, num='Notebook3 Figure 4a', figsize=(10, 9),
        gridspec_kw={'width_ratios': [3, 2]},
    )
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True
    fig.canvas.layout.height = '1050px'

    # ── Left panel: scatter plot ───────────────────────────────────────────────
    rng_n  = np.random.default_rng(99)
    y_init = DEFAULT_SLOPE * x + rng_n.normal(0, DEFAULT_NOISE, N)
    scat   = ax_scatter.scatter(x, y_init, s=30, alpha=0.6, color='steelblue',
                                edgecolors='k', linewidth=0.3)

    (trend_line,) = ax_scatter.plot([], [], color='red', linewidth=2,
                                    linestyle='--', label='Trend')

    ax_scatter.set_xlabel('x', fontsize=11)
    ax_scatter.set_ylabel('y', fontsize=11)
    ax_scatter.grid(True, alpha=0.2)
    ax_scatter.legend(fontsize=9)

    # ── Right panel: formula display ───────────────────────────────────────────
    ax_formula.axis('off')
    formula_text = ax_formula.text(
        0.04, 0.97, '',
        transform=ax_formula.transAxes,
        ha='left', va='top',
        fontsize=9.5, family='monospace',
        linespacing=1.7,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f9f9e8',
                  edgecolor='#bbb', linewidth=1.2),
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    # ── Interpretation helper ─────────────────────────────────────────────────
    def interpret(r):
        a = abs(r)
        direction = 'positive' if r >= 0 else 'negative'
        if a < 0.1:   strength = 'negligible'
        elif a < 0.3: strength = 'weak'
        elif a < 0.5: strength = 'moderate'
        elif a < 0.7: strength = 'strong'
        else:         strength = 'very strong'
        return f'{strength} {direction}' if a >= 0.1 else 'negligible (no linear relationship)'

    # ── Update function ───────────────────────────────────────────────────────
    noise_rng = np.random.default_rng()

    def update(slope, noise, relationship):
        base_noise = noise_rng.normal(0, noise, N)
        if relationship == 'Linear':
            y = slope * x + base_noise
        elif relationship == 'Quadratic':
            y = slope * (x - 5) ** 2 + base_noise
        else:
            y = slope * np.sin(x) + base_noise

        x_mean = x.mean()
        y_mean = y.mean()
        dx = x - x_mean
        dy = y - y_mean
        numerator   = np.sum(dx * dy)
        sum_dx2     = np.sum(dx ** 2)
        sum_dy2     = np.sum(dy ** 2)
        denominator = np.sqrt(sum_dx2 * sum_dy2)
        r = numerator / denominator if denominator != 0 else 0.0

        scat.set_offsets(np.column_stack([x, y]))
        ax_scatter.set_ylim(y.min() - 1, y.max() + 1)

        x_ends   = np.array([x.min(), x.max()])
        ls_slope = numerator / sum_dx2 if sum_dx2 != 0 else 0
        y_ends   = y_mean + ls_slope * (x_ends - x_mean)
        trend_line.set_data(x_ends, y_ends)

        ax_scatter.set_title(
            f"Pearson r = {r:.3f}   ({interpret(r)})", fontsize=11,
        )

        formula_str = (
            f' The Pearson r formula:\n'
            f'\n'
            f'       Σ (xᵢ - x̄)(yᵢ - ȳ)\n'
            f' r = ─────────────────────────────\n'
            f'     √[Σ(xᵢ-x̄)²  ×  Σ(yᵢ-ȳ)²]\n'
            f'\n'
            f' ─────────────────────────────────\n'
            f' Step 1: compute the means\n'
            f'   x̄ = {x_mean:.3f}\n'
            f'   ȳ = {y_mean:.3f}\n'
            f'\n'
            f' Step 2: deviations from the mean\n'
            f'   For each point, subtract the mean:\n'
            f'   (xᵢ - x̄) and (yᵢ - ȳ)\n'
            f'   (computed for all {N} points)\n'
            f'\n'
            f' Step 3: numerator\n'
            f'   Σ(xᵢ - x̄)(yᵢ - ȳ) = {numerator:.2f}\n'
            f'   {"positive → same direction" if numerator > 0 else "negative → opposite directions" if numerator < 0 else "zero → no relationship"}\n'
            f'\n'
            f' Step 4: denominator\n'
            f'   Σ(xᵢ - x̄)² = {sum_dx2:.2f}\n'
            f'   Σ(yᵢ - ȳ)² = {sum_dy2:.2f}\n'
            f'   √({sum_dx2:.2f} × {sum_dy2:.2f})\n'
            f'            = {denominator:.2f}\n'
            f'\n'
            f' Step 5: divide\n'
            f'   r = {numerator:.2f} / {denominator:.2f}\n'
            f'     = {r:.3f}\n'
            f'\n'
            f' ─────────────────────────────────\n'
            f' Interpretation: {interpret(r)}'
        )

        formula_text.set_text(formula_str)
        fig.canvas.draw_idle()

    # ── Widgets ───────────────────────────────────────────────────────────────
    slope_s = FloatSlider(
        value=DEFAULT_SLOPE, min=-5.0, max=5.0, step=0.1,
        description='Slope',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='340px'),
        continuous_update=True,
    )
    noise_s = FloatSlider(
        value=DEFAULT_NOISE, min=0.1, max=10.0, step=0.1,
        description='Noise',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='340px'),
        continuous_update=True,
    )
    slope_b = widgets.BoundedFloatText(
        value=DEFAULT_SLOPE, min=-5.0, max=5.0, step=0.1,
        description='', layout=widgets.Layout(width='90px'),
    )
    noise_b = widgets.BoundedFloatText(
        value=DEFAULT_NOISE, min=0.1, max=10.0, step=0.1,
        description='', layout=widgets.Layout(width='90px'),
    )
    widgets.jslink((slope_s, 'value'), (slope_b, 'value'))
    widgets.jslink((noise_s, 'value'), (noise_b, 'value'))

    rel_dropdown = Dropdown(
        options=['Linear', 'Quadratic', 'Sine'],
        value=DEFAULT_REL,
        description='Relationship:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='250px'),
    )

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        slope_s.value      = DEFAULT_SLOPE
        noise_s.value      = DEFAULT_NOISE
        rel_dropdown.value = DEFAULT_REL

    reset_btn.on_click(on_reset)

    # ── Layout ────────────────────────────────────────────────────────────────
    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Dataset controls</b>'),
        HBox([slope_s, slope_b]),
        HBox([noise_s, noise_b]),
        rel_dropdown,
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        'slope':        slope_s,
        'noise':        noise_s,
        'relationship': rel_dropdown,
    })

    display(controls, out)