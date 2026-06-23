"""
Figure 2a — Interactive Cohen's d Explorer
==========================================

Visualises how Cohen's d quantifies the separation between two normal
distributions. Two overlapping bell curves are shown with the overlap
region shaded in purple.

The user controls the mean and standard deviation of each distribution.
The figure updates live to show:

    - The two distribution curves with shaded overlap
    - Dashed vertical lines marking each mean
    - The full Cohen's d formula with actual numbers substituted in,
      updating in real time so users can follow the calculation step by step
    - A plain-English interpretation of the resulting d value

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_2a import show
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
from scipy.stats import norm
import ipywidgets as widgets
from ipywidgets import interactive_output, FloatSlider, VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the interactive Figure 2a Cohen's d explorer."""

    plt.close('Notebook3 Figure 2a')

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_MU1   = 50.0;  DEFAULT_SIG1 = 10.0
    DEFAULT_MU2   = 70.0;  DEFAULT_SIG2 = 10.0

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    # Two panels: left = the distribution plot, right = the live formula
    fig, (ax_plot, ax_formula) = plt.subplots(
        1, 2, num='Notebook3 Figure 2a', figsize=(12, 5),
        gridspec_kw={'width_ratios': [2, 1]},   # plot gets 2/3 of width
    )
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # x range wide enough for any reasonable parameter combination
    x = np.linspace(-20, 140, 1000)

    # ── Left panel: distribution curves ───────────────────────────────────────
    y1_init = norm.pdf(x, DEFAULT_MU1, DEFAULT_SIG1)
    y2_init = norm.pdf(x, DEFAULT_MU2, DEFAULT_SIG2)

    (curve1,) = ax_plot.plot(x, y1_init, color='steelblue', linewidth=2.5,
                             label='Distribution 1')
    (curve2,) = ax_plot.plot(x, y2_init, color='tomato', linewidth=2.5,
                             label='Distribution 2')

    # Shaded fills stored in a list so they can be removed and redrawn
    fills = []

    def redraw_fills(y1, y2):
        for f in fills:
            f.remove()
        fills.clear()
        fills.append(ax_plot.fill_between(x, y1, alpha=0.15, color='steelblue'))
        fills.append(ax_plot.fill_between(x, y2, alpha=0.15, color='tomato'))
        # Overlap region — minimum of the two curves at each x point
        fills.append(ax_plot.fill_between(x, np.minimum(y1, y2),
                                          alpha=0.40, color='purple',
                                          label='Overlap'))

    redraw_fills(y1_init, y2_init)

    vline1 = ax_plot.axvline(DEFAULT_MU1, color='steelblue', linewidth=1.8,
                             linestyle='--', alpha=0.8)
    vline2 = ax_plot.axvline(DEFAULT_MU2, color='tomato',    linewidth=1.8,
                             linestyle='--', alpha=0.8)

    ax_plot.set_xlabel('Value', fontsize=11)
    ax_plot.set_ylabel('Probability density', fontsize=11)
    ax_plot.set_ylim(0, 0.12)
    ax_plot.legend(fontsize=9, loc='upper right')
    ax_plot.grid(True, alpha=0.2)

    # ── Right panel: live formula ──────────────────────────────────────────────
    # Turn off all axes decorations — this panel is pure text
    ax_formula.axis('off')

    # Single text object that we update via set_text() on every slider change.
    # Using a monospace font makes the numbers align cleanly.
    formula_text = ax_formula.text(
        0.05, 0.97, '',
        transform=ax_formula.transAxes,
        ha='left', va='top',
        fontsize=10.5, family='monospace',
        linespacing=1.8,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f9f9e8',
                  edgecolor='#bbb', linewidth=1.2),
    )

    plt.tight_layout()

    # ── Interpretation helper ─────────────────────────────────────────────────
    def interpret(d):
        if d < 0.2:   return 'Negligible\n(almost total overlap)'
        elif d < 0.5: return 'Small\n(heavy overlap)'
        elif d < 0.8: return 'Medium\n(moderate overlap)'
        elif d < 1.2: return 'Large\n(well separated)'
        else:         return 'Very large\n(minimal overlap)'

    # ── Update function ───────────────────────────────────────────────────────
    def update(mu1, sig1, mu2, sig2):
        y1 = norm.pdf(x, mu1, sig1)
        y2 = norm.pdf(x, mu2, sig2)

        # Update curves and fills
        curve1.set_ydata(y1)
        curve2.set_ydata(y2)
        redraw_fills(y1, y2)

        # Move mean lines
        vline1.set_xdata([mu1, mu1])
        vline2.set_xdata([mu2, mu2])

        # Rescale y-axis so tall narrow curves aren't clipped
        peak = max(norm.pdf(mu1, mu1, sig1), norm.pdf(mu2, mu2, sig2))
        ax_plot.set_ylim(0, peak * 1.4)

        # ── Compute each step of the formula with real numbers ────────────────
        gap          = abs(mu1 - mu2)
        sig1_sq      = sig1 ** 2
        sig2_sq      = sig2 ** 2
        avg_var      = (sig1_sq + sig2_sq) / 2
        sigma_pooled = np.sqrt(avg_var)
        d            = gap / sigma_pooled

        # Build the formula string showing each calculation step explicitly.
        # The goal is that a user can follow each line top-to-bottom and
        # see exactly how the final d value is produced from the parameters.
        formula_str = (
            f' Step 1: gap between means\n'
            f' |μ₁ - μ₂| = |{mu1:.1f} - {mu2:.1f}|\n'
            f'          = {gap:.2f}\n'
            f'\n'
            f' Step 2: pooled standard deviation\n'
            f' σ_pooled = √( (σ₁² + σ₂²) / 2 )\n'
            f'         = √( ({sig1:.1f}² + {sig2:.1f}²) / 2 )\n'
            f'         = √( ({sig1_sq:.1f} + {sig2_sq:.1f}) / 2 )\n'
            f'         = √( {avg_var:.2f} )\n'
            f'         = {sigma_pooled:.3f}\n'
            f'\n'
            f' Step 3: Cohen\'s d\n'
            f' d = gap / σ_pooled\n'
            f'   = {gap:.2f} / {sigma_pooled:.3f}\n'
            f'   = {d:.3f}\n'
            f'\n'
            f' ─────────────────────\n'
            f' Result: {interpret(d)}'
        )

        formula_text.set_text(formula_str)

        ax_plot.set_title(
            f"Cohen's d = {d:.3f}  —  {interpret(d).splitlines()[0]}",
            fontsize=11,
        )
        ax_plot.legend(fontsize=9, loc='upper right')
        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    def make_slider(val, lo, hi, step, desc):
        return FloatSlider(value=val, min=lo, max=hi, step=step,
                           description=desc,
                           style={'description_width': '90px'},
                           layout=widgets.Layout(width='320px'),
                           continuous_update=True)

    def make_box(val, lo, hi, step):
        return widgets.BoundedFloatText(value=val, min=lo, max=hi, step=step,
                                        description='',
                                        layout=widgets.Layout(width='90px'))

    mu1_s  = make_slider(DEFAULT_MU1,  -50, 130, 0.5, 'Mean 1 (μ₁)')
    sig1_s = make_slider(DEFAULT_SIG1,    1,  40, 0.5, 'Std 1 (σ₁)')
    mu2_s  = make_slider(DEFAULT_MU2,  -50, 130, 0.5, 'Mean 2 (μ₂)')
    sig2_s = make_slider(DEFAULT_SIG2,    1,  40, 0.5, 'Std 2 (σ₂)')

    mu1_b  = make_box(DEFAULT_MU1,  -50, 130, 0.5)
    sig1_b = make_box(DEFAULT_SIG1,    1,  40, 0.5)
    mu2_b  = make_box(DEFAULT_MU2,  -50, 130, 0.5)
    sig2_b = make_box(DEFAULT_SIG2,    1,  40, 0.5)

    for s, b in [(mu1_s, mu1_b), (sig1_s, sig1_b),
                 (mu2_s, mu2_b), (sig2_s, sig2_b)]:
        widgets.jslink((s, 'value'), (b, 'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        mu1_s.value  = DEFAULT_MU1;  sig1_s.value = DEFAULT_SIG1
        mu2_s.value  = DEFAULT_MU2;  sig2_s.value = DEFAULT_SIG2

    reset_btn.on_click(on_reset)

    # ── Layout ────────────────────────────────────────────────────────────────
    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    def col(label_text, colour, mu_s, mu_b, sig_s, sig_b):
        return VBox([
            HTML(f'<b style="color:{colour}">{label_text}</b>'),
            HBox([mu_s,  mu_b]),
            HBox([sig_s, sig_b]),
        ])

    controls = VBox([
        HBox([
            col('Distribution 1', 'steelblue', mu1_s, mu1_b, sig1_s, sig1_b),
            col('Distribution 2', 'tomato',    mu2_s, mu2_b, sig2_s, sig2_b),
        ]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        'mu1': mu1_s, 'sig1': sig1_s,
        'mu2': mu2_s, 'sig2': sig2_s,
    })

    display(controls, out)