"""
Figure 1 — Two Normal Distributions Explorer
=============================================

Two side-by-side histograms showing normal distributions with different
means and standard deviations:

    Distribution A: tight bell curve  (mean=50, std=5)
    Distribution B: shifted right     (mean=70, std=5)

The user can adjust the mean, standard deviation, and sample size for
each distribution independently. A vertical line marks the mean and dashed
lines mark +/- 1 standard deviation on each panel.

Summary statistics (mean, median, std, min, max) are printed to the cell
output whenever the sliders are changed.

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
from ipywidgets import interactive_output, IntSlider, FloatSlider, VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the interactive Figure 1 normal distribution explorer."""

    plt.close('Notebook3 Figure 1')

    # ── Default parameter values ──────────────────────────────────────────────
    DEFAULT_N     = 1000
    DEFAULT_MU_A  = 50.0
    DEFAULT_SIG_A = 5.0
    DEFAULT_MU_B  = 70.0
    DEFAULT_SIG_B = 5.0

    COLOURS = ['steelblue', 'seagreen']

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook3 Figure 1', figsize=(10, 5), sharey=True)
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Helper: rebuild one panel ─────────────────────────────────────────────
    def draw_panel(ax, samples, colour, label):
        """Clear ax and redraw the histogram with mean and std markers."""
        ax.cla()

        ax.hist(samples, bins=40, color=colour,
                edgecolor='white', linewidth=0.4, alpha=0.85)

        # Solid vertical line at the sample mean
        ax.axvline(samples.mean(), color='black', linewidth=2,
                   linestyle='-', label=f'Mean = {samples.mean():.1f}')

        # Dashed lines at +/- 1 standard deviation
        ax.axvline(samples.mean() + samples.std(), color='black', linewidth=1.5,
                   linestyle='--', label=f'+/- 1 std  (std={samples.std():.1f})')
        ax.axvline(samples.mean() - samples.std(), color='black',
                   linewidth=1.5, linestyle='--')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Value')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Count')

    # Output widget for live summary statistics
    stats_out = widgets.Output()

    # ── Update function ───────────────────────────────────────────────────────
    def update(n, mu_a, sig_a, mu_b, sig_b):
        rng = np.random.default_rng()

        dist_a = rng.normal(loc=mu_a, scale=sig_a, size=n)
        dist_b = rng.normal(loc=mu_b, scale=sig_b, size=n)

        configs = [
            (dist_a, COLOURS[0], f'A: mean={mu_a:.1f}, std={sig_a:.1f}'),
            (dist_b, COLOURS[1], f'B: mean={mu_b:.1f}, std={sig_b:.1f}'),
        ]

        for ax, (dist, colour, label) in zip(axes, configs):
            draw_panel(ax, dist, colour, label)

        axes[0].set_ylabel('Count')

        fig.suptitle(
            'Figure 1: Two Normal distributions — comparing mean and standard deviation',
            fontsize=11,
        )

        stats_out.clear_output(wait=True)
        with stats_out:
            for name, dist in zip(
                [f'A (μ={mu_a:.1f}, σ={sig_a:.1f})',
                 f'B (μ={mu_b:.1f}, σ={sig_b:.1f})'],
                [dist_a, dist_b],
            ):
                print(f'Dist {name}:  '
                      f'mean={dist.mean():.1f}  median={np.median(dist):.1f}  '
                      f'std={dist.std():.1f}  min={dist.min():.1f}  max={dist.max():.1f}')

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    def make_float(val, lo, hi, step, desc):
        return FloatSlider(value=val, min=lo, max=hi, step=step,
                           description=desc, style={'description_width': '80px'},
                           layout=widgets.Layout(width='300px'), continuous_update=True)

    def make_box(val, lo, hi, step):
        return widgets.BoundedFloatText(value=val, min=lo, max=hi, step=step,
                                        description='', layout=widgets.Layout(width='90px'))

    n_slider = IntSlider(value=DEFAULT_N, min=50, max=5000, step=50,
                         description='Samples (n)', style={'description_width': '90px'},
                         layout=widgets.Layout(width='310px'), continuous_update=True)
    n_box    = widgets.BoundedIntText(value=DEFAULT_N, min=50, max=5000, step=50,
                                      description='', layout=widgets.Layout(width='90px'))
    widgets.jslink((n_slider, 'value'), (n_box, 'value'))

    mu_a_s  = make_float(DEFAULT_MU_A,  0, 100, 0.5, 'Mean A (μ)')
    sig_a_s = make_float(DEFAULT_SIG_A, 0.5, 30, 0.5, 'Std A (σ)')
    mu_b_s  = make_float(DEFAULT_MU_B,  0, 100, 0.5, 'Mean B (μ)')
    sig_b_s = make_float(DEFAULT_SIG_B, 0.5, 30, 0.5, 'Std B (σ)')

    mu_a_b  = make_box(DEFAULT_MU_A,  0, 100, 0.5)
    sig_a_b = make_box(DEFAULT_SIG_A, 0.5, 30, 0.5)
    mu_b_b  = make_box(DEFAULT_MU_B,  0, 100, 0.5)
    sig_b_b = make_box(DEFAULT_SIG_B, 0.5, 30, 0.5)

    for s, b in [(mu_a_s, mu_a_b), (sig_a_s, sig_a_b),
                 (mu_b_s, mu_b_b), (sig_b_s, sig_b_b)]:
        widgets.jslink((s, 'value'), (b, 'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        n_slider.value = DEFAULT_N
        mu_a_s.value   = DEFAULT_MU_A
        sig_a_s.value   = DEFAULT_SIG_A
        mu_b_s.value   = DEFAULT_MU_B
        sig_b_s.value   = DEFAULT_SIG_B

    reset_btn.on_click(on_reset)

    # ── Layout ────────────────────────────────────────────────────────────────
    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    def col(label_text, mu_s, mu_b, sig_s, sig_b, colour):
        return VBox([
            HTML(f'<b style="color:{colour}">Distribution {label_text}</b>'),
            HBox([mu_s, mu_b]),
            HBox([sig_s, sig_b]),
        ])

    controls = VBox([
        HBox([n_slider, n_box]),
        sep,
        HBox([
            col('A', mu_a_s, mu_a_b, sig_a_s, sig_a_b, COLOURS[0]),
            col('B', mu_b_s, mu_b_b, sig_b_s, sig_b_b, COLOURS[1]),
        ]),
        reset_btn,
        stats_out,
    ])

    out = interactive_output(update, {
        'n':    n_slider,
        'mu_a': mu_a_s, 'sig_a': sig_a_s,
        'mu_b': mu_b_s, 'sig_b': sig_b_s,
    })

    display(controls, out)