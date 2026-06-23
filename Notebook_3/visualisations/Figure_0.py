"""
Figure 0 — Interactive Normal and Uniform Distribution Explorer
===============================================================

Supports the introductory text on random number generation and statistical
distributions. Two side-by-side panels update live as the user adjusts
the distribution parameters:

Left panel:  Normal distribution — user controls the mean (μ) and
             standard deviation (σ). The histogram of sampled values is
             overlaid with the theoretical probability density curve.

Right panel: Uniform distribution — user controls the lower bound (a)
             and upper bound (b). The histogram is overlaid with the flat
             theoretical density line.

Both panels use the same sample size (n = 1000) so the shapes are stable
and easy to compare.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_0 import show
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
from ipywidgets import interactive_output, FloatSlider, VBox, HBox, Label
from IPython.display import display
from scipy.stats import norm, uniform


def show():
    """Render the interactive Figure 0 distribution explorer."""

    plt.close('Notebook3 Figure 0')

    # Fixed random seed so the initial histogram is the same for every user.
    # We reseed inside update() too so dragging sliders shows meaningful change.
    rng = np.random.default_rng(42)

    # Sample size — large enough for a smooth histogram, small enough to be fast
    N = 1000

    # Default parameter values
    DEFAULT_MU    =  0.0   # Normal: mean
    DEFAULT_SIGMA =  1.0   # Normal: standard deviation
    DEFAULT_A     =  0.0   # Uniform: lower bound
    DEFAULT_B     =  1.0   # Uniform: upper bound

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook3 Figure 0', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    ax_norm = axes[0]   # Left panel  — normal distribution
    ax_unif = axes[1]   # Right panel — uniform distribution

    # ── Left panel: normal distribution ──────────────────────────────────────

    # Draw the initial histogram.  density=True scales the y-axis to probability
    # density so it can be compared directly with the theoretical curve.
    norm_samples = rng.normal(loc=DEFAULT_MU, scale=DEFAULT_SIGMA, size=N)
    norm_counts, norm_bins, norm_patches = ax_norm.hist(
        norm_samples, bins=40, density=True,
        color='steelblue', alpha=0.6, label='Sampled values',
    )

    # Overlay the theoretical probability density function (PDF).
    # We store the line so set_ydata() can update it when parameters change.
    x_norm = np.linspace(-10, 10, 300)
    (norm_curve,) = ax_norm.plot(
        x_norm,
        norm.pdf(x_norm, loc=DEFAULT_MU, scale=DEFAULT_SIGMA),
        color='steelblue', linewidth=2.5,
        label=f'PDF  μ={DEFAULT_MU:.1f}, σ={DEFAULT_SIGMA:.1f}',
    )

    ax_norm.set_title('Normal Distribution')
    ax_norm.set_xlabel('Value')
    ax_norm.set_ylabel('Probability density')
    ax_norm.set_xlim(-10, 10)
    ax_norm.set_ylim(0, 1.0)
    ax_norm.legend(fontsize=8)
    ax_norm.grid(True, alpha=0.3)

    # ── Right panel: uniform distribution ────────────────────────────────────

    unif_samples = rng.uniform(low=DEFAULT_A, high=DEFAULT_B, size=N)
    unif_counts, unif_bins, unif_patches = ax_unif.hist(
        unif_samples, bins=40, density=True,
        color='tomato', alpha=0.6, label='Sampled values',
    )

    # The uniform PDF is a flat horizontal line at height 1/(b-a)
    x_unif = np.linspace(-6, 6, 300)
    (unif_curve,) = ax_unif.plot(
        x_unif,
        uniform.pdf(x_unif, loc=DEFAULT_A, scale=DEFAULT_B - DEFAULT_A),
        color='tomato', linewidth=2.5,
        label=f'PDF  a={DEFAULT_A:.1f}, b={DEFAULT_B:.1f}',
    )

    ax_unif.set_title('Uniform Distribution')
    ax_unif.set_xlabel('Value')
    ax_unif.set_ylabel('Probability density')
    ax_unif.set_xlim(-6, 6)
    ax_unif.set_ylim(0, 4.0)
    ax_unif.legend(fontsize=8)
    ax_unif.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── Helper: rebuild histogram bars ────────────────────────────────────────
    # Matplotlib histograms have no mutation API — we must remove the old bars
    # and draw new ones, similar to fill_between in earlier figures.
    def redraw_hist(ax, samples, bins, color, alpha):
        """Remove existing histogram patches and draw fresh ones."""
        # Remove all existing bar patches from the axes
        for patch in ax.patches:
            patch.remove()
        # Draw the new histogram and return the patches for future removal
        _, _, patches = ax.hist(
            samples, bins=bins, density=True,
            color=color, alpha=alpha, label='Sampled values',
        )
        return patches

    # ── Update function ───────────────────────────────────────────────────────
    # Called by interactive_output whenever any slider changes.
    def update(mu, sigma, a, b):
        # Guard against invalid uniform range — b must be greater than a
        if b <= a:
            return

        # Generate fresh samples using a new seed each call so the histogram
        # shape reflects the parameters rather than frozen noise
        rng_local = np.random.default_rng()
        norm_samples = rng_local.normal(loc=mu, scale=sigma, size=N)
        unif_samples = rng_local.uniform(low=a, high=b, size=N)

        # Rebuild histograms
        redraw_hist(ax_norm, norm_samples, 40, 'steelblue', 0.6)
        redraw_hist(ax_unif, unif_samples, 40, 'tomato',    0.6)

        # Update the theoretical PDF curves in place
        norm_curve.set_ydata(norm.pdf(x_norm, loc=mu, scale=sigma))
        norm_curve.set_label(f'PDF  μ={mu:.2f}, σ={sigma:.2f}')

        unif_curve.set_ydata(uniform.pdf(x_unif, loc=a, scale=b - a))
        unif_curve.set_label(f'PDF  a={a:.2f}, b={b:.2f}')

        # Rescale y-axis so tall narrow normal curves don't get clipped
        ax_norm.set_ylim(0, max(1.0, norm.pdf(mu, loc=mu, scale=sigma) * 1.3))
        # Uniform density is 1/(b-a); add headroom above the flat line
        ax_unif.set_ylim(0, max(4.0, (1 / (b - a)) * 1.3))

        ax_norm.legend(fontsize=8)
        ax_unif.legend(fontsize=8)
        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────

    mu_slider = FloatSlider(
        value=DEFAULT_MU, min=-5.0, max=5.0, step=0.1,
        description='Mean (μ)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='340px'),
        continuous_update=True,
    )
    sigma_slider = FloatSlider(
        value=DEFAULT_SIGMA, min=0.1, max=5.0, step=0.1,
        description='Std dev (σ)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='340px'),
        continuous_update=True,
    )
    a_slider = FloatSlider(
        value=DEFAULT_A, min=-5.0, max=4.9, step=0.1,
        description='Lower (a)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='340px'),
        continuous_update=True,
    )
    b_slider = FloatSlider(
        value=DEFAULT_B, min=-4.9, max=5.0, step=0.1,
        description='Upper (b)',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='340px'),
        continuous_update=True,
    )

    # Paired text boxes for precise value entry
    mu_box    = widgets.BoundedFloatText(value=DEFAULT_MU,    min=-5.0, max=5.0,  step=0.1, description='', layout=widgets.Layout(width='90px'))
    sigma_box = widgets.BoundedFloatText(value=DEFAULT_SIGMA, min=0.1,  max=5.0,  step=0.1, description='', layout=widgets.Layout(width='90px'))
    a_box     = widgets.BoundedFloatText(value=DEFAULT_A,     min=-5.0, max=4.9,  step=0.1, description='', layout=widgets.Layout(width='90px'))
    b_box     = widgets.BoundedFloatText(value=DEFAULT_B,     min=-4.9, max=5.0,  step=0.1, description='', layout=widgets.Layout(width='90px'))

    widgets.jslink((mu_slider,    'value'), (mu_box,    'value'))
    widgets.jslink((sigma_slider, 'value'), (sigma_box, 'value'))
    widgets.jslink((a_slider,     'value'), (a_box,     'value'))
    widgets.jslink((b_slider,     'value'), (b_box,     'value'))

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(
        description='Reset',
        button_style='warning',
        layout=widgets.Layout(width='100px'),
    )

    def on_reset(b):
        mu_slider.value    = DEFAULT_MU
        sigma_slider.value = DEFAULT_SIGMA
        a_slider.value     = DEFAULT_A
        b_slider.value     = DEFAULT_B

    reset_btn.on_click(on_reset)

    # ── Layout ────────────────────────────────────────────────────────────────
    # A thin horizontal rule visually separates the two sets of controls
    sep = widgets.HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        Label('Normal distribution', style={'font_weight': 'bold'}),
        HBox([mu_slider,    mu_box]),
        HBox([sigma_slider, sigma_box]),
        sep,
        Label('Uniform distribution', style={'font_weight': 'bold'}),
        HBox([a_slider, a_box]),
        HBox([b_slider, b_box]),
        reset_btn,
    ])

    out = interactive_output(update, {
        'mu': mu_slider, 'sigma': sigma_slider,
        'a':  a_slider,  'b':     b_slider,
    })

    display(controls, out)