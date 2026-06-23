"""
Figure 1a — Interactive Distribution Explorer
=============================================

Six panels showing the six common distribution shapes described in
section 7.2. Each panel shows a histogram of sampled values overlaid
with the theoretical density curve where applicable.

The user can switch between distributions using a dropdown and adjust
the relevant parameters using sliders. All panels update live.

Distributions covered:
    Normal      — mean (μ) and standard deviation (σ)
    Uniform     — lower bound (a) and upper bound (b)
    Right-skewed — shape parameter (α) using a log-normal distribution
    Left-skewed  — shape parameter using a reflected log-normal
    Bimodal     — means of two sub-populations and their mixing proportion
    Exponential — rate parameter (λ)

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_1a import show
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
from scipy import stats


def show():
    """Render the interactive Figure 1a distribution explorer."""

    plt.close('Notebook3 Figure 1a')

    N = 2000   # number of samples — large enough for smooth histograms
    rng = np.random.default_rng(42)

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULTS = {
        'Normal':      dict(mu=0.0,   sigma=1.0),
        'Uniform':     dict(a=-3.0,   b=3.0),
        'Right-skewed': dict(skew=1.0),
        'Left-skewed':  dict(skew=1.0),
        'Bimodal':     dict(mu1=-2.0, mu2=2.0, mix=0.5),
        'Exponential': dict(lam=1.0),
    }

    # ── Build the figure — single axes, redrawn on every change ───────────────
    fig, ax = plt.subplots(num='Notebook3 Figure 1a', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, left=0.12, bottom=0.12)

    # ── Sliders for all parameters (shown/hidden by dropdown) ─────────────────
    mu_s    = FloatSlider(value=0.0,  min=-5.0, max=5.0,  step=0.1,
                          description='Mean (μ)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    sigma_s = FloatSlider(value=1.0,  min=0.1,  max=5.0,  step=0.1,
                          description='Std dev (σ)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    a_s     = FloatSlider(value=-3.0, min=-6.0, max=0.0,  step=0.1,
                          description='Lower (a)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    b_s     = FloatSlider(value=3.0,  min=0.0,  max=6.0,  step=0.1,
                          description='Upper (b)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    skew_s  = FloatSlider(value=1.0,  min=0.2,  max=3.0,  step=0.1,
                          description='Skew (σ)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    mu1_s   = FloatSlider(value=-2.0, min=-5.0, max=0.0,  step=0.1,
                          description='Mean 1 (μ₁)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    mu2_s   = FloatSlider(value=2.0,  min=0.0,  max=5.0,  step=0.1,
                          description='Mean 2 (μ₂)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    mix_s   = FloatSlider(value=0.5,  min=0.1,  max=0.9,  step=0.05,
                          description='Mix ratio',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)
    lam_s   = FloatSlider(value=1.0,  min=0.1,  max=5.0,  step=0.1,
                          description='Rate (λ)',
                          style={'description_width': '110px'},
                          layout=widgets.Layout(width='340px'),
                          continuous_update=True)

    # ── Dropdown to select distribution ───────────────────────────────────────
    dist_dd = Dropdown(
        options=['Normal', 'Uniform', 'Right-skewed',
                 'Left-skewed', 'Bimodal', 'Exponential'],
        value='Normal',
        description='Distribution:',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='340px'),
    )

    # ── Description label that updates with the distribution ──────────────────
    desc_label = HTML(value='')

    DESCRIPTIONS = {
        'Normal':
            '<b>Normal (Gaussian)</b> — symmetric bell curve. '
            'Adjust μ to shift left/right; adjust σ to widen or narrow.',
        'Uniform':
            '<b>Uniform</b> — all values equally likely between a and b. '
            'No peak, no tail — perfectly flat.',
        'Right-skewed':
            '<b>Right-skewed</b> — long tail to the right. '
            'Increase σ to make the tail longer and the skew more pronounced.',
        'Left-skewed':
            '<b>Left-skewed</b> — long tail to the left. '
            'Increase σ to make the skew more pronounced.',
        'Bimodal':
            '<b>Bimodal</b> — two peaks from two mixed subpopulations. '
            'Adjust μ₁ and μ₂ to move the peaks; mix ratio controls their relative size.',
        'Exponential':
            '<b>Exponential</b> — peak at zero, long right tail. '
            'Higher λ = faster decay = shorter typical values.',
    }

    # ── Slider panels — one per distribution ──────────────────────────────────
    panels = {
        'Normal':      VBox([HBox([mu_s]),    HBox([sigma_s])]),
        'Uniform':     VBox([HBox([a_s]),     HBox([b_s])]),
        'Right-skewed': VBox([HBox([skew_s])]),
        'Left-skewed':  VBox([HBox([skew_s])]),
        'Bimodal':     VBox([HBox([mu1_s]),   HBox([mu2_s]),  HBox([mix_s])]),
        'Exponential': VBox([HBox([lam_s])]),
    }

    # Container that holds whichever panel is currently active
    slider_container = VBox([panels['Normal']])

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    # ── Update function ───────────────────────────────────────────────────────
    def update(dist, mu, sigma, a, b, skew, mu1, mu2, mix, lam):
        ax.cla()   # clear the axes and redraw from scratch

        colour = {
            'Normal': 'steelblue', 'Uniform': 'seagreen',
            'Right-skewed': 'tomato', 'Left-skewed': 'darkorange',
            'Bimodal': 'purple', 'Exponential': 'goldenrod',
        }[dist]

        # ── Generate samples and PDF curve ────────────────────────────────────
        if dist == 'Normal':
            samples = rng.normal(mu, sigma, N)
            xs      = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
            pdf     = stats.norm.pdf(xs, mu, sigma)
            ax.axvline(mu, color='black', linewidth=2, linestyle='-',
                       label=f'Mean = {mu:.1f}')
            ax.axvline(mu - sigma, color='black', linewidth=1.5,
                       linestyle='--', label=f'±1σ  (σ={sigma:.1f})')
            ax.axvline(mu + sigma, color='black', linewidth=1.5, linestyle='--')
            title = f'Normal distribution  μ={mu:.1f},  σ={sigma:.1f}'

        elif dist == 'Uniform':
            if b <= a:
                b = a + 0.1
            samples = rng.uniform(a, b, N)
            xs      = np.linspace(a - 0.5, b + 0.5, 300)
            pdf     = stats.uniform.pdf(xs, a, b - a)
            ax.axvline((a + b) / 2, color='black', linewidth=2, linestyle='-',
                       label=f'Mean = {(a+b)/2:.1f}')
            title = f'Uniform distribution  a={a:.1f},  b={b:.1f}'

        elif dist == 'Right-skewed':
            samples = np.exp(rng.normal(0, skew, N))
            xs      = np.linspace(0.001, np.percentile(samples, 99), 300)
            pdf     = stats.lognorm.pdf(xs, skew)
            ax.axvline(np.median(samples), color='black', linewidth=2,
                       linestyle='-', label=f'Median = {np.median(samples):.2f}')
            ax.axvline(np.mean(samples), color='red', linewidth=1.5,
                       linestyle='--', label=f'Mean = {np.mean(samples):.2f}')
            title = f'Right-skewed distribution  (log-normal, σ={skew:.1f})'

        elif dist == 'Left-skewed':
            samples = -np.exp(rng.normal(0, skew, N))
            xs      = np.linspace(np.percentile(samples, 1), -0.001, 300)
            pdf     = stats.lognorm.pdf(-xs, skew)
            ax.axvline(np.median(samples), color='black', linewidth=2,
                       linestyle='-', label=f'Median = {np.median(samples):.2f}')
            ax.axvline(np.mean(samples), color='red', linewidth=1.5,
                       linestyle='--', label=f'Mean = {np.mean(samples):.2f}')
            title = f'Left-skewed distribution  (reflected log-normal, σ={skew:.1f})'

        elif dist == 'Bimodal':
            n1 = int(N * mix)
            n2 = N - n1
            samples = np.concatenate([
                rng.normal(mu1, 0.8, n1),
                rng.normal(mu2, 0.8, n2),
            ])
            xs  = np.linspace(mu1 - 4, mu2 + 4, 300)
            pdf = (mix * stats.norm.pdf(xs, mu1, 0.8) +
                   (1 - mix) * stats.norm.pdf(xs, mu2, 0.8))
            ax.axvline(mu1, color='black', linewidth=1.5, linestyle='--',
                       label=f'μ₁={mu1:.1f}')
            ax.axvline(mu2, color='black', linewidth=1.5, linestyle=':',
                       label=f'μ₂={mu2:.1f}')
            title = (f'Bimodal distribution  μ₁={mu1:.1f},  μ₂={mu2:.1f},  '
                     f'mix={mix:.0%}/{1-mix:.0%}')

        else:  # Exponential
            samples = rng.exponential(1 / lam, N)
            xs      = np.linspace(0, np.percentile(samples, 99), 300)
            pdf     = stats.expon.pdf(xs, scale=1/lam)
            ax.axvline(1/lam, color='black', linewidth=2, linestyle='-',
                       label=f'Mean = 1/λ = {1/lam:.2f}')
            title = f'Exponential distribution  λ={lam:.1f}'

        # ── Draw histogram and PDF overlay ────────────────────────────────────
        ax.hist(samples, bins=50, density=True, color=colour,
                alpha=0.55, edgecolor='white', linewidth=0.3)

        if dist not in ('Bimodal',):
            ax.plot(xs, pdf, color=colour, linewidth=2.5,
                    label='Theoretical PDF')
        else:
            ax.plot(xs, pdf, color='black', linewidth=2.5,
                    linestyle='-', label='Combined PDF')

        # Summary stats annotation
        ax.text(0.98, 0.97,
                f'mean   = {np.mean(samples):.3f}\n'
                f'median = {np.median(samples):.3f}\n'
                f'std    = {np.std(samples):.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          edgecolor='#aaa', alpha=0.9))

        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Probability density', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

        desc_label.value = (
            f'<div style="font-size:0.9em; color:#444; margin-top:4px;">'
            f'{DESCRIPTIONS[dist]}</div>'
        )

        fig.canvas.draw_idle()

    # ── Handle dropdown change — show/hide the right sliders ──────────────────
    def on_dist_change(change):
        slider_container.children = [panels[change['new']]]
        # Reset sliders to defaults for the newly selected distribution
        d = change['new']
        if d == 'Normal':
            mu_s.value = 0.0;  sigma_s.value = 1.0
        elif d == 'Uniform':
            a_s.value = -3.0;  b_s.value = 3.0
        elif d in ('Right-skewed', 'Left-skewed'):
            skew_s.value = 1.0
        elif d == 'Bimodal':
            mu1_s.value = -2.0;  mu2_s.value = 2.0;  mix_s.value = 0.5
        elif d == 'Exponential':
            lam_s.value = 1.0

    dist_dd.observe(on_dist_change, names='value')

    def on_reset(b):
        on_dist_change({'new': dist_dd.value})

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HBox([dist_dd, reset_btn]),
        slider_container,
        sep,
        desc_label,
    ])

    out = interactive_output(update, {
        'dist':  dist_dd,
        'mu':    mu_s,   'sigma': sigma_s,
        'a':     a_s,    'b':     b_s,
        'skew':  skew_s,
        'mu1':   mu1_s,  'mu2':  mu2_s,  'mix': mix_s,
        'lam':   lam_s,
    })

    display(controls, out)