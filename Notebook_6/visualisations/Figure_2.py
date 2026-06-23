"""
Figure 2 — Interactive Gaussian Probability Density Function Explorer
=====================================================================
An interactive figure that lets users explore how the Gaussian (normal)
distribution changes as its parameters are adjusted, and evaluate it at a
specific feature value x_1.

The axis ranges and defaults are set to reflect the spam email worked example:
    Spam class:     mu=40,  sigma=8   (short emails from unknown senders)
    Not-spam class: mu=310, sigma=40  (longer emails from known contacts)

x_1 represents email length in words — so only non-negative values are used.

Three controllable parameters:

    mu    (μ) — the mean: controls where the curve is centred
    sigma (σ) — the standard deviation: controls how spread out the curve is
    x_1       — the feature value to evaluate: shown as a vertical marker;
                 the height of the curve at x_1 is P(x_1 | Y), the likelihood

The live equation panel substitutes all three values into the PDF formula and
shows the fully computed numeric result — so users can see exactly what
the maths produces for any combination of parameters.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_2 import show
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
from ipywidgets import interactive_output
from IPython.display import display

# ── x-axis range for the plot ────────────────────────────────────────────────
# Email word counts are non-negative and realistically range from near 0
# to around 500 words. We use 0 to 500 so the axis always makes sense
# in the context of the spam worked example.
X_PLOT = np.linspace(0, 500, 1000)


def gaussian_pdf(x, mu, sigma):
    """
    Evaluate the Gaussian probability density function at each point in x.

    Parameters
    ----------
    x     : scalar or array of x values to evaluate at
    mu    : mean of the distribution (centre of the bell curve)
    sigma : standard deviation (controls the width of the bell curve)

    Returns
    -------
    Scalar or array of probability density values.
    """
    # The coefficient normalises the curve so the total area underneath = 1.
    coefficient = 1.0 / np.sqrt(2 * np.pi * sigma ** 2)
    # The exponent is always <= 0, so exp(...) is always in (0, 1].
    # It equals 1 (maximum) when x == mu, falling off symmetrically either side.
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * np.exp(exponent)


def show():
    """
    Render Figure 2: Interactive Gaussian PDF Explorer with x_1 evaluation.

    Defaults match the spam class from the worked example (mu=40, sigma=8).
    Sliders control mu, sigma, and x_1.
    The curve, vertical marker, and equation all update live.
    """
    # Never call plt.close() here — interactive_output manages the canvas.

    # ── Default parameter values — spam class from the worked example ────────
    mu_init    = 40.0    # mean email length for spam (words)
    sigma_init = 8.0     # spread of email lengths for spam (words)
    x1_init    = 42.0    # the query email from the worked example (words)

    # ── Build figure and axes once ───────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook6 Figure 2', figsize=(10, 6))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    y_init = gaussian_pdf(X_PLOT, mu_init, sigma_init)

    # ── Bell curve and shaded area ───────────────────────────────────────────
    # Store the line so we can mutate it via set_ydata() on each update.
    (line,) = ax.plot(X_PLOT, y_init, color='steelblue', lw=2.5, zorder=3)

    # fill_between has no mutation API — store in a list so we can
    # call .remove() and redraw it each time.
    fill_container = [ax.fill_between(X_PLOT, y_init, alpha=0.15,
                                      color='steelblue', zorder=2)]

    # ── Vertical dashed line marking mu ─────────────────────────────────────
    mean_line = ax.axvline(mu_init, color='tomato', lw=1.5,
                           linestyle='--', alpha=0.7, zorder=4,
                           label=r'Mean $\mu$')

    # ── Vertical marker for x_1 ─────────────────────────────────────────────
    # This is the email length being evaluated — the orange line shows where
    # the query value sits relative to the class distribution.
    x1_line = ax.axvline(x1_init, color='darkorange', lw=2.0,
                         linestyle='-', alpha=0.9, zorder=5,
                         label=r'$x_1$ (query email length - spam class)')

    # ── Dot on the curve at x_1 ─────────────────────────────────────────────
    # The height of this dot is P(x_1 | Y) — the likelihood value.
    y1_init = gaussian_pdf(x1_init, mu_init, sigma_init)
    (dot,) = ax.plot(x1_init, y1_init, 'o', color='darkorange',
                     ms=9, zorder=6)

    # ── Dotted drop line from dot to x-axis ─────────────────────────────────
    # Helps users read the likelihood value off the y-axis.
    (h_line,) = ax.plot([x1_init, x1_init], [0, y1_init],
                        color='darkorange', lw=1.2,
                        linestyle=':', zorder=4, alpha=0.7)

    # ── Axis limits and labels ───────────────────────────────────────────────
    ax.set_xlim(0, 500)
    ax.set_ylim(0, gaussian_pdf(mu_init, mu_init, sigma_init) * 1.2)
    ax.set_xlabel(r'$x_1$ — email length (words)', fontsize=12)
    ax.set_ylabel(r'$P(x_1 \mid \mu, \sigma)$', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10, loc='upper left')

    # ── Live equation annotation ─────────────────────────────────────────────
    eq_text = ax.text(
        0.97, 0.97,
        _make_equation_string(mu_init, sigma_init, x1_init),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#cccccc', alpha=0.95),
        zorder=7,
    )

    # ── Title ────────────────────────────────────────────────────────────────
    title = ax.set_title(
        rf'Gaussian PDF: $\mu = {mu_init:.1f}$ words, '
        rf'$\sigma = {sigma_init:.1f}$, '
        rf'$x_1 = {x1_init:.1f}$ words',
        fontsize=11,
    )

    plt.tight_layout()

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════

    def make_slider(label, val, lo, hi, step, width='300px'):
        # Each slider is paired with a BoundedFloatText box so users
        # can type exact values as well as drag.
        sl = widgets.FloatSlider(
            value=val, min=lo, max=hi, step=step,
            description=label,
            style={'description_width': '90px'},
            layout=widgets.Layout(width=width),
            readout=False,
        )
        bx = widgets.BoundedFloatText(
            value=val, min=lo, max=hi, step=step,
            layout=widgets.Layout(width='85px'),
        )
        # jslink keeps slider and text box in sync without a Python round-trip.
        widgets.jslink((sl, 'value'), (bx, 'value'))
        return sl, bx

    # Ranges reflect realistic email word counts (0–500) and spreads (1–100).
    mu_slider,    mu_box    = make_slider(r'μ (mean)',       mu_init,     1.0, 490.0, 1.0)
    sigma_slider, sigma_box = make_slider(r'σ (std dev)',    sigma_init,  1.0, 100.0, 1.0)
    x1_slider,    x1_box    = make_slider(r'x₁ (length)',   x1_init,     1.0, 499.0, 1.0)

    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='75px'),
    )

    def on_reset(_):
        # Return to the spam class defaults from the worked example.
        mu_slider.value    = mu_init
        sigma_slider.value = sigma_init
        x1_slider.value    = x1_init

    reset_btn.on_click(on_reset)

    # Single horizontal row keeps vertical footprint minimal.
    controls = widgets.HBox([
        mu_slider, mu_box,
        widgets.Label('  '),
        sigma_slider, sigma_box,
        widgets.Label('  '),
        x1_slider, x1_box,
        widgets.Label('  '),
        reset_btn,
    ])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # Called automatically by interactive_output on every slider change.
    # ════════════════════════════════════════════════════════════════════════
    def update(mu, sigma, x1):
        y = gaussian_pdf(X_PLOT, mu, sigma)

        # Update the bell curve
        line.set_ydata(y)

        # Redraw the filled area (fill_between has no mutation API)
        fill_container[0].remove()
        fill_container[0] = ax.fill_between(X_PLOT, y, alpha=0.15,
                                            color='steelblue', zorder=2)

        # Move the mean marker
        mean_line.set_xdata([mu, mu])

        # Move the x_1 marker, dot, and drop line
        x1_val = gaussian_pdf(x1, mu, sigma)
        x1_line.set_xdata([x1, x1])
        dot.set_xdata([x1])
        dot.set_ydata([x1_val])
        h_line.set_xdata([x1, x1])
        h_line.set_ydata([0, x1_val])

        # Rescale y-axis to current peak plus 20% headroom so the curve
        # always fills the axes sensibly regardless of sigma.
        peak = gaussian_pdf(mu, mu, sigma)
        ax.set_ylim(0, peak * 1.2)

        # Update live equation and title
        eq_text.set_text(_make_equation_string(mu, sigma, x1))
        title.set_text(
            rf'Gaussian PDF: $\mu = {mu:.1f}$ words, '
            rf'$\sigma = {sigma:.1f}$, '
            rf'$x_1 = {x1:.1f}$ words'
        )

        fig.canvas.draw_idle()

    out = interactive_output(
        update,
        {'mu': mu_slider, 'sigma': sigma_slider, 'x1': x1_slider},
    )

    display(controls, out)


def _make_equation_string(mu, sigma, x1):
    """
    Build the live equation string with mu, sigma, and x1 substituted in,
    showing the full numeric result of P(x1 | mu, sigma).
    """
    sigma2      = sigma ** 2
    coefficient = 1.0 / np.sqrt(2 * np.pi * sigma2)
    exponent    = -((x1 - mu) ** 2) / (2 * sigma2)
    result      = coefficient * np.exp(exponent)

    return (
        # Line 1 — symbolic form
        r'$P(x_1 \mid \mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}'
        r'\exp\!\left(-\frac{(x_1-\mu)^2}{2\sigma^2}\right)$'
        '\n\n'
        # Line 2 — numbers substituted in
        rf'$= \frac{{1}}{{\sqrt{{2\pi \cdot {sigma2:.1f}}}}}'
        rf'\exp\!\left(-\frac{{({x1:.1f} - {mu:.1f})^2}}{{{2*sigma2:.1f}}}\right)$'
        '\n\n'
        # Line 3 — fully evaluated result
        rf'$= {coefficient:.5f} \times \exp({exponent:.4f})$'
        '\n\n'
        rf'$\approx {result:.6f}$'
    )