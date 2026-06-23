"""
Figure 1b — Interactive Gaussian Likelihood Explorer
=====================================================
Shows a histogram of synthetic email lengths for the spam class, overlaid
with a fitted Gaussian curve. Users can adjust the mean and standard
deviation manually to see how the fit changes, then evaluate the likelihood
at a specific email length x_1.

The likelihood P(x_1 | Y) is shown directly on the plot:
  - A vertical orange line marks x_1
  - A horizontal dashed line from the curve height at x_1 to the y-axis
  - The full Gaussian PDF equation with numbers substituted in is annotated

Controls:
  - mu slider       — shift the Gaussian mean
  - sigma slider    — adjust the Gaussian spread
  - x_1 slider      — the email length to evaluate the likelihood at
  - Snap to best fit — sets mu and sigma to the sample mean and std of the data
  - Reset           — returns all sliders to their default values

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_1b import show
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

# ── Generate synthetic spam email lengths ────────────────────────────────────
# We use a fixed seed so the histogram looks the same every time the cell runs.
# The spam class from the worked example has mean=40 words, std=8 words.
rng = np.random.default_rng(42)
N_SAMPLES  = 200
TRUE_MU    = 40.0
TRUE_SIGMA = 8.0
EMAIL_LENGTHS = rng.normal(TRUE_MU, TRUE_SIGMA, N_SAMPLES)
EMAIL_LENGTHS = np.clip(EMAIL_LENGTHS, 1, 500)   # email lengths must be positive

# ── Best-fit parameters computed from the data ───────────────────────────────
# These are the values the model would actually learn from this training data.
BEST_MU    = float(np.mean(EMAIL_LENGTHS))
BEST_SIGMA = float(np.std(EMAIL_LENGTHS))

# ── x-axis range for the curve ───────────────────────────────────────────────
X_CURVE = np.linspace(0, 100, 500)

# ── Histogram bin edges — fixed so the bars don't jump as sliders move ───────
BINS = np.arange(0, 101, 4)


def gaussian_pdf(x, mu, sigma):
    """Evaluate the Gaussian PDF at x given mu and sigma."""
    coeff   = 1.0 / np.sqrt(2 * np.pi * sigma ** 2)
    exp_val = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coeff * exp_val


def show():
    """
    Render Figure 1b: Interactive Gaussian Likelihood Explorer.

    Histogram of spam email lengths with an overlaid Gaussian curve.
    Sliders control mu, sigma, and x_1. The likelihood at x_1 is shown
    directly on the plot with a full annotated equation.
    """
    # ── Default parameter values ─────────────────────────────────────────────
    mu_init    = TRUE_MU
    sigma_init = TRUE_SIGMA
    x1_init    = 42.0

    # ── Build figure and axes once ───────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook6 Figure 1b', figsize=(10, 5))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ── Draw the histogram (static — data doesn't change) ────────────────────
    # density=True scales the histogram so its area integrates to 1,
    # matching the scale of the Gaussian PDF curve drawn on top.
    ax.hist(EMAIL_LENGTHS, bins=BINS, density=True,
            color='steelblue', alpha=0.35, edgecolor='white',
            linewidth=0.5, zorder=2, label='Spam email lengths (histogram)')

    # ── Initial Gaussian curve ───────────────────────────────────────────────
    y_curve_init = gaussian_pdf(X_CURVE, mu_init, sigma_init)
    (curve_line,) = ax.plot(X_CURVE, y_curve_init,
                            color='steelblue', lw=2.5, zorder=3,
                            label=r'Fitted Gaussian $P(x_1 \mid Y=\mathrm{spam})$')

    # ── Vertical line for mu ─────────────────────────────────────────────────
    mu_line = ax.axvline(mu_init, color='tomato', lw=1.5,
                         linestyle='--', alpha=0.7, zorder=4,
                         label=r'Mean $\mu$')

    # ── Vertical line for x_1 ────────────────────────────────────────────────
    x1_vline = ax.axvline(x1_init, color='darkorange', lw=2.0,
                          linestyle='-', alpha=0.9, zorder=5,
                          label=r'$x_1$ (query email length)')

    # ── Dot on the curve at x_1 ──────────────────────────────────────────────
    y1_init = gaussian_pdf(x1_init, mu_init, sigma_init)
    (dot,) = ax.plot(x1_init, y1_init, 'o', color='darkorange',
                     ms=10, zorder=7)

    # ── Horizontal dashed line from dot to y-axis ────────────────────────────
    # This helps the user read off the likelihood value on the y-axis.
    (h_line,) = ax.plot([0, x1_init], [y1_init, y1_init],
                        color='darkorange', lw=1.2,
                        linestyle=':', zorder=4, alpha=0.8)

    # ── Vertical drop line from dot to x-axis ────────────────────────────────
    (v_drop,) = ax.plot([x1_init, x1_init], [0, y1_init],
                        color='darkorange', lw=1.2,
                        linestyle=':', zorder=4, alpha=0.5)

    # ── Equation annotation ───────────────────────────────────────────────────
    eq_text = ax.text(
        0.98, 0.97,
        _make_eq_string(mu_init, sigma_init, x1_init),
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#cccccc', alpha=0.95),
        zorder=8,
    )

    # ── y-axis annotation showing the likelihood value ────────────────────────
    # A small text label on the y-axis at the height of the dot.
    y_label = ax.text(
        0.5, y1_init,
        f'  {y1_init:.5f}',
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        color='darkorange',
        va='center',
        zorder=8,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    title = ax.set_title(
        rf'Spam email lengths — Gaussian fit: $\mu={mu_init:.1f}$, '
        rf'$\sigma={sigma_init:.1f}$, $x_1={x1_init:.1f}$ words',
        fontsize=11,
    )

    # ── Axes settings ─────────────────────────────────────────────────────────
    ax.set_xlim(0, 100)
    ax.set_ylim(0, gaussian_pdf(BEST_MU, BEST_MU, max(BEST_SIGMA * 0.5, 1)) * 1.4)
    ax.set_xlabel(r'Email length $x_1$ (words)', fontsize=12)
    ax.set_ylabel(r'Probability density  $P(x_1 \mid \mu, \sigma)$', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════

    def make_slider(label, val, lo, hi, step, width='260px'):
        sl = widgets.FloatSlider(
            value=val, min=lo, max=hi, step=step,
            description=label,
            style={'description_width': '90px'},
            layout=widgets.Layout(width=width),
            readout=False,
        )
        bx = widgets.BoundedFloatText(
            value=val, min=lo, max=hi, step=step,
            layout=widgets.Layout(width='80px'),
        )
        widgets.jslink((sl, 'value'), (bx, 'value'))
        return sl, bx

    mu_slider,    mu_box    = make_slider(r'μ (mean)',      mu_init,    1.0,  90.0, 0.5)
    sigma_slider, sigma_box = make_slider(r'σ (std dev)',   sigma_init, 0.5,  30.0, 0.5)
    x1_slider,    x1_box    = make_slider(r'x₁ (length)',  x1_init,    1.0,  99.0, 1.0)

    # ── Snap to best fit button ───────────────────────────────────────────────
    snap_btn = widgets.Button(
        description=f'Snap to best fit  (μ={BEST_MU:.1f}, σ={BEST_SIGMA:.1f})',
        button_style='info',
        layout=widgets.Layout(width='280px'),
    )

    def on_snap(_):
        # Set sliders to the sample mean and std computed from the data —
        # these are the values the model would actually learn during training.
        mu_slider.value    = round(BEST_MU,    1)
        sigma_slider.value = round(BEST_SIGMA, 1)

    snap_btn.on_click(on_snap)

    # ── Reset button ──────────────────────────────────────────────────────────
    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_reset(_):
        mu_slider.value    = mu_init
        sigma_slider.value = sigma_init
        x1_slider.value    = x1_init

    reset_btn.on_click(on_reset)

    # ── Layout ────────────────────────────────────────────────────────────────
    controls = widgets.HBox([
        mu_slider, mu_box,
        widgets.Label('  '),
        sigma_slider, sigma_box,
        widgets.Label('  '),
        x1_slider, x1_box,
        widgets.Label('  '),
        snap_btn,
        widgets.Label(' '),
        reset_btn,
    ])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # ════════════════════════════════════════════════════════════════════════
    def update(mu, sigma, x1):
        # Recompute the Gaussian curve
        y_curve = gaussian_pdf(X_CURVE, mu, sigma)
        curve_line.set_ydata(y_curve)

        # Move the mean marker
        mu_line.set_xdata([mu, mu])

        # Move x_1 marker and dot
        x1_val = gaussian_pdf(x1, mu, sigma)
        x1_vline.set_xdata([x1, x1])
        dot.set_xdata([x1])
        dot.set_ydata([x1_val])

        # Update horizontal and vertical drop lines
        h_line.set_xdata([0, x1])
        h_line.set_ydata([x1_val, x1_val])
        v_drop.set_xdata([x1, x1])
        v_drop.set_ydata([0, x1_val])

        # Update y-axis likelihood label
        y_label.set_position((0.5, x1_val))
        y_label.set_text(f'  {x1_val:.5f}')

        # Rescale y-axis so curve always fills the axes sensibly.
        # Peak of the Gaussian is at x=mu with height 1/sqrt(2*pi*sigma^2).
        peak = gaussian_pdf(mu, mu, sigma)
        ax.set_ylim(0, peak * 1.4)

        # Update annotation and title
        eq_text.set_text(_make_eq_string(mu, sigma, x1))
        title.set_text(
            rf'Spam email lengths — Gaussian fit: $\mu={mu:.1f}$, '
            rf'$\sigma={sigma:.1f}$, $x_1={x1:.1f}$ words'
        )

        fig.canvas.draw_idle()

    out = interactive_output(
        update,
        {'mu': mu_slider, 'sigma': sigma_slider, 'x1': x1_slider},
    )

    display(controls, out)


def _make_eq_string(mu, sigma, x1):
    """
    Build the annotated equation string with current values substituted in.
    Shows the symbolic form, numeric substitution, and final result.
    """
    sigma2  = sigma ** 2
    coeff   = 1.0 / np.sqrt(2 * np.pi * sigma2)
    exp_arg = -((x1 - mu) ** 2) / (2 * sigma2)
    result  = coeff * np.exp(exp_arg)

    return (
        r'$P(x_1 \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}'
        r'\exp\!\left(-\frac{(x_1 - \mu)^2}{2\sigma^2}\right)$'
        '\n\n'
        rf'$= \frac{{1}}{{\sqrt{{2\pi \cdot {sigma2:.2f}}}}}'
        rf'\exp\!\left(-\frac{{({x1:.1f} - {mu:.1f})^2}}{{{2*sigma2:.2f}}}\right)$'
        '\n\n'
        rf'$= {coeff:.5f} \times \exp({exp_arg:.4f})$'
        '\n\n'
        rf'$\approx \mathbf{{{result:.6f}}}$'
    )