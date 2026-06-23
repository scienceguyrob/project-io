"""
Figure 1c — Gaussian Misfit on Uniformly Distributed Data
==========================================================
Demonstrates the consequences of Naïve Bayes' normality assumption when
the underlying data is not Gaussian.

The feature shown is the hour of day an email was sent (0–23). Spam emails
are sent by automated bots operating globally across all time zones, so the
sending hour is approximately uniformly distributed — any hour is equally
likely. Fitting a Gaussian to this data produces a poor model that:

  - Overestimates the likelihood of emails sent around midday
  - Underestimates the likelihood of emails sent late at night or early morning
  - Creates phantom decision boundaries that do not reflect reality

Users can adjust mu and sigma manually to see that no Gaussian can fit
uniform data well, then evaluate the likelihood at a specific hour x_1 to
see how the misfit affects the computed value.

Controls:
  - mu slider       — shift the Gaussian mean
  - sigma slider    — adjust the Gaussian spread
  - x_1 slider      — the sending hour to evaluate the likelihood at
  - Snap to best fit — sets mu and sigma to the sample mean and std
  - Reset           — returns all sliders to their defaults

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_1c import show
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

# ── Generate uniformly distributed sending hours ─────────────────────────────
# Spam bots operate across all time zones, so any hour is equally likely.
# We generate 200 integer hours in [0, 23] from a uniform distribution.
rng = np.random.default_rng(7)
N_SAMPLES    = 200
SEND_HOURS   = rng.integers(0, 24, N_SAMPLES).astype(float)

# ── True uniform likelihood ───────────────────────────────────────────────────
# For a uniform distribution over [0, 23], every hour has equal probability.
# The probability density is 1 / (23 - 0) = 1/23 ≈ 0.0435.
UNIFORM_DENSITY = 1.0 / 24.0

# ── Best-fit Gaussian parameters (what the model actually learns) ─────────────
# These are the mean and std of the uniform data — the model fits these
# even though the data is not Gaussian. The mean will be ~11.5 (midday)
# and the std will be ~6.9 (the std of a uniform distribution over 0–23).
BEST_MU    = float(np.mean(SEND_HOURS))
BEST_SIGMA = float(np.std(SEND_HOURS))

# ── x-axis for curve and uniform line ────────────────────────────────────────
X_CURVE = np.linspace(-2, 26, 500)
X_UNIFORM = np.array([0, 23])


def gaussian_pdf(x, mu, sigma):
    """Evaluate the Gaussian PDF at x given mu and sigma."""
    coeff   = 1.0 / np.sqrt(2 * np.pi * sigma ** 2)
    exp_val = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coeff * exp_val


def show():
    """
    Render Figure 1c: Gaussian Misfit on Uniformly Distributed Data.

    Histogram of spam sending hours with a fitted Gaussian curve and the
    true uniform distribution shown for comparison. The misfit between
    the two illustrates the danger of assuming normality.
    """
    # ── Default parameter values — snap to best fit on load ──────────────────
    mu_init    = round(BEST_MU, 1)
    sigma_init = round(BEST_SIGMA, 1)
    x1_init    = 2.0    # 2am — a late-night sending hour

    # ── Build figure and axes once ───────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook6 Figure 1c', figsize=(10, 5))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ── Histogram (static) ───────────────────────────────────────────────────
    # density=True scales the bars to match the PDF scale.
    # Bins span 0–24 with width 1 so each bar represents one hour.
    ax.hist(SEND_HOURS, bins=np.arange(0, 25, 1), density=True,
            color='steelblue', alpha=0.35, edgecolor='white',
            linewidth=0.5, zorder=2, label='Spam sending hour (histogram)')

    # ── True uniform distribution (static dashed line) ───────────────────────
    # This is what the likelihood should look like — flat across all hours.
    ax.axhline(UNIFORM_DENSITY, color='seagreen', lw=2.0,
               linestyle='--', alpha=0.85, zorder=3,
               label=rf'True uniform density ($\frac{{1}}{{24}} \approx {UNIFORM_DENSITY:.4f}$)')

    # ── Fitted Gaussian curve ─────────────────────────────────────────────────
    y_curve_init = gaussian_pdf(X_CURVE, mu_init, sigma_init)
    (curve_line,) = ax.plot(X_CURVE, y_curve_init,
                            color='tomato', lw=2.5, zorder=4,
                            label=r'Fitted Gaussian $P(x_1 \mid Y=\mathrm{spam})$')

    # ── Vertical line for mu ──────────────────────────────────────────────────
    mu_line = ax.axvline(mu_init, color='tomato', lw=1.5,
                         linestyle='--', alpha=0.5, zorder=4,
                         label=r'Gaussian mean $\mu$')

    # ── Vertical line for x_1 ─────────────────────────────────────────────────
    x1_vline = ax.axvline(x1_init, color='darkorange', lw=2.0,
                          linestyle='-', alpha=0.9, zorder=5,
                          label=r'$x_1$ (query sending hour)')

    # ── Dot on Gaussian curve at x_1 ─────────────────────────────────────────
    y1_gauss_init = gaussian_pdf(x1_init, mu_init, sigma_init)
    (dot_gauss,) = ax.plot(x1_init, y1_gauss_init, 'o',
                           color='tomato', ms=10, zorder=7)

    # ── Dot on uniform line at x_1 ────────────────────────────────────────────
    # Shows what the true likelihood should be at this hour.
    (dot_uniform,) = ax.plot(x1_init, UNIFORM_DENSITY, 's',
                             color='seagreen', ms=10, zorder=7,
                             label=r'True likelihood at $x_1$')

    # ── Horizontal dashed line from Gaussian dot to y-axis ───────────────────
    (h_gauss,) = ax.plot([0, x1_init], [y1_gauss_init, y1_gauss_init],
                         color='tomato', lw=1.2, linestyle=':', zorder=4, alpha=0.8)

    # ── Horizontal dotted line from uniform dot to y-axis ────────────────────
    (h_uniform,) = ax.plot([0, x1_init], [UNIFORM_DENSITY, UNIFORM_DENSITY],
                           color='seagreen', lw=1.2, linestyle=':', zorder=4, alpha=0.6)

    # ── Vertical drop line from Gaussian dot to x-axis ───────────────────────
    (v_drop,) = ax.plot([x1_init, x1_init], [0, y1_gauss_init],
                        color='darkorange', lw=1.2, linestyle=':', zorder=4, alpha=0.5)

    # ── Shaded region showing the error between Gaussian and uniform ──────────
    # This makes the misfit visually obvious — it fills the gap between what
    # the Gaussian says and what the true uniform distribution says at x_1.
    err_container = [ax.fill_between(
        [x1_init - 0.15, x1_init + 0.15],
        [UNIFORM_DENSITY, UNIFORM_DENSITY],
        [y1_gauss_init, y1_gauss_init],
        alpha=0.25, color='gold', zorder=6,
    )]

    # ── Equation and error annotation ─────────────────────────────────────────
    eq_text = ax.text(
        0.98, 0.97,
        _make_eq_string(mu_init, sigma_init, x1_init),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#cccccc', alpha=0.95),
        zorder=8,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    title = ax.set_title(
        rf'Spam sending hour — Gaussian fit: $\mu={mu_init:.1f}$h, '
        rf'$\sigma={sigma_init:.1f}$h, $x_1={x1_init:.0f}$h',
        fontsize=11,
    )

    # ── Axes settings ─────────────────────────────────────────────────────────
    ax.set_xlim(-0.5, 23.5)
    peak_init = gaussian_pdf(mu_init, mu_init, sigma_init)
    ax.set_ylim(0, max(peak_init, UNIFORM_DENSITY) * 1.4)
    ax.set_xlabel(r'Hour of day email was sent  $x_1$  (0 = midnight, 12 = noon)', fontsize=11)
    ax.set_ylabel(r'Probability density  $P(x_1 \mid \mu, \sigma)$', fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8.5, loc='upper left')
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

    mu_slider,    mu_box    = make_slider(r'μ (mean hr)',   mu_init,    0.0,  23.0, 0.5)
    sigma_slider, sigma_box = make_slider(r'σ (std dev)',   sigma_init, 0.5,  12.0, 0.5)
    x1_slider,    x1_box    = make_slider(r'x₁ (hour)',    x1_init,    0.0,  23.0, 1.0)

    snap_btn = widgets.Button(
        description=f'Snap to best fit  (μ={BEST_MU:.1f}h, σ={BEST_SIGMA:.1f}h)',
        button_style='info',
        layout=widgets.Layout(width='300px'),
    )

    def on_snap(_):
        mu_slider.value    = round(BEST_MU, 1)
        sigma_slider.value = round(BEST_SIGMA, 1)

    snap_btn.on_click(on_snap)

    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_reset(_):
        mu_slider.value    = mu_init
        sigma_slider.value = sigma_init
        x1_slider.value    = x1_init

    reset_btn.on_click(on_reset)

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
        # Update Gaussian curve
        y_curve = gaussian_pdf(X_CURVE, mu, sigma)
        curve_line.set_ydata(y_curve)

        # Move mean marker
        mu_line.set_xdata([mu, mu])

        # Move x_1 marker and dots
        y1_gauss = gaussian_pdf(x1, mu, sigma)
        x1_vline.set_xdata([x1, x1])
        dot_gauss.set_xdata([x1])
        dot_gauss.set_ydata([y1_gauss])
        dot_uniform.set_xdata([x1])

        # Update drop and horizontal lines
        h_gauss.set_xdata([0, x1])
        h_gauss.set_ydata([y1_gauss, y1_gauss])
        h_uniform.set_xdata([0, x1])
        v_drop.set_xdata([x1, x1])
        v_drop.set_ydata([0, y1_gauss])

        # Update the error shading between Gaussian and uniform at x_1
        err_container[0].remove()
        lo_y = min(y1_gauss, UNIFORM_DENSITY)
        hi_y = max(y1_gauss, UNIFORM_DENSITY)
        err_container[0] = ax.fill_between(
            [x1 - 0.15, x1 + 0.15],
            [lo_y, lo_y],
            [hi_y, hi_y],
            alpha=0.25, color='gold', zorder=6,
        )

        # Rescale y-axis
        peak = gaussian_pdf(mu, mu, sigma)
        ax.set_ylim(0, max(peak, UNIFORM_DENSITY) * 1.4)

        # Update annotation and title
        eq_text.set_text(_make_eq_string(mu, sigma, x1))
        title.set_text(
            rf'Spam sending hour — Gaussian fit: $\mu={mu:.1f}$h, '
            rf'$\sigma={sigma:.1f}$h, $x_1={x1:.0f}$h'
        )

        fig.canvas.draw_idle()

    out = interactive_output(
        update,
        {'mu': mu_slider, 'sigma': sigma_slider, 'x1': x1_slider},
    )

    display(controls, out)


def _make_eq_string(mu, sigma, x1):
    """
    Build the annotated equation string showing both the Gaussian likelihood
    and the true uniform likelihood at x_1, so the user can compare them.
    """
    sigma2   = sigma ** 2
    coeff    = 1.0 / np.sqrt(2 * np.pi * sigma2)
    exp_arg  = -((x1 - mu) ** 2) / (2 * sigma2)
    gauss_ll = coeff * np.exp(exp_arg)
    error    = gauss_ll - UNIFORM_DENSITY
    sign     = '+' if error >= 0 else '-'

    return (
        r'$\mathbf{Gaussian\ assumed:}$'
        '\n'
        rf'$P(x_1={x1:.0f} \mid \mu,\sigma) = {coeff:.5f} \times \exp({exp_arg:.4f})$'
        '\n'
        rf'$\approx \mathbf{{{gauss_ll:.5f}}}$'
        '\n\n'
        r'$\mathbf{True\ uniform:}$'
        '\n'
        rf'$P(x_1={x1:.0f}) = 1/24 \approx \mathbf{{{UNIFORM_DENSITY:.5f}}}$'
        '\n\n'
        rf'$\mathbf{{Error:}}\ {sign}{abs(error):.5f}$'
    )