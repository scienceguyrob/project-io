"""
Figure 10 — Interactive Logit / Sigmoid Transformation Explorer
===============================================================

Three linked panels that show the relationship between the sigmoid,
the logit, and how a straight line in log-odds space becomes a curve
in probability space:

Left panel:   a straight line in log-odds (z) space — this is what the
              linear part of logistic regression fits. The user can
              adjust slope and intercept.

Middle panel: the same data transformed through the sigmoid — the straight
              line in log-odds space becomes the familiar S-shaped curve
              in probability space.

Right panel:  the logit transformation going the other way — starting with
              a probability and showing how the logit maps it back to a
              linear (log-odds) value.

The user controls the slope and intercept of the linear model, seeing
in real time how changes in the linear part affect the sigmoid curve.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_10 import show
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
    """Render the interactive Figure 10 logit/sigmoid transformation explorer."""

    plt.close('Notebook5 Figure 10')

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # ── Fixed ranges ──────────────────────────────────────────────────────────
    x_range  = np.linspace(-5, 5, 300)    # feature values (x axis for left panel)
    p_range  = np.linspace(0.001, 0.999, 300)  # probability values for right panel

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_SLOPE     = 1.0
    DEFAULT_INTERCEPT = 0.0

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3, num='Notebook5 Figure 10', figsize=(14, 5),
        gridspec_kw={'width_ratios': [1, 1, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    ax_logodds, ax_prob, ax_logit = axes

    # ── Left panel: log-odds space (linear) ───────────────────────────────────
    z_init = DEFAULT_SLOPE * x_range + DEFAULT_INTERCEPT
    logodds_line, = ax_logodds.plot(x_range, z_init, color='steelblue',
                                     linewidth=2.5)
    ax_logodds.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax_logodds.set_xlabel('Feature value  x', fontsize=10)
    ax_logodds.set_ylabel('Log-odds  z = β₀ + β₁x', fontsize=10)
    ax_logodds.set_title('Step 1: Linear model\nin log-odds space', fontsize=10)
    ax_logodds.set_xlim(-5.2, 5.2)
    ax_logodds.grid(True, alpha=0.2)

    # Annotation explaining log-odds
    ax_logodds.text(0.05, 0.95,
                    'z can be any real number\n(unbounded — suits a linear model)',
                    transform=ax_logodds.transAxes, fontsize=8,
                    va='top', color='steelblue',
                    bbox=dict(boxstyle='round', facecolor='#eef4ff',
                              edgecolor='steelblue', alpha=0.8))

    # Mutable dot tracking a specific x point through all three panels
    tracker_logodds = ax_logodds.scatter([], [], color='tomato', s=100,
                                          zorder=5, edgecolors='k', lw=0.8)
    vline_logodds   = ax_logodds.axvline(x=0, color='tomato', linewidth=1.2,
                                          linestyle='--', alpha=0.0)

    # ── Middle panel: probability space (sigmoid curve) ───────────────────────
    p_init = sigmoid(z_init)
    prob_line, = ax_prob.plot(x_range, p_init, color='seagreen', linewidth=2.5)
    ax_prob.axhline(0.5, color='red', linewidth=1.5, linestyle='--',
                    label='Threshold = 0.5')
    ax_prob.axhline(0,   color='gray', linewidth=0.8, linestyle=':')
    ax_prob.axhline(1,   color='gray', linewidth=0.8, linestyle=':')

    # Shade class regions
    ax_prob.fill_between(x_range, p_init, 0.5,
                          where=(p_init < 0.5), alpha=0.12, color='steelblue')
    ax_prob.fill_between(x_range, p_init, 0.5,
                          where=(p_init >= 0.5), alpha=0.12, color='tomato')

    ax_prob.set_xlabel('Feature value  x', fontsize=10)
    ax_prob.set_ylabel('P(Y=1 | x)  — probability', fontsize=10)
    ax_prob.set_title('Step 2: Apply sigmoid\nstraight line → S-curve', fontsize=10)
    ax_prob.set_xlim(-5.2, 5.2)
    ax_prob.set_ylim(-0.05, 1.1)
    ax_prob.legend(fontsize=8)
    ax_prob.grid(True, alpha=0.2)

    ax_prob.text(0.05, 0.95,
                 'Probability is now bounded\nbetween 0 and 1',
                 transform=ax_prob.transAxes, fontsize=8,
                 va='top', color='seagreen',
                 bbox=dict(boxstyle='round', facecolor='#efffef',
                           edgecolor='seagreen', alpha=0.8))

    tracker_prob = ax_prob.scatter([], [], color='tomato', s=100,
                                    zorder=5, edgecolors='k', lw=0.8)
    hline_prob,  = ax_prob.plot([], [], color='tomato', linewidth=1.2,
                                 linestyle='--', alpha=0.6)
    vline_prob   = ax_prob.axvline(x=0, color='tomato', linewidth=1.2,
                                    linestyle='--', alpha=0.0)

    # ── Right panel: logit (inverse sigmoid) ──────────────────────────────────
    logit_vals = logit(p_range)
    ax_logit.plot(p_range, logit_vals, color='darkorange', linewidth=2.5,
                  label='logit(p) = log(p / (1-p))')
    ax_logit.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax_logit.axvline(0.5, color='red', linewidth=1.5, linestyle='--',
                     label='p = 0.5  →  z = 0')

    tracker_logit = ax_logit.scatter([], [], color='tomato', s=100,
                                      zorder=5, edgecolors='k', lw=0.8)
    hline_logit,  = ax_logit.plot([], [], color='tomato', linewidth=1.2,
                                   linestyle='--', alpha=0.6)
    vline_logit,  = ax_logit.plot([], [], color='tomato', linewidth=1.2,
                                   linestyle='--', alpha=0.6)

    ax_logit.set_xlabel('Probability  P(Y=1)', fontsize=10)
    ax_logit.set_ylabel('Log-odds  z  (logit)', fontsize=10)
    ax_logit.set_title('Step 3: Logit (inverse)\nprobability → log-odds', fontsize=10)
    ax_logit.set_xlim(-0.05, 1.05)
    ax_logit.set_ylim(-6, 6)
    ax_logit.legend(fontsize=8)
    ax_logit.grid(True, alpha=0.2)

    ax_logit.text(0.05, 0.95,
                  'The logit maps probability\nback to log-odds (undoes sigmoid)',
                  transform=ax_logit.transAxes, fontsize=8,
                  va='top', color='darkorange',
                  bbox=dict(boxstyle='round', facecolor='#fff8ee',
                            edgecolor='darkorange', alpha=0.8))

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    # ── Readout ───────────────────────────────────────────────────────────────
    readout = widgets.Output()

    # ── Update function ───────────────────────────────────────────────────────
    def update(slope, intercept, x_track):
        z_vals = slope * x_range + intercept
        p_vals = sigmoid(z_vals)

        # Update left panel line
        logodds_line.set_ydata(z_vals)
        ax_logodds.set_ylim(min(-6, z_vals.min() - 0.5),
                             max(6, z_vals.max() + 0.5))

        # Update middle panel sigmoid curve and shading
        prob_line.set_ydata(p_vals)

        # Redraw shading — remove old collections and add new
        while len(ax_prob.collections) > 0:
            ax_prob.collections[0].remove()
        ax_prob.fill_between(x_range, p_vals, 0.5,
                              where=(p_vals < 0.5), alpha=0.12, color='steelblue')
        ax_prob.fill_between(x_range, p_vals, 0.5,
                              where=(p_vals >= 0.5), alpha=0.12, color='tomato')

        # Tracking point through all three panels
        z_track = slope * x_track + intercept
        p_track = sigmoid(z_track)

        # Left: dot on log-odds line
        tracker_logodds.set_offsets([[x_track, z_track]])
        vline_logodds.set_xdata([x_track, x_track])
        vline_logodds.set_alpha(0.5)

        # Middle: dot on sigmoid curve
        tracker_prob.set_offsets([[x_track, p_track]])
        hline_prob.set_data([-5.2, x_track], [p_track, p_track])
        vline_prob.set_xdata([x_track, x_track])
        vline_prob.set_alpha(0.5)

        # Right: dot on logit curve
        tracker_logit.set_offsets([[p_track, z_track]])
        hline_logit.set_data([-0.05, p_track], [z_track, z_track])
        vline_logit.set_data([p_track, p_track], [-6, z_track])

        pred = 1 if p_track > 0.5 else 0

        fig.suptitle(
            f'Logit ↔ Sigmoid transformation  |  '
            f'Model: z = {intercept:.2f} + {slope:.2f}·x\n'
            f'At x = {x_track:.1f}:  z = {z_track:.3f}  →  '
            f'P(Y=1) = {p_track:.3f}  →  predict class {pred}',
            fontsize=10,
        )

        readout.clear_output(wait=True)
        with readout:
            odds = p_track / (1 - p_track) if p_track < 1 else float('inf')
            print(f'At x = {x_track:.2f}:')
            print(f'  Step 1 — log-odds:    z = {intercept:.2f} + {slope:.2f} × {x_track:.2f} = {z_track:.4f}')
            print(f'  Step 2 — sigmoid:     P(Y=1) = 1 / (1 + e^{-z_track:.3f}) = {p_track:.4f}')
            print(f'  Odds:                 {p_track:.3f} / {1-p_track:.3f} = {odds:.3f}')
            print(f'  Step 3 — logit check: log({p_track:.3f}/{1-p_track:.3f}) = {z_track:.4f}  ✓')
            print(f'  Prediction:           class {pred}  '
                  f'({"confident" if abs(p_track - 0.5) > 0.3 else "uncertain"})')

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    slope_s = FloatSlider(value=DEFAULT_SLOPE, min=-3.0, max=3.0, step=0.1,
                           description='Slope (β₁)',
                           style={'description_width': '100px'},
                           layout=widgets.Layout(width='340px'),
                           continuous_update=True)
    intercept_s = FloatSlider(value=DEFAULT_INTERCEPT, min=-4.0, max=4.0, step=0.1,
                               description='Intercept (β₀)',
                               style={'description_width': '100px'},
                               layout=widgets.Layout(width='340px'),
                               continuous_update=True)
    x_track_s = FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.1,
                             description='Track x =',
                             style={'description_width': '100px'},
                             layout=widgets.Layout(width='340px'),
                             continuous_update=True)

    slope_b     = widgets.BoundedFloatText(value=DEFAULT_SLOPE,     min=-3.0, max=3.0, step=0.1,
                                            description='', layout=widgets.Layout(width='80px'))
    intercept_b = widgets.BoundedFloatText(value=DEFAULT_INTERCEPT, min=-4.0, max=4.0, step=0.1,
                                            description='', layout=widgets.Layout(width='80px'))
    x_track_b   = widgets.BoundedFloatText(value=0.0, min=-5.0, max=5.0, step=0.1,
                                            description='', layout=widgets.Layout(width='80px'))

    for s, b in [(slope_s, slope_b), (intercept_s, intercept_b), (x_track_s, x_track_b)]:
        widgets.jslink((s, 'value'), (b, 'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        slope_s.value     = DEFAULT_SLOPE
        intercept_s.value = DEFAULT_INTERCEPT
        x_track_s.value   = 0.0

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Adjust the linear model and track a point through all three transformations:</b>'),
        HBox([slope_s,     slope_b]),
        HBox([intercept_s, intercept_b]),
        HBox([x_track_s,   x_track_b]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        'slope':     slope_s,
        'intercept': intercept_s,
        'x_track':   x_track_s,
    })

    display(controls, out)
    display(readout)