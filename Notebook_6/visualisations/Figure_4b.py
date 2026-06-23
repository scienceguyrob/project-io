"""
Figure 4b — Interactive Decision Stump: Data and Error Landscape
================================================================
Two interactive panels showing a decision stump applied to the fruit
classification problem (apple vs orange by diameter).

Panel 1 (left)  — The raw data: apple and orange diameters plotted as a
                  1-D strip of points. A vertical threshold line shows the
                  current split. Misclassified points are highlighted in gold.
                  Shaded regions show which class is predicted on each side.

Panel 2 (right) — The error landscape: misclassification count plotted across
                  all possible threshold values. A marker tracks the current
                  theta. The optimal threshold is marked in green.

Controls:
  - theta slider    — drag the decision boundary through the data
  - Snap to optimal — jump to the threshold that minimises errors
  - Reset           — return to the default threshold

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_4b import show
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
import matplotlib.patches as mpatches
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display

# ── Generate synthetic fruit data ────────────────────────────────────────────
# Apples tend to be smaller; oranges tend to be larger.
# Deliberate overlap means no single threshold achieves zero errors.
rng = np.random.default_rng(42)
N = 40

apple_diameters  = rng.normal(6.5, 0.8, N)
orange_diameters = rng.normal(8.5, 0.9, N)

# Label: 0 = Apple, 1 = Orange
X_fruit = np.concatenate([apple_diameters, orange_diameters])
y_fruit = np.array([0] * N + [1] * N)

# ── Pre-compute the error landscape ─────────────────────────────────────────
# Try every candidate threshold (midpoint between consecutive sorted values)
# and count misclassifications — this is exactly what the stump learner does.
sorted_vals = np.sort(np.unique(X_fruit))
candidates  = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0


def count_errors(theta):
    """
    Count misclassifications for a stump that predicts:
        Orange (1) if diameter >= theta
        Apple  (0) if diameter <  theta
    """
    preds = (X_fruit >= theta).astype(int)
    return int(np.sum(preds != y_fruit))


errors = np.array([count_errors(t) for t in candidates])

# ── Optimal threshold ────────────────────────────────────────────────────────
opt_idx   = int(np.argmin(errors))
OPT_THETA = float(candidates[opt_idx])
OPT_ERRS  = int(errors[opt_idx])

# ── Axis limits ──────────────────────────────────────────────────────────────
THETA_INIT = 7.5
THETA_MIN  = float(X_fruit.min()) - 0.3
THETA_MAX  = float(X_fruit.max()) + 0.3

# ── Colours ──────────────────────────────────────────────────────────────────
COL_APPLE  = 'steelblue'
COL_ORANGE = 'tomato'
COL_MISS   = 'gold'
COL_THRESH = '#333333'


def get_colors(theta):
    """
    Return a face colour for each point:
      correctly classified → class colour
      misclassified        → gold
    """
    preds = (X_fruit >= theta).astype(int)
    return [
        (COL_APPLE if yi == 0 else COL_ORANGE) if yi == pi else COL_MISS
        for yi, pi in zip(y_fruit, preds)
    ]


def show():
    """
    Render Figure 4b: Data strip and error landscape for the fruit stump.
    """
    plt.close('Notebook6 Figure 4b')
    fig, (ax_data, ax_err) = plt.subplots(
        1, 2, num='Notebook6 Figure 4b', figsize=(10, 5)
    )

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Data strip
    # ════════════════════════════════════════════════════════════════════════

    # Jitter y positions slightly so overlapping points are visible.
    rng2     = np.random.default_rng(0)
    y_jitter = rng2.uniform(-0.18, 0.18, len(X_fruit))

    init_cols = get_colors(THETA_INIT)

    # scatter returns a PathCollection — we update face colours via
    # set_facecolors() on each slider change rather than redrawing.
    sc = ax_data.scatter(
        X_fruit, y_jitter,
        c=init_cols, s=65,
        edgecolors='k', lw=0.4, zorder=3,
    )

    # Threshold line — mutated via set_xdata() on each update.
    thresh_line = ax_data.axvline(
        THETA_INIT, color=COL_THRESH, lw=2.5,
        linestyle='--', zorder=4,
        label=r'Threshold $\theta$',
    )

    # Shaded regions showing predicted class either side of the threshold.
    # Stored in lists so we can call .remove() and redraw them each update
    # (axvspan has no mutation API).
    left_fill  = [ax_data.axvspan(THETA_MIN - 1, THETA_INIT,
                                  alpha=0.07, color=COL_APPLE,  zorder=1)]
    right_fill = [ax_data.axvspan(THETA_INIT, THETA_MAX + 1,
                                  alpha=0.07, color=COL_ORANGE, zorder=1)]

    # Predict-class labels in the shaded regions
    apple_label  = ax_data.text(
        THETA_INIT - 0.5, 0.35, 'Predict:\nApple',
        ha='right', va='top', fontsize=8.5,
        color=COL_APPLE, fontweight='bold', zorder=5,
    )
    orange_label = ax_data.text(
        THETA_INIT + 0.05, 0.35, 'Predict:\nOrange',
        ha='left', va='top', fontsize=8.5,
        color=COL_ORANGE, fontweight='bold', zorder=5,
    )

    ax_data.set_xlim(THETA_MIN, THETA_MAX)
    ax_data.set_ylim(-0.5, 0.5)
    ax_data.set_yticks([])
    ax_data.set_xlabel('Diameter (cm)', fontsize=11)
    ax_data.grid(True, alpha=0.2, axis='x')

    ax_data.legend(handles=[
        mpatches.Patch(color=COL_APPLE,  label='Apple (correct)'),
        mpatches.Patch(color=COL_ORANGE, label='Orange (correct)'),
        mpatches.Patch(color=COL_MISS,   label='Misclassified'),
    ], fontsize=8.5, loc='upper left')

    n_init = count_errors(THETA_INIT)
    data_title = ax_data.set_title(
        rf'Fruit data  —  $\theta = {THETA_INIT:.2f}$ cm  |  '
        rf'errors: {n_init} / {len(X_fruit)}',
        fontsize=10,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Error landscape
    # ════════════════════════════════════════════════════════════════════════

    # Full error curve — static, drawn once.
    ax_err.plot(candidates, errors, color='#4a6fa5', lw=2.0, zorder=3,
                label='Misclassification count')

    # Optimal threshold marker — static green line.
    ax_err.axvline(OPT_THETA, color='seagreen', lw=1.5,
                   linestyle='--', alpha=0.8, zorder=2,
                   label=f'Optimal  θ = {OPT_THETA:.2f} cm  ({OPT_ERRS} errors)')

    # Moving dot tracking the current theta — mutated via set_xdata/set_ydata.
    (err_dot,) = ax_err.plot(
        THETA_INIT, count_errors(THETA_INIT),
        'o', color=COL_THRESH, ms=9, zorder=5,
        label=r'Current $\theta$',
    )

    # Vertical drop line from the dot to the x-axis — helps read the error count.
    (err_drop,) = ax_err.plot(
        [THETA_INIT, THETA_INIT], [0, count_errors(THETA_INIT)],
        color=COL_THRESH, lw=1.0, linestyle=':', zorder=4, alpha=0.6,
    )

    ax_err.set_xlabel('Threshold θ (cm)', fontsize=11)
    ax_err.set_ylabel('Misclassifications', fontsize=11)
    ax_err.set_ylim(0, max(errors) + 3)
    ax_err.legend(fontsize=8.5)
    ax_err.grid(True, alpha=0.2)

    err_title = ax_err.set_title(
        rf'Error landscape  —  current errors: {n_init}  |  '
        rf'minimum possible: {OPT_ERRS}',
        fontsize=10,
    )

    plt.suptitle(
        'Figure 4b: Decision Stump — fruit classification by diameter',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════

    theta_slider = widgets.FloatSlider(
        value=THETA_INIT, min=THETA_MIN, max=THETA_MAX, step=0.05,
        description='theta (cm)',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='420px'),
        readout=False,
    )
    theta_box = widgets.BoundedFloatText(
        value=THETA_INIT, min=THETA_MIN, max=THETA_MAX, step=0.05,
        layout=widgets.Layout(width='85px'),
    )
    widgets.jslink((theta_slider, 'value'), (theta_box, 'value'))

    snap_btn = widgets.Button(
        description=f'Snap to optimal  (theta = {OPT_THETA:.2f} cm)',
        button_style='success',
        layout=widgets.Layout(width='260px'),
    )
    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_snap(_):
        theta_slider.value = round(OPT_THETA, 2)

    def on_reset(_):
        theta_slider.value = THETA_INIT

    snap_btn.on_click(on_snap)
    reset_btn.on_click(on_reset)

    controls = widgets.HBox([
        theta_slider, theta_box,
        widgets.Label('  '),
        snap_btn,
        widgets.Label(' '),
        reset_btn,
    ])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # ════════════════════════════════════════════════════════════════════════
    def update(theta):
        n_err = count_errors(theta)

        # ── Panel 1 ──────────────────────────────────────────────────────────
        # Update point colours
        sc.set_facecolors(get_colors(theta))

        # Move threshold line
        thresh_line.set_xdata([theta, theta])

        # Remove and redraw shaded regions (axvspan has no mutation API)
        left_fill[0].remove()
        right_fill[0].remove()
        left_fill[0]  = ax_data.axvspan(THETA_MIN - 1, theta,
                                        alpha=0.07, color=COL_APPLE,  zorder=1)
        right_fill[0] = ax_data.axvspan(theta, THETA_MAX + 1,
                                        alpha=0.07, color=COL_ORANGE, zorder=1)

        # Move predict-class labels to sit either side of the new threshold
        apple_label.set_position((theta - 0.08, 0.35))
        apple_label.set_ha('right')
        orange_label.set_position((theta + 0.08, 0.35))

        data_title.set_text(
            rf'Fruit data  —  $\theta = {theta:.2f}$ cm  |  '
            rf'errors: {n_err} / {len(X_fruit)}'
        )

        # ── Panel 2 ──────────────────────────────────────────────────────────
        err_dot.set_xdata([theta])
        err_dot.set_ydata([n_err])
        err_drop.set_xdata([theta, theta])
        err_drop.set_ydata([0, n_err])

        err_title.set_text(
            rf'Error landscape  —  current errors: {n_err}  |  '
            rf'minimum possible: {OPT_ERRS}'
        )

        fig.canvas.draw_idle()

    out = interactive_output(update, {'theta': theta_slider})
    display(controls, out)