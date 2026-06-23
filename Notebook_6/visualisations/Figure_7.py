"""
Figure 7 — Interactive Information Gain Explorer
=================================================
Visualises how information gain is computed for a split on a single feature,
using the fruit classification example (apple vs orange by diameter).

Two panels update live as the threshold theta is adjusted:

  Panel 1 (left)  — The 1D data strip showing the current split. Each side
                    displays the class counts and entropy for that child group.
                    The parent entropy is shown in the title.

  Panel 2 (right) — The full information gain calculation shown step by step
                    with current values substituted in — parent entropy,
                    left child entropy, right child entropy, weighted sum,
                    and final IG — all updating live.

Controls:
  - theta slider    — drag the split point through the data
  - Snap to best IG — jump to the threshold that maximises information gain
  - Reset           — return to the default threshold

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_7 import show
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

# ── Reproduce the same fruit dataset as Figure 4b ────────────────────────────
rng = np.random.default_rng(42)
N = 40
apple_diameters  = rng.normal(6.5, 0.8, N)
orange_diameters = rng.normal(8.5, 0.9, N)
X_fruit = np.concatenate([apple_diameters, orange_diameters])
y_fruit = np.array([0] * N + [1] * N)   # 0 = Apple, 1 = Orange

THETA_INIT = 7.5
THETA_MIN  = float(X_fruit.min()) - 0.3
THETA_MAX  = float(X_fruit.max()) + 0.3

COL_APPLE  = 'steelblue'
COL_ORANGE = 'tomato'
COL_MISS   = 'gold'


def entropy(labels):
    """
    Compute entropy H of a label array.
    Returns 0.0 for empty arrays or perfectly pure groups.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = np.bincount(labels, minlength=2)
    probs  = counts / n
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
    return float(-np.sum(terms))


def information_gain(theta):
    """
    Compute information gain for a split at theta.
    Returns a dict with all intermediate values needed for display.
    """
    n      = len(y_fruit)
    left   = y_fruit[X_fruit <  theta]
    right  = y_fruit[X_fruit >= theta]
    n_l, n_r = len(left), len(right)

    h_parent   = entropy(y_fruit)
    h_left     = entropy(left)
    h_right    = entropy(right)
    w_left     = n_l / n if n > 0 else 0
    w_right    = n_r / n if n > 0 else 0
    h_children = w_left * h_left + w_right * h_right
    ig         = h_parent - h_children

    return dict(
        n=n, n_l=n_l, n_r=n_r,
        h_parent=h_parent,
        h_left=h_left,     h_right=h_right,
        w_left=w_left,     w_right=w_right,
        h_children=h_children, ig=ig,
        n_apple_left=int(np.sum(left  == 0)),
        n_orange_left=int(np.sum(left == 1)),
        n_apple_right=int(np.sum(right == 0)),
        n_orange_right=int(np.sum(right== 1)),
    )


# ── Pre-compute IG across all candidate thresholds ───────────────────────────
sorted_vals = np.sort(np.unique(X_fruit))
candidates  = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
ig_vals     = np.array([information_gain(t)['ig'] for t in candidates])
opt_idx     = int(np.argmax(ig_vals))
OPT_THETA   = float(candidates[opt_idx])
OPT_IG      = float(ig_vals[opt_idx])


def _count_label(d, side):
    """Compact count + entropy label for one child group."""
    if side == 'left':
        return (f"n = {d['n_l']}\n"
                f"Apple: {d['n_apple_left']}  Orange: {d['n_orange_left']}\n"
                f"H = {d['h_left']:.4f} bits")
    else:
        return (f"n = {d['n_r']}\n"
                f"Apple: {d['n_apple_right']}  Orange: {d['n_orange_right']}\n"
                f"H = {d['h_right']:.4f} bits")


def _calc_text(d):
    """Step-by-step IG calculation string with current values."""
    return (
        "Step 1 — Parent entropy\n"
        f"  H(D) = {d['h_parent']:.4f} bits\n\n"

        "Step 2 — Left child entropy\n"
        f"  {d['n_apple_left']} apple, {d['n_orange_left']} orange  (n = {d['n_l']})\n"
        f"  H(left) = {d['h_left']:.4f} bits\n\n"

        "Step 3 — Right child entropy\n"
        f"  {d['n_apple_right']} apple, {d['n_orange_right']} orange  (n = {d['n_r']})\n"
        f"  H(right) = {d['h_right']:.4f} bits\n\n"

        "Step 4 — Weighted child entropy\n"
        f"  ({d['n_l']}/{d['n']}) x {d['h_left']:.4f}\n"
        f"  + ({d['n_r']}/{d['n']}) x {d['h_right']:.4f}\n"
        f"  = {d['h_children']:.4f} bits\n\n"

        "Step 5 — Information Gain\n"
        f"  IG = {d['h_parent']:.4f} - {d['h_children']:.4f}\n"
        f"  IG = {d['ig']:.4f} bits"
    )


def show():
    """
    Render Figure 7: Interactive Information Gain Explorer.
    """
    plt.close('Notebook6 Figure 7')
    fig, (ax_data, ax_calc) = plt.subplots(
        1, 2, num='Notebook6 Figure 7', figsize=(10, 6),
        gridspec_kw={'width_ratios': [1.5, 1]}
    )

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Data strip
    # ════════════════════════════════════════════════════════════════════════
    rng2     = np.random.default_rng(0)
    y_jitter = rng2.uniform(-0.18, 0.18, len(X_fruit))

    def get_colors(theta):
        preds = (X_fruit >= theta).astype(int)
        return [
            (COL_APPLE if yi == 0 else COL_ORANGE) if yi == pi else COL_MISS
            for yi, pi in zip(y_fruit, preds)
        ]

    d_init = information_gain(THETA_INIT)

    sc = ax_data.scatter(
        X_fruit, y_jitter,
        c=get_colors(THETA_INIT), s=55,
        edgecolors='k', lw=0.4, zorder=3,
    )

    thresh_line = ax_data.axvline(
        THETA_INIT, color='#333', lw=2.5, linestyle='--', zorder=4,
    )

    left_fill  = [ax_data.axvspan(THETA_MIN - 1, THETA_INIT,
                                  alpha=0.07, color=COL_APPLE,  zorder=1)]
    right_fill = [ax_data.axvspan(THETA_INIT, THETA_MAX + 1,
                                  alpha=0.07, color=COL_ORANGE, zorder=1)]

    # Child count/entropy labels either side of the threshold
    left_lbl = ax_data.text(
        THETA_INIT - 0.12, 0.48, _count_label(d_init, 'left'),
        ha='right', va='top', fontsize=8.5, color='#222', zorder=5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#ccc', alpha=0.92),
    )
    right_lbl = ax_data.text(
        THETA_INIT + 0.12, 0.48, _count_label(d_init, 'right'),
        ha='left', va='top', fontsize=8.5, color='#222', zorder=5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#ccc', alpha=0.92),
    )

    ax_data.set_xlim(THETA_MIN, THETA_MAX)
    ax_data.set_ylim(-0.55, 0.65)
    ax_data.set_yticks([])
    ax_data.set_xlabel('Diameter (cm)', fontsize=11)
    ax_data.grid(True, alpha=0.2, axis='x')
    ax_data.legend(handles=[
        mpatches.Patch(color=COL_APPLE,  label='Apple'),
        mpatches.Patch(color=COL_ORANGE, label='Orange'),
        mpatches.Patch(color=COL_MISS,   label='Misclassified'),
    ], fontsize=8.5, loc='upper left')

    data_title = ax_data.set_title(
        rf"H(D) = {d_init['h_parent']:.4f} bits  |  "
        rf"theta = {THETA_INIT:.2f} cm  |  "
        rf"IG = {d_init['ig']:.4f} bits",
        fontsize=10,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Step-by-step calculation
    # ════════════════════════════════════════════════════════════════════════
    ax_calc.axis('off')

    calc_text = ax_calc.text(
        0.05, 0.97, _calc_text(d_init),
        transform=ax_calc.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9ff',
                  edgecolor='#c8d0e8', alpha=1.0),
        zorder=3,
        linespacing=1.6,
    )

    ax_calc.set_title('Information Gain — step by step', fontsize=10)

    plt.suptitle(
        'Figure 7: Information Gain Explorer — fruit classification by diameter',
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
        description=f'Snap to best IG  (theta = {OPT_THETA:.2f} cm)',
        button_style='success',
        layout=widgets.Layout(width='270px'),
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
        d = information_gain(theta)

        # Panel 1
        sc.set_facecolors(get_colors(theta))
        thresh_line.set_xdata([theta, theta])

        left_fill[0].remove()
        right_fill[0].remove()
        left_fill[0]  = ax_data.axvspan(THETA_MIN - 1, theta,
                                        alpha=0.07, color=COL_APPLE,  zorder=1)
        right_fill[0] = ax_data.axvspan(theta, THETA_MAX + 1,
                                        alpha=0.07, color=COL_ORANGE, zorder=1)

        left_lbl.set_position((theta - 0.12, 0.48))
        left_lbl.set_text(_count_label(d, 'left'))
        right_lbl.set_position((theta + 0.12, 0.48))
        right_lbl.set_text(_count_label(d, 'right'))

        data_title.set_text(
            rf"H(D) = {d['h_parent']:.4f} bits  |  "
            rf"theta = {theta:.2f} cm  |  "
            rf"IG = {d['ig']:.4f} bits"
        )

        # Panel 2
        calc_text.set_text(_calc_text(d))

        fig.canvas.draw_idle()

    out = interactive_output(update, {'theta': theta_slider})
    display(controls, out)