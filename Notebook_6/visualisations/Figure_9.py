"""
Figure 9 — Gini Impurity on a Continuous Feature
=================================================
Identical in structure to Figure 8, but replaces information gain (entropy)
with Gini impurity as the split quality measure.

Shows how Gini impurity reduction is evaluated for a continuous feature by
visualising the class distributions (apple vs orange diameter) and letting
users drag a candidate threshold through them.

The step-by-step calculation panel shows:
  - Parent Gini impurity G(D)
  - Left and right child Gini impurities
  - Weighted child Gini impurity
  - Gini reduction (equivalent to information gain but using Gini)

Controls:
  - theta slider         — drag the split point through the distributions
  - Snap to best Gini    — jump to the candidate with the highest Gini reduction
  - Reset                — return to the default threshold

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_9 import show
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
from scipy.stats import gaussian_kde

# ── Reproduce the same fruit dataset ─────────────────────────────────────────
rng = np.random.default_rng(42)
N = 40
apple_diameters  = rng.normal(6.5, 0.8, N)
orange_diameters = rng.normal(8.5, 0.9, N)
X_fruit = np.concatenate([apple_diameters, orange_diameters])
y_fruit = np.array([0] * N + [1] * N)   # 0 = Apple, 1 = Orange

THETA_INIT = 7.5
THETA_MIN  = float(X_fruit.min()) - 0.5
THETA_MAX  = float(X_fruit.max()) + 0.5

COL_APPLE  = 'steelblue'
COL_ORANGE = 'tomato'


# ── Gini impurity functions ───────────────────────────────────────────────────

def gini(labels):
    """
    Compute Gini impurity G of a label array.

    G = 1 - sum(p_k^2)

    Returns 0.0 for empty arrays or perfectly pure groups.
    A group split 50/50 returns 0.5 — the maximum for two classes.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = np.bincount(labels, minlength=2)
    probs  = counts / n
    return float(1.0 - np.sum(probs ** 2))


def gini_reduction(theta):
    """
    Compute Gini reduction for a split at theta, returning all intermediate
    values needed for the step-by-step display panel.
    """
    n      = len(y_fruit)
    left   = y_fruit[X_fruit <  theta]
    right  = y_fruit[X_fruit >= theta]
    n_l, n_r = len(left), len(right)

    g_parent   = gini(y_fruit)
    g_left     = gini(left)
    g_right    = gini(right)
    w_left     = n_l / n if n > 0 else 0
    w_right    = n_r / n if n > 0 else 0
    g_children = w_left * g_left + w_right * g_right
    reduction  = g_parent - g_children

    return dict(
        n=n, n_l=n_l, n_r=n_r,
        g_parent=g_parent,
        g_left=g_left,     g_right=g_right,
        w_left=w_left,     w_right=w_right,
        g_children=g_children, reduction=reduction,
        n_apple_left=int(np.sum(left  == 0)),
        n_orange_left=int(np.sum(left == 1)),
        n_apple_right=int(np.sum(right == 0)),
        n_orange_right=int(np.sum(right== 1)),
    )


# ── Pre-compute Gini reduction across all candidate thresholds ────────────────
sorted_vals = np.sort(np.unique(X_fruit))
candidates  = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
gini_vals   = np.array([gini_reduction(t)['reduction'] for t in candidates])
opt_idx     = int(np.argmax(gini_vals))
OPT_THETA   = float(candidates[opt_idx])
OPT_GINI    = float(gini_vals[opt_idx])

# ── KDE curves for smooth distribution display ────────────────────────────────
X_PLOT     = np.linspace(THETA_MIN, THETA_MAX, 400)
kde_apple  = gaussian_kde(apple_diameters,  bw_method=0.4)
kde_orange = gaussian_kde(orange_diameters, bw_method=0.4)
Y_APPLE    = kde_apple(X_PLOT)
Y_ORANGE   = kde_orange(X_PLOT)
Y_MAX      = max(Y_APPLE.max(), Y_ORANGE.max()) * 1.25


def _calc_text(d):
    """Step-by-step Gini reduction calculation as plain monospace text."""
    # Show the Gini formula for each group so users can see the arithmetic.
    # For the parent: G = 1 - (p_apple^2 + p_orange^2)
    n = d['n']
    p_apple_parent  = (N) / n          # 40 apples out of 80
    p_orange_parent = (N) / n          # 40 oranges out of 80

    p_apple_l  = d['n_apple_left']  / d['n_l'] if d['n_l'] > 0 else 0
    p_orange_l = d['n_orange_left'] / d['n_l'] if d['n_l'] > 0 else 0
    p_apple_r  = d['n_apple_right'] / d['n_r'] if d['n_r'] > 0 else 0
    p_orange_r = d['n_orange_right']/ d['n_r'] if d['n_r'] > 0 else 0

    return (
        "Step 1 — Parent Gini impurity\n"
        f"  G = 1 - (p_apple^2 + p_orange^2)\n"
        f"  G = 1 - ({p_apple_parent:.3f}^2 + {p_orange_parent:.3f}^2)\n"
        f"  G(D) = {d['g_parent']:.4f}\n\n"

        "Step 2 — Left child  (diameter < theta)\n"
        f"  Apple: {d['n_apple_left']}  Orange: {d['n_orange_left']}"
        f"  (n={d['n_l']})\n"
        f"  G = 1 - ({p_apple_l:.3f}^2 + {p_orange_l:.3f}^2)\n"
        f"  G(left) = {d['g_left']:.4f}\n\n"

        "Step 3 — Right child  (diameter >= theta)\n"
        f"  Apple: {d['n_apple_right']}  Orange: {d['n_orange_right']}"
        f"  (n={d['n_r']})\n"
        f"  G = 1 - ({p_apple_r:.3f}^2 + {p_orange_r:.3f}^2)\n"
        f"  G(right) = {d['g_right']:.4f}\n\n"

        "Step 4 — Weighted child Gini\n"
        f"  ({d['n_l']}/{d['n']}) x {d['g_left']:.4f}\n"
        f"  + ({d['n_r']}/{d['n']}) x {d['g_right']:.4f}\n"
        f"  = {d['g_children']:.4f}\n\n"

        "Step 5 — Gini Reduction\n"
        f"  Reduction = {d['g_parent']:.4f} - {d['g_children']:.4f}\n"
        f"  Reduction = {d['reduction']:.4f}"
    )


def show():
    """
    Render Figure 9: Gini impurity on a continuous feature with distributions.
    """
    plt.close('Notebook6 Figure 9')
    fig, (ax_dist, ax_calc) = plt.subplots(
        1, 2, num='Notebook6 Figure 9', figsize=(11, 6),
        gridspec_kw={'width_ratios': [1.6, 1]}
    )

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Distribution plot
    # ════════════════════════════════════════════════════════════════════════

    # ── KDE curves (static) ───────────────────────────────────────────────────
    ax_dist.plot(X_PLOT, Y_APPLE,  color=COL_APPLE,  lw=2.5, zorder=4,
                 label='Apple distribution')
    ax_dist.plot(X_PLOT, Y_ORANGE, color=COL_ORANGE, lw=2.5, zorder=4,
                 label='Orange distribution')

    # ── Histogram bars behind the KDE curves ─────────────────────────────────
    bins = np.linspace(THETA_MIN, THETA_MAX, 20)
    ax_dist.hist(apple_diameters,  bins=bins, density=True,
                 color=COL_APPLE,  alpha=0.18, zorder=2)
    ax_dist.hist(orange_diameters, bins=bins, density=True,
                 color=COL_ORANGE, alpha=0.18, zorder=2)

    # ── Candidate threshold tick marks on the x-axis ─────────────────────────
    for c in candidates:
        ax_dist.axvline(c, ymin=0, ymax=0.025,
                        color='#aaa', lw=0.8, zorder=2)

    ax_dist.text(candidates[5], -Y_MAX * 0.055,
                 'candidate\nthresholds',
                 ha='center', va='top', fontsize=7.5,
                 color='#999', style='italic')

    # ── Threshold line ────────────────────────────────────────────────────────
    thresh_line = ax_dist.axvline(
        THETA_INIT, color='#222', lw=2.5, linestyle='--', zorder=5,
        label=r'Current threshold $\theta$',
    )

    # ── Shaded left/right regions ─────────────────────────────────────────────
    def make_fills(theta):
        left_mask  = X_PLOT <= theta
        right_mask = X_PLOT >= theta
        fl = ax_dist.fill_between(X_PLOT, 0, Y_APPLE,
                                  where=left_mask,
                                  alpha=0.22, color=COL_APPLE, zorder=3)
        fr = ax_dist.fill_between(X_PLOT, 0, Y_ORANGE,
                                  where=right_mask,
                                  alpha=0.22, color=COL_ORANGE, zorder=3)
        return [fl], [fr]

    left_fill, right_fill = make_fills(THETA_INIT)

    # ── Predict-class labels ──────────────────────────────────────────────────
    left_pred_txt = ax_dist.text(
        THETA_INIT - 0.15, Y_MAX * 0.88,
        'Predict:\nApple',
        ha='right', va='top', fontsize=9,
        color=COL_APPLE, fontweight='bold', zorder=6,
    )
    right_pred_txt = ax_dist.text(
        THETA_INIT + 0.15, Y_MAX * 0.88,
        'Predict:\nOrange',
        ha='left', va='top', fontsize=9,
        color=COL_ORANGE, fontweight='bold', zorder=6,
    )

    # ── Axes settings ─────────────────────────────────────────────────────────
    ax_dist.set_xlim(THETA_MIN, THETA_MAX)
    ax_dist.set_ylim(-Y_MAX * 0.08, Y_MAX)
    ax_dist.set_xlabel('Diameter (cm)', fontsize=11)
    ax_dist.set_ylabel('Probability density', fontsize=11)
    ax_dist.legend(fontsize=9, loc='upper left')
    ax_dist.grid(True, alpha=0.15)

    d_init = gini_reduction(THETA_INIT)
    dist_title = ax_dist.set_title(
        rf"theta = {THETA_INIT:.2f} cm  |  "
        rf"Gini reduction = {d_init['reduction']:.4f}  |  "
        rf"Best possible = {OPT_GINI:.4f}  (theta = {OPT_THETA:.2f} cm)",
        fontsize=9.5,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Step-by-step Gini calculation
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

    ax_calc.set_title('Gini Impurity — step by step', fontsize=10)

    plt.suptitle(
        'Figure 9: Gini Impurity on a continuous feature — '
        'evaluating candidate thresholds on the fruit diameter distribution',
        fontsize=10,
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
        description=f'Snap to best Gini  (theta = {OPT_THETA:.2f} cm)',
        button_style='success',
        layout=widgets.Layout(width='280px'),
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
        d = gini_reduction(theta)

        thresh_line.set_xdata([theta, theta])

        left_fill[0].remove()
        right_fill[0].remove()
        new_l, new_r = make_fills(theta)
        left_fill[0]  = new_l[0]
        right_fill[0] = new_r[0]

        left_pred_txt.set_position((theta - 0.15, Y_MAX * 0.88))
        right_pred_txt.set_position((theta + 0.15, Y_MAX * 0.88))

        dist_title.set_text(
            rf"theta = {theta:.2f} cm  |  "
            rf"Gini reduction = {d['reduction']:.4f}  |  "
            rf"Best possible = {OPT_GINI:.4f}  (theta = {OPT_THETA:.2f} cm)"
        )
        calc_text.set_text(_calc_text(d))

        fig.canvas.draw_idle()

    out = interactive_output(update, {'theta': theta_slider})
    display(controls, out)