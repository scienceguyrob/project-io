"""
Figure 8 — Information Gain on a Continuous Feature
====================================================
Shows how information gain is evaluated for a continuous feature by
visualising the class distributions (apple vs orange diameter) and
letting users drag a candidate threshold through them.

The plot makes clear what the algorithm is actually doing:
  - The overlapping histograms show the distribution of each class
  - Candidate threshold midpoints are marked on the x-axis as ticks
  - The vertical threshold line divides the distributions into two groups
  - Shaded regions show which class dominates on each side
  - The IG calculation panel updates live at each candidate position

Controls:
  - theta slider    — drag the split point through the distributions
  - Snap to best IG — jump to the candidate with the highest IG
  - Reset           — return to the default threshold

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_8 import show
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


# ── Entropy and IG ────────────────────────────────────────────────────────────
def entropy(labels):
    """Compute entropy H of a label array. Returns 0 for empty or pure groups."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = np.bincount(labels, minlength=2)
    probs  = counts / n
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
    return float(-np.sum(terms))


def information_gain(theta):
    """Compute IG for a split at theta, returning all intermediate values."""
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
        h_left=h_left,    h_right=h_right,
        w_left=w_left,    w_right=w_right,
        h_children=h_children, ig=ig,
        n_apple_left=int(np.sum(left  == 0)),
        n_orange_left=int(np.sum(left == 1)),
        n_apple_right=int(np.sum(right == 0)),
        n_orange_right=int(np.sum(right== 1)),
    )


# ── Pre-compute candidate thresholds and their IG values ─────────────────────
sorted_vals = np.sort(np.unique(X_fruit))
candidates  = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
ig_vals     = np.array([information_gain(t)['ig'] for t in candidates])
opt_idx     = int(np.argmax(ig_vals))
OPT_THETA   = float(candidates[opt_idx])
OPT_IG      = float(ig_vals[opt_idx])

# ── KDE curves for smooth distribution display ────────────────────────────────
X_PLOT  = np.linspace(THETA_MIN, THETA_MAX, 400)
kde_apple  = gaussian_kde(apple_diameters,  bw_method=0.4)
kde_orange = gaussian_kde(orange_diameters, bw_method=0.4)
Y_APPLE  = kde_apple(X_PLOT)
Y_ORANGE = kde_orange(X_PLOT)
Y_MAX    = max(Y_APPLE.max(), Y_ORANGE.max()) * 1.25


def _calc_text(d):
    """Step-by-step IG calculation as plain monospace text."""
    return (
        "Step 1 — Parent entropy\n"
        f"  H(D) = {d['h_parent']:.4f} bits\n\n"

        "Step 2 — Left child  (diameter < theta)\n"
        f"  Apple: {d['n_apple_left']}  Orange: {d['n_orange_left']}"
        f"  (n={d['n_l']})\n"
        f"  H(left) = {d['h_left']:.4f} bits\n\n"

        "Step 3 — Right child  (diameter >= theta)\n"
        f"  Apple: {d['n_apple_right']}  Orange: {d['n_orange_right']}"
        f"  (n={d['n_r']})\n"
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
    Render Figure 8: IG on a continuous feature with overlapping distributions.
    """
    plt.close('Notebook6 Figure 8')
    fig, (ax_dist, ax_calc) = plt.subplots(
        1, 2, num='Notebook6 Figure 8', figsize=(10, 6),
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
    # These show the user exactly which values the algorithm evaluates.
    # We draw them as small vertical lines just below the x-axis.
    for c in candidates:
        ax_dist.axvline(c, ymin=0, ymax=0.025,
                        color='#aaa', lw=0.8, zorder=2)

    # One labelled candidate tick to explain what they are
    ax_dist.text(candidates[5], -Y_MAX * 0.055,
                 'candidate\nthresholds',
                 ha='center', va='top', fontsize=7.5,
                 color='#999', style='italic')

    # ── Threshold line ────────────────────────────────────────────────────────
    thresh_line = ax_dist.axvline(
        THETA_INIT, color='#222', lw=2.5, linestyle='--', zorder=5,
        label=r'Current threshold $\theta$',
    )

    # ── Shaded left/right regions under each KDE curve ───────────────────────
    # Left of threshold — Apple region (blue shading)
    # Right of threshold — Orange region (red shading)
    # These are fill_between objects stored in lists for .remove() on update.
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

    d_init = information_gain(THETA_INIT)
    dist_title = ax_dist.set_title(
        rf"theta = {THETA_INIT:.2f} cm  |  "
        rf"IG = {d_init['ig']:.4f} bits  |  "
        rf"Best possible IG = {OPT_IG:.4f} bits  (theta = {OPT_THETA:.2f} cm)",
        fontsize=9.5,
    )

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Step-by-step IG calculation
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
        'Figure 8: IG on a continuous feature — '
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

        # Move threshold line
        thresh_line.set_xdata([theta, theta])

        # Remove and redraw shaded regions
        left_fill[0].remove()
        right_fill[0].remove()
        new_l, new_r = make_fills(theta)
        left_fill[0]  = new_l[0]
        right_fill[0] = new_r[0]

        # Move predict labels
        left_pred_txt.set_position((theta - 0.15, Y_MAX * 0.88))
        right_pred_txt.set_position((theta + 0.15, Y_MAX * 0.88))

        # Update title and calc panel
        dist_title.set_text(
            rf"theta = {theta:.2f} cm  |  "
            rf"IG = {d['ig']:.4f} bits  |  "
            rf"Best possible IG = {OPT_IG:.4f} bits  (theta = {OPT_THETA:.2f} cm)"
        )
        calc_text.set_text(_calc_text(d))

        fig.canvas.draw_idle()

    out = interactive_output(update, {'theta': theta_slider})
    display(controls, out)