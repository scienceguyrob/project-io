"""
Figure 11 — Interactive WCSS Explorer: Drag the Centroid
=========================================================
Demonstrates what WCSS is measuring at the most basic level: the sum of
squared distances from every point in a cluster to its centroid.

A single centroid is draggable. As it moves, the spoke lines from every
data point update in real time, and the WCSS readout changes. The learner
can discover for themselves that WCSS is minimised when the centroid sits
at the mean of the data — and that any other position increases it.

The annotation panel on the right shows the current WCSS alongside the
WCSS at the true mean, so the learner can see how far from optimal the
current centroid position is.

Interaction
-----------
Click and drag the centroid star to any position. The spokes and WCSS
update live.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_11 import show
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
from sklearn.datasets import make_blobs


# ── Dataset ───────────────────────────────────────────────────────────────────
# A single compact cluster so the WCSS concept is unambiguous.
# The true mean is the theoretical minimum — the learner discovers this
# by dragging the centroid around.
X_data, _ = make_blobs(n_samples=80, centers=1, cluster_std=1.2, random_state=7)

# Pre-compute the true mean and the minimum achievable WCSS
TRUE_MEAN = X_data.mean(axis=0)
WCSS_MIN  = float(np.sum((X_data - TRUE_MEAN) ** 2))

# ── Initial centroid position — offset from the mean so the learner
# immediately sees a non-optimal state and has somewhere to drag toward
CENTROID_INIT = TRUE_MEAN + np.array([2.5, 2.0])

HIT_RADIUS = 0.35

# ── Axis limits ───────────────────────────────────────────────────────────────
PAD  = 1.5
XLIM = (X_data[:, 0].min() - PAD, X_data[:, 0].max() + PAD)
YLIM = (X_data[:, 1].min() - PAD, X_data[:, 1].max() + PAD)


def _wcss(centroid):
    """Sum of squared distances from every data point to the centroid."""
    return float(np.sum((X_data - centroid) ** 2))


def _annotation(centroid):
    """Build the annotation string for the right-hand panel."""
    w     = _wcss(centroid)
    delta = w - WCSS_MIN
    pct   = 100.0 * delta / WCSS_MIN if WCSS_MIN > 0 else 0.0

    return (
        "Within-Cluster Sum of Squares\n"
        "─────────────────────────────────\n\n"
        r"$\mathrm{WCSS} = \sum_{i=1}^{n} \|\mathbf{x}_i - \boldsymbol{\mu}\|^2$"
        "\n\n"
        f"Current centroid:\n"
        f"  $({centroid[0]:.2f},\\ {centroid[1]:.2f})$\n\n"
        f"Current WCSS:\n"
        f"  ${w:.2f}$\n\n"
        f"Minimum possible WCSS:\n"
        f"  ${WCSS_MIN:.2f}$\n"
        f"  (at the true mean)\n\n"
        f"Excess above minimum:\n"
        f"  ${delta:.2f}$  (+{pct:.1f}%)\n\n"
        "─────────────────────────────────\n\n"
        "The centroid that minimises WCSS\n"
        "is always the mean of the data.\n"
        "Drag toward the dense centre\n"
        "to see WCSS fall."
    )


def show():
    """Render Figure 11: draggable centroid WCSS explorer."""
    plt.close('Notebook8 Figure 11')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 11',
        figsize=(10, 6.5),
        gridspec_kw={'width_ratios': [1.4, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    centroid  = [CENTROID_INIT.copy()]
    dragging  = [False]

    # ── Static elements ───────────────────────────────────────────────────────

    # Data points — never move
    ax_plot.scatter(
        X_data[:, 0], X_data[:, 1],
        s=35, color='lightsteelblue', edgecolors='k',
        lw=0.3, alpha=0.9, zorder=2, label='Data points',
    )

    # True mean marker — fixed reference so the learner knows where to aim
    ax_plot.scatter(
        *TRUE_MEAN, marker='+', s=200, color='seagreen',
        lw=2.5, zorder=5, label='True mean (WCSS minimum)',
    )

    # ── Dynamic elements ──────────────────────────────────────────────────────

    # Spoke lines from each data point to the centroid
    spoke_lines = []
    for pt in X_data:
        line, = ax_plot.plot(
            [pt[0], centroid[0][0]],
            [pt[1], centroid[0][1]],
            color='tomato', lw=0.6, alpha=0.4, zorder=1,
        )
        spoke_lines.append((pt, line))

    # Draggable centroid star
    scat_centroid = ax_plot.scatter(
        *centroid[0], marker='*', s=400,
        color='tomato', edgecolors='black',
        lw=0.8, zorder=6, label='Centroid (drag me)',
    )

    plot_title = ax_plot.set_title(
        f'WCSS = {_wcss(centroid[0]):.2f}   '
        f'(minimum = {WCSS_MIN:.2f} at true mean)',
        fontsize=10,
    )

    ax_plot.set_xlim(*XLIM)
    ax_plot.set_ylim(*YLIM)
    ax_plot.set_xlabel('Feature 1', fontsize=10)
    ax_plot.set_ylabel('Feature 2', fontsize=10)
    ax_plot.grid(True, alpha=0.2)
    ax_plot.legend(fontsize=9, loc='upper left',
                   framealpha=1.0, edgecolor='#cccccc')

    # ── Annotation panel ──────────────────────────────────────────────────────
    ax_ann.set_axis_off()
    ann_text = ax_ann.text(
        0.05, 0.97,
        _annotation(centroid[0]),
        transform=ax_ann.transAxes,
        fontsize=9.5, va='top', ha='left',
        linespacing=1.7,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 11: WCSS — drag the centroid to see how position '
        'affects the within-cluster sum of squares',
        fontsize=11,
    )
    plt.subplots_adjust(wspace=0.08, top=0.88)

    # ── Drag interaction ──────────────────────────────────────────────────────
    def _on_press(event):
        if event.inaxes is not ax_plot:
            return
        cursor = np.array([event.xdata, event.ydata])
        if np.linalg.norm(cursor - centroid[0]) < HIT_RADIUS:
            dragging[0] = True

    def _on_release(event):
        dragging[0] = False

    def _on_motion(event):
        if not dragging[0] or event.inaxes is not ax_plot:
            return

        x = float(np.clip(event.xdata, XLIM[0], XLIM[1]))
        y = float(np.clip(event.ydata, YLIM[0], YLIM[1]))
        centroid[0] = np.array([x, y])

        # Update centroid marker
        scat_centroid.set_offsets([centroid[0]])

        # Update all spoke lines
        for pt, line in spoke_lines:
            line.set_xdata([pt[0], centroid[0][0]])
            line.set_ydata([pt[1], centroid[0][1]])

        # Update title and annotation
        plot_title.set_text(
            f'WCSS = {_wcss(centroid[0]):.2f}   '
            f'(minimum = {WCSS_MIN:.2f} at true mean)'
        )
        ann_text.set_text(_annotation(centroid[0]))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event',  _on_motion)

    plt.show()