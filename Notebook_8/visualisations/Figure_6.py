"""
Figure 6 — Interactive Mahalanobis Distance Explorer
=====================================================
Demonstrates that Mahalanobis distance accounts for the correlation structure
of the data, while Euclidean distance does not.

A single draggable test point sits on the left panel, overlaid on a
correlated 2D data cloud. As the point moves, the right panel updates with
a step-by-step annotation showing exactly how the Mahalanobis distance is
computed from the estimated covariance matrix.

The key insight: two points at the same Euclidean distance from the centre
can have very different Mahalanobis distances depending on whether they lie
along or across the grain of the data.

Interaction
-----------
Click and drag the red test point to any position. Both the Euclidean and
Mahalanobis distances update live, along with the full calculation breakdown.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_6 import show
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


# ── Data generation ───────────────────────────────────────────────────────────
# Covariance matrix must be symmetric positive semi-definite (PSD).
# [[5,4],[4,3]] has determinant 5*3 - 4*4 = -1, so it is NOT PSD and causes
# a RuntimeWarning. We use [[3,2],[2,2]] instead: determinant = 3*2 - 2*2 = 2,
# which is positive, giving a valid strongly correlated covariance matrix.
RNG      = np.random.default_rng(3)
COV_TRUE = np.array([[3.0, 2.0], [2.0, 2.0]])
CENTRE   = np.array([0.0, 0.0])
N_PTS    = 300
DATA     = RNG.multivariate_normal(CENTRE, COV_TRUE, N_PTS)

# Estimate S from the data rather than using COV_TRUE directly — this is what
# a practitioner would do and keeps the demonstration honest.
S     = np.cov(DATA.T)
S_INV = np.linalg.inv(S)

# ── Initial test point ────────────────────────────────────────────────────────
# Placed along the main axis of the data cloud so the learner can drag it
# perpendicular and immediately see the Mahalanobis distance rise.
PT_INIT = np.array([2.0, 0.5])

# ── Plot limits ───────────────────────────────────────────────────────────────
XLIM = (-8.0, 8.0)
YLIM = (-8.0, 8.0)

HIT_RADIUS = 0.4

# ── Colours ───────────────────────────────────────────────────────────────────
COL_DATA = 'lightsteelblue'
COL_PT   = 'tomato'
COL_LINE = '#444444'


def _mahalanobis(x, mu, s_inv):
    """sqrt( (x - mu)^T  S^{-1}  (x - mu) )"""
    diff = x - mu
    return float(np.sqrt(diff @ s_inv @ diff))


def _annotation(pt, mu, s_inv):
    """
    Build the step-by-step Mahalanobis calculation string.
    Matplotlib's mathtext parser does not support \\begin{pmatrix} or
    other LaTeX environments, so the inverse covariance matrix is displayed
    as two plain-text rows rather than a matrix environment.
    """
    diff    = pt - mu
    s_inv_d = s_inv @ diff
    quad    = float(diff @ s_inv_d)
    d_mah   = float(np.sqrt(max(quad, 0.0)))   # guard against tiny negatives
    d_euc   = float(np.linalg.norm(diff))

    return (
        "Mahalanobis distance\n"
        "─────────────────────────────────\n\n"
        r"$d_M = \sqrt{(\mathbf{x}-\mu)^\top \mathbf{S}^{-1}(\mathbf{x}-\mu)}$"
        "\n\n"
        f"$\\mathbf{{x}} = ({pt[0]:.2f},\\ {pt[1]:.2f})$\n"
        f"$\\mu = ({mu[0]:.2f},\\ {mu[1]:.2f})$\n\n"
        f"$\\mathbf{{x}} - \\mu = ({diff[0]:.2f},\\ {diff[1]:.2f})$\n\n"
        # Matrix displayed as two plain rows — \begin{} not supported
        "$\\mathbf{S}^{-1}$:\n"
        f"  [{s_inv[0,0]:.3f},  {s_inv[0,1]:.3f}]\n"
        f"  [{s_inv[1,0]:.3f},  {s_inv[1,1]:.3f}]\n\n"
        "$\\mathbf{S}^{-1}(\\mathbf{x}-\\mu)$\n"
        f"$= ({s_inv_d[0]:.3f},\\ {s_inv_d[1]:.3f})$\n\n"
        "$(\\mathbf{x}-\\mu)^\\top \\mathbf{S}^{-1}(\\mathbf{x}-\\mu)$\n"
        f"$= {quad:.3f}$\n\n"
        f"$d_M = \\sqrt{{{quad:.3f}}}$\n"
        f"$= \\mathbf{{{d_mah:.3f}}}$\n\n"
        "─────────────────────────────────\n\n"
        "Euclidean distance\n"
        f"$d_E = \\sqrt{{{diff[0]:.2f}^2 + {diff[1]:.2f}^2}}$\n"
        f"$= \\mathbf{{{d_euc:.3f}}}$"
    )


def show():
    """Render Figure 6: interactive Mahalanobis distance explorer."""
    plt.close('Notebook8 Figure 6')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 6',
        figsize=(10, 7),
        gridspec_kw={'width_ratios': [1.3, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    pt       = [PT_INIT.copy()]
    dragging = [False]

    # ── Left panel: data cloud ────────────────────────────────────────────────
    ax_plot.scatter(
        DATA[:, 0], DATA[:, 1],
        s=12, color=COL_DATA, edgecolors='none', alpha=0.6,
        label='Data', zorder=2,
    )
    ax_plot.scatter(
        *CENTRE, s=140, color='black', marker='+', lw=2,
        zorder=5, label='Centre $\\mu$',
    )

    (conn_line,) = ax_plot.plot(
        [CENTRE[0], pt[0][0]], [CENTRE[1], pt[0][1]],
        color=COL_LINE, lw=1.5, ls='--', zorder=4,
    )

    scat_pt = ax_plot.scatter(
        *pt[0], s=140, color=COL_PT, edgecolors='k', lw=0.8,
        zorder=6, label='Test point (drag me)',
    )

    d_mah_init = _mahalanobis(pt[0], CENTRE, S_INV)
    d_euc_init = float(np.linalg.norm(pt[0] - CENTRE))

    plot_title = ax_plot.set_title(
        f'Euclidean = {d_euc_init:.3f}   |   Mahalanobis = {d_mah_init:.3f}',
        fontsize=10,
    )

    ax_plot.set_xlim(*XLIM)
    ax_plot.set_ylim(*YLIM)
    ax_plot.set_aspect('equal')
    ax_plot.set_xlabel('Feature 1', fontsize=11)
    ax_plot.set_ylabel('Feature 2', fontsize=11)
    ax_plot.grid(True, alpha=0.2)
    ax_plot.legend(fontsize=9, loc='upper left',
                   framealpha=1.0, edgecolor='#cccccc')

    # ── Right panel: annotation ───────────────────────────────────────────────
    ax_ann.set_axis_off()
    ann_text = ax_ann.text(
        0.04, 0.97,
        _annotation(pt[0], CENTRE, S_INV),
        transform=ax_ann.transAxes,
        fontsize=9, va='top', ha='left',
        linespacing=1.65,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    plt.suptitle(
        'Figure 6: Mahalanobis distance — drag the point to explore '
        'how direction relative to the data cloud affects distance',
        fontsize=11,
    )
    plt.subplots_adjust(wspace=0.08)

    # ── Drag interaction ──────────────────────────────────────────────────────
    def _on_press(event):
        if event.inaxes is not ax_plot:
            return
        cursor = np.array([event.xdata, event.ydata])
        if np.linalg.norm(cursor - pt[0]) < HIT_RADIUS:
            dragging[0] = True

    def _on_release(event):
        dragging[0] = False

    def _on_motion(event):
        if not dragging[0] or event.inaxes is not ax_plot:
            return

        x = float(np.clip(event.xdata, XLIM[0] + 0.1, XLIM[1] - 0.1))
        y = float(np.clip(event.ydata, YLIM[0] + 0.1, YLIM[1] - 0.1))
        pt[0] = np.array([x, y])

        d_mah = _mahalanobis(pt[0], CENTRE, S_INV)
        d_euc = float(np.linalg.norm(pt[0] - CENTRE))

        scat_pt.set_offsets([pt[0]])
        conn_line.set_xdata([CENTRE[0], pt[0][0]])
        conn_line.set_ydata([CENTRE[1], pt[0][1]])

        plot_title.set_text(
            f'Euclidean = {d_euc:.3f}   |   Mahalanobis = {d_mah:.3f}'
        )
        ann_text.set_text(_annotation(pt[0], CENTRE, S_INV))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event',  _on_motion)

    plt.show()