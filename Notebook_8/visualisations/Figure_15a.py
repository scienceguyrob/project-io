"""
Figure 15a — Interactive Linkage Criterion Explorer
====================================================
Demonstrates how the four main linkage criteria define distance between
clusters differently, using a simple two-cluster example with draggable
points.

Two small clusters (A and B) are shown. The user can drag any point in
either cluster. As points move, the linkage distance between the two
clusters is recomputed live for all four criteria simultaneously, and
visual annotations show exactly which pair of points is driving each
distance measurement:

  Single   — the shortest line between any point in A and any point in B
  Complete — the longest line between any point in A and any point in B
  Average  — the mean of all pairwise distances between A and B
  Ward     — the increase in WCSS that would result from merging A and B

The annotation panel on the right shows the numeric distances for all
four criteria so the learner can see how they diverge as the clusters
are reshaped.

Interaction
-----------
Click and drag any point in either cluster. All four distance measurements
and their visual annotations update live.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_15a import show
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


# ── Initial cluster positions ─────────────────────────────────────────────────
# Cluster A: four points on the left
# Cluster B: four points on the right
# Deliberately asymmetric so single and complete linkage give very different
# values from the start — making the contrast immediately visible.
A_INIT = np.array([
    [1.0, 3.0],
    [1.5, 4.5],
    [2.0, 2.5],
    [2.5, 4.0],
], dtype=float)

B_INIT = np.array([
    [5.5, 2.5],
    [6.0, 4.0],
    [7.0, 3.0],
    [6.5, 5.0],
], dtype=float)

# ── Visual parameters ─────────────────────────────────────────────────────────
COL_A      = 'steelblue'
COL_B      = 'tomato'
COL_SINGLE = 'seagreen'
COL_COMP   = 'darkorange'
COL_AVG    = 'mediumpurple'
COL_WARD   = '#333333'

HIT_RADIUS = 0.35
XLIM = (0.0, 9.0)
YLIM = (0.5, 6.5)


# ── Distance functions ────────────────────────────────────────────────────────

def _pairwise(A, B):
    """All pairwise Euclidean distances between points in A and points in B."""
    return np.array([
        np.linalg.norm(a - b)
        for a in A for b in B
    ])


def _single(A, B):
    """Shortest distance between any point in A and any point in B."""
    return float(np.min(_pairwise(A, B)))


def _complete(A, B):
    """Longest distance between any point in A and any point in B."""
    return float(np.max(_pairwise(A, B)))


def _average(A, B):
    """Mean of all pairwise distances between A and B."""
    return float(np.mean(_pairwise(A, B)))


def _ward(A, B):
    """
    Increase in total WCSS that would result from merging clusters A and B.

    Ward distance = n_A * n_B / (n_A + n_B) * ||mu_A - mu_B||^2

    This is the standard Ward(D) formula. It penalises merges that would
    pull the combined centroid far from one or both of the original centroids,
    weighted by cluster size.
    """
    n_a, n_b   = len(A), len(B)
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    return float((n_a * n_b) / (n_a + n_b) * np.linalg.norm(mu_a - mu_b) ** 2)


def _single_pair(A, B):
    """Return the pair of points (one from A, one from B) with minimum distance."""
    best_d, best_i, best_j = np.inf, 0, 0
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            d = np.linalg.norm(a - b)
            if d < best_d:
                best_d, best_i, best_j = d, i, j
    return best_i, best_j


def _complete_pair(A, B):
    """Return the pair of points (one from A, one from B) with maximum distance."""
    best_d, best_i, best_j = -np.inf, 0, 0
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            d = np.linalg.norm(a - b)
            if d > best_d:
                best_d, best_i, best_j = d, i, j
    return best_i, best_j


def _annotation_text(A, B):
    """Build the annotation panel string with all four distances."""
    d_s = _single(A, B)
    d_c = _complete(A, B)
    d_a = _average(A, B)
    d_w = _ward(A, B)

    return (
        "Linkage distances\n"
        "─────────────────────────────────\n\n"
        "Single (min pairwise):\n"
        f"  shortest line between\n"
        f"  any point in A and B\n"
        f"  $d_{{single}} = {d_s:.3f}$\n\n"
        "Complete (max pairwise):\n"
        f"  longest line between\n"
        f"  any point in A and B\n"
        f"  $d_{{complete}} = {d_c:.3f}$\n\n"
        "Average (mean pairwise):\n"
        f"  mean of all pairwise\n"
        f"  distances between A and B\n"
        f"  $d_{{average}} = {d_a:.3f}$\n\n"
        "Ward (WCSS increase):\n"
        f"  cost of merging A and B\n"
        f"  weighted by cluster size\n"
        f"  $d_{{ward}} = {d_w:.3f}$\n\n"
        "─────────────────────────────────\n\n"
        "Drag any point to see how\n"
        "each criterion responds\n"
        "differently to the same move."
    )


def show():
    """Render Figure 15a: interactive linkage criterion explorer."""
    plt.close('Notebook8 Figure 15a')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 15a',
        figsize=(10, 7),
        gridspec_kw={'width_ratios': [1.5, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # Mutable cluster arrays — modified in place as points are dragged
    A = [A_INIT.copy()]
    B = [B_INIT.copy()]

    # Track which point is being dragged: ('A', idx) or ('B', idx) or None
    dragging = [None]

    # ── Static cluster labels ─────────────────────────────────────────────────
    ax_plot.text(
        A_INIT[:, 0].mean(), YLIM[1] - 0.2, 'Cluster A',
        ha='center', va='top', fontsize=11,
        color=COL_A, fontweight='bold',
    )
    ax_plot.text(
        B_INIT[:, 0].mean(), YLIM[1] - 0.2, 'Cluster B',
        ha='center', va='top', fontsize=11,
        color=COL_B, fontweight='bold',
    )

    # ── Scatter plots for both clusters ───────────────────────────────────────
    scat_a = ax_plot.scatter(
        A[0][:, 0], A[0][:, 1],
        s=120, color=COL_A, edgecolors='k', lw=0.8,
        zorder=4, label='Cluster A (drag points)',
    )
    scat_b = ax_plot.scatter(
        B[0][:, 0], B[0][:, 1],
        s=120, color=COL_B, edgecolors='k', lw=0.8,
        zorder=4, label='Cluster B (drag points)',
    )

    # ── Centroid markers ──────────────────────────────────────────────────────
    scat_mu_a = ax_plot.scatter(
        *A[0].mean(axis=0), marker='+', s=200,
        color=COL_A, lw=2.5, zorder=5,
    )
    scat_mu_b = ax_plot.scatter(
        *B[0].mean(axis=0), marker='+', s=200,
        color=COL_B, lw=2.5, zorder=5,
    )

    # ── Linkage distance lines ────────────────────────────────────────────────
    # Single: solid green line between nearest pair
    si, sj     = _single_pair(A[0], B[0])
    line_single, = ax_plot.plot(
        [A[0][si, 0], B[0][sj, 0]],
        [A[0][si, 1], B[0][sj, 1]],
        color=COL_SINGLE, lw=2.2, ls='-', zorder=3,
        label=f'Single  d={_single(A[0], B[0]):.2f}',
    )

    # Complete: dashed orange line between furthest pair
    ci, cj     = _complete_pair(A[0], B[0])
    line_complete, = ax_plot.plot(
        [A[0][ci, 0], B[0][cj, 0]],
        [A[0][ci, 1], B[0][cj, 1]],
        color=COL_COMP, lw=2.2, ls='--', zorder=3,
        label=f'Complete  d={_complete(A[0], B[0]):.2f}',
    )

    # Ward: dotted line between centroids, thickness encodes the Ward distance
    line_ward, = ax_plot.plot(
        [A[0].mean(axis=0)[0], B[0].mean(axis=0)[0]],
        [A[0].mean(axis=0)[1], B[0].mean(axis=0)[1]],
        color=COL_WARD, lw=2.2, ls=':', zorder=3,
        label=f'Ward  d={_ward(A[0], B[0]):.2f}',
    )

    ax_plot.set_xlim(*XLIM)
    ax_plot.set_ylim(*YLIM)
    ax_plot.set_xlabel('Feature 1', fontsize=10)
    ax_plot.set_ylabel('Feature 2', fontsize=10)
    ax_plot.set_title(
        'Drag any point to see how each linkage criterion responds\n'
        'Green = Single   Orange = Complete   Dotted = Ward',
        fontsize=10,
    )
    ax_plot.legend(fontsize=8, loc='lower right',
                   framealpha=1.0, edgecolor='#cccccc')
    ax_plot.grid(True, alpha=0.2)

    plot_title = ax_plot.set_title(
        'Drag any point — green = Single, orange dashed = Complete, '
        'dotted = Ward (centroid–centroid)',
        fontsize=9,
    )

    # ── Annotation panel ──────────────────────────────────────────────────────
    ax_ann.set_axis_off()
    ann_text = ax_ann.text(
        0.04, 0.97,
        _annotation_text(A[0], B[0]),
        transform=ax_ann.transAxes,
        fontsize=9, va='top', ha='left',
        linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 15a: Linkage criteria — how each one measures distance between clusters',
        fontsize=11,
    )
    plt.subplots_adjust(wspace=0.08, top=0.88)

    # ── Drag interaction ──────────────────────────────────────────────────────
    def _on_press(event):
        if event.inaxes is not ax_plot or event.xdata is None:
            return
        cursor = np.array([event.xdata, event.ydata])
        # Check cluster A points first, then B
        for idx, pt in enumerate(A[0]):
            if np.linalg.norm(cursor - pt) < HIT_RADIUS:
                dragging[0] = ('A', idx)
                return
        for idx, pt in enumerate(B[0]):
            if np.linalg.norm(cursor - pt) < HIT_RADIUS:
                dragging[0] = ('B', idx)
                return

    def _on_release(event):
        dragging[0] = None

    def _on_motion(event):
        if dragging[0] is None or event.inaxes is not ax_plot:
            return
        if event.xdata is None:
            return

        x = float(np.clip(event.xdata, XLIM[0] + 0.1, XLIM[1] - 0.1))
        y = float(np.clip(event.ydata, YLIM[0] + 0.1, YLIM[1] - 0.1))

        cluster, idx = dragging[0]
        if cluster == 'A':
            A[0][idx] = [x, y]
        else:
            B[0][idx] = [x, y]

        _redraw()

    def _redraw():
        Ac, Bc = A[0], B[0]

        # Update scatter positions
        scat_a.set_offsets(Ac)
        scat_b.set_offsets(Bc)

        # Update centroid markers
        scat_mu_a.set_offsets([Ac.mean(axis=0)])
        scat_mu_b.set_offsets([Bc.mean(axis=0)])

        # Update Single line
        si, sj = _single_pair(Ac, Bc)
        line_single.set_xdata([Ac[si, 0], Bc[sj, 0]])
        line_single.set_ydata([Ac[si, 1], Bc[sj, 1]])
        line_single.set_label(f'Single  d={_single(Ac, Bc):.2f}')

        # Update Complete line
        ci, cj = _complete_pair(Ac, Bc)
        line_complete.set_xdata([Ac[ci, 0], Bc[cj, 0]])
        line_complete.set_ydata([Ac[ci, 1], Bc[cj, 1]])
        line_complete.set_label(f'Complete  d={_complete(Ac, Bc):.2f}')

        # Update Ward line (centroid to centroid)
        mu_a = Ac.mean(axis=0)
        mu_b = Bc.mean(axis=0)
        line_ward.set_xdata([mu_a[0], mu_b[0]])
        line_ward.set_ydata([mu_a[1], mu_b[1]])
        line_ward.set_label(f'Ward  d={_ward(Ac, Bc):.2f}')

        # Refresh legend
        ax_plot.legend(fontsize=8, loc='lower right',
                       framealpha=1.0, edgecolor='#cccccc')

        # Update annotation
        ann_text.set_text(_annotation_text(Ac, Bc))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event',  _on_motion)

    plt.show()