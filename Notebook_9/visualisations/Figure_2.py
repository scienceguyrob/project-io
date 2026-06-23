"""
Figure 2 — The OPTICS Reachability Plot
=========================================
Demonstrates how OPTICS replaces DBSCAN's single fixed ε with a
reachability plot that encodes the clustering structure across every
possible ε value at once.

Uses the same dataset as Figure 1 (one dense cluster, one sparse cluster,
and scattered noise points), fitted once with OPTICS. Two panels are shown:

  - Top panel: the points in feature space, ordered and coloured exactly
    as OPTICS processes them
  - Bottom panel: the reachability plot — reachability distance for each
    point, in OPTICS processing order

A single draggable horizontal line on the reachability plot acts as a
cutoff threshold. Points whose reachability distance falls below the line
are coloured by which "valley" (cluster) they belong to; points above the
line are coloured as noise. The same colouring is mirrored on the top
scatter panel, so dragging the cutoff up and down shows directly how a
single reachability plot can reproduce the result of running DBSCAN at
many different ε values.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_2 import show
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
from sklearn.cluster import OPTICS

from visualisations.Figure_1 import X, GROUP, MIN_PTS


# ── Colour palette ─────────────────────────────────────────────────────────
COL_NOISE  = '#bbbbbb'
COL_DENSE  = 'steelblue'
COL_SPARSE = 'tomato'
COL_LINE   = '#444444'

# ── Fit OPTICS once ──────────────────────────────────────────────────────────
# OPTICS, unlike DBSCAN, does not need an ε at fit time — it computes the
# reachability distance for every point and an ordering that groups points
# from the same dense region together. min_samples plays the same role as
# DBSCAN's min_samples (= minPts) — it is the k used in core-dist.
_optics = OPTICS(min_samples=MIN_PTS)
_optics.fit(X)

# The processing order OPTICS assigns to each point, and the reachability
# distance of each point in that order. The first point in the ordering has
# reachability distance = inf by definition (it has no predecessor), so we
# replace it with a large finite value purely for plotting purposes.
ORDER = _optics.ordering_
REACH = _optics.reachability_[ORDER].copy()
_finite_max = REACH[np.isfinite(REACH)].max()
REACH[np.isinf(REACH)] = _finite_max * 1.15  # plot-only placeholder for the first point

N = len(ORDER)
X_ordered     = X[ORDER]
GROUP_ordered = GROUP[ORDER]

# ── Axis limits for the scatter panel ───────────────────────────────────────
PAD = 1.0
XLIM = (X[:, 0].min() - PAD, X[:, 0].max() + PAD)
YLIM = (X[:, 1].min() - PAD, X[:, 1].max() + PAD)

# ── Reachability plot y-limits ──────────────────────────────────────────────
REACH_YMAX = REACH.max() * 1.1


def _assign_colours(cutoff):
    """
    Given a reachability cutoff, decide a colour for every point in
    OPTICS order.

    Points with reachability distance above the cutoff are noise (grey).
    Points at or below the cutoff form contiguous runs in the ordering —
    each run is a "valley" in the reachability plot. We colour each valley
    according to whichever ground-truth group (dense / sparse) is the
    majority within it, so the colouring stays interpretable as the cutoff
    moves. Runs with no clear majority (e.g. a single scattered noise
    point that happens to fall below the cutoff) are coloured grey too.

    Returns
    -------
    colours : list of str, length N — one colour per point in OPTICS order
    """
    below = REACH <= cutoff
    colours = [COL_NOISE] * N

    # Walk through the ordering, grouping consecutive "below cutoff" points
    # into runs. Each run corresponds to one valley in the reachability plot.
    i = 0
    while i < N:
        if not below[i]:
            i += 1
            continue

        # Find the extent of this run of "below cutoff" points
        j = i
        while j < N and below[j]:
            j += 1

        run_groups = GROUP_ordered[i:j]
        n_dense  = int((run_groups == 'dense').sum())
        n_sparse = int((run_groups == 'sparse').sum())

        # Require a reasonable run length and a clear majority before
        # treating it as a real cluster — otherwise leave it as noise.
        if (j - i) >= MIN_PTS and (n_dense > 0 or n_sparse > 0):
            run_colour = COL_DENSE if n_dense >= n_sparse else COL_SPARSE
            for k in range(i, j):
                colours[k] = run_colour
        # else: leave this run as noise (grey)

        i = j

    return colours


def show():
    """Render Figure 2: the OPTICS reachability plot with a draggable cutoff."""
    plt.close('Notebook9 Figure 2')

    fig, (ax_scatter, ax_reach) = plt.subplots(
        2, 1,
        num='Notebook9 Figure 2',
        figsize=(9, 8),
        gridspec_kw={'height_ratios': [1, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    fig.suptitle(
        'Figure 2: The reachability plot encodes clustering structure\n'
        'across every ε value at once — drag the line to set a cutoff',
        fontsize=11,
        y=0.98,
    )
    plt.subplots_adjust(hspace=0.35, top=0.88, bottom=0.08)

    # ── Reachability bar plot (static structure, drawn once) ────────────────
    # Drawn as a bar plot, which is the conventional presentation of a
    # reachability plot: one bar per point, in OPTICS processing order,
    # height = reachability distance.
    bar_x = np.arange(N)
    ax_reach.bar(bar_x, REACH, width=1.0, color=COL_NOISE, edgecolor='none', zorder=2)
    bars = ax_reach.containers[0]

    ax_reach.set_xlim(-1, N)
    ax_reach.set_ylim(0, REACH_YMAX)
    ax_reach.set_xlabel('Cluster order (as visited by OPTICS)', fontsize=10)
    ax_reach.set_ylabel('Reachability distance', fontsize=10)
    ax_reach.set_title('Reachability plot', fontsize=10)
    ax_reach.grid(True, alpha=0.15, axis='y')

    # Draggable cutoff line, starting partway up the visible range.
    initial_cutoff = REACH_YMAX * 0.35
    cutoff_line = ax_reach.axhline(
        initial_cutoff, color=COL_LINE, linewidth=1.6,
        linestyle='--', zorder=3,
    )
    cutoff_label = ax_reach.text(
        0.99, 0.95, '', transform=ax_reach.transAxes,
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9),
    )

    # ── Scatter panel (points redrawn on each update) ────────────────────────
    scatter_artist = ax_scatter.scatter(
        X_ordered[:, 0], X_ordered[:, 1],
        c=[COL_NOISE] * N, s=55, edgecolors='k', lw=0.4, zorder=4,
    )
    ax_scatter.set_xlim(*XLIM)
    ax_scatter.set_ylim(*YLIM)
    ax_scatter.set_xlabel('Feature 1', fontsize=10)
    ax_scatter.set_ylabel('Feature 2', fontsize=10)
    ax_scatter.set_title('Points coloured by current cutoff', fontsize=10)
    ax_scatter.grid(True, alpha=0.15)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COL_DENSE, markeredgecolor='k',
                   markersize=9, label='Dense-cluster valley'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COL_SPARSE, markeredgecolor='k',
                   markersize=9, label='Sparse-cluster valley'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COL_NOISE, markeredgecolor='k',
                   markersize=9, label='Above cutoff (noise)'),
    ]
    ax_scatter.legend(
        handles=legend_handles, fontsize=8,
        loc='lower left', framealpha=1.0, edgecolor='#cccccc',
    )

    # ── Redraw colours for a given cutoff ────────────────────────────────────
    def _update(cutoff):
        colours = _assign_colours(cutoff)

        # Update bar colours
        for bar, colour in zip(bars, colours):
            bar.set_color(colour)

        # Update scatter colours
        scatter_artist.set_color(colours)

        n_clusters = (
            (np.array(colours) == COL_DENSE).any().astype(int)
            + (np.array(colours) == COL_SPARSE).any().astype(int)
        )
        n_noise = sum(1 for c in colours if c == COL_NOISE)

        cutoff_label.set_text(
            f'cutoff = {cutoff:.3f}\n'
            f'clusters: {n_clusters}   noise: {n_noise}'
        )

        fig.canvas.draw_idle()

    _update(initial_cutoff)

    # ── Drag interaction ──────────────────────────────────────────────────────
    # The cutoff line can be picked up and dragged vertically within the
    # reachability axes. This is the same button_press / motion / release
    # pattern used for draggable elements in earlier figures.
    _dragging = {'active': False}

    def _on_press(event):
        if event.inaxes != ax_reach:
            return
        # Only start dragging if the click is close to the current line,
        # so that clicking elsewhere on the plot doesn't move the line.
        y_line = cutoff_line.get_ydata()[0]
        if event.ydata is not None and abs(event.ydata - y_line) < REACH_YMAX * 0.03:
            _dragging['active'] = True

    def _on_motion(event):
        if not _dragging['active']:
            return
        if event.inaxes != ax_reach or event.ydata is None:
            return
        new_cutoff = float(np.clip(event.ydata, 0, REACH_YMAX))
        cutoff_line.set_ydata([new_cutoff, new_cutoff])
        _update(new_cutoff)

    def _on_release(event):
        _dragging['active'] = False

    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_motion)
    fig.canvas.mpl_connect('button_release_event', _on_release)