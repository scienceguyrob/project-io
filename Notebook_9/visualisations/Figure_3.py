"""
Figure 3 — Core Distance vs Reachability Distance
===================================================
Demonstrates the relationship:

    reach-dist_minPts(p, o) = max(core-dist_minPts(o), d(p, o))

A small, fixed scatter of points represents the neighbourhood around a
chosen anchor point o. One point, p, is draggable. As p is moved:

  - The actual distance d(p, o) is recalculated and shown as a line
    from p to o
  - o's core distance (distance to its minPts-th nearest neighbour,
    NOT counting p) is shown as a fixed dashed circle around o —
    it does not change as p moves, since it depends only on the
    other fixed points
  - The reachability distance reach-dist(p, o) = max(core-dist(o), d(p, o))
    is shown as a second circle (solid) around o, which always sits at
    or outside the core-distance circle
  - An annotation panel reports both values numerically and explains
    which term — core-dist(o) or d(p, o) — currently "wins" the max

Dragging p from inside the core-distance circle to outside it (and back)
shows the moment at which reach-dist(p, o) switches from being floored at
core-dist(o) to tracking d(p, o) directly.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_3 import show
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


# ── Fixed analysis parameter ────────────────────────────────────────────────
# minPts = 4 here (small, so the figure stays uncluttered). core-dist(o) is
# defined as the distance from o to its minPts-th nearest neighbour, counted
# among the FIXED points only — the draggable point p never participates in
# this calculation, which is what keeps the core-distance circle static.
MIN_PTS = 4

# ── Fixed neighbourhood points ──────────────────────────────────────────────
# o is the anchor point, at the origin. The other fixed points form a small
# neighbourhood around it at varying distances, chosen so that o's 4th
# nearest neighbour (core-dist) sits at a clearly visible radius.
O_POINT = np.array([0.0, 0.0])

FIXED_POINTS = np.array([
    [0.35,  0.10],   # close
    [-0.20, 0.40],   # close
    [0.55, -0.35],   # medium
    [-0.50, -0.30],  # medium  <- this is o's 4th nearest neighbour (core-dist)
    [0.90,  0.60],   # far
    [-0.85, 0.55],   # far
])

# core-dist(o) = distance from o to its MIN_PTS-th nearest neighbour among
# FIXED_POINTS. Computed once, since FIXED_POINTS never change.
_fixed_dists = np.linalg.norm(FIXED_POINTS - O_POINT, axis=1)
CORE_DIST_O  = float(np.sort(_fixed_dists)[MIN_PTS - 1])

# ── Draggable point's starting position ─────────────────────────────────────
P_START = np.array([0.15, -0.55])

# ── Axis limits ──────────────────────────────────────────────────────────────
ALL_X = np.concatenate([FIXED_POINTS[:, 0], [O_POINT[0], P_START[0]]])
ALL_Y = np.concatenate([FIXED_POINTS[:, 1], [O_POINT[1], P_START[1]]])
PAD   = 0.6
XLIM  = (ALL_X.min() - PAD, ALL_X.max() + PAD)
YLIM  = (ALL_Y.min() - PAD, ALL_Y.max() + PAD)

COL_O           = 'steelblue'
COL_FIXED       = '#999999'
COL_P           = 'tomato'
COL_CORE_CIRCLE = 'goldenrod'
COL_REACH_CIRCLE = 'mediumpurple'
COL_DIST_LINE   = '#444444'


def _annotation_text(d_po, reach):
    """Build the annotation panel string."""
    if d_po <= CORE_DIST_O:
        winner = (
            "reach-dist(p, o) = core-dist(o)\n\n"
            "p is INSIDE o's core-distance\n"
            "circle: d(p, o) is smaller than\n"
            "core-dist(o), so the reachability\n"
            "distance is floored at core-dist(o).\n"
            "Moving p around inside this circle\n"
            "does not change reach-dist(p, o)."
        )
    else:
        winner = (
            "reach-dist(p, o) = d(p, o)\n\n"
            "p is OUTSIDE o's core-distance\n"
            "circle: d(p, o) is now larger than\n"
            "core-dist(o), so the reachability\n"
            "distance simply tracks the actual\n"
            "distance from p to o."
        )

    return (
        "Core distance vs reachability distance\n"
        "─────────────────────────────────\n\n"
        f"minPts = {MIN_PTS}  (fixed)\n\n"
        f"core-dist(o)   = {CORE_DIST_O:.3f}\n"
        f"  (distance from o to its\n"
        f"   {MIN_PTS}th nearest fixed\n"
        f"   neighbour — does not\n"
        f"   depend on p)\n\n"
        f"d(p, o)        = {d_po:.3f}\n"
        f"  (actual distance from\n"
        f"   the draggable point p\n"
        f"   to o)\n\n"
        f"reach-dist(p, o) = max(core-dist(o), d(p, o))\n"
        f"                 = {reach:.3f}\n\n"
        "─────────────────────────────────\n\n"
        f"{winner}"
    )


def show():
    """Render Figure 3: core distance vs reachability distance, with draggable p."""
    plt.close('Notebook9 Figure 3')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook9 Figure 3',
        figsize=(10, 7),
        gridspec_kw={'width_ratios': [1.4, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    ax_ann.set_axis_off()

    ann_text = ax_ann.text(
        0.04, 0.97, '',
        transform=ax_ann.transAxes,
        fontsize=9, va='top', ha='left',
        linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 3: Reachability distance is floored at core-dist(o) —\n'
        'drag point p to see how reach-dist(p, o) responds',
        fontsize=11,
        y=0.98,
    )
    plt.subplots_adjust(wspace=0.08, top=0.86)

    ax_plot.set_xlim(*XLIM)
    ax_plot.set_ylim(*YLIM)
    ax_plot.set_aspect('equal')
    ax_plot.set_xlabel('Feature 1', fontsize=10)
    ax_plot.set_ylabel('Feature 2', fontsize=10)
    ax_plot.grid(True, alpha=0.15)

    # ── Static elements: fixed points, o, core-distance circle ──────────────
    ax_plot.scatter(
        FIXED_POINTS[:, 0], FIXED_POINTS[:, 1],
        marker='o', s=55, color=COL_FIXED, edgecolors='k',
        lw=0.4, alpha=0.85, zorder=3,
        label='Fixed neighbourhood points',
    )
    ax_plot.scatter(
        O_POINT[0], O_POINT[1],
        marker='o', s=90, color=COL_O, edgecolors='k',
        lw=0.6, zorder=5, label='o (anchor point)',
    )
    ax_plot.annotate(
        'o', O_POINT, textcoords='offset points', xytext=(8, 8), fontsize=11,
    )

    # core-dist(o) circle — static, since it depends only on FIXED_POINTS
    core_circle = mpatches.Circle(
        O_POINT, radius=CORE_DIST_O,
        facecolor='none', edgecolor=COL_CORE_CIRCLE,
        linestyle='--', linewidth=1.6, zorder=2,
        label=f'core-dist(o) = {CORE_DIST_O:.2f}',
    )
    ax_plot.add_patch(core_circle)

    # ── Dynamic elements: reach-dist circle, distance line, point p ──────────
    reach_circle = mpatches.Circle(
        O_POINT, radius=CORE_DIST_O,
        facecolor='none', edgecolor=COL_REACH_CIRCLE,
        linestyle='-', linewidth=1.8, zorder=2,
    )
    ax_plot.add_patch(reach_circle)

    dist_line, = ax_plot.plot(
        [O_POINT[0], P_START[0]], [O_POINT[1], P_START[1]],
        color=COL_DIST_LINE, linewidth=1.2, linestyle=':', zorder=4,
    )

    p_scatter = ax_plot.scatter(
        P_START[0], P_START[1],
        marker='o', s=90, color=COL_P, edgecolors='k',
        lw=0.6, zorder=6, label='p (drag me)',
    )
    p_label = ax_plot.annotate(
        'p', P_START, textcoords='offset points', xytext=(8, 8), fontsize=11,
    )

    ax_plot.legend(
        fontsize=8, loc='upper right', framealpha=1.0, edgecolor='#cccccc',
    )

    # ── State for the draggable point ────────────────────────────────────────
    state = {'p': P_START.copy(), 'dragging': False}

    def _update(p_pos):
        d_po  = float(np.linalg.norm(p_pos - O_POINT))
        reach = max(CORE_DIST_O, d_po)

        # Reachability circle radius always equals reach-dist(p, o), so it
        # sits exactly on the core-dist circle when p is inside it, and
        # expands outward to meet p once p moves beyond core-dist(o).
        reach_circle.set_radius(reach)

        # Visually distinguish the two regimes: while reach-dist is floored
        # at core-dist(o) (p inside the circle), draw the reach circle in a
        # muted style since it coincides with the core-dist circle; once
        # d(p, o) takes over, draw it more prominently.
        if d_po <= CORE_DIST_O:
            reach_circle.set_alpha(0.35)
        else:
            reach_circle.set_alpha(0.9)

        dist_line.set_data([O_POINT[0], p_pos[0]], [O_POINT[1], p_pos[1]])
        p_scatter.set_offsets([p_pos])
        p_label.set_position((8, 8))
        p_label.xy = p_pos

        ax_plot.set_title(
            f'd(p, o) = {d_po:.3f}    core-dist(o) = {CORE_DIST_O:.3f}    '
            f'reach-dist(p, o) = {reach:.3f}',
            fontsize=9,
        )

        ann_text.set_text(_annotation_text(d_po, reach))

        fig.canvas.draw_idle()

    _update(state['p'])

    # ── Drag interaction ──────────────────────────────────────────────────────
    # Standard button_press / motion / release pattern: clicking near p
    # picks it up, motion while held updates its position, release drops it.
    PICK_RADIUS = 0.18  # in data units — generous enough to click p easily

    def _on_press(event):
        if event.inaxes != ax_plot or event.xdata is None:
            return
        click = np.array([event.xdata, event.ydata])
        if np.linalg.norm(click - state['p']) <= PICK_RADIUS:
            state['dragging'] = True

    def _on_motion(event):
        if not state['dragging']:
            return
        if event.inaxes != ax_plot or event.xdata is None:
            return
        state['p'] = np.array([event.xdata, event.ydata])
        _update(state['p'])

    def _on_release(event):
        state['dragging'] = False

    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_motion)
    fig.canvas.mpl_connect('button_release_event', _on_release)