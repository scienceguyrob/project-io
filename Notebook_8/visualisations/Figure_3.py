"""
Figure 3 — Interactive Euclidean Distance Explorer
===================================================
Two points, P1 and P2, are draggable. As either point moves, the Euclidean
distance line and the step-by-step formula annotation on the right update
in real time.

The right-hand annotation panel shows the full calculation:

    d = √( (x₂ - x₁)² + (y₂ - y₁)² )

with the current numeric values substituted at each stage, so the learner
can see exactly how the formula produces the displayed distance.

Interaction
-----------
Click and drag either point to move it. The distance line, annotation,
and title all update live.

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
from matplotlib.lines import Line2D


# ── Initial point positions ───────────────────────────────────────────────────
P1_INIT = np.array([1.5, 1.5])
P2_INIT = np.array([4.5, 5.5])

# ── Axes limits — kept fixed so the annotation panel never rescales ───────────
XLIM = (0.0, 7.5)
YLIM = (0.0, 7.5)

# ── Drag detection radius (in data units) ────────────────────────────────────
HIT_RADIUS = 0.35

# ── Colours ───────────────────────────────────────────────────────────────────
COL_P1   = 'steelblue'
COL_P2   = 'tomato'
COL_LINE = '#333333'


def _euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def _annotation_lines(p1, p2):
    """
    Build the step-by-step formula string shown in the annotation panel.
    Each line substitutes the current numeric values so the learner can
    follow the calculation from formula to result.
    """
    dx  = p2[0] - p1[0]
    dy  = p2[1] - p1[1]
    dx2 = dx ** 2
    dy2 = dy ** 2
    d   = np.sqrt(dx2 + dy2)

    sign_x = '+' if dx >= 0 else '\u2212'   # unicode minus for cleaner display
    sign_y = '+' if dy >= 0 else '\u2212'

    return (
        "Euclidean distance\n"
        "──────────────────────────────\n\n"
        r"$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$"
        "\n\n"
        f"$x_1 = {p1[0]:.2f},\\ x_2 = {p2[0]:.2f}$\n"
        f"$y_1 = {p1[1]:.2f},\\ y_2 = {p2[1]:.2f}$\n\n"
        f"$x_2 - x_1 = {abs(dx):.2f}$\n"
        f"$y_2 - y_1 = {abs(dy):.2f}$\n\n"
        f"$(x_2-x_1)^2 = {dx2:.2f}$\n"
        f"$(y_2-y_1)^2 = {dy2:.2f}$\n\n"
        f"$dx^2 + dy^2 = {dx2 + dy2:.2f}$\n\n"
        r"$d = \sqrt{" + f"{dx2 + dy2:.2f}" + r"}$"
        "\n\n"
        f"$d = \\mathbf{{{d:.3f}}}$"
    )


def show():
    """Render Figure 3: interactive Euclidean distance explorer."""
    plt.close('Notebook8 Figure 3')

    # ── Layout: left = scatter panel, right = annotation panel ───────────────
    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 3',
        figsize=(10, 6),
        gridspec_kw={'width_ratios': [1.4, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # Mutable state — using a list so the closure can reassign elements
    pts = [P1_INIT.copy(), P2_INIT.copy()]
    dragging = [None]   # index of the point currently being dragged, or None

    # ── Initial plot elements ─────────────────────────────────────────────────
    scat_p1 = ax_plot.scatter(*pts[0], s=140, color=COL_P1, zorder=5,
                               label='P1 (drag me)')
    scat_p2 = ax_plot.scatter(*pts[1], s=140, color=COL_P2, zorder=5,
                               label='P2 (drag me)')

    (dist_line,) = ax_plot.plot(
        [pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
        color=COL_LINE, lw=2.2, zorder=3,
        label=f'd = {_euclidean(pts[0], pts[1]):.3f}'
    )

    # Midpoint label showing the distance value along the line
    mid = (pts[0] + pts[1]) / 2
    dist_label = ax_plot.text(
        mid[0] + 0.12, mid[1] + 0.12,
        f'd = {_euclidean(pts[0], pts[1]):.3f}',
        fontsize=10, color=COL_LINE, zorder=6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', alpha=0.9),
    )

    ax_plot.set_xlim(*XLIM)
    ax_plot.set_ylim(*YLIM)
    ax_plot.set_xlabel('x', fontsize=11)
    ax_plot.set_ylabel('y', fontsize=11)
    ax_plot.grid(True, alpha=0.25)
    ax_plot.legend(fontsize=9, loc='upper left', framealpha=1.0,
                   edgecolor='#cccccc')
    plot_title = ax_plot.set_title(
        f'Euclidean distance: {_euclidean(pts[0], pts[1]):.3f}  —  drag either point',
        fontsize=10
    )

    # ── Annotation panel ──────────────────────────────────────────────────────
    ax_ann.set_axis_off()
    ann_text = ax_ann.text(
        0.05, 0.95,
        _annotation_lines(pts[0], pts[1]),
        transform=ax_ann.transAxes,
        fontsize=10, va='top', ha='left',
        linespacing=1.7,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    plt.suptitle(
        'Figure 3: Euclidean distance — drag either point to explore',
        fontsize=11
    )
    plt.tight_layout()

    # ── Drag interaction ──────────────────────────────────────────────────────
    def _on_press(event):
        if event.inaxes is not ax_plot:
            return
        cursor = np.array([event.xdata, event.ydata])
        # Check which point (if any) is within the hit radius
        for i, p in enumerate(pts):
            if np.linalg.norm(cursor - p) < HIT_RADIUS:
                dragging[0] = i
                return

    def _on_release(event):
        dragging[0] = None

    def _on_motion(event):
        if dragging[0] is None or event.inaxes is not ax_plot:
            return

        # Clamp to axes bounds so the point cannot leave the visible area
        x = float(np.clip(event.xdata, XLIM[0] + 0.1, XLIM[1] - 0.1))
        y = float(np.clip(event.ydata, YLIM[0] + 0.1, YLIM[1] - 0.1))
        pts[dragging[0]] = np.array([x, y])

        p1, p2 = pts[0], pts[1]
        d = _euclidean(p1, p2)
        mid = (p1 + p2) / 2

        # Update scatter positions
        scat_p1.set_offsets([p1])
        scat_p2.set_offsets([p2])

        # Update distance line
        dist_line.set_xdata([p1[0], p2[0]])
        dist_line.set_ydata([p1[1], p2[1]])

        # Update midpoint label
        dist_label.set_position((mid[0] + 0.12, mid[1] + 0.12))
        dist_label.set_text(f'd = {d:.3f}')

        # Update plot title
        plot_title.set_text(
            f'Euclidean distance: {d:.3f}  —  drag either point'
        )

        # Update annotation panel
        ann_text.set_text(_annotation_lines(p1, p2))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event',  _on_motion)

    plt.show()