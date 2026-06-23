"""
Figure 4 — Interactive Manhattan Distance Explorer
===================================================
Two points, P1 and P2, are draggable. As either point moves, the axis-aligned
Manhattan path and the step-by-step formula annotation on the right update
in real time.

The Manhattan path is drawn as two segments:
  - A horizontal step from P1 to the x-coordinate of P2  (|dx|)
  - A vertical step from there up to P2                  (|dy|)

The right-hand annotation panel shows the full calculation:

    d = |x₂ - x₁| + |y₂ - y₁|

with current numeric values substituted at each stage.

Interaction
-----------
Click and drag either point to move it. The path segments, annotation,
and title all update live.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_4 import show
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


# ── Initial point positions ───────────────────────────────────────────────────
P1_INIT = np.array([1.5, 1.5])
P2_INIT = np.array([4.5, 5.5])

# ── Axes limits ───────────────────────────────────────────────────────────────
XLIM = (0.0, 7.5)
YLIM = (0.0, 7.5)

# ── Drag detection radius (in data units) ────────────────────────────────────
HIT_RADIUS = 0.35

# ── Colours ───────────────────────────────────────────────────────────────────
COL_P1   = 'steelblue'
COL_P2   = 'tomato'
COL_DX   = 'seagreen'
COL_DY   = 'darkorange'
COL_CORN = '#888888'    # corner marker where the two segments meet


def _manhattan(a, b):
    return np.sum(np.abs(a - b))


def _annotation_lines(p1, p2):
    """
    Build the step-by-step formula string for the annotation panel.
    Substitutes current numeric values at each stage so the learner
    can follow from formula to result.
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    d  = dx + dy

    return (
        "Manhattan distance\n"
        "──────────────────────────────\n\n"
        r"$d = |x_2 - x_1| + |y_2 - y_1|$"
        "\n\n"
        f"$x_1 = {p1[0]:.2f},\\ x_2 = {p2[0]:.2f}$\n"
        f"$y_1 = {p1[1]:.2f},\\ y_2 = {p2[1]:.2f}$\n\n"
        f"$|x_2 - x_1|$\n"
        f"$= |{p2[0]:.2f} - {p1[0]:.2f}| = {dx:.3f}$\n\n"
        f"$|y_2 - y_1|$\n"
        f"$= |{p2[1]:.2f} - {p1[1]:.2f}| = {dy:.3f}$\n\n"
        f"$d = {dx:.3f} + {dy:.3f}$\n\n"
        f"$d = \\mathbf{{{d:.3f}}}$"
    )


def show():
    """Render Figure 4: interactive Manhattan distance explorer."""
    plt.close('Notebook8 Figure 4')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 4',
        figsize=(10, 6),
        gridspec_kw={'width_ratios': [1.4, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    pts      = [P1_INIT.copy(), P2_INIT.copy()]
    dragging = [None]

    # ── Initial plot elements ─────────────────────────────────────────────────
    scat_p1 = ax_plot.scatter(*pts[0], s=140, color=COL_P1, zorder=5,
                               label='P1 (drag me)')
    scat_p2 = ax_plot.scatter(*pts[1], s=140, color=COL_P2, zorder=5,
                               label='P2 (drag me)')

    # Corner point — where the horizontal step ends and the vertical begins
    corner = np.array([pts[1][0], pts[0][1]])

    (dx_line,) = ax_plot.plot(
        [pts[0][0], corner[0]], [pts[0][1], corner[1]],
        color=COL_DX, lw=2.5, zorder=3,
        label=f'|dx| = {abs(pts[1][0] - pts[0][0]):.3f}',
        solid_capstyle='round',
    )
    (dy_line,) = ax_plot.plot(
        [corner[0], pts[1][0]], [corner[1], pts[1][1]],
        color=COL_DY, lw=2.5, zorder=3,
        label=f'|dy| = {abs(pts[1][1] - pts[0][1]):.3f}',
        solid_capstyle='round',
    )

    # Small square marker at the corner so the right angle is obvious
    corner_dot = ax_plot.scatter(
        *corner, s=55, color=COL_CORN, zorder=4, marker='s'
    )

    # dx label — centred below the horizontal segment
    dx_lbl = ax_plot.text(
        (pts[0][0] + corner[0]) / 2, pts[0][1] - 0.25,
        f'|dx| = {abs(pts[1][0] - pts[0][0]):.3f}',
        fontsize=9, color=COL_DX, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.85),
    )

    # dy label — to the right of the vertical segment
    dy_lbl = ax_plot.text(
        corner[0] + 0.15, (corner[1] + pts[1][1]) / 2,
        f'|dy| = {abs(pts[1][1] - pts[0][1]):.3f}',
        fontsize=9, color=COL_DY, ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.85),
    )

    # Total distance label near P2
    d_init = _manhattan(pts[0], pts[1])
    dist_label = ax_plot.text(
        pts[1][0] + 0.15, pts[1][1] + 0.15,
        f'd = {d_init:.3f}',
        fontsize=10, color='#333333', zorder=6,
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
        f'Manhattan distance: {d_init:.3f}  —  drag either point',
        fontsize=10,
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
        'Figure 4: Manhattan distance — drag either point to explore',
        fontsize=11,
    )
    # tight_layout is skipped here — it triggers a mathtext render pass before
    # the canvas is ready when ax_ann contains multi-line LaTeX, causing a
    # ValueError in Matplotlib's mathtext parser on some backends.
    plt.subplots_adjust(wspace=0.15)

    # ── Drag interaction ──────────────────────────────────────────────────────
    def _on_press(event):
        if event.inaxes is not ax_plot:
            return
        cursor = np.array([event.xdata, event.ydata])
        for i, p in enumerate(pts):
            if np.linalg.norm(cursor - p) < HIT_RADIUS:
                dragging[0] = i
                return

    def _on_release(event):
        dragging[0] = None

    def _on_motion(event):
        if dragging[0] is None or event.inaxes is not ax_plot:
            return

        x = float(np.clip(event.xdata, XLIM[0] + 0.1, XLIM[1] - 0.1))
        y = float(np.clip(event.ydata, YLIM[0] + 0.1, YLIM[1] - 0.1))
        pts[dragging[0]] = np.array([x, y])

        p1, p2 = pts[0], pts[1]
        corner = np.array([p2[0], p1[1]])
        dx     = abs(p2[0] - p1[0])
        dy     = abs(p2[1] - p1[1])
        d      = dx + dy

        # Update scatter positions
        scat_p1.set_offsets([p1])
        scat_p2.set_offsets([p2])

        # Update path segments
        dx_line.set_xdata([p1[0], corner[0]])
        dx_line.set_ydata([p1[1], corner[1]])
        dy_line.set_xdata([corner[0], p2[0]])
        dy_line.set_ydata([corner[1], p2[1]])

        # Update corner marker
        corner_dot.set_offsets([corner])

        # Update segment labels
        dx_lbl.set_position(((p1[0] + corner[0]) / 2, p1[1] - 0.25))
        dx_lbl.set_text(f'|dx| = {dx:.3f}')

        dy_lbl.set_position((corner[0] + 0.15, (corner[1] + p2[1]) / 2))
        dy_lbl.set_text(f'|dy| = {dy:.3f}')

        # Update legend
        dx_line.set_label(f'|dx| = {dx:.3f}')
        dy_line.set_label(f'|dy| = {dy:.3f}')
        ax_plot.legend(fontsize=9, loc='upper left', framealpha=1.0,
                       edgecolor='#cccccc')

        # Update total distance label near P2
        dist_label.set_position((p2[0] + 0.15, p2[1] + 0.15))
        dist_label.set_text(f'd = {d:.3f}')

        # Update title and annotation
        plot_title.set_text(f'Manhattan distance: {d:.3f}  —  drag either point')
        ann_text.set_text(_annotation_lines(p1, p2))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event',  _on_motion)

    plt.show()