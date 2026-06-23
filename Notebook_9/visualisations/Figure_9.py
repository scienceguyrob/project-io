"""
Figure 9 — Vectors: Direction and Magnitude
==============================================
An interactive 2D vector explorer. A single draggable point defines a
vector v = [x, y], drawn as an arrow from the origin to that point.

As the point is dragged:

  - The arrow updates live to point from (0, 0) to the current (x, y).
  - A dashed right-angle triangle is drawn with legs of length |x| and
    |y|, visually connecting the vector to the Pythagorean magnitude
    formula.
  - An annotation panel shows the current vector [x, y], its magnitude
    ||v|| = sqrt(x^2 + y^2), and the angle the arrow makes with the
    x-axis, recomputed live.

A second, faint reference arrow shows the vector's direction normalised
to a fixed length, so that changes in direction vs changes in magnitude
can be visually distinguished — moving the point further along the same
ray keeps the reference arrow's direction unchanged, while moving it to a
different ray rotates it.

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
from IPython.display import display


# ── Initial vector ───────────────────────────────────────────────────────────
# Matches the [3, 1] example used in the surrounding markdown.
INITIAL_VECTOR = np.array([3.0, 1.0])

# ── Axis limits ──────────────────────────────────────────────────────────────
LIMIT = 5.0
XLIM = (-LIMIT, LIMIT)
YLIM = (-LIMIT, LIMIT)

# Length of the faint "direction only" reference arrow
REFERENCE_LENGTH = 1.5

COL_VECTOR    = 'steelblue'
COL_TRIANGLE  = '#999999'
COL_REFERENCE = '#bbbbbb'


def _annotation_text(v):
    """Build the annotation panel string for the current vector v."""
    x, y = v
    magnitude = float(np.linalg.norm(v))

    # Angle from the positive x-axis, in degrees, in the range (-180, 180].
    angle_deg = float(np.degrees(np.arctan2(y, x)))

    return (
        "Vector properties\n"
        "─────────────────────────────────\n\n"
        f"v = [{x:.2f}, {y:.2f}]\n\n"
        "─────────────────────────────────\n\n"
        "Magnitude:\n"
        f"  ||v|| = sqrt(x² + y²)\n"
        f"        = sqrt({x:.2f}² + {y:.2f}²)\n"
        f"        = sqrt({x*x:.2f} + {y*y:.2f})\n"
        f"        = {magnitude:.3f}\n\n"
        "─────────────────────────────────\n\n"
        "Direction:\n"
        f"  angle from x-axis = {angle_deg:.1f}°"
    )


def show():
    """Render Figure 9: interactive vector explorer."""
    plt.close('Notebook9 Figure 9')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook9 Figure 9',
        figsize=(10, 5.5),
        gridspec_kw={'width_ratios': [1.3, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    ax_ann.set_axis_off()

    ann_text = ax_ann.text(
        0.04, 0.97, '',
        transform=ax_ann.transAxes,
        fontsize=10, va='top', ha='left',
        family='monospace',
        linespacing=1.7,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 9: A vector has a direction and a magnitude —\n'
        'drag the point to see how both change',
        fontsize=11,
        y=0.98,
    )
    plt.subplots_adjust(wspace=0.08, top=0.86)

    ax_plot.set_xlim(*XLIM)
    ax_plot.set_ylim(*YLIM)
    ax_plot.set_aspect('equal')
    ax_plot.axhline(0, color='black', lw=0.8, zorder=1)
    ax_plot.axvline(0, color='black', lw=0.8, zorder=1)
    ax_plot.set_xlabel('$X$', fontsize=11)
    ax_plot.set_ylabel('$Y$', fontsize=11)
    ax_plot.grid(True, alpha=0.15)

    # ── State ─────────────────────────────────────────────────────────────
    state = {'v': INITIAL_VECTOR.copy(), 'dragging': False}

    # Dashed right-angle triangle legs (|x| and |y|), redrawn on each update.
    leg_x, = ax_plot.plot([], [], color=COL_TRIANGLE, linestyle='--', linewidth=1.2, zorder=2)
    leg_y, = ax_plot.plot([], [], color=COL_TRIANGLE, linestyle='--', linewidth=1.2, zorder=2)

    # Faint reference arrow showing direction only, at a fixed length —
    # created once via annotate() and updated via remove()/re-add, since
    # FancyArrow / annotate arrows don't support set_data() directly.
    reference_arrow = {'artist': None}

    # Main vector arrow — same approach (annotate with arrowprops), since
    # arrows need to be recreated to change both position and length.
    main_arrow = {'artist': None}

    point_marker, = ax_plot.plot([], [], 'o', color=COL_VECTOR, markersize=9,
                                  markeredgecolor='k', markeredgewidth=0.6, zorder=5)

    legend_handles = [
        plt.Line2D([0], [0], color=COL_VECTOR, linewidth=2, label='Vector v = [x, y]'),
        plt.Line2D([0], [0], color=COL_REFERENCE, linewidth=2,
                   label='Direction only (fixed length)'),
        plt.Line2D([0], [0], color=COL_TRIANGLE, linestyle='--', linewidth=1.2,
                   label='|x| and |y| (for magnitude)'),
    ]
    ax_plot.legend(handles=legend_handles, fontsize=8, loc='upper left',
                   framealpha=1.0, edgecolor='#cccccc')

    def _update():
        v = state['v']
        x, y = v
        magnitude = float(np.linalg.norm(v))

        # ── Main vector arrow ────────────────────────────────────────────
        if main_arrow['artist'] is not None:
            main_arrow['artist'].remove()
        main_arrow['artist'] = ax_plot.annotate(
            '', xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=COL_VECTOR, lw=2.5,
                            mutation_scale=18),
            zorder=4,
        )

        # ── Reference arrow (direction only, fixed length) ────────────────
        if reference_arrow['artist'] is not None:
            reference_arrow['artist'].remove()
        if magnitude > 1e-9:
            ref_x = (x / magnitude) * REFERENCE_LENGTH
            ref_y = (y / magnitude) * REFERENCE_LENGTH
        else:
            ref_x, ref_y = 0.0, 0.0
        reference_arrow['artist'] = ax_plot.annotate(
            '', xy=(ref_x, ref_y), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=COL_REFERENCE, lw=2.5,
                            mutation_scale=14, alpha=0.8),
            zorder=3,
        )

        # ── Right-angle triangle legs ──────────────────────────────────────
        # Horizontal leg: from (0, y) to (x, y) — represents |x|
        # Vertical leg:   from (x, 0) to (x, y) — represents |y|
        leg_x.set_data([0, x], [y, y])
        leg_y.set_data([x, x], [0, y])

        # ── Draggable point marker ─────────────────────────────────────────
        point_marker.set_data([x], [y])

        ann_text.set_text(_annotation_text(v))

        fig.canvas.draw_idle()

    _update()

    # ── Drag interaction ──────────────────────────────────────────────────────
    PICK_RADIUS = 0.4  # in data units

    def _on_press(event):
        if event.inaxes != ax_plot or event.xdata is None:
            return
        click = np.array([event.xdata, event.ydata])
        if np.linalg.norm(click - state['v']) <= PICK_RADIUS:
            state['dragging'] = True

    def _on_motion(event):
        if not state['dragging']:
            return
        if event.inaxes != ax_plot or event.xdata is None:
            return
        state['v'] = np.array([event.xdata, event.ydata])
        _update()

    def _on_release(event):
        state['dragging'] = False

    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_motion)
    fig.canvas.mpl_connect('button_release_event', _on_release)

    # ── Reset button ─────────────────────────────────────────────────────────
    reset_button = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='100px'),
    )

    def _on_reset(button):
        state['v'] = INITIAL_VECTOR.copy()
        state['dragging'] = False
        _update()

    reset_button.on_click(_on_reset)

    display(reset_button)