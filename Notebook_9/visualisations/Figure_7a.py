"""
Figure 7a — Variance and Covariance, Live
============================================
Lets the user drag points in a small 2D dataset and watch Var(X), Var(Y),
and Cov(X, Y) recompute live, with each term's contribution shown directly
on the scatter plot.

A small set of draggable points is shown on a scatter plot, with dotted
guide lines through the mean of X and the mean of Y, dividing the plot
into four quadrants. As a point is dragged:

  - Its deviation from the mean in X and in Y is shown as a pair of
    dashed lines (one horizontal, one vertical) connecting it back to
    the mean lines.
  - The point is coloured according to the SIGN of its contribution to
    Cov(X, Y) — i.e. the sign of (Xi - X_bar)(Yi - Y_bar) — so points in
    the top-right and bottom-left quadrants (positive contribution) are
    one colour, and points in the top-left and bottom-right quadrants
    (negative contribution) are another.
  - An annotation panel recomputes Var(X), Var(Y), and Cov(X, Y) from
    scratch using the current point positions, and shows the running
    sums term by term.

This figure is deliberately small (8 points) so that every term in the
sums shown in the annotation panel can be related back to a specific,
visible point on the plot.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_7a import show
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


# ── Initial dataset ──────────────────────────────────────────────────────────
# 8 points, deliberately spread across all four quadrants relative to their
# own mean, so that both positive and negative covariance contributions are
# visible from the start.
INITIAL_POINTS = np.array([
    [1.0, 1.2],
    [2.0, 1.8],
    [3.0, 3.0],
    [1.5, 2.6],
    [4.0, 1.0],
    [0.5, 2.2],
    [3.5, 3.4],
    [2.5, 0.6],
])

PAD = 1.0
XLIM = (INITIAL_POINTS[:, 0].min() - PAD, INITIAL_POINTS[:, 0].max() + PAD)
YLIM = (INITIAL_POINTS[:, 1].min() - PAD, INITIAL_POINTS[:, 1].max() + PAD)

COL_POS = 'steelblue'   # points whose (Xi-X_bar)(Yi-Y_bar) > 0
COL_NEG = 'tomato'      # points whose (Xi-X_bar)(Yi-Y_bar) < 0
COL_ZERO = '#999999'    # points exactly on a mean line


def _variance(values):
    """Sample variance, Bessel-corrected — same formula as Section 6.1."""
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _covariance(x, y):
    """Sample covariance — same formula as Section 6.2."""
    n = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n
    return sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y)) / (n - 1)


def _annotation_text(points):
    """Build the annotation panel string showing the live calculation."""
    x = points[:, 0]
    y = points[:, 1]
    n = len(x)
    x_bar = x.mean()
    y_bar = y.mean()

    var_x = _variance(x)
    var_y = _variance(y)
    cov_xy = _covariance(x, y)

    lines = [
        "Live calculation",
        "─────────────────────────────────",
        "",
        f"n = {n}    X̄ = {x_bar:.2f}    Ȳ = {y_bar:.2f}",
        "",
        "─────────────────────────────────",
        "",
        f"Var(X) = {var_x:.3f}",
        f"Var(Y) = {var_y:.3f}",
        f"Cov(X,Y) = {cov_xy:.3f}",
        "",
        "─────────────────────────────────",
        "",
        "Per-point contributions to Cov(X,Y):",
        "(Xi - X̄)(Yi - Ȳ)",
        "",
    ]

    for i, (xi, yi) in enumerate(zip(x, y)):
        dx = xi - x_bar
        dy = yi - y_bar
        product = dx * dy
        sign = '+' if product > 0 else ('-' if product < 0 else '0')
        lines.append(f"  P{i+1}: ({dx:+.2f})({dy:+.2f}) = {product:+.2f}  [{sign}]")

    return "\n".join(lines)


def show():
    """Render Figure 7a: live variance/covariance with draggable points."""
    plt.close('Notebook9 Figure 7a')

    # Reset button is created and displayed first, so it appears ABOVE
    # the figure in the notebook's output.
    reset_button = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='100px'),
    )
    display(reset_button)

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook9 Figure 7a',
        figsize=(10, 6.5),
        gridspec_kw={'width_ratios': [1.4, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    ax_ann.set_axis_off()

    ann_text = ax_ann.text(
        0.02, 0.98, '',
        transform=ax_ann.transAxes,
        fontsize=8.5, va='top', ha='left',
        family='monospace',
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 7a: Variance and covariance, live —\n'
        'drag any point and watch the calculation update',
        fontsize=11,
        y=0.98,
    )
    plt.subplots_adjust(wspace=0.05, top=0.86)

    ax_plot.set_xlim(*XLIM)
    ax_plot.set_ylim(*YLIM)
    ax_plot.set_xlabel('$X$', fontsize=11)
    ax_plot.set_ylabel('$Y$', fontsize=11)
    ax_plot.grid(True, alpha=0.15)

    # ── State ─────────────────────────────────────────────────────────────
    state = {'points': INITIAL_POINTS.copy(), 'dragging': None}

    # Mean lines — redrawn (not just moved) on each update, since the mean
    # itself changes as points move.
    mean_vline = ax_plot.axvline(0, color='#444444', linestyle='--', linewidth=1.2, zorder=2)
    mean_hline = ax_plot.axhline(0, color='#444444', linestyle='--', linewidth=1.2, zorder=2)

    # Deviation guide lines for the point currently being dragged —
    # created empty and updated in place.
    dev_vline, = ax_plot.plot([], [], color='#888888', linestyle=':', linewidth=1.2, zorder=2)
    dev_hline, = ax_plot.plot([], [], color='#888888', linestyle=':', linewidth=1.2, zorder=2)

    scatter_artist = ax_plot.scatter(
        state['points'][:, 0], state['points'][:, 1],
        s=70, edgecolors='k', lw=0.6, zorder=4,
    )

    point_labels = [
        ax_plot.annotate(f'P{i+1}', pt, textcoords='offset points',
                          xytext=(6, 6), fontsize=9)
        for i, pt in enumerate(state['points'])
    ]

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_POS,
                   markeredgecolor='k', markersize=9,
                   label='Positive contribution to Cov(X,Y)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_NEG,
                   markeredgecolor='k', markersize=9,
                   label='Negative contribution to Cov(X,Y)'),
        plt.Line2D([0], [0], color='#444444', linestyle='--', linewidth=1.2,
                   label='Mean of X / mean of Y'),
    ]
    ax_plot.legend(handles=legend_handles, fontsize=8, loc='upper left',
                   framealpha=1.0, edgecolor='#cccccc')

    def _update(highlight_idx=None):
        points = state['points']
        x = points[:, 0]
        y = points[:, 1]
        x_bar = x.mean()
        y_bar = y.mean()

        # Move the mean lines to their current positions
        mean_vline.set_xdata([x_bar, x_bar])
        mean_hline.set_ydata([y_bar, y_bar])

        # Colour each point by the SIGN of its contribution to Cov(X,Y)
        colours = []
        for xi, yi in zip(x, y):
            product = (xi - x_bar) * (yi - y_bar)
            if product > 0:
                colours.append(COL_POS)
            elif product < 0:
                colours.append(COL_NEG)
            else:
                colours.append(COL_ZERO)

        scatter_artist.set_offsets(points)
        scatter_artist.set_color(colours)

        for label, pt in zip(point_labels, points):
            label.set_position((6, 6))
            label.xy = pt

        # Deviation guide lines for the highlighted (dragged) point only
        if highlight_idx is not None:
            xi, yi = points[highlight_idx]
            dev_vline.set_data([xi, xi], [y_bar, yi])
            dev_hline.set_data([x_bar, xi], [yi, yi])
        else:
            dev_vline.set_data([], [])
            dev_hline.set_data([], [])

        ann_text.set_text(_annotation_text(points))

        fig.canvas.draw_idle()

    _update()

    # ── Drag interaction ──────────────────────────────────────────────────────
    PICK_RADIUS = 0.25  # in data units

    def _on_press(event):
        if event.inaxes != ax_plot or event.xdata is None:
            return
        click = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(state['points'] - click, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= PICK_RADIUS:
            state['dragging'] = nearest

    def _on_motion(event):
        if state['dragging'] is None:
            return
        if event.inaxes != ax_plot or event.xdata is None:
            return
        state['points'][state['dragging']] = [event.xdata, event.ydata]
        _update(highlight_idx=state['dragging'])

    def _on_release(event):
        state['dragging'] = None
        _update()

    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_motion)
    fig.canvas.mpl_connect('button_release_event', _on_release)

    # ── Reset button callback ────────────────────────────────────────────────
    # Restores the original dataset (a fresh copy, so the constant
    # INITIAL_POINTS is never mutated by dragging).
    def _on_reset(button):
        state['points'] = INITIAL_POINTS.copy()
        state['dragging'] = None
        _update()

    reset_button.on_click(_on_reset)