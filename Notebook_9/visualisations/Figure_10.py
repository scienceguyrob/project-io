"""
Figure 10 — Matrix-Vector Multiplication and Eigenvectors
============================================================
Lets the user drag a point to define a vector v = [x, y], and shows the
result of multiplying it by a fixed matrix M, side by side, with the
matrix-vector multiplication worked out numerically in an annotation
panel.

Three panels:

  - Left panel: the original vector v, drawn as an arrow from the origin.
    Draggable.
  - Middle panel: the transformed vector Mv, drawn as an arrow from the
    origin on its own set of axes (same scale as the left panel).
  - Right panel: an annotation showing the matrix M, the multiplication
    Mv worked out term by term, and a clear statement of whether v is
    (approximately) an eigenvector of M — i.e. whether Mv points in the
    same (or exactly opposite) direction as v.

When v is close to being an eigenvector, both arrows are drawn in a
shared highlight colour and the annotation reports the corresponding
eigenvalue (the ratio ||Mv|| / ||v||, signed according to whether Mv
points the same way as v or the opposite way).

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_10 import show
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


# ── Fixed matrix and initial vector ─────────────────────────────────────────
# Same matrix and vectors used as worked examples in Section 7.1.
M = np.array([[2.0, 1.0],
               [1.0, 2.0]])

INITIAL_VECTOR = np.array([3.0, 1.0])

# Angular tolerance (in degrees) for treating v as "approximately" an
# eigenvector — i.e. how close the angle between v and Mv must be to
# 0 deg (same direction) or 180 deg (opposite direction).
EIGENVECTOR_TOLERANCE_DEG = 3.0

# ── Axis limits ──────────────────────────────────────────────────────────────
LIMIT = 15.0
XLIM = (-LIMIT, LIMIT)
YLIM = (-LIMIT, LIMIT)


def _clamp_vector(v):
    """
    Clamp v so that BOTH v and Mv = M @ v stay within [-LIMIT, LIMIT] on
    each axis.

    Works by binary search on a scaling factor applied to v: scaling v
    towards the origin scales Mv by the same factor (matrix multiplication
    is linear), so a single scalar search is sufficient regardless of
    which of v's or Mv's components is the one that would overflow.
    """
    v = np.asarray(v, dtype=float)
    Mv = M @ v

    max_component = max(
        np.abs(v).max() if v.any() else 0.0,
        np.abs(Mv).max() if Mv.any() else 0.0,
    )

    if max_component <= LIMIT or max_component == 0.0:
        return v

    scale = LIMIT / max_component
    return v * scale

COL_V          = 'steelblue'
COL_MV         = 'seagreen'
COL_EIGEN      = 'goldenrod'


def _angle_between(u, v):
    """
    Angle between two vectors, in degrees, in the range [0, 180].

    Uses the dot product formula cos(theta) = (u . v) / (||u|| ||v||).
    np.clip guards against floating-point values slightly outside
    [-1, 1], which would otherwise make arccos return NaN.
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-9 or norm_v < 1e-9:
        return 0.0
    cos_theta = np.dot(u, v) / (norm_u * norm_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _classify(v, Mv):
    """
    Decide whether v is approximately an eigenvector of M, based on the
    angle between v and Mv.

    Returns (is_eigenvector, eigenvalue_estimate, relationship), where
    relationship is 'same direction', 'opposite direction', or None.
    """
    angle = _angle_between(v, Mv)
    norm_v = np.linalg.norm(v)
    norm_Mv = np.linalg.norm(Mv)

    if angle <= EIGENVECTOR_TOLERANCE_DEG:
        eigenvalue = norm_Mv / norm_v if norm_v > 1e-9 else 0.0
        return True, eigenvalue, 'same direction'
    elif angle >= 180 - EIGENVECTOR_TOLERANCE_DEG:
        eigenvalue = -norm_Mv / norm_v if norm_v > 1e-9 else 0.0
        return True, eigenvalue, 'opposite direction'
    else:
        return False, None, None


def _annotation_text(v, Mv):
    """Build the annotation panel string for the current vector v."""
    x, y = v
    mvx, mvy = Mv

    is_eigen, eigenvalue, relationship = _classify(v, Mv)
    angle = _angle_between(v, Mv)

    lines = [
        "Matrix-vector multiplication",
        "─────────────────────────────────",
        "",
        "M = [ 2  1 ]",
        "    [ 1  2 ]",
        "",
        f"v = [{x:.2f}, {y:.2f}]",
        "",
        "─────────────────────────────────",
        "",
        "Mv = M v",
        "",
        f"  row 1: (2 × {x:.2f}) + (1 × {y:.2f}) = {mvx:.2f}",
        f"  row 2: (1 × {x:.2f}) + (2 × {y:.2f}) = {mvy:.2f}",
        "",
        f"Mv = [{mvx:.2f}, {mvy:.2f}]",
        "",
        "─────────────────────────────────",
        "",
        f"Angle between v and Mv: {angle:.1f}°",
        "",
    ]

    if is_eigen:
        lines.append("[YES] v is (approximately) an eigenvector of M")
        lines.append(f"      Mv points in the {relationship} as v")
        lines.append(f"      eigenvalue ≈ {eigenvalue:.2f}")
    else:
        lines.append("[NO]  v is NOT an eigenvector of M")
        lines.append("      Mv points in a different direction —")
        lines.append("      M has rotated v, not just scaled it")

    return "\n".join(lines)


def show():
    """Render Figure 10: matrix-vector multiplication and eigenvectors."""
    plt.close('Notebook9 Figure 10')

    fig, (ax_v, ax_mv, ax_ann) = plt.subplots(
        1, 3,
        num='Notebook9 Figure 10',
        figsize=(12, 6),
        gridspec_kw={'width_ratios': [1, 1, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    ax_ann.set_axis_off()

    ann_text = ax_ann.text(
        0.04, 0.97, '',
        transform=ax_ann.transAxes,
        fontsize=9.5, va='top', ha='left',
        family='monospace',
        linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    fig.suptitle(
        'Figure 10: Matrix-vector multiplication — drag v and watch Mv\n'
        'Highlighted in gold when v is (approximately) an eigenvector of M',
        fontsize=11,
        y=0.98,
    )
    plt.subplots_adjust(wspace=0.15, top=0.84)

    for ax, title in [(ax_v, 'v (drag me)'), (ax_mv, 'Mv = result')]:
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', lw=0.8, zorder=1)
        ax.axvline(0, color='black', lw=0.8, zorder=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('$X$', fontsize=10)
        ax.set_ylabel('$Y$', fontsize=10)
        ax.grid(True, alpha=0.15)

    # ── State ─────────────────────────────────────────────────────────────
    state = {'v': INITIAL_VECTOR.copy(), 'dragging': False}

    v_arrow  = {'artist': None}
    mv_arrow = {'artist': None}

    v_marker, = ax_v.plot([], [], 'o', color=COL_V, markersize=9,
                           markeredgecolor='k', markeredgewidth=0.6, zorder=5)

    # text_x and text_y are created below, but referenced here — Python
    # closures resolve names at call time, so this is fine as long as
    # _update() is only called after they exist.
    def _update(sync_text=True):
        v = state['v']
        Mv = M @ v

        is_eigen, _, _ = _classify(v, Mv)
        v_colour  = COL_EIGEN if is_eigen else COL_V
        mv_colour = COL_EIGEN if is_eigen else COL_MV

        # ── v arrow (left panel) ─────────────────────────────────────────
        if v_arrow['artist'] is not None:
            v_arrow['artist'].remove()
        v_arrow['artist'] = ax_v.annotate(
            '', xy=(v[0], v[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=v_colour, lw=2.5,
                            mutation_scale=18),
            zorder=4,
        )
        v_marker.set_data([v[0]], [v[1]])

        # ── Mv arrow (middle panel) ───────────────────────────────────────
        if mv_arrow['artist'] is not None:
            mv_arrow['artist'].remove()
        mv_arrow['artist'] = ax_mv.annotate(
            '', xy=(Mv[0], Mv[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=mv_colour, lw=2.5,
                            mutation_scale=18),
            zorder=4,
        )

        ann_text.set_text(_annotation_text(v, Mv))

        # Keep the text boxes in sync with the current vector — e.g. after
        # a drag, or after clamping has adjusted a typed-in value.
        # sync_text=False is used when called FROM a text box's own
        # observer, to avoid the observer retriggering itself.
        if sync_text:
            text_x.value = round(float(v[0]), 3)
            text_y.value = round(float(v[1]), 3)

        fig.canvas.draw_idle()

    # ── Drag interaction (left panel only) ───────────────────────────────────
    PICK_RADIUS = 0.5  # in data units

    def _on_press(event):
        if event.inaxes != ax_v or event.xdata is None:
            return
        click = np.array([event.xdata, event.ydata])
        if np.linalg.norm(click - state['v']) <= PICK_RADIUS:
            state['dragging'] = True

    def _on_motion(event):
        if not state['dragging']:
            return
        if event.inaxes != ax_v or event.xdata is None:
            return
        state['v'] = _clamp_vector([event.xdata, event.ydata])
        _update()

    def _on_release(event):
        state['dragging'] = False

    fig.canvas.mpl_connect('button_press_event', _on_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_motion)
    fig.canvas.mpl_connect('button_release_event', _on_release)

    # ── Text input boxes for x and y ─────────────────────────────────────────
    # Let the user type exact values for v = [x, y] directly, as an
    # alternative to dragging. Values are clamped the same way as drags,
    # so Mv always stays within the visible axes.
    text_x = widgets.FloatText(
        value=float(INITIAL_VECTOR[0]),
        description='x:',
        step=0.5,
        layout=widgets.Layout(width='140px'),
    )
    text_y = widgets.FloatText(
        value=float(INITIAL_VECTOR[1]),
        description='y:',
        step=0.5,
        layout=widgets.Layout(width='140px'),
    )

    def _on_text_change(change):
        new_v = _clamp_vector([text_x.value, text_y.value])
        state['v'] = new_v
        # sync_text=False avoids re-triggering this observer when _update()
        # writes the (possibly clamped) values back into the text boxes.
        _update(sync_text=False)
        text_x.value = round(float(new_v[0]), 3)
        text_y.value = round(float(new_v[1]), 3)

    text_x.observe(_on_text_change, names='value')
    text_y.observe(_on_text_change, names='value')

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

    # Initial draw, now that text_x/text_y exist for _update() to sync with.
    _update()

    display(widgets.HBox([text_x, text_y, reset_button]))