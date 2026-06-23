"""
Figure 8 — Interactive Cosine Similarity Explorer
==================================================
Visualises cosine similarity as the angle between two word-count vectors
in 2D document space.

Two document vectors are displayed as arrows from the origin. The user can
edit the word counts for each document using sliders (one per word, up to
three words). The figure shows:

  Left panel  — the two vectors as arrows in 2D. The x-axis represents
                the first word dimension; the y-axis the second. The third
                word dimension contributes to the dot product and norms
                shown in the annotation but is not displayed spatially,
                keeping the plot readable.

  Right panel — step-by-step annotation showing the dot product, norms,
                and final cosine similarity with current values substituted.

The opening state reproduces the document A / document B example from the
text: A = (1, 2, 1), B = (3, 6, 3), cosine similarity = 1.0.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_8 import show
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
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display


# ── Vocabulary ────────────────────────────────────────────────────────────────
WORDS = ['"data"', '"model"', '"learn"']

# ── Initial vectors (from the worked example in the markdown) ─────────────────
A_INIT = [1, 2, 1]
B_INIT = [3, 6, 3]

# ── Colours ───────────────────────────────────────────────────────────────────
COL_A   = 'steelblue'
COL_B   = 'tomato'
COL_ANG = 'seagreen'


def _cosine(a, b):
    """Return dot product, norms, and cosine similarity."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot = float(np.dot(a, b))
    na  = float(np.linalg.norm(a))
    nb  = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return dot, na, nb, 0.0
    return dot, na, nb, dot / (na * nb)


def _annotation(a, b):
    """Step-by-step cosine similarity calculation string."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot, na, nb, cos = _cosine(a, b)

    dot_terms = ' + '.join(
        f'({int(a[i])}×{int(b[i])})' for i in range(len(a))
    )
    dot_vals = ' + '.join(
        f'{int(a[i] * b[i])}' for i in range(len(a))
    )
    na_terms = ' + '.join(f'{a[i]:.0f}²' for i in range(len(a)))
    nb_terms = ' + '.join(f'{b[i]:.0f}²' for i in range(len(b)))

    angle_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    return (
        "Cosine similarity\n"
        "─────────────────────────────────\n\n"
        r"$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}"
        r"{\|\mathbf{a}\| \, \|\mathbf{b}\|}$"
        "\n\n"
        "Step 1: dot product\n"
        f"$\\mathbf{{a}} \\cdot \\mathbf{{b}}$\n"
        f"$= {dot_terms}$\n"
        f"$= {dot_vals} = {dot:.2f}$\n\n"
        "Step 2: norms\n"
        f"$\\|\\mathbf{{a}}\\| = \\sqrt{{{na_terms}}}$\n"
        f"$= \\sqrt{{{na**2:.2f}}} = {na:.3f}$\n\n"
        f"$\\|\\mathbf{{b}}\\| = \\sqrt{{{nb_terms}}}$\n"
        f"$= \\sqrt{{{nb**2:.2f}}} = {nb:.3f}$\n\n"
        "Step 3: combine\n"
        f"$\\cos(\\theta) = \\frac{{{dot:.2f}}}"
        f"{{{na:.3f} \\times {nb:.3f}}}$\n\n"
        f"$= \\mathbf{{{cos:.3f}}}$\n\n"
        f"Angle: $\\theta = {angle_deg:.1f}°$\n\n"
        "─────────────────────────────────\n\n"
        "Cosine distance\n"
        f"$d = 1 - {cos:.3f} = \\mathbf{{{1 - cos:.3f}}}$"
    )


def _draw_vectors(ax, a, b):
    """
    Draw both vectors as arrows in 2D using the first two word dimensions
    as fixed world-space axes. The third dimension contributes to the
    dot product and norms in the annotation but is not plotted, so that
    neither arrow moves when only the other vector's sliders are changed.

    Labels for A, B, and the angle are placed at fixed positions in axes
    coordinates below the plot area so they never overlap regardless of
    vector direction or magnitude.
    """
    ax.clear()

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    # Use only the first two components for the 2D display. These axes are
    # fixed to the word dimensions and do not change when slider values change,
    # so each arrow responds only to its own sliders.
    a2d = (a[0], a[1])
    b2d = (b[0], b[1])

    dot, na, nb, cos = _cosine(a, b)
    angle_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    arrowprops = dict(arrowstyle='->', lw=2.5,
                      mutation_scale=20, shrinkA=0, shrinkB=0)

    # Vector arrows
    ax.annotate(
        '', xy=a2d, xytext=(0, 0),
        arrowprops={**arrowprops, 'color': COL_A},
    )
    ax.annotate(
        '', xy=b2d, xytext=(0, 0),
        arrowprops={**arrowprops, 'color': COL_B},
    )

    # ── Fixed-position labels in axes coordinates ─────────────────────────────
    # Placed below the plot area so they never overlap the arrows or each other
    # regardless of vector direction or magnitude.
    ax.text(
        0.05, -0.16,
        f'A = {[int(x) for x in a]}   $\\|\\mathbf{{a}}\\|$ = {na:.2f}',
        color=COL_A, fontsize=9, ha='left', va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COL_A, alpha=0.9, lw=1.2),
    )
    ax.text(
        0.55, -0.16,
        f'B = {[int(x) for x in b]}   $\\|\\mathbf{{b}}\\|$ = {nb:.2f}',
        color=COL_B, fontsize=9, ha='left', va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COL_B, alpha=0.9, lw=1.2),
    )

    # Angle label — fixed position below vector labels
    ax.text(
        0.05, -0.27,
        f'$\\theta = {angle_deg:.1f}°$     '
        f'$\\cos\\theta = {cos:.3f}$     '
        f'distance $= {1 - cos:.3f}$',
        color=COL_ANG, fontsize=9, ha='left', va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=COL_ANG, alpha=0.9, lw=1.2),
    )

    # Arc showing the angle between the two 2D projections
    if angle_deg > 0.5:
        r_arc = min(np.linalg.norm(a2d), np.linalg.norm(b2d)) * 0.35
        if r_arc > 0.05:
            theta1 = float(np.degrees(np.arctan2(a2d[1], a2d[0])))
            theta2 = float(np.degrees(np.arctan2(b2d[1], b2d[0])))
            if theta1 > theta2:
                theta1, theta2 = theta2, theta1
            arc = mpatches.Arc(
                (0, 0), 2 * r_arc, 2 * r_arc,
                angle=0, theta1=theta1, theta2=theta2,
                color=COL_ANG, lw=1.8,
            )
            ax.add_patch(arc)

    # Origin dot
    ax.scatter(0, 0, s=60, color='black', zorder=5)

    # Axis limits driven by the longer of the two vectors
    lim = max(np.linalg.norm(a2d), np.linalg.norm(b2d), 1.0) * 1.5 + 1.0
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.axhline(0, color='#ccc', lw=0.8, ls=':')
    ax.axvline(0, color='#ccc', lw=0.8, ls=':')
    ax.set_xlabel(f'Word axis 1: {WORDS[0]}', fontsize=10)
    ax.set_ylabel(f'Word axis 2: {WORDS[1]}', fontsize=10)
    ax.set_title(
        'Document vectors in word-count space\n'
        f'(third dimension {WORDS[2]} contributes to '
        'calculation but is not plotted)',
        fontsize=10,
    )
    ax.grid(True, alpha=0.15)

    handles = [
        mpatches.Patch(color=COL_A,   label='Document A'),
        mpatches.Patch(color=COL_B,   label='Document B'),
        mpatches.Patch(color=COL_ANG, label='Angle θ'),
    ]
    ax.legend(handles=handles, fontsize=9, loc='upper right',
              framealpha=1.0, edgecolor='#cccccc')


def show():
    """Render Figure 8: interactive cosine similarity explorer."""
    plt.close('Notebook8 Figure 8')

    fig, (ax_vec, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 8',
        figsize=(10, 8),
        gridspec_kw={'width_ratios': [1.3, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # ── Annotation panel ──────────────────────────────────────────────────────
    ax_ann.set_axis_off()
    ann_text = ax_ann.text(
        0.04, 0.97,
        _annotation(A_INIT, B_INIT),
        transform=ax_ann.transAxes,
        fontsize=9, va='top', ha='left',
        linespacing=1.65,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    _draw_vectors(ax_vec, A_INIT, B_INIT)

    _, _, _, cos_init = _cosine(A_INIT, B_INIT)
    fig_title = fig.suptitle(
        f'Figure 8: Cosine similarity = {cos_init:.3f}   '
        f'Cosine distance = {1 - cos_init:.3f}',
        fontsize=11,
    )
    plt.subplots_adjust(wspace=0.1, top=0.88, bottom=0.22)

    # ── Widgets ───────────────────────────────────────────────────────────────
    # Six sliders: word counts for each of the three words in each document.
    # Initial values reproduce the worked example from the markdown text.
    sliders_a = [
        widgets.IntSlider(
            value=A_INIT[i], min=0, max=10, step=1,
            description=f'A: {WORDS[i]}',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='340px'),
        )
        for i in range(3)
    ]
    sliders_b = [
        widgets.IntSlider(
            value=B_INIT[i], min=0, max=10, step=1,
            description=f'B: {WORDS[i]}',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='340px'),
        )
        for i in range(3)
    ]

    def _update(a0, a1, a2, b0, b1, b2):
        a = [a0, a1, a2]
        b = [b0, b1, b2]
        _, _, _, cos = _cosine(a, b)
        _draw_vectors(ax_vec, a, b)
        ann_text.set_text(_annotation(a, b))
        fig_title.set_text(
            f'Figure 8: Cosine similarity = {cos:.3f}   '
            f'Cosine distance = {1 - cos:.3f}'
        )
        fig.canvas.draw_idle()

    out = interactive_output(
        _update,
        {
            'a0': sliders_a[0], 'a1': sliders_a[1], 'a2': sliders_a[2],
            'b0': sliders_b[0], 'b1': sliders_b[1], 'b2': sliders_b[2],
        },
    )

    col_a = widgets.VBox(sliders_a)
    col_b = widgets.VBox(sliders_b)
    controls = widgets.HBox([col_a, col_b])
    display(controls, out)