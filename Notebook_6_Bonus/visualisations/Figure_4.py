"""
Figure 4 — Searching for Weights: the MP Neuron Learning AND
================================================================

Shows a small MP neuron (two inputs, one output) together with the full
AND truth table, so the reader can experience "searching for weights"
directly rather than reading about it.

The diagram has two parts:

1. A small network diagram (left) showing the two input nodes, the
   neuron body, and the output arrow, identical in spirit to Figure 2.
   There are no input sliders here — x_1 and x_2 are not adjustable,
   since the whole point of this figure is to search over weights and
   threshold, not over inputs. The diagram exists purely to show the
   structure of the neuron whose parameters the sliders control.

2. The full AND truth table (right), always showing all four input
   combinations at once, each with its target output, its weighted sum z
   under the current weights, the neuron's predicted output, and a
   green tick or red cross showing whether that row is currently
   correct. This table updates live as the weight and threshold sliders
   move, so the reader can see immediately which rows a given choice of
   weights gets right and which it gets wrong. A one-line summary beneath
   the table reports the total number of rows currently wrong.

Sliders control w_1, w_2 (from -2 to 2) and theta (from -2 to 4). A
"Reset" button returns all three to the values used in the worked AND
example in Section 7.2.

The key teaching point: learning is a search through the space of
possible weights and thresholds. As the reader moves the sliders, they
should see the tick/cross pattern in the table change, making it
concrete that most combinations of weights and threshold get at least
one row wrong, and that finding a combination with zero errors is
exactly what "learning AND" means for this neuron.

Usage
-----
From a Jupyter notebook cell::

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
from matplotlib.patches import FancyArrowPatch, Circle
import ipywidgets as widgets
from ipywidgets import interactive_output, FloatSlider, VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the interactive Figure 4 weight-search-over-AND explorer."""

    plt.close('Notebook6bonus Figure 4')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_INPUT     = '#4C78A8'
    COL_INPUT_E   = '#2c4a66'
    COL_NEURON    = '#E45756'
    COL_NEURON_E  = '#8c2f30'
    COL_FIRE      = '#54A24B'
    COL_FIRE_E    = '#2f5c2a'
    COL_SILENT    = '#9c9c9c'
    COL_SILENT_E  = '#5e5e5e'
    COL_TEXT      = '#222222'
    COL_ARROW     = '#666666'
    COL_ARROW_NEG = '#b23b3b'
    COL_WRONG     = '#b23b3b'

    # The tick and cross are drawn as small line-based glyphs rather than
    # the unicode check-mark/cross characters. Some renderer/font
    # combinations (in particular the ipympl backend saving to PNG with a
    # plain Arial fallback) do not have a glyph for U+2713 CHECK MARK and
    # print a "Glyph missing from font(s)" warning, silently drawing
    # nothing in its place. Drawing the symbols as plot lines instead
    # sidesteps the font issue entirely and renders identically everywhere.
    def draw_tick(ax, x, y, size=0.018, color=COL_FIRE_E):
        ln, = ax.plot([x - size, x - size * 0.25, x + size],
                       [y - size * 0.05, y - size * 0.9, y + size * 0.9],
                       color=color, lw=2.4, solid_capstyle='round',
                       transform=ax.transAxes, zorder=4)
        return [ln]

    def draw_cross(ax, x, y, size=0.014, color=COL_WRONG):
        ln1, = ax.plot([x - size, x + size], [y - size, y + size], color=color,
                        lw=2.4, solid_capstyle='round', transform=ax.transAxes,
                        zorder=4)
        ln2, = ax.plot([x - size, x + size], [y + size, y - size], color=color,
                        lw=2.4, solid_capstyle='round', transform=ax.transAxes,
                        zorder=4)
        return [ln1, ln2]

    AND_TABLE = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    COL_LABELS = ['$x_1$', '$x_2$', 'target', 'z', 'predicted', 'correct?']

    FIXED_X = (1.0, 1.0)   # input pair traced in the network diagram (not user-adjustable)
    DEFAULT_W = [1.0, 1.0]
    DEFAULT_THETA = 1.5

    # ── Figure layout: network (left), table (right) ────────────────────────
    fig, (ax_net, ax_table) = plt.subplots(
        1, 2, num='Notebook6bonus Figure 4', figsize=(12.5, 5.2),
        gridspec_kw={'width_ratios': [1.0, 1.4]}, constrained_layout=True)

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    fig.suptitle('Figure 4: Searching for weights — the MP neuron learning AND',
                 fontsize=13, color=COL_TEXT)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 1: small static network diagram (structure only; x1, x2 fixed)
    # ══════════════════════════════════════════════════════════════════════
    ax_net.set_xlim(-1.2, 6.2)
    ax_net.set_ylim(-2.6, 2.6)
    ax_net.set_aspect('equal')
    ax_net.axis('off')
    ax_net.set_title('MP neuron structure', fontsize=11.5, color=COL_TEXT)

    in1_pos = np.array([0.4, 1.3])
    in2_pos = np.array([0.4, -1.3])
    neuron_c = np.array([3.1, 0.0])
    neuron_r = 1.0
    out_x = 5.6

    for pos, lab in zip([in1_pos, in2_pos], ['$x_1$', '$x_2$']):
        ax_net.add_patch(Circle(pos, 0.32, facecolor=COL_INPUT,
                                 edgecolor=COL_INPUT_E, lw=1.3, zorder=5))
        ax_net.text(*pos, lab, ha='center', va='center', color='white',
                    fontsize=11, fontweight='bold', zorder=6)

    neuron_patch = Circle(neuron_c, neuron_r, facecolor=COL_NEURON, alpha=0.85,
                           edgecolor=COL_NEURON_E, lw=1.6, zorder=5)
    ax_net.add_patch(neuron_patch)
    neuron_z_txt = ax_net.text(neuron_c[0], neuron_c[1] + 0.25, '', fontsize=10.5,
                                ha='center', va='center', color='white', zorder=6)
    neuron_theta_txt = ax_net.text(neuron_c[0], neuron_c[1] - 0.25, '', fontsize=10,
                                    ha='center', va='center', color='white', zorder=6)

    w_arrows, w_texts = [], []
    for pos, wlab in zip([in1_pos, in2_pos], [r'$w_1$', r'$w_2$']):
        d = neuron_c - pos
        d = d / np.linalg.norm(d)
        start = pos + d * 0.32
        end = neuron_c - d * neuron_r
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                 color=COL_ARROW, lw=1.4, zorder=3,
                                 shrinkA=0, shrinkB=0)
        ax_net.add_patch(arrow)
        mid = (start + end) / 2
        perp = np.array([-d[1], d[0]]) * 0.3
        txt = ax_net.text(mid[0] + perp[0], mid[1] + perp[1], wlab, fontsize=10.5,
                           ha='center', va='center', color=COL_TEXT)
        w_arrows.append(arrow)
        w_texts.append(txt)

    out_start = neuron_c + np.array([neuron_r, 0])
    out_arrow = FancyArrowPatch(out_start, (out_x, 0), arrowstyle='-|>',
                                 mutation_scale=18, color=COL_SILENT, lw=2.2,
                                 zorder=3)
    ax_net.add_patch(out_arrow)
    out_label = ax_net.text((out_start[0] + out_x) / 2, 0.4, '', fontsize=11,
                             ha='center', fontweight='bold')

    ax_net.text(neuron_c[0], -2.2,
                f'(diagram fixed at $x_1$={FIXED_X[0]:.0f}, $x_2$={FIXED_X[1]:.0f})',
                ha='center', fontsize=8.5, color='#888888')

    # ══════════════════════════════════════════════════════════════════════
    # Panel 2: full AND truth table, evaluated live against current weights
    # ══════════════════════════════════════════════════════════════════════
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')
    ax_table.set_title('AND truth table (live)', fontsize=11.5, color=COL_TEXT)

    n_cols = len(COL_LABELS)
    n_rows = len(AND_TABLE)
    row_h = 1.0 / (n_rows + 1.8)

    for j, lab in enumerate(COL_LABELS):
        ax_table.text((j + 0.5) / n_cols, 1 - 0.6 * row_h, lab, ha='center',
                       va='center', fontweight='bold', fontsize=10.5)
    ax_table.plot([0.02, 0.98], [1 - 1.15 * row_h] * 2, color='#bbbbbb', lw=1.0)

    row_cells = []
    mark_positions = []   # axes-fraction (x, y) for each row's tick/cross glyph
    for i, (x1, x2, target) in enumerate(AND_TABLE):
        y = 1 - (i + 1.9) * row_h
        cells = {
            'x1': ax_table.text((0 + 0.5) / n_cols, y, str(x1), ha='center',
                                 va='center', fontsize=10.5),
            'x2': ax_table.text((1 + 0.5) / n_cols, y, str(x2), ha='center',
                                 va='center', fontsize=10.5),
            'target': ax_table.text((2 + 0.5) / n_cols, y, str(target),
                                     ha='center', va='center', fontsize=10.5),
            'z': ax_table.text((3 + 0.5) / n_cols, y, '', ha='center',
                                va='center', fontsize=10.5, color='#555555'),
            'pred': ax_table.text((4 + 0.5) / n_cols, y, '', ha='center',
                                   va='center', fontsize=10.5),
        }
        row_cells.append(cells)
        mark_positions.append(((5 + 0.5) / n_cols, y))

    error_summary_txt = ax_table.text(0.5, 1 - (n_rows + 2.6) * row_h, '',
                                       ha='center', va='center', fontsize=11.5,
                                       fontweight='bold')

    mark_artists = []   # cleared and redrawn each update, since they're plot lines

    # ── Update function ───────────────────────────────────────────────────────
    def update(w1, w2, theta):
        x1, x2 = FIXED_X
        z_sel = w1 * x1 + w2 * x2
        fires_sel = z_sel >= theta

        for arrow, txt, w, x in zip(w_arrows, w_texts, [w1, w2], [x1, x2]):
            is_active = x > 0
            colour = COL_ARROW_NEG if w < 0 else COL_ARROW
            arrow.set_color(colour if is_active else '#cccccc')
            arrow.set_linewidth(1.0 + 1.6 * min(abs(w), 2.0))
            txt.set_color(COL_TEXT if is_active else '#bbbbbb')

        neuron_z_txt.set_text(rf'$z = {z_sel:.2f}$')
        neuron_theta_txt.set_text(rf'$\theta = {theta:.2f}$')

        if fires_sel:
            out_arrow.set_color(COL_FIRE)
            out_label.set_text('$y = 1$')
            out_label.set_color(COL_FIRE_E)
        else:
            out_arrow.set_color(COL_SILENT)
            out_label.set_text('$y = 0$')
            out_label.set_color(COL_SILENT_E)

        # clear previous tick/cross glyphs before redrawing
        for artist in mark_artists:
            artist.remove()
        mark_artists.clear()

        n_errors = 0
        for (rx1, rx2, target), cells, (mx, my) in zip(AND_TABLE, row_cells,
                                                         mark_positions):
            z = w1 * rx1 + w2 * rx2
            pred = 1 if z >= theta else 0
            correct = (pred == target)
            if not correct:
                n_errors += 1

            cells['z'].set_text(f'{z:.2f}')
            cells['pred'].set_text(str(pred))

            if correct:
                mark_artists.extend(draw_tick(ax_table, mx, my))
            else:
                mark_artists.extend(draw_cross(ax_table, mx, my))

        if n_errors == 0:
            error_summary_txt.set_text('0 errors — this neuron correctly computes AND')
            error_summary_txt.set_color(COL_FIRE_E)
        else:
            row_word = 'row' if n_errors == 1 else 'rows'
            error_summary_txt.set_text(f'{n_errors} of 4 {row_word} wrong')
            error_summary_txt.set_color(COL_WRONG)

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    w1_s = FloatSlider(value=DEFAULT_W[0], min=-2.0, max=2.0, step=0.1,
                        description='w_1', style={'description_width': '55px'},
                        layout=widgets.Layout(width='260px'))
    w2_s = FloatSlider(value=DEFAULT_W[1], min=-2.0, max=2.0, step=0.1,
                        description='w_2', style={'description_width': '55px'},
                        layout=widgets.Layout(width='260px'))
    theta_s = FloatSlider(value=DEFAULT_THETA, min=-2.0, max=4.0, step=0.1,
                           description='theta',
                           style={'description_width': '55px'},
                           layout=widgets.Layout(width='260px'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                                layout=widgets.Layout(width='100px'))

    def on_reset(b):
        w1_s.value, w2_s.value = DEFAULT_W
        theta_s.value = DEFAULT_THETA

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Search for weights that solve AND</b>'),
        HBox([VBox([HTML('<i>Weights</i>'), w1_s, w2_s]),
              VBox([HTML('<i>Threshold</i>'), theta_s])]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {'w1': w1_s, 'w2': w2_s, 'theta': theta_s})

    display(controls, out)