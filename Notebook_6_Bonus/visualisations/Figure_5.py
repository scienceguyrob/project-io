"""
Figure 5 — Searching for Weights: a Two-Hidden-Neuron Network Learning XOR
=============================================================================

Shows the three-neuron network from Section 9 (two hidden neurons, h_1 and
h_2, feeding a single output neuron y) as a diagram, together with a live
XOR truth table, so the reader can explore the same "search for working
parameters" experience as Figure 4, but for the larger network needed for
XOR.

The diagram has two parts:

1. A network diagram (left) showing the two inputs, the two hidden
   neurons, and the output neuron, with every connection drawn and
   labelled with its current weight. Each neuron also displays its
   current threshold. This mirrors the structure introduced in
   Section 9.2, with h_1 playing the OR-like role and h_2 playing the
   AND-like role by default, though the reader is free to move every
   slider away from those defaults.

2. The full XOR truth table (right), always showing all four input
   combinations at once, each with the value of h_1, h_2, the output y,
   the target, and a green tick or red cross showing whether that row is
   currently correct. A one-line summary beneath the table reports the
   total number of rows wrong.

Nine sliders give full control over the network: w11, w12 (inputs into
h_1), theta_1 (threshold of h_1), w21, w22 (inputs into h_2), theta_2
(threshold of h_2), v1, v2 (h_1 and h_2 into the output neuron), and
theta_y (threshold of the output neuron). A "Reset" button returns all
nine to the worked values from Section 9.2, which solve XOR exactly.

The key teaching point: solving XOR requires getting nine separate
numbers right at once, not just two or three as with AND or OR. Moving
any single slider away from its working value is usually enough to break
at least one row of the truth table. The figure is meant to make that
fragility, and the size of the search problem, tangible by contrast with
the much smaller search involved in Figure 4.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_5 import show
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
    """Render the interactive Figure 5 XOR-network weight-search explorer."""

    plt.close('Notebook6bonus Figure 5')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_INPUT     = '#4C78A8'
    COL_INPUT_E   = '#2c4a66'
    COL_HIDDEN    = '#F2B701'
    COL_HIDDEN_E  = '#9c7a00'
    COL_OUTPUT    = '#54A24B'
    COL_OUTPUT_E  = '#2f5c2a'
    COL_TEXT      = '#222222'
    COL_POS       = '#666666'
    COL_NEG       = '#b23b3b'
    COL_FIRE_E    = '#2f5c2a'
    COL_WRONG     = '#b23b3b'

    # Tick/cross drawn as plot lines rather than unicode glyphs, since some
    # renderer/font combinations are missing the U+2713 CHECK MARK glyph
    # and silently drop it with a console warning instead of drawing
    # anything. Lines render identically everywhere.
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

    XOR_TABLE = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    COL_LABELS = ['$x_1$', '$x_2$', '$h_1$', '$h_2$', '$y$', 'target', 'correct?']

    # Defaults: the worked solution from Section 9.2
    DEFAULTS = dict(w11=1.0, w12=1.0, theta1=0.5,
                     w21=1.0, w22=1.0, theta2=1.5,
                     v1=1.0, v2=-2.0, thetay=0.5)

    # ── Figure layout: network (left), table (right) ────────────────────────
    fig, (ax_net, ax_table) = plt.subplots(
        1, 2, num='Notebook6bonus Figure 5', figsize=(13.5, 6.4),
        gridspec_kw={'width_ratios': [1.15, 1.0]}, constrained_layout=True)

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    fig.suptitle('Figure 5: Searching for weights — a two-hidden-neuron '
                 'network learning XOR', fontsize=13, color=COL_TEXT)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 1: network diagram
    # ══════════════════════════════════════════════════════════════════════
    ax_net.set_xlim(-1, 11)
    ax_net.set_ylim(-4.2, 4.2)
    ax_net.set_aspect('equal')
    ax_net.axis('off')
    ax_net.set_title('Network structure', fontsize=11.5, color=COL_TEXT)

    in1 = np.array([0.5, 2.1])
    in2 = np.array([0.5, -2.1])
    h1c = np.array([4.6, 2.3])
    h2c = np.array([4.6, -2.3])
    outc = np.array([8.7, 0.0])
    r_in, r_hidden, r_out = 0.35, 0.58, 0.62

    for pos, lab in zip([in1, in2], ['$x_1$', '$x_2$']):
        ax_net.add_patch(Circle(pos, r_in, facecolor=COL_INPUT,
                                 edgecolor=COL_INPUT_E, lw=1.3, zorder=5))
        ax_net.text(*pos, lab, ha='center', va='center', color='white',
                    fontsize=11, fontweight='bold', zorder=6)

    h1_patch = Circle(h1c, r_hidden, facecolor=COL_HIDDEN, edgecolor=COL_HIDDEN_E,
                       lw=1.4, zorder=5)
    h2_patch = Circle(h2c, r_hidden, facecolor=COL_HIDDEN, edgecolor=COL_HIDDEN_E,
                       lw=1.4, zorder=5)
    ax_net.add_patch(h1_patch)
    ax_net.add_patch(h2_patch)
    ax_net.text(h1c[0], h1c[1] + 0.18, '$h_1$', ha='center', va='center',
                color='white', fontsize=11, fontweight='bold', zorder=6)
    ax_net.text(h2c[0], h2c[1] + 0.18, '$h_2$', ha='center', va='center',
                color='white', fontsize=11, fontweight='bold', zorder=6)
    h1_theta_txt = ax_net.text(h1c[0], h1c[1] - 0.22, '', ha='center', va='center',
                                color='white', fontsize=8.5, zorder=6)
    h2_theta_txt = ax_net.text(h2c[0], h2c[1] - 0.22, '', ha='center', va='center',
                                color='white', fontsize=8.5, zorder=6)

    out_patch = Circle(outc, r_out, facecolor=COL_OUTPUT, edgecolor=COL_OUTPUT_E,
                        lw=1.4, zorder=5)
    ax_net.add_patch(out_patch)
    ax_net.text(outc[0], outc[1] + 0.2, '$y$', ha='center', va='center',
                color='white', fontsize=12, fontweight='bold', zorder=6)
    outy_theta_txt = ax_net.text(outc[0], outc[1] - 0.22, '', ha='center',
                                  va='center', color='white', fontsize=8.5, zorder=6)

    # input -> hidden arrows (4 total: x1->h1, x2->h1, x1->h2, x2->h2)
    def make_arrow(start_pos, end_pos, r_start, r_end):
        d = end_pos - start_pos
        d = d / np.linalg.norm(d)
        start = start_pos + d * r_start
        end = end_pos - d * r_end
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                                 color=COL_POS, lw=1.4, zorder=3,
                                 shrinkA=0, shrinkB=0)
        return arrow, start, end

    arrow_x1h1, s_x1h1, e_x1h1 = make_arrow(in1, h1c, r_in, r_hidden)
    arrow_x2h1, s_x2h1, e_x2h1 = make_arrow(in2, h1c, r_in, r_hidden)
    arrow_x1h2, s_x1h2, e_x1h2 = make_arrow(in1, h2c, r_in, r_hidden)
    arrow_x2h2, s_x2h2, e_x2h2 = make_arrow(in2, h2c, r_in, r_hidden)
    for a in [arrow_x1h1, arrow_x2h1, arrow_x1h2, arrow_x2h2]:
        ax_net.add_patch(a)

    arrow_h1y, s_h1y, e_h1y = make_arrow(h1c, outc, r_hidden, r_out)
    arrow_h2y, s_h2y, e_h2y = make_arrow(h2c, outc, r_hidden, r_out)
    ax_net.add_patch(arrow_h1y)
    ax_net.add_patch(arrow_h2y)

    def label_pos(start, end, offset_scale=0.30, side=1):
        mid = (start + end) / 2
        d = end - start
        d = d / np.linalg.norm(d)
        perp = np.array([-d[1], d[0]]) * offset_scale * side
        return mid + perp

    w11_txt = ax_net.text(*label_pos(s_x1h1, e_x1h1, offset_scale=0.30, side=1), '',
                           fontsize=9, ha='center', color=COL_TEXT)
    w12_txt = ax_net.text(*(label_pos(s_x2h1, e_x2h1, offset_scale=0.42, side=-1)
                             + np.array([0.26, 0])), '',
                           fontsize=9, ha='center', color=COL_TEXT)
    w21_txt = ax_net.text(*(label_pos(s_x1h2, e_x1h2, offset_scale=0.42, side=1)
                             + np.array([0.26, 0])), '',
                           fontsize=9, ha='center', color=COL_TEXT)
    w22_txt = ax_net.text(*label_pos(s_x2h2, e_x2h2, offset_scale=0.30, side=-1), '',
                           fontsize=9, ha='center', color=COL_TEXT)
    v1_txt = ax_net.text(*label_pos(s_h1y, e_h1y, side=1), '', fontsize=9.5,
                          ha='center', color=COL_TEXT)
    v2_txt = ax_net.text(*label_pos(s_h2y, e_h2y, side=-1), '', fontsize=9.5,
                          ha='center', color=COL_TEXT)

    ax_net.text(4.8, -3.7, 'arrow thickness = weight magnitude   '
                           'red = negative (inhibitory) weight',
                ha='center', fontsize=8.5, color='#888888')

    # ══════════════════════════════════════════════════════════════════════
    # Panel 2: live XOR truth table
    # ══════════════════════════════════════════════════════════════════════
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')
    ax_table.set_title('XOR truth table (live)', fontsize=11.5, color=COL_TEXT)

    n_cols = len(COL_LABELS)
    n_rows = len(XOR_TABLE)
    row_h = 1.0 / (n_rows + 1.8)

    for j, lab in enumerate(COL_LABELS):
        ax_table.text((j + 0.5) / n_cols, 1 - 0.6 * row_h, lab, ha='center',
                       va='center', fontweight='bold', fontsize=10)
    ax_table.plot([0.02, 0.98], [1 - 1.15 * row_h] * 2, color='#bbbbbb', lw=1.0)

    row_cells = []
    mark_positions = []
    for i, (x1, x2, target) in enumerate(XOR_TABLE):
        y = 1 - (i + 1.9) * row_h
        cells = {
            'x1': ax_table.text((0 + 0.5) / n_cols, y, str(x1), ha='center',
                                 va='center', fontsize=10.5),
            'x2': ax_table.text((1 + 0.5) / n_cols, y, str(x2), ha='center',
                                 va='center', fontsize=10.5),
            'h1': ax_table.text((2 + 0.5) / n_cols, y, '', ha='center',
                                 va='center', fontsize=10.5, color='#555555'),
            'h2': ax_table.text((3 + 0.5) / n_cols, y, '', ha='center',
                                 va='center', fontsize=10.5, color='#555555'),
            'pred': ax_table.text((4 + 0.5) / n_cols, y, '', ha='center',
                                   va='center', fontsize=10.5),
            'target': ax_table.text((5 + 0.5) / n_cols, y, str(target),
                                     ha='center', va='center', fontsize=10.5),
        }
        row_cells.append(cells)
        mark_positions.append(((6 + 0.5) / n_cols, y))

    error_summary_txt = ax_table.text(0.5, 1 - (n_rows + 2.6) * row_h, '',
                                       ha='center', va='center', fontsize=11.5,
                                       fontweight='bold')

    mark_artists = []

    # ── Update function ───────────────────────────────────────────────────────
    def update(w11, w12, theta1, w21, w22, theta2, v1, v2, thetay):
        for arrow, w in [(arrow_x1h1, w11), (arrow_x2h1, w12),
                          (arrow_x1h2, w21), (arrow_x2h2, w22),
                          (arrow_h1y, v1), (arrow_h2y, v2)]:
            arrow.set_color(COL_NEG if w < 0 else COL_POS)
            arrow.set_linewidth(1.0 + 1.6 * min(abs(w), 2.0))

        w11_txt.set_text(f'{w11:.1f}')
        w12_txt.set_text(f'{w12:.1f}')
        w21_txt.set_text(f'{w21:.1f}')
        w22_txt.set_text(f'{w22:.1f}')
        v1_txt.set_text(f'{v1:.1f}')
        v2_txt.set_text(f'{v2:.1f}')

        h1_theta_txt.set_text(rf'$\theta$={theta1:.1f}')
        h2_theta_txt.set_text(rf'$\theta$={theta2:.1f}')
        outy_theta_txt.set_text(rf'$\theta$={thetay:.1f}')

        for artist in mark_artists:
            artist.remove()
        mark_artists.clear()

        n_errors = 0
        for (x1, x2, target), cells, (mx, my) in zip(XOR_TABLE, row_cells,
                                                       mark_positions):
            z_h1 = w11 * x1 + w12 * x2
            h1 = 1 if z_h1 >= theta1 else 0
            z_h2 = w21 * x1 + w22 * x2
            h2 = 1 if z_h2 >= theta2 else 0
            z_y = v1 * h1 + v2 * h2
            pred = 1 if z_y >= thetay else 0
            correct = (pred == target)
            if not correct:
                n_errors += 1

            cells['h1'].set_text(str(h1))
            cells['h2'].set_text(str(h2))
            cells['pred'].set_text(str(pred))

            if correct:
                mark_artists.extend(draw_tick(ax_table, mx, my))
            else:
                mark_artists.extend(draw_cross(ax_table, mx, my))

        if n_errors == 0:
            error_summary_txt.set_text('0 errors — this network correctly computes XOR')
            error_summary_txt.set_color(COL_FIRE_E)
        else:
            row_word = 'row is' if n_errors == 1 else 'rows are'
            error_summary_txt.set_text(f'{n_errors} of 4 {row_word} wrong')
            error_summary_txt.set_color(COL_WRONG)

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    def wslider(default, desc):
        return FloatSlider(value=default, min=-2.0, max=2.0, step=0.1,
                            description=desc, style={'description_width': '55px'},
                            layout=widgets.Layout(width='230px'))

    def tslider(default, desc):
        return FloatSlider(value=default, min=-2.0, max=4.0, step=0.1,
                            description=desc, style={'description_width': '55px'},
                            layout=widgets.Layout(width='230px'))

    w11_s = wslider(DEFAULTS['w11'], 'w11')
    w12_s = wslider(DEFAULTS['w12'], 'w12')
    theta1_s = tslider(DEFAULTS['theta1'], 'theta1')

    w21_s = wslider(DEFAULTS['w21'], 'w21')
    w22_s = wslider(DEFAULTS['w22'], 'w22')
    theta2_s = tslider(DEFAULTS['theta2'], 'theta2')

    v1_s = wslider(DEFAULTS['v1'], 'v1')
    v2_s = wslider(DEFAULTS['v2'], 'v2')
    thetay_s = tslider(DEFAULTS['thetay'], 'theta_y')

    reset_btn = widgets.Button(description='Reset to solution',
                                button_style='warning',
                                layout=widgets.Layout(width='150px'))

    def on_reset(b):
        w11_s.value, w12_s.value, theta1_s.value = (DEFAULTS['w11'],
                                                      DEFAULTS['w12'],
                                                      DEFAULTS['theta1'])
        w21_s.value, w22_s.value, theta2_s.value = (DEFAULTS['w21'],
                                                      DEFAULTS['w22'],
                                                      DEFAULTS['theta2'])
        v1_s.value, v2_s.value, thetay_s.value = (DEFAULTS['v1'],
                                                    DEFAULTS['v2'],
                                                    DEFAULTS['thetay'])

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Search for weights and thresholds that solve XOR</b>'),
        HBox([
            VBox([HTML('<i>Neuron h_1 (OR-like)</i>'), w11_s, w12_s, theta1_s]),
            VBox([HTML('<i>Neuron h_2 (AND-like)</i>'), w21_s, w22_s, theta2_s]),
            VBox([HTML('<i>Output neuron y</i>'), v1_s, v2_s, thetay_s]),
        ]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        'w11': w11_s, 'w12': w12_s, 'theta1': theta1_s,
        'w21': w21_s, 'w22': w22_s, 'theta2': theta2_s,
        'v1': v1_s, 'v2': v2_s, 'thetay': thetay_s,
    })

    display(controls, out)