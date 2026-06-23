"""
Figure 4a — The Working XOR Network
======================================

Shows a static diagram of the three-neuron network from Section 9.2,
with the weights and thresholds already set to the worked solution that
correctly computes XOR. Unlike Figure 5, this diagram has no sliders or
interactivity — it is a fixed reference picture of the finished network,
useful for a reader who wants to see the solved structure on its own
before (or instead of) exploring how it can be broken in Figure 5.

The diagram shows two inputs (x_1, x_2), two hidden neurons (h_1, h_2),
and one output neuron (y), with every connection drawn and labelled with
its weight. Each neuron is labelled with its threshold. The connection
from h_2 to the output neuron is drawn in red, since its weight is
negative (inhibitory) — the mechanism that makes the network treat "both
hidden neurons firing" differently from "only h_1 firing".

The key teaching point: this is what a correct XOR solution actually
looks like as a finished object — nine fixed numbers (six weights, three
thresholds) arranged in a particular structure. It is meant to be read
alongside the worked calculations in Section 9.2 and 9.3, as the diagram
those calculations describe.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_4a import show
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


def show():
    """Render the static Figure 4a working-XOR-network diagram."""

    plt.close('Notebook6bonus Figure 4a')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_INPUT    = '#4C78A8'
    COL_INPUT_E  = '#2c4a66'
    COL_HIDDEN   = '#F2B701'
    COL_HIDDEN_E = '#9c7a00'
    COL_OUTPUT   = '#54A24B'
    COL_OUTPUT_E = '#2f5c2a'
    COL_TEXT     = '#222222'
    COL_POS      = '#666666'
    COL_NEG      = '#b23b3b'

    # The worked solution from Section 9.2
    W11, W12, THETA1 = 1.0, 1.0, 0.5     # h_1: OR-like
    W21, W22, THETA2 = 1.0, 1.0, 1.5     # h_2: AND-like
    V1, V2, THETAY = 1.0, -2.0, 0.5      # output neuron

    fig, ax = plt.subplots(num='Notebook6bonus Figure 4a', figsize=(10.5, 6.4),
                            constrained_layout=True)

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    fig.suptitle('Figure 4a: The working XOR network', fontsize=13, color=COL_TEXT)

    ax.set_xlim(-1, 11)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Node positions ───────────────────────────────────────────────────────
    in1 = np.array([0.5, 2.1])
    in2 = np.array([0.5, -2.1])
    h1c = np.array([4.6, 2.3])
    h2c = np.array([4.6, -2.3])
    outc = np.array([8.7, 0.0])
    r_in, r_hidden, r_out = 0.35, 0.58, 0.62

    # ── Nodes ────────────────────────────────────────────────────────────────
    for pos, lab in zip([in1, in2], ['$x_1$', '$x_2$']):
        ax.add_patch(Circle(pos, r_in, facecolor=COL_INPUT, edgecolor=COL_INPUT_E,
                             lw=1.3, zorder=5))
        ax.text(*pos, lab, ha='center', va='center', color='white', fontsize=11,
                fontweight='bold', zorder=6)

    ax.add_patch(Circle(h1c, r_hidden, facecolor=COL_HIDDEN, edgecolor=COL_HIDDEN_E,
                         lw=1.4, zorder=5))
    ax.add_patch(Circle(h2c, r_hidden, facecolor=COL_HIDDEN, edgecolor=COL_HIDDEN_E,
                         lw=1.4, zorder=5))
    ax.text(h1c[0], h1c[1] + 0.18, '$h_1$', ha='center', va='center', color='white',
            fontsize=11, fontweight='bold', zorder=6)
    ax.text(h2c[0], h2c[1] + 0.18, '$h_2$', ha='center', va='center', color='white',
            fontsize=11, fontweight='bold', zorder=6)
    ax.text(h1c[0], h1c[1] - 0.22, rf'$\theta$={THETA1:.1f}', ha='center',
            va='center', color='white', fontsize=8.5, zorder=6)
    ax.text(h2c[0], h2c[1] - 0.22, rf'$\theta$={THETA2:.1f}', ha='center',
            va='center', color='white', fontsize=8.5, zorder=6)

    ax.add_patch(Circle(outc, r_out, facecolor=COL_OUTPUT, edgecolor=COL_OUTPUT_E,
                         lw=1.4, zorder=5))
    ax.text(outc[0], outc[1] + 0.2, '$y$', ha='center', va='center', color='white',
            fontsize=12, fontweight='bold', zorder=6)
    ax.text(outc[0], outc[1] - 0.22, rf'$\theta$={THETAY:.1f}', ha='center',
            va='center', color='white', fontsize=8.5, zorder=6)

    # ── Connections ──────────────────────────────────────────────────────────
    def draw_connection(start_pos, end_pos, r_start, r_end, weight, label_side,
                         label_shift=np.array([0.0, 0.0])):
        d = end_pos - start_pos
        d = d / np.linalg.norm(d)
        start = start_pos + d * r_start
        end = end_pos - d * r_end
        colour = COL_NEG if weight < 0 else COL_POS
        lw = 1.2 + 1.6 * min(abs(weight), 2.0)
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                 color=colour, lw=lw, zorder=3, shrinkA=0, shrinkB=0)
        ax.add_patch(arrow)

        mid = (start + end) / 2
        perp = np.array([-d[1], d[0]]) * 0.32 * label_side
        ax.text(*(mid + perp + label_shift), f'{weight:.1f}', fontsize=9.5,
                ha='center', color=COL_TEXT, zorder=4)

    draw_connection(in1, h1c, r_in, r_hidden, W11, label_side=1)
    draw_connection(in2, h1c, r_in, r_hidden, W12, label_side=-1,
                     label_shift=np.array([0.26, 0]))
    draw_connection(in1, h2c, r_in, r_hidden, W21, label_side=1,
                     label_shift=np.array([0.26, 0]))
    draw_connection(in2, h2c, r_in, r_hidden, W22, label_side=-1)
    draw_connection(h1c, outc, r_hidden, r_out, V1, label_side=1)
    draw_connection(h2c, outc, r_hidden, r_out, V2, label_side=-1)

    # ── Legend / footnote ────────────────────────────────────────────────────
    ax.text(4.8, -3.7, 'arrow thickness = weight magnitude   '
                       'red = negative (inhibitory) weight',
            ha='center', fontsize=8.5, color='#888888')

    ax.text(0.5, 3.6, '$h_1$ detects "at least one input is 1" (OR-like)',
            ha='left', fontsize=9, color='#555555')
    ax.text(0.5, -3.3, '$h_2$ detects "both inputs are 1" (AND-like)',
            ha='left', fontsize=9, color='#555555')
    ax.text(8.7, 1.0, '$y$ fires when $h_1$\nfires but $h_2$ does not',
            ha='center', fontsize=9, color='#555555')