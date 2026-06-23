"""
Figure 6 — Structure of a Multilayer Perceptron
==================================================

Shows a static diagram of a small Multilayer Perceptron (MLP) with two
hidden layers, used to introduce the general layered structure described
in Section 10.2. The network drawn has 3 input nodes, 4 neurons in the
first hidden layer, 3 neurons in the second hidden layer, and 2 output
nodes, fully connected between every pair of adjacent layers.

This is a structural reference diagram rather than an interactive tool.
There are no sliders, since the point of the figure is to show the shape
of the network — how layers are connected and what "fully connected"
looks like at a glance — not to let the reader search over weights, as
in Figures 4 and 5. The specific layer sizes (3, 4, 3, 2) are
illustrative; the diagram is meant to generalise to MLPs of any size.

Each node is colour-coded by layer type: steelblue for input nodes,
goldenrod for hidden nodes, seagreen for output nodes. Connections are
drawn as thin grey lines to keep the fully-connected structure legible
without overwhelming the figure, since a network this size already has
3*4 + 4*3 + 3*2 = 30 individual connections. One connection is picked
out and labelled with a generic weight symbol w_ij, to anchor the
abstract idea of "every connection has its own weight" to a single
concrete example. A label above each hidden layer notes that an
activation function is applied at every neuron in that layer, tying this
diagram back to the activation functions covered in Sections 5.3, 5.4,
and 10.3.

The key teaching point: an MLP is built from the same components as a
single MP neuron or the two-hidden-neuron XOR network, just repeated and
stacked into multiple fully connected layers. The diagram is meant to
make that repetition, and the resulting growth in the number of weights,
visually obvious.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_6 import show
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
    """Render the static Figure 6 MLP architecture diagram."""

    plt.close('Notebook6bonus Figure 6')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_INPUT   = '#4C78A8'
    COL_HIDDEN  = '#DAA520'
    COL_OUTPUT  = '#54A24B'
    COL_LINE    = '#aaaaaa'
    COL_TEXT    = '#222222'
    COL_HIGHLIGHT = '#b23b3b'

    # ── Network architecture ─────────────────────────────────────────────────
    layers = [
        {'n': 3, 'x': 0.0, 'labels': ['$x_1$', '$x_2$', '$x_3$'],
         'color': COL_INPUT, 'name': 'Input layer'},
        {'n': 4, 'x': 3.2, 'labels': [r'$h^{(1)}_1$', r'$h^{(1)}_2$',
                                       r'$h^{(1)}_3$', r'$h^{(1)}_4$'],
         'color': COL_HIDDEN, 'name': 'Hidden layer 1'},
        {'n': 3, 'x': 6.4, 'labels': [r'$h^{(2)}_1$', r'$h^{(2)}_2$', r'$h^{(2)}_3$'],
         'color': COL_HIDDEN, 'name': 'Hidden layer 2'},
        {'n': 2, 'x': 9.6, 'labels': ['$y_1$', '$y_2$'],
         'color': COL_OUTPUT, 'name': 'Output layer'},
    ]
    r = 0.34
    spacing = 1.05

    def node_ys(n):
        total_h = (n - 1) * spacing
        return [total_h / 2 - i * spacing for i in range(n)]

    layer_positions = []
    for layer in layers:
        ys = node_ys(layer['n'])
        layer_positions.append([(layer['x'], y) for y in ys])

    fig, ax = plt.subplots(num='Notebook6bonus Figure 6', figsize=(12.5, 7.2),
                            constrained_layout=True)
    ax.axis('off')
    ax.set_aspect('equal')

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    fig.suptitle('Figure 6: Structure of a Multilayer Perceptron with two '
                 'hidden layers', fontsize=13, color=COL_TEXT)

    # ── Connections (drawn first, so nodes sit on top) ───────────────────────
    highlighted_edge = None
    for li in range(len(layers) - 1):
        for (x1, y1) in layer_positions[li]:
            for (x2, y2) in layer_positions[li + 1]:
                ax.plot([x1 + r, x2 - r], [y1, y2], color=COL_LINE, lw=0.7,
                        alpha=0.55, zorder=1)

    # Pick one connection (first input -> first hidden-1 neuron) to highlight
    hx1, hy1 = layer_positions[0][0]
    hx2, hy2 = layer_positions[1][0]
    ax.plot([hx1 + r, hx2 - r], [hy1, hy2], color=COL_HIGHLIGHT, lw=2.0, zorder=2)
    mid_x, mid_y = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    ax.annotate(r'weight $w_{ij}$', xy=(mid_x, mid_y), xytext=(mid_x - 0.3, mid_y + 1.1),
                fontsize=10, color=COL_HIGHLIGHT, ha='center',
                arrowprops=dict(arrowstyle='-', color=COL_HIGHLIGHT, lw=0.9))

    # ── Nodes ────────────────────────────────────────────────────────────────
    for layer, positions in zip(layers, layer_positions):
        for (x, y), lab in zip(positions, layer['labels']):
            ax.add_patch(Circle((x, y), r, facecolor=layer['color'],
                                 edgecolor='black', lw=0.8, zorder=5, alpha=0.92))
            ax.text(x, y, lab, ha='center', va='center', fontsize=8.5,
                    color='white', zorder=6)

    # ── Layer name labels (below each column) ───────────────────────────────
    bottom_y = -2.7
    for layer in layers:
        ax.text(layer['x'], bottom_y, layer['name'], ha='center', fontsize=10.5,
                 color=COL_TEXT)

    # ── "Activation function applied" annotation above hidden layers ────────
    top_y = 2.5
    for layer in layers[1:3]:
        ax.text(layer['x'], top_y, 'Activation function\napplied', ha='center',
                fontsize=8.5, color='#666666', style='italic')

    # ── Fully-connected note ─────────────────────────────────────────────────
    ax.text((layers[0]['x'] + layers[-1]['x']) / 2, bottom_y - 0.7,
             'Every node is connected to every node in the next layer '
             '(fully connected / dense)',
             ha='center', fontsize=9, color='#888888')

    ax.set_xlim(-1.0, 10.6)
    ax.set_ylim(-3.9, 3.2)