"""
Figure 3 — AND, OR, and XOR Plotted as Points
================================================

Shows the truth tables of AND, OR, and XOR as three side-by-side scatter
plots, one gate per panel, with x_1 on the horizontal axis and x_2 on the
vertical axis. Each of the four possible input combinations — (0,0),
(0,1), (1,0), (1,1) — is plotted as a point, labelled with its own
coordinate pair directly above it. A filled circle marks a combination
whose correct output is 1; a hollow circle marks a combination whose
correct output is 0. An overall legend beneath the three panels explains
this convention once, rather than repeating it on every panel.

A single straight line is drawn on the AND and OR panels, chosen so
that it cleanly separates the filled points from the hollow points. XOR
cannot be separated this way at all: its two filled points sit on
opposite corners of the square from each other, so no single straight
line can put both of them on one side and both hollow points on the
other. The XOR panel therefore shows two lines instead of one, bounding
a diagonal strip that contains the two filled points and excludes the
two hollow points. No commentary on linear separability is given in
this figure — the lines are present simply as a visual seed, to be
returned to and explained properly later in the notebook once the idea
of linear separability has been introduced. The two-line treatment of
XOR also foreshadows the two-hidden-neuron solution built later in
Section 9, where each of these two lines becomes the decision boundary
of one hidden neuron.

Toggle buttons let the reader show one gate at a time, or all three at
once (the default). This is a static reference diagram rather than a
parameter-exploration tool, since the truth tables themselves do not
change, so toggles rather than sliders are used to let the reader focus
on a single gate without altering any underlying data.

The key teaching point: AND, OR, and XOR all look superficially similar
— two binary inputs, four possible combinations — but the geometric
arrangement of their "true" and "false" points on the plane is
different. This sets up the discussion of linear separability later in
the notebook.

Usage
-----
From a Jupyter notebook cell::

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
import ipywidgets as widgets
from ipywidgets import VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the Figure 3 AND / OR / XOR truth table scatter plots."""

    plt.close('Notebook6bonus Figure 3')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_FILL    = '#54A24B'   # seagreen — output 1
    COL_FILL_E  = '#2f5c2a'
    COL_EMPTY_E = '#B23B3B'   # hollow marker edge — output 0
    COL_LINE    = '#4C78A8'   # steelblue — the separating / attempted line
    COL_TEXT    = '#222222'
    COL_MUTED   = '#cfcfcf'

    # ── Truth tables ─────────────────────────────────────────────────────────
    gates = {
        'AND': {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
        'OR':  {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        'XOR': {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    }
    gate_order = ['AND', 'OR', 'XOR']

    # Lines are given in the form x2 = m*x1 + c. AND and OR each get a single
    # line that cleanly separates the filled points from the hollow points.
    # XOR cannot be separated by any single line — its filled points sit on
    # opposite corners from each other — so it is given two lines instead,
    # one on each side of the diagonal of filled points. Together the two
    # lines bound the strip containing the filled points, foreshadowing the
    # two-hidden-neuron solution built in Section 9.
    gate_lines = {
        'AND': [dict(m=-1.0, c=1.5)],                 # separates (1,1) from the rest
        'OR':  [dict(m=-1.0, c=0.5)],                 # separates (0,0) from the rest
        'XOR': [dict(m=-1.0, c=0.5), dict(m=-1.0, c=1.5)],  # bounds the filled strip
    }

    # ── Figure: one axis per gate ────────────────────────────────────────────
    # constrained_layout (rather than tight_layout) is used because it
    # reserves real space for figure-level artists such as suptitle() and
    # legend() and keeps recomputing that space as the canvas is resized.
    # tight_layout only lays out the axes once, at creation time, which is
    # why the title and legend could end up clipped or pushed off-canvas
    # when the ipympl widget canvas was resized afterwards.
    fig, axes = plt.subplots(1, 3, num='Notebook6bonus Figure 3',
                              figsize=(13.5, 5.0), constrained_layout=True)

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    fig.suptitle('Figure 3: AND, OR, and XOR plotted as points', fontsize=13,
                 color=COL_TEXT)

    gate_artists = {}

    for ax, name in zip(axes, gate_order):
        table = gates[name]
        points = []
        labels = []

        for (x1, x2), y in table.items():
            if y == 1:
                pt = ax.scatter(x1, x2, s=260, facecolor=COL_FILL,
                                 edgecolor=COL_FILL_E, linewidth=1.6, zorder=5)
            else:
                pt = ax.scatter(x1, x2, s=260, facecolor='white',
                                 edgecolor=COL_EMPTY_E, linewidth=1.8, zorder=5)
            points.append(pt)
            lbl = ax.annotate(f'({x1},{x2})', (x1, x2), textcoords='offset points',
                               xytext=(0, 22), ha='center', fontsize=9.5,
                               color='#555555')
            labels.append(lbl)

        # the example line(s): one for AND/OR, two for XOR (see gate_lines above)
        lines_for_gate = []
        x_line = np.array([-0.6, 1.6])
        for line_spec in gate_lines[name]:
            y_line = line_spec['m'] * x_line + line_spec['c']
            line, = ax.plot(x_line, y_line, color=COL_LINE, lw=2.0,
                             linestyle='--', zorder=3, alpha=0.85)
            lines_for_gate.append(line)

        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.85)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        title = ax.set_title(name, fontsize=13, color=COL_TEXT)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)

        gate_artists[name] = {
            'points': points, 'labels': labels, 'lines': lines_for_gate,
            'title': title, 'axis': ax,
        }

    # ── Overall legend beneath the three plots ──────────────────────────────
    # Proxy artists, since the legend explains the marker convention used
    # across all three panels rather than referring to one specific point.
    # loc='outside lower center' tells constrained_layout to treat the
    # legend as a real layout element with its own reserved row, rather
    # than an overlay positioned in raw figure coordinates — this is what
    # keeps it on-canvas and correctly placed as the figure is resized.
    legend_fill = plt.Line2D([0], [0], marker='o', linestyle='', markersize=11,
                              markerfacecolor=COL_FILL, markeredgecolor=COL_FILL_E,
                              markeredgewidth=1.6, label='Gate returns 1')
    legend_empty = plt.Line2D([0], [0], marker='o', linestyle='', markersize=11,
                               markerfacecolor='white', markeredgecolor=COL_EMPTY_E,
                               markeredgewidth=1.8, label='Gate returns 0')
    fig.legend(handles=[legend_fill, legend_empty], loc='outside lower center',
               ncol=2, frameon=False, fontsize=10.5)

    # ── Toggle logic: show one gate, or all three (default) ────────────────
    def set_visible(name, visible):
        art = gate_artists[name]
        alpha_pt = 1.0 if visible else 0.08
        alpha_line = 0.85 if visible else 0.04
        for pt in art['points']:
            pt.set_alpha(alpha_pt)
        for lbl in art['labels']:
            lbl.set_color('#555555' if visible else COL_MUTED)
        for ln in art['lines']:
            ln.set_alpha(alpha_line)
        art['title'].set_color(COL_TEXT if visible else COL_MUTED)
        art['axis'].set_facecolor('white' if visible else '#fbfbfb')

    def show_all():
        for name in gate_order:
            set_visible(name, True)
        fig.canvas.draw_idle()

    def show_only(name):
        for other in gate_order:
            set_visible(other, other == name)
        fig.canvas.draw_idle()

    # ── Toggle buttons ───────────────────────────────────────────────────────
    btn_and = widgets.ToggleButton(value=False, description='AND only',
                                    layout=widgets.Layout(width='110px'))
    btn_or = widgets.ToggleButton(value=False, description='OR only',
                                   layout=widgets.Layout(width='110px'))
    btn_xor = widgets.ToggleButton(value=False, description='XOR only',
                                    layout=widgets.Layout(width='110px'))
    btn_all = widgets.Button(description='Show all', button_style='warning',
                              layout=widgets.Layout(width='110px'))

    toggle_buttons = {'AND': btn_and, 'OR': btn_or, 'XOR': btn_xor}

    def on_toggle(change, name):
        if change['new']:
            for other_name, other_btn in toggle_buttons.items():
                if other_name != name and other_btn.value:
                    other_btn.value = False
            show_only(name)
        else:
            if not any(b.value for b in toggle_buttons.values()):
                show_all()

    btn_and.observe(lambda c: on_toggle(c, 'AND'), names='value')
    btn_or.observe(lambda c: on_toggle(c, 'OR'), names='value')
    btn_xor.observe(lambda c: on_toggle(c, 'XOR'), names='value')

    def on_show_all(b):
        for btn in toggle_buttons.values():
            btn.value = False
        show_all()

    btn_all.on_click(on_show_all)

    controls = VBox([
        HTML('<b>Focus on a single gate</b>'),
        HBox([btn_and, btn_or, btn_xor, btn_all]),
    ])

    display(controls)