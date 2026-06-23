"""
Figure 1 — Structure of a Biological Neuron
=============================================

Shows a single biological neuron drawn schematically, with its four main
structural parts clearly labelled: the dendrites, the cell body (soma),
the axon (with its myelin sheath and nodes of Ranvier), and the axon
terminals. A zoomed inset panel on the right shows a single axon terminal
approaching the dendrite of a second, partially drawn neuron, with the
synaptic cleft — the small gap between the two cells — labelled explicitly.

The dendrites and axon terminals are drawn as recursively branching,
organically curved structures rather than straight lines, so the diagram
reads as a real cell rather than a wiring schematic. The myelin sheath is
drawn as a series of separate segments along the axon, with the small gaps
between segments labelled as a node of Ranvier.

A set of toggle buttons lets the user highlight each of the four main
parts in turn (dendrites, soma, axon, axon terminals), dimming the rest of
the diagram so that the relevant structure stands out. This is a
labelling and orientation aid rather than a parameter-exploration tool,
so there are no sliders — the underlying geometry of the neuron does not
change, only which part of it is highlighted.

The key teaching point: a neuron is a single cell with a clear division of
labour. Dendrites receive, the soma integrates, the axon transmits, and
the axon terminals hand the signal on to the next cell across a synaptic
gap. This structural picture is the direct biological basis for the
weights, summation, and output of the artificial neuron introduced later
in the notebook.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_1 import show
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
from matplotlib.path import Path
import matplotlib.patches as mpatches
import ipywidgets as widgets
from ipywidgets import VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the Figure 1 biological neuron diagram with part-highlighting controls."""

    plt.close('Notebook6bonus Figure 1')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_DENDRITE  = '#4C78A8'
    COL_SOMA      = '#E45756'
    COL_SOMA_E    = '#8c2f30'
    COL_AXON      = '#54A24B'
    COL_MYELIN    = '#d7d7d7'
    COL_MYELIN_E  = '#999999'
    COL_TERMINAL  = '#B279A2'
    COL_TERMINAL_E= '#6b3f63'
    COL_NEXT      = '#9c9c9c'
    COL_TEXT      = '#222222'

    rng = np.random.default_rng(7)

    # ── Figure and axes: main diagram (left) + synapse inset (right) ──────────
    fig = plt.figure(num='Notebook6bonus Figure 1', figsize=(13, 6.4))
    ax_main = fig.add_axes([0.02, 0.06, 0.66, 0.88])
    ax_inset = fig.add_axes([0.70, 0.18, 0.28, 0.64])

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    ax_main.set_xlim(-1.6, 12.6)
    ax_main.set_ylim(-4.6, 4.6)
    ax_main.set_aspect('equal')
    ax_main.axis('off')

    ax_inset.set_xlim(0, 10)
    ax_inset.set_ylim(0, 10)
    ax_inset.set_aspect('equal')
    ax_inset.axis('off')

    # ── Recursive branch-drawing helper (shared by dendrites and terminals) ──
    def branch(ax, start, angle, length, depth, width, color, tips=None,
               spread=(0.35, 0.75), jitter=0.15, n_children_fn=None,
               zorder=4):
        """Recursively draw a branching structure as a sequence of gentle
        curved segments, optionally collecting the leaf-tip coordinates."""
        if depth == 0 or length < 0.13:
            if tips is not None:
                tips.append(start.copy())
            return
        end = start + length * np.array([np.cos(angle), np.sin(angle)])
        mid = ((start + end) / 2
               + length * 0.15 * np.array([-np.sin(angle), np.cos(angle)])
               * rng.uniform(-1, 1))
        path = Path([start, mid, end], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color,
                                    lw=width, capstyle='round', zorder=zorder)
        ax.add_patch(patch)

        if n_children_fn is not None:
            n_children = n_children_fn(depth)
        else:
            n_children = 2 if depth > 1 else rng.integers(1, 3)

        angle_jit = rng.uniform(-jitter, jitter)
        for _ in range(n_children):
            new_angle = (angle
                         + rng.uniform(*spread) * rng.choice([-1, 1])
                         + angle_jit)
            branch(ax, end, new_angle, length * 0.68, depth - 1,
                   width * 0.72, color, tips=tips, spread=spread,
                   jitter=jitter, n_children_fn=n_children_fn, zorder=zorder)
        if depth == 1 and tips is not None:
            tips.append(end.copy())

    # ── Soma (cell body) ────────────────────────────────────────────────────
    soma_center = np.array([3.0, 0.0])
    soma_rx, soma_ry = 1.0, 0.85
    th = np.linspace(0, 2 * np.pi, 200)
    wobble = 1 + 0.06 * np.sin(5 * th + 1.3) + 0.04 * np.sin(3 * th)
    soma_x = soma_center[0] + soma_rx * wobble * np.cos(th)
    soma_y = soma_center[1] + soma_ry * wobble * np.sin(th)
    soma_patch, = ax_main.fill(soma_x, soma_y, color=COL_SOMA, alpha=0.85,
                                zorder=5, ec=COL_SOMA_E, lw=1.3)
    nucleus = plt.Circle(soma_center + np.array([0.05, 0.05]), 0.35,
                          color=COL_SOMA_E, alpha=0.55, zorder=6)
    ax_main.add_patch(nucleus)

    # ── Dendrites: branching tree on the left of the soma ──────────────────
    dendrite_patches_start = len(ax_main.patches)
    base_angles = np.linspace(np.pi * 0.55, np.pi * 1.45, 5)
    for a in base_angles:
        start = soma_center + 0.85 * np.array([np.cos(a), np.sin(a)])
        branch(ax_main, start, a, 1.35, depth=4, width=2.6, color=COL_DENDRITE)
    dendrite_patches_end = len(ax_main.patches)

    # ── Axon: long cable extending right, tapering wobble ──────────────────
    axon_patches_start = len(ax_main.patches)
    axon_start = soma_center + np.array([soma_rx * 0.95, -0.05])
    axon_end = np.array([9.1, -0.05])
    n_pts = 300
    t = np.linspace(0, 1, n_pts)
    wave = 0.10 * np.sin(t * 4 * np.pi) * (1 - t)
    axon_x = axon_start[0] + t * (axon_end[0] - axon_start[0])
    axon_y = axon_start[1] + t * (axon_end[1] - axon_start[1]) + wave
    axon_line, = ax_main.plot(axon_x, axon_y, color=COL_AXON, lw=5,
                               solid_capstyle='round', zorder=2)

    # Myelin sheath: segmented ellipses with small gaps (nodes of Ranvier)
    n_segments = 8
    seg_len = (axon_end[0] - axon_start[0] - 0.6) / n_segments
    gap = 0.12
    for i in range(n_segments):
        seg_start_x = axon_start[0] + 0.3 + i * seg_len
        cx = seg_start_x + seg_len / 2 - gap / 2
        idx = np.argmin(np.abs(axon_x - cx))
        cy = axon_y[idx]
        ell = mpatches.Ellipse((cx, cy), width=seg_len - gap, height=0.62,
                                facecolor=COL_MYELIN, edgecolor=COL_MYELIN_E,
                                lw=0.8, zorder=3, alpha=0.95)
        ax_main.add_patch(ell)
    axon_patches_end = len(ax_main.patches)

    node_x = axon_start[0] + 0.3 + 3 * seg_len
    node_label = ax_main.annotate(
        'Node of Ranvier\n(gap in the myelin sheath)', xy=(node_x, 0.08),
        xytext=(node_x - 0.5, 2.0), fontsize=8, color='#666666', ha='center',
        arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8), zorder=7)

    # ── Axon terminals: branching at the end of the axon ────────────────────
    terminal_patches_start = len(ax_main.patches)
    raw_tips = []
    terminal_branch_start = np.array([axon_end[0], axon_y[-1]])
    branch(ax_main, terminal_branch_start, 0.05, 1.0, depth=3, width=2.6,
           color=COL_TERMINAL, tips=raw_tips, spread=(0.35, 0.65), jitter=0.08)

    # de-duplicate near-identical tip coordinates (recursion can revisit a
    # near-leaf node more than once at shallow depth)
    unique_tips = []
    for tip in raw_tips:
        if not any(np.linalg.norm(tip - u) < 0.05 for u in unique_tips):
            unique_tips.append(tip)

    for tip in unique_tips:
        dot = plt.Circle(tip, 0.11, color=COL_TERMINAL, zorder=5,
                          ec=COL_TERMINAL_E, lw=0.8)
        ax_main.add_patch(dot)
    terminal_patches_end = len(ax_main.patches)

    # pick the tip nearest the inset for the zoomed synapse view
    synapse_tip = max(unique_tips, key=lambda p: p[0])

    # ── Labels on the main diagram ──────────────────────────────────────────
    label_style = dict(fontsize=10.5, color=COL_TEXT, ha='center', zorder=8)
    lbl_dendrite = ax_main.annotate('Dendrites', xy=(0.0, 2.6), xytext=(-0.6, 3.9),
                                     **label_style,
                                     arrowprops=dict(arrowstyle='-', color='#888888', lw=0.8))
    lbl_soma = ax_main.annotate('Cell body\n(soma)',
                                 xy=(soma_center[0] + 0.25, soma_center[1] - soma_ry + 0.05),
                                 xytext=(soma_center[0] + 0.4, -3.2), **label_style,
                                 arrowprops=dict(arrowstyle='-', color='#888888', lw=0.8))
    lbl_axon = ax_main.annotate('Axon', xy=(6.0, axon_y[int(n_pts*0.55)]),
                                 xytext=(6.0, -2.3), **label_style,
                                 arrowprops=dict(arrowstyle='-', color='#888888', lw=0.8))
    lbl_terminal = ax_main.annotate('Axon terminals\n(synaptic terminals)',
                                     xy=(synapse_tip[0]-0.1, synapse_tip[1]+0.1),
                                     xytext=(9.6, 2.6), **label_style,
                                     arrowprops=dict(arrowstyle='-', color='#888888', lw=0.8))

    ax_main.set_title('Figure 1: A single biological neuron', fontsize=12, color=COL_TEXT,
                       pad=10)

    # ── Synapse inset: zoomed view of one terminal meeting the next neuron ──
    # incoming terminal (button) on the left of the inset
    term_pos = np.array([2.1, 5.0])
    ax_inset.add_patch(plt.Circle(term_pos, 0.55, color=COL_TERMINAL,
                                   ec=COL_TERMINAL_E, lw=1.2, zorder=5))
    # short stub of axon leading into the terminal
    ax_inset.plot([0.2, term_pos[0]-0.4], [5.0, 5.0], color=COL_TERMINAL,
                  lw=4, solid_capstyle='round', zorder=3)

    # neurotransmitter dots drifting across the synaptic cleft
    nt_x = rng.uniform(2.9, 4.0, 14)
    nt_y = rng.uniform(4.4, 5.6, 14)
    ax_inset.scatter(nt_x, nt_y, s=18, color='#F2B701', zorder=6,
                      edgecolors='#9c7a00', linewidths=0.4)

    # synaptic cleft gap (shaded band)
    ax_inset.axvspan(2.65, 3.7, color='#f0f0f0', zorder=1)
    ax_inset.annotate('Synaptic cleft\n(the gap)', xy=(3.15, 3.6),
                       xytext=(3.15, 1.4), fontsize=8.5, color='#666666',
                       ha='center',
                       arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8))

    # receiving dendrite of the next neuron, partially drawn on the right
    next_start = np.array([7.5, 5.0])
    branch(ax_inset, next_start, np.pi, 1.6, depth=3, width=2.2, color=COL_NEXT,
           spread=(0.3, 0.55), jitter=0.05)
    ax_inset.annotate('Dendrite of the\nnext neuron', xy=(6.6, 5.5),
                       xytext=(6.6, 8.4), fontsize=8.5, color='#666666',
                       ha='center',
                       arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8))

    ax_inset.annotate('Axon terminal', xy=term_pos, xytext=(1.3, 8.2),
                       fontsize=8.5, color='#666666', ha='center',
                       arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8))
    ax_inset.annotate('Neurotransmitters', xy=(3.4, 5.9), xytext=(3.4, 9.4),
                       fontsize=8.5, color='#666666', ha='center',
                       arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8))

    ax_inset.set_title('Zoomed view: the synapse', fontsize=10.5, color=COL_TEXT)

    # ── Part registry for the highlight toggles ─────────────────────────────
    parts = {
        'dendrites': {
            'patches': ax_main.patches[dendrite_patches_start:dendrite_patches_end],
            'labels': [lbl_dendrite],
        },
        'soma': {
            'patches': [soma_patch, nucleus],
            'labels': [lbl_soma],
        },
        'axon': {
            'patches': ax_main.patches[axon_patches_start:axon_patches_end],
            'labels': [lbl_axon, node_label],
            'lines': [axon_line],
        },
        'terminals': {
            'patches': ax_main.patches[terminal_patches_start:terminal_patches_end],
            'labels': [lbl_terminal],
        },
    }

    DIM_ALPHA = 0.12

    def reset_highlight():
        for name, group in parts.items():
            base_alpha = 0.85 if name == 'soma' else 1.0
            for p in group['patches']:
                p.set_alpha(base_alpha)
            for ln in group.get('lines', []):
                ln.set_alpha(1.0)
            for lb in group['labels']:
                lb.set_color(COL_TEXT)
                lb.set_fontweight('normal')
        fig.canvas.draw_idle()

    def highlight(part_name):
        for name, group in parts.items():
            is_target = (name == part_name)
            base_alpha = 0.85 if name == 'soma' else 1.0
            a_patch = base_alpha if is_target else DIM_ALPHA
            a_line = 1.0 if is_target else DIM_ALPHA
            for p in group['patches']:
                p.set_alpha(a_patch)
            for ln in group.get('lines', []):
                ln.set_alpha(a_line)
            for lb in group['labels']:
                lb.set_color(COL_TEXT if is_target else '#bbbbbb')
                lb.set_fontweight('bold' if is_target else 'normal')
        fig.canvas.draw_idle()

    # ── Toggle buttons ───────────────────────────────────────────────────────
    btn_dendrites = widgets.ToggleButton(value=False, description='Dendrites',
                                          layout=widgets.Layout(width='110px'))
    btn_soma = widgets.ToggleButton(value=False, description='Cell body',
                                     layout=widgets.Layout(width='110px'))
    btn_axon = widgets.ToggleButton(value=False, description='Axon',
                                     layout=widgets.Layout(width='110px'))
    btn_terminals = widgets.ToggleButton(value=False, description='Terminals',
                                          layout=widgets.Layout(width='110px'))
    btn_reset = widgets.Button(description='Show all', button_style='warning',
                                layout=widgets.Layout(width='110px'))

    toggle_buttons = {'dendrites': btn_dendrites, 'soma': btn_soma,
                       'axon': btn_axon, 'terminals': btn_terminals}

    def on_toggle(change, name):
        if change['new']:
            for other_name, other_btn in toggle_buttons.items():
                if other_name != name and other_btn.value:
                    other_btn.value = False
            highlight(name)
        else:
            if not any(b.value for b in toggle_buttons.values()):
                reset_highlight()

    btn_dendrites.observe(lambda c: on_toggle(c, 'dendrites'), names='value')
    btn_soma.observe(lambda c: on_toggle(c, 'soma'), names='value')
    btn_axon.observe(lambda c: on_toggle(c, 'axon'), names='value')
    btn_terminals.observe(lambda c: on_toggle(c, 'terminals'), names='value')

    def on_reset(b):
        for btn in toggle_buttons.values():
            btn.value = False
        reset_highlight()

    btn_reset.on_click(on_reset)

    controls = VBox([
        HTML('<b>Highlight a part of the neuron</b>'),
        HBox([btn_dendrites, btn_soma, btn_axon, btn_terminals, btn_reset]),
    ])

    display(controls)