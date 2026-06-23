"""
Figure 2 — Structure of the McCulloch-Pitts Neuron
=====================================================

Shows a single McCulloch-Pitts (MP) neuron drawn as a network diagram:
three labelled input nodes (x_1, x_2, x_3) on the left, each connected by
a labelled, weighted arrow (w_1, w_2, w_3) to a central neuron body, and a
single labelled output arrow on the right carrying the neuron's binary
output y.

The neuron body is labelled "Sum + Threshold" and displays the weighted
sum value z and the current threshold theta in short form, sized to
always fit inside the circle. The full step-by-step calculation — each
weighted term, the running total, and the comparison against theta — is
shown separately in a light-coloured annotation box below the neuron
body, connected to it by a short dashed leader line. Keeping this longer
text outside the coloured circle avoids the white-on-tomato legibility
problem that occurs when a long calculation string is rendered in white
text and overflows the edge of the neuron body.

Interactive sliders let the user set each input x_1, x_2, x_3 (binary:
0 or 1), each weight w_1, w_2, w_3, and the threshold theta. As these are
changed, the diagram updates live: the weighted sum z is recomputed and
displayed inside the neuron body, the full term-by-term calculation and
comparison against theta are shown in the annotation box below, and the
output arrow and label change colour depending on whether the neuron
fires (z >= theta, output 1) or stays silent (z < theta, output 0). The
annotation box itself changes colour to a light green when the neuron
fires, giving a second visual confirmation beyond the output arrow. A
small numeric readout beneath the whole diagram also shows the
calculation as printed text, so the arithmetic behind the diagram is
always visible in more than one form.

The key teaching point: the MP neuron is nothing more than a weighted sum
compared against a threshold. Moving the sliders should make it obvious
that increasing a weight or an input makes that input's contribution
larger, that negative weights pull the sum down (inhibition), and that
the threshold sets the bar the weighted sum has to clear before the
neuron fires.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_2 import show
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
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import ipywidgets as widgets
from ipywidgets import interactive_output, FloatSlider, VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the interactive Figure 2 McCulloch-Pitts neuron diagram."""

    plt.close('Notebook6bonus Figure 2')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_INPUT      = '#4C78A8'   # steelblue
    COL_INPUT_E    = '#2c4a66'
    COL_NEURON     = '#E45756'   # tomato
    COL_NEURON_E   = '#8c2f30'
    COL_FIRE       = '#54A24B'   # seagreen
    COL_FIRE_E     = '#2f5c2a'
    COL_SILENT     = '#9c9c9c'
    COL_SILENT_E   = '#5e5e5e'
    COL_TEXT       = '#222222'
    COL_ARROW      = '#666666'
    COL_ARROW_NEG  = '#b23b3b'   # inhibitory (negative weight) arrows shown in red

    # ── Fixed geometry ──────────────────────────────────────────────────────
    input_x   = 0.5
    input_ys  = [2.2, 0.0, -2.2]
    in_labels = [r'$x_1$', r'$x_2$', r'$x_3$']
    w_labels  = [r'$w_1$', r'$w_2$', r'$w_3$']
    neuron_center = np.array([5.5, 0.0])
    neuron_r  = 1.45
    output_x  = 9.6

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_X = [1.0, 1.0, 0.0]
    DEFAULT_W = [1.0, 1.0, 1.0]
    DEFAULT_THETA = 1.5

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook6bonus Figure 2', figsize=(10.5, 7.6))
    ax.set_xlim(-1.0, 11.2)
    ax.set_ylim(-5.0, 3.4)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    title = ax.set_title('Figure 2: Structure of the McCulloch-Pitts neuron',
                          fontsize=13, color=COL_TEXT, pad=12)

    # ── Static input nodes ───────────────────────────────────────────────────
    input_circles = []
    input_texts = []
    for y, lab in zip(input_ys, in_labels):
        c = Circle((input_x, y), 0.32, facecolor=COL_INPUT, edgecolor=COL_INPUT_E,
                   lw=1.3, zorder=5)
        ax.add_patch(c)
        t = ax.text(input_x, y, lab, ha='center', va='center', fontsize=12,
                    color='white', zorder=6, fontweight='bold')
        input_circles.append(c)
        input_texts.append(t)

    # ── Neuron body ──────────────────────────────────────────────────────────
    neuron_circ = Circle(neuron_center, neuron_r, facecolor=COL_NEURON, alpha=0.85,
                          edgecolor=COL_NEURON_E, lw=1.6, zorder=5)
    ax.add_patch(neuron_circ)

    neuron_title_txt = ax.text(neuron_center[0], neuron_center[1] + 0.45,
                                'Sum + Threshold', fontsize=10.5, ha='center',
                                va='center', color='white', fontweight='bold',
                                zorder=6)
    neuron_z_txt = ax.text(neuron_center[0], neuron_center[1] - 0.05,
                            '', fontsize=10.5, ha='center', va='center',
                            color='white', zorder=6)
    neuron_theta_txt = ax.text(neuron_center[0], neuron_center[1] - 0.55,
                                '', fontsize=10.5, ha='center', va='center',
                                color='white', zorder=6)

    # ── Calculation annotation box below the neuron (dark text, light panel) ─
    # This sits clear of the tomato-coloured neuron body, so the text never
    # has to compete with the circle's edge or fade into a white-on-white
    # area. A short leader line connects it back to the neuron body.
    calc_box_center = np.array([neuron_center[0], neuron_center[1] - 2.55])
    calc_leader = FancyArrowPatch(
        (neuron_center[0], neuron_center[1] - neuron_r),
        (calc_box_center[0], calc_box_center[1] + 0.62),
        arrowstyle='-', color='#999999', lw=1.0, linestyle=(0, (3, 2)),
        zorder=2)
    ax.add_patch(calc_leader)

    calc_box = FancyBboxPatch(
        (calc_box_center[0] - 3.1, calc_box_center[1] - 0.75),
        6.2, 1.5,
        boxstyle='round,pad=0.08,rounding_size=0.12',
        facecolor='#f7f7f7', edgecolor='#bbbbbb', lw=1.0, zorder=4)
    ax.add_patch(calc_box)

    calc_txt = ax.text(calc_box_center[0], calc_box_center[1], '',
                        fontsize=10.5, ha='center', va='center',
                        color=COL_TEXT, zorder=5, linespacing=1.8)
    weight_arrows = []
    weight_texts = []
    for y, wlab in zip(input_ys, w_labels):
        start = np.array([input_x + 0.32, y])
        direction = neuron_center - start
        direction = direction / np.linalg.norm(direction)
        end = neuron_center - direction * neuron_r
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                 color=COL_ARROW, lw=1.4, zorder=3,
                                 shrinkA=0, shrinkB=0)
        ax.add_patch(arrow)
        mid = (start + end) / 2
        perp = np.array([-direction[1], direction[0]]) * 0.32
        txt = ax.text(mid[0] + perp[0], mid[1] + perp[1], wlab, fontsize=11,
                       ha='center', va='center', color=COL_TEXT)
        weight_arrows.append(arrow)
        weight_texts.append(txt)

    # ── Output arrow (mutable: colour, style, and label change live) ────────
    out_start = neuron_center + np.array([neuron_r, 0])
    out_end = np.array([output_x, 0])
    out_arrow = FancyArrowPatch(out_start, out_end, arrowstyle='-|>',
                                 mutation_scale=20, color=COL_SILENT, lw=2.4,
                                 zorder=3)
    ax.add_patch(out_arrow)
    out_label = ax.text((out_start[0] + out_end[0]) / 2, 0.45,
                         'output $y$', fontsize=11.5, ha='center',
                         color=COL_TEXT, fontweight='bold')

    # output state badge (firing / silent), positioned beyond the arrow tip
    out_badge = ax.text(output_x + 0.55, 0.0, '', fontsize=12, ha='left',
                         va='center', color=COL_TEXT, fontweight='bold')

    # ── Readout panel beneath the diagram ────────────────────────────────────
    readout = widgets.Output()

    # ── Update function ───────────────────────────────────────────────────────
    def update(x1, x2, x3, w1, w2, w3, theta):
        xs = [x1, x2, x3]
        ws = [w1, w2, w3]
        terms = [w * x for w, x in zip(ws, xs)]
        z = sum(terms)
        fires = z >= theta

        # update input node fill: a binary "no signal" input is shown faded
        for c, t, x in zip(input_circles, input_texts, xs):
            if x > 0:
                c.set_facecolor(COL_INPUT)
                c.set_alpha(1.0)
            else:
                c.set_facecolor(COL_INPUT)
                c.set_alpha(0.35)

        # update weight arrows: colour shows sign, thickness shows magnitude
        for arrow, txt, w, x in zip(weight_arrows, weight_texts, ws, xs):
            is_active = x > 0
            colour = COL_ARROW_NEG if w < 0 else COL_ARROW
            arrow.set_color(colour if is_active else '#cccccc')
            arrow.set_linewidth(1.0 + 1.6 * min(abs(w), 2.0))
            txt.set_color(COL_TEXT if is_active else '#bbbbbb')

        # update neuron body text (kept short so it always fits inside the
        # circle; the step-by-step arithmetic now lives in the calculation
        # box below, drawn on a light background so it never disappears)
        neuron_z_txt.set_text(rf'$z = \sum_i w_i x_i = {z:.2f}$')
        neuron_theta_txt.set_text(rf'$\theta$ = {theta:.2f}')

        # build the step-by-step calculation text for the box below the neuron
        term_str = ' + '.join(f'({w:.1f})({x:.0f})' for w, x in zip(ws, xs))
        if fires:
            compare_str = f'{z:.2f}  \u2265  {theta:.2f}   \u2192   fires (y = 1)'
        else:
            compare_str = f'{z:.2f}  <  {theta:.2f}   \u2192   silent (y = 0)'
        calc_txt.set_text(
            f'$z = {term_str} = {z:.2f}$\n'
            f'compare to threshold:   {compare_str}'
        )
        calc_txt.set_color(COL_FIRE_E if fires else COL_TEXT)
        calc_box.set_edgecolor(COL_FIRE if fires else '#bbbbbb')
        calc_box.set_facecolor('#eef6ec' if fires else '#f7f7f7')

        # update output arrow, label, and badge
        if fires:
            out_arrow.set_color(COL_FIRE)
            out_label.set_text('output $y = 1$')
            out_label.set_color(COL_FIRE_E)
            out_badge.set_text('NEURON FIRES')
            out_badge.set_color(COL_FIRE_E)
        else:
            out_arrow.set_color(COL_SILENT)
            out_label.set_text('output $y = 0$')
            out_label.set_color(COL_SILENT_E)
            out_badge.set_text('neuron silent')
            out_badge.set_color(COL_SILENT_E)

        # printed step-by-step readout
        readout.clear_output(wait=True)
        with readout:
            print(f"{'Input':<10}{'Weight':<10}{'Contribution (w_i * x_i)':<28}")
            print('-' * 48)
            for i, (x, w, term) in enumerate(zip(xs, ws, terms), start=1):
                print(f'x_{i} = {x:<4.0f}{"w_" + str(i) + " = " + format(w, ".2f"):<10}'
                      f'{term:<28.2f}')
            print('-' * 48)
            print(f'z = sum of contributions = {z:.2f}')
            print(f'threshold theta = {theta:.2f}')
            if fires:
                print(f'z >= theta, so the neuron FIRES -> output y = 1')
            else:
                print(f'z < theta, so the neuron stays SILENT -> output y = 0')

        fig.canvas.draw_idle()

    # ── Sliders: inputs (binary), weights, threshold ─────────────────────────
    def make_input_slider(label, default):
        return FloatSlider(value=default, min=0.0, max=1.0, step=1.0,
                            description=label,
                            style={'description_width': '50px'},
                            layout=widgets.Layout(width='240px'),
                            continuous_update=True)

    def make_weight_slider(label, default):
        return FloatSlider(value=default, min=-2.0, max=2.0, step=0.1,
                            description=label,
                            style={'description_width': '50px'},
                            layout=widgets.Layout(width='240px'),
                            continuous_update=True)

    x1_s = make_input_slider('x_1', DEFAULT_X[0])
    x2_s = make_input_slider('x_2', DEFAULT_X[1])
    x3_s = make_input_slider('x_3', DEFAULT_X[2])

    w1_s = make_weight_slider('w_1', DEFAULT_W[0])
    w2_s = make_weight_slider('w_2', DEFAULT_W[1])
    w3_s = make_weight_slider('w_3', DEFAULT_W[2])

    theta_s = FloatSlider(value=DEFAULT_THETA, min=-2.0, max=4.0, step=0.1,
                           description='theta',
                           style={'description_width': '50px'},
                           layout=widgets.Layout(width='240px'),
                           continuous_update=True)

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                                layout=widgets.Layout(width='100px'))

    def on_reset(b):
        x1_s.value, x2_s.value, x3_s.value = DEFAULT_X
        w1_s.value, w2_s.value, w3_s.value = DEFAULT_W
        theta_s.value = DEFAULT_THETA

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>McCulloch-Pitts neuron — adjust inputs, weights, and threshold</b>'),
        HBox([VBox([HTML('<i>Inputs (0 or 1)</i>'), x1_s, x2_s, x3_s]),
              VBox([HTML('<i>Weights</i>'), w1_s, w2_s, w3_s]),
              VBox([HTML('<i>Threshold</i>'), theta_s])]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        'x1': x1_s, 'x2': x2_s, 'x3': x3_s,
        'w1': w1_s, 'w2': w2_s, 'w3': w3_s,
        'theta': theta_s,
    })

    display(controls, out)
    display(readout)