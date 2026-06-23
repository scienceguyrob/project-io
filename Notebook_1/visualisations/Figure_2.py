"""
Figure 2 — Interactive Straight-Line Explorer
==============================================

Displays y = mx + c with sliders for m (slope) and c (intercept).
The plot is built once and updated in place — no flickering.

Usage in notebook (run %matplotlib widget first):
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
import ipywidgets as widgets
from ipywidgets import interactive_output, FloatSlider, VBox


def show():
    """
    Render the interactive Figure 2 plot inline in a Jupyter notebook.

    The figure is built once and all subsequent slider interactions mutate
    existing plot artists (line data, dot position, labels) rather than
    redrawing the figure — this keeps updates smooth and flicker-free.

    Requires %matplotlib widget to have been called in the notebook first.
    """

    # ── Fixed x-values ───────────────────────────────────────────────────────
    # Computed once at setup and reused on every slider update.
    # np.linspace(start, stop, num) returns 'num' evenly spaced values.
    x = np.linspace(-5, 5, 200)

    # ── Build the figure ONCE ─────────────────────────────────────────────────
    # All artists (line, intercept dot) are stored in variables so they can
    # be mutated later without triggering a full redraw.
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.resizable = True
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = False
    fig.canvas.toolbar_position = 'right'

    # The line y = mx + c — starts at y = x (i.e. m=1, c=0)
    # The comma after 'line' unpacks the single-element list returned by ax.plot()
    (line,) = ax.plot(x, x, color='royalblue', linewidth=2, label='y = 1x + 0')

    # Red dot marking the y-intercept: the point (0, c) where the line crosses y-axis
    (intercept,) = ax.plot(
        [0], [0], 'o',
        color='red', zorder=5, markersize=9,
        label='y-intercept: (0, 0)'
    )

    # Dashed reference lines at x=0 and y=0 (the axes themselves)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Figure 2: Straight line y = 1x + 0')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Fix the y-axis range so the plot never jumps or rescales during interaction
    ax.set_ylim(-12, 12)

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    # Called automatically by interactive_output whenever a slider value changes.
    # Only mutates existing artists — never creates new ones or calls plt.show().
    def update(m, c):
        # Recalculate y-values for the new slope and intercept
        y = m * x + c

        # Move the line to its new position
        line.set_ydata(y)

        # Move the intercept dot to (0, c)
        intercept.set_ydata([c])

        # Update legend labels to reflect current m and c values
        line.set_label(f'y = {m}x + {c}')
        intercept.set_label(f'y-intercept: (0, {c})')

        # Update the figure title
        ax.set_title(f'Figure 2: Straight line y = {m}x + {c}')
        ax.legend(loc='upper left')

        # draw_idle() schedules a lightweight repaint — much cheaper than plt.show()
        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    # FloatSlider creates a draggable slider widget.
    #   value           = starting value
    #   min / max       = allowed range
    #   step            = increment per tick
    #   description     = label shown beside the slider
    #   continuous_update = True means the plot updates while dragging,
    #                       not just on release
    m_slider = FloatSlider(
        value=1.0, min=-5.0, max=5.0, step=0.1,
        description='m (slope)',
        style={'description_width': '90px'},
        continuous_update=True
    )
    c_slider = FloatSlider(
        value=0.0, min=-10.0, max=10.0, step=0.5,
        description='c (intercept)',
        style={'description_width': '90px'},
        continuous_update=True
    )

    # ── Wire sliders to the update function ───────────────────────────────────
    # interactive_output links each slider to the matching parameter in update().
    # VBox stacks the two sliders vertically above the plot.
    controls = VBox([m_slider, c_slider])
    out = interactive_output(update, {'m': m_slider, 'c': c_slider})

    display(controls, out)