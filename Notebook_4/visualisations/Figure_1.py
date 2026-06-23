"""
Figure 1 — Raw Fruit Data: Apples vs Oranges by Diameter
=========================================================

A 1D scatter plot showing synthetic fruit diameter measurements for two
classes — apples and oranges — plotted along a single horizontal axis.
All points sit on y = 0 so the focus is entirely on the diameter values
and how much the two classes overlap.

This figure introduces the classification dataset used throughout Lab 4.

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


def show():
    """Render the static Figure 1 fruit data scatter plot."""

    plt.close('Notebook4 Figure 1')

    # ── Data generation ───────────────────────────────────────────────────────
    # A fixed seed ensures the data is identical every time show() is called,
    # so the plot always looks the same regardless of when the cell is run.
    rng = np.random.default_rng(0)

    # rng.normal(loc, scale, size) draws samples from a normal distribution.
    # loc  = mean diameter (cm)
    # scale = standard deviation — controls how spread out the values are
    apple_diameters  = rng.normal(loc=7.5, scale=0.8, size=30)   # apples: smaller
    orange_diameters = rng.normal(loc=9.5, scale=0.9, size=30)   # oranges: larger

    # Combine into a single feature array x and a label array y.
    # np.concatenate joins the two arrays end-to-end into one flat array.
    x = np.concatenate([apple_diameters, orange_diameters])

    # Labels: 0 = Apple, 1 = Orange.
    # np.array([0] * 30 + [1] * 30) creates [0, 0, ..., 1, 1, ...] with 30 of each.
    y = np.array([0] * 30 + [1] * 30)

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(num='Notebook4 Figure 1', figsize=(5, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # All points sit at y = 0 — this is a 1D scatter plot.
    # The y position carries no information; only the x (diameter) matters.
    ax.scatter(apple_diameters,  [0] * 30, color='green',  s=70, alpha=0.7,
               edgecolors='k', lw=0.4, label='Apple (label=0)', zorder=3)
    ax.scatter(orange_diameters, [0] * 30, color='orange', s=70, alpha=0.7,
               edgecolors='k', lw=0.4, label='Orange (label=1)', zorder=3)

    ax.set_xlabel('Diameter (cm)')
    ax.set_yticks([])   # hide y-axis ticks — the y position is meaningless here
    ax.set_title('Figure 1: Raw fruit data — apples vs oranges by diameter')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.show()

    # Print a summary so users can see the key statistics in the cell output
    print(f"Total samples: {len(x)}  |  Apples: {(y==0).sum()}  |  Oranges: {(y==1).sum()}")
    print(f"Apple range:   {apple_diameters.min():.2f} – {apple_diameters.max():.2f} cm")
    print(f"Orange range:  {orange_diameters.min():.2f} – {orange_diameters.max():.2f} cm")