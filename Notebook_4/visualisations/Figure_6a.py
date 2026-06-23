"""
Figure 6a — Squared Error Loss Function
========================================

A simple static plot showing the squared error loss function L(θ) = θ²,
illustrating why squaring the error is a natural way to measure how wrong
a prediction is.

Specific points are marked so users can read off the loss values directly
and see that larger mistakes are penalised more heavily than smaller ones.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_6a import show
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
    """Render the static Figure 6a squared error loss illustration."""

    plt.close('Notebook4 Figure 6a')

    theta = np.linspace(-4, 4, 300)
    loss  = theta ** 2

    fig, ax = plt.subplots(num='Notebook4 Figure 6a', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    ax.plot(theta, loss, color='steelblue', linewidth=2.5,
            label='L(θ) = θ²  (squared error)')

    # Each tuple: (theta value, label text, label x position, label y position)
    # Positions are set explicitly so nothing overlaps the axis or curve.
    points = [
        (-3, 'θ = -3,  loss = 9',             -1.8, 11.5),
        (-2, 'θ = -2,  loss = 4',             -1.5,  6.0),
        ( 0, 'θ = 0,   loss = 0  ← correct',  0.2,  1.5),
        ( 2, 'θ = 2,   loss = 4',              2.2,  6.0),
        ( 3, 'θ = 3,   loss = 9',              2.2, 11.5),
    ]

    for t, label, x_off, y_off in points:
        ax.scatter([t], [t**2], color='tomato', s=80, zorder=5)
        ax.annotate(label, xy=(t, t**2),
                    xytext=(x_off, y_off),
                    fontsize=8.5, color='tomato',
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=0.8))

    ax.axvline(0, color='green', linewidth=1.8, linestyle='--',
               label='Correct answer: θ = 0  →  loss = 0')

    ax.set_xlabel('Parameter value θ', fontsize=11)
    ax.set_ylabel('Loss  L(θ) = θ²', fontsize=11)
    ax.set_title('Figure 6a: Squared error loss — larger mistakes are penalised more heavily',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()