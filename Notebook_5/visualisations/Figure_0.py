"""
Figure 0 — Simple Regression Prediction Illustration
=====================================================

A minimal plot showing the fitted line y = 50 + 3x with a single
prediction highlighted: for a house with floor area x = 80 m², the
model predicts a sale price of ŷ = 290 £k.

Dashed red lines show how to read the prediction off the plot —
a vertical line from x = 80 up to the fitted line, then a horizontal
line across to the y-axis.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_0 import show
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
    """Render the static Figure 0 prediction illustration."""

    plt.close('Notebook5 Figure 0')

    beta_0, beta_1 = 50, 3
    x_plot = np.linspace(30, 160, 300)
    y_plot = beta_0 + beta_1 * x_plot

    fig, ax = plt.subplots(num='Notebook5 Figure 0', figsize=(8, 4))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # The fitted line: ŷ = β₀ + β₁x
    ax.plot(x_plot, y_plot, color='black', linewidth=2,
            label=r'$\hat{y} = 50 + 3x$')

    # Dashed red lines showing how the prediction is read off the plot
    ax.plot([80, 80], [0,  290], color='tomato', linewidth=1.5, linestyle='--')
    ax.plot([0,  80], [290, 290], color='tomato', linewidth=1.5, linestyle='--')

    # Mark the prediction point
    ax.scatter([80], [290], color='tomato', s=100, zorder=5)
    ax.text(83, 285, r'$\hat{y} = 290$ £k', fontsize=10, color='tomato')
    ax.text(82,  15, r'$x = 80$ m²',        fontsize=10, color='tomato')

    ax.set_xlabel('Floor area (m²)')
    ax.set_ylabel('Predicted price (£k)')
    ax.set_title(
        r'$\hat{y} = 50 + 3 \times x$  —  example prediction at $x = 80$ m²'
    )
    ax.set_xlim(30, 160)
    ax.set_ylim(0, 540)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()