"""
Figure 4a — Rise and Run: How Slope is Measured
=================================================

Two side-by-side panels illustrating the concept of slope using rise and run:

Left panel:  a straight line with the rise and run clearly marked, showing
             that slope = rise / run and is constant everywhere on the line.

Right panel: a curve with a tangent line drawn at one specific point,
             showing that the slope of a curve is measured at a single point
             using the tangent line — and is different at every point.

This figure introduces the idea of slope before gradient descent is discussed.

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


def show():
    """Render the static Figure 4a rise and run illustration."""

    plt.close('Notebook4 Figure 4a')

    fig, axes = plt.subplots(1, 2, num='Notebook4 Figure 4a', figsize=(10, 4))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: rise and run on a straight line ───────────────────────────
    ax = axes[0]

    # A simple straight line from (0,1) to (4,3) — slope = rise/run = 2/4 = 0.5
    x = np.array([0, 4])
    y = np.array([1, 3])
    ax.plot(x, y, color='steelblue', linewidth=2.5)

    # Run arrow — horizontal distance traveled along the x-axis
    ax.annotate('', xy=(4, 1), xytext=(0, 1),
                arrowprops=dict(arrowstyle='->', color='tomato', lw=2))
    ax.text(2, 0.7, 'run = 4', ha='center', color='tomato', fontsize=11)

    # Rise arrow — vertical distance gained along the y-axis
    ax.annotate('', xy=(4, 3), xytext=(4, 1),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(4.3, 2, 'rise = 2', ha='left', color='green', fontsize=11)

    # Slope calculation label
    ax.text(1.5, 2.5, 'slope = rise / run\n       = 2 / 4 = 0.5',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Slope of a straight line\n'
                 '(same everywhere on the line)', fontsize=11)
    ax.grid(True, alpha=0.2)

    # ── Right panel: rise and run on a curve ──────────────────────────────────
    ax = axes[1]

    # A gentle curve: y = x²/4
    x_curve = np.linspace(0, 4, 300)
    y_curve  = x_curve ** 2 / 4
    ax.plot(x_curve, y_curve, color='steelblue', linewidth=2.5, label='Curve')

    # Choose a specific point on the curve to measure the slope at
    # Using a smaller x value so the rise arrow stays within the plot area
    pt_x  = 1.5
    pt_y  = pt_x ** 2 / 4
    slope = pt_x / 2   # derivative of x²/4 = x/2

    # Draw the tangent line at that point — a short segment centred on the point
    half = 0.7
    tx   = np.array([pt_x - half, pt_x + half])
    ty   = pt_y + slope * (tx - pt_x)
    ax.plot(tx, ty, color='orange', linewidth=2, linestyle='--',
            label='Tangent line at this point')

    # Mark the point itself
    ax.scatter([pt_x], [pt_y], color='black', s=80, zorder=5)

    # Run arrow along the tangent
    ax.annotate('', xy=(pt_x + half, pt_y), xytext=(pt_x, pt_y),
                arrowprops=dict(arrowstyle='->', color='tomato', lw=2))
    ax.text(pt_x + half / 2, pt_y - 0.08, 'run', ha='center',
            color='tomato', fontsize=10)

    # Rise arrow along the tangent
    ax.annotate('', xy=(pt_x + half, pt_y + slope * half),
                xytext=(pt_x + half, pt_y),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(pt_x + half + 0.1, pt_y + slope * half / 2, 'rise',
            ha='left', color='green', fontsize=10)

    # Slope label
    ax.text(0.15, 0.9,
            f'At this point:\nslope = rise / run\n      = {slope:.2f}',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, 1.4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Slope of a curve — measured at one point\n'
                 'using the tangent line (different at every point)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Figure 4a: Rise and run — how slope is measured on a line and on a curve',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()