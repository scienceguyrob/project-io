"""
Figure 9 — The Dartboard Analogy for Bias and Variance
=======================================================================
Illustrates the four combinations of bias and variance using a dartboard
analogy. Each panel shows 40 dots, where each dot represents the prediction
made by one model trained on a different random training sample. The red
cross marks the bullseye (the true target value) and the gold star marks
the mean of all 40 predictions.

  Spread of dots around the gold star = variance
  Distance of gold star from bullseye = bias

The four panels cover the full combination space:

  Low bias, low variance   — dots cluster tightly around the bullseye.
                             The ideal outcome.
  Low bias, high variance  — dots are centred on the bullseye on average
                             but are scattered widely. The mean is right
                             but any individual prediction is unreliable.
  High bias, low variance  — dots cluster tightly, but around the wrong
                             point. The model is consistently wrong in
                             the same direction.
  High bias, high variance — dots are both off-centre and scattered.
                             The worst outcome: wrong and inconsistent.

This figure is entirely self-contained and requires no external variables.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_9 import show
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


# Fixed random seed so the dart positions are the same on every notebook run.
RANDOM_SEED = 5

# Number of dart throws (model predictions) shown per panel.
N_DARTS = 40

# Dartboard ring radii, drawn from outermost to innermost. Each ring is
# rendered with a progressively lighter alpha so the board reads as a
# series of concentric zones rather than a solid filled circle.
RING_RADII  = [1.00, 0.75, 0.50, 0.25]
RING_ALPHAS = [0.05, 0.08, 0.12, 0.20]   # outermost to innermost
RING_COLOUR = 'gray'

# Axis limits: slightly larger than the outermost ring radius so the ring
# edge is not clipped by the axes boundary.
AXIS_LIM = 1.2

# Marker sizes and line widths for the bullseye and mean-prediction markers.
BULLSEYE_SIZE  = 180
BULLSEYE_LW    = 3
MEAN_PRED_SIZE = 120
MEAN_PRED_LW   = 0.8
DART_SIZE      = 40
DART_LW        = 0.3

# The four bias-variance scenarios. Each entry is:
#   (title, bias_centre, spread, colour)
# bias_centre is the [x, y] offset of the cluster from the bullseye at (0, 0).
# spread is the standard deviation of the Gaussian scatter around that centre.
SCENARIOS = [
    ('Low bias\nLow variance',    [0.00, 0.00], 0.15, 'seagreen'),
    ('Low bias\nHigh variance',   [0.00, 0.00], 0.60, 'steelblue'),
    ('High bias\nLow variance',   [0.55, 0.45], 0.15, 'goldenrod'),
    ('High bias\nHigh variance',  [0.50, 0.40], 0.55, 'tomato'),
]


def show():
    """Render Figure 9: the bias-variance dartboard analogy."""
    plt.close('Notebook11 Figure 9')

    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(1, 4, figsize=(10, 4), num='Notebook11 Figure 9')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    for ax, (title, bias_centre, spread, colour) in zip(axes, SCENARIOS):

        # Each dart is drawn from a 2D Gaussian centred at bias_centre with
        # standard deviation spread in both x and y. bias_centre = [0, 0]
        # means low bias (cluster centred on the bullseye); larger spread
        # values correspond to higher variance.
        darts = rng.normal(bias_centre, spread, (N_DARTS, 2))

        # Draw the board rings from largest to smallest so smaller rings
        # are rendered on top and remain visible.
        for radius, alpha in zip(RING_RADII, RING_ALPHAS):
            ax.add_patch(plt.Circle(
                (0, 0), radius,
                color=RING_COLOUR, fill=True, alpha=alpha,
            ))
            ax.add_patch(plt.Circle(
                (0, 0), radius,
                color=RING_COLOUR, fill=False, lw=0.8,
            ))

        # Individual model predictions.
        ax.scatter(
            darts[:, 0], darts[:, 1],
            color=colour, s=DART_SIZE, alpha=0.8,
            edgecolors='k', lw=DART_LW, zorder=4,
        )

        # Bullseye: the true target value (always at the origin).
        ax.scatter(
            0, 0,
            s=BULLSEYE_SIZE, color='red', marker='+',
            lw=BULLSEYE_LW, zorder=6,
        )

        # Mean prediction across the 40 models: distance from the bullseye
        # is the empirical bias for this scenario.
        ax.scatter(
            np.mean(darts[:, 0]), np.mean(darts[:, 1]),
            s=MEAN_PRED_SIZE, color='gold', marker='*',
            edgecolors='k', lw=MEAN_PRED_LW, zorder=5,
        )

        ax.set_xlim(-AXIS_LIM, AXIS_LIM)
        ax.set_ylim(-AXIS_LIM, AXIS_LIM)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10, color=colour)

    fig.suptitle(
        'Figure 9: Bias-Variance dartboard analogy\n'
        'Red + = true target.  Gold \u2605 = mean prediction.  Dots = individual models.',
        fontsize=11,
    )
    plt.tight_layout()