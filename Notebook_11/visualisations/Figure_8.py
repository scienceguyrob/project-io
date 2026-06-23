"""
Figure 8 — Polynomial Regression: Underfitting to Overfitting
=======================================================================
Illustrates the bias-variance trade-off using polynomial regression at
three degrees on a small synthetic dataset. Each panel shows a different
point on the complexity spectrum:

  Degree 1  — high bias, low variance (underfitting): the model is too
              simple to capture the true sinusoidal pattern and is
              systematically wrong on both training and test data.

  Degree 4  — balanced: enough capacity to approximate the true function
              without fitting the noise.

  Degree 20 — low bias, high variance (overfitting): the model has enough
              capacity to pass through every training point, including the
              noise, but generalises poorly to new data.

The true underlying function (sin(2x) + 0.5x, shown as a dashed line) is
known because the data is synthetic, which allows the fit to be compared
directly against ground truth rather than just against held-out test error.

This figure is entirely self-contained and requires no external variables.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_8 import show
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

import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.exceptions import RankWarning


# ── Dataset parameters ───────────────────────────────────────────────────────
# A small sample size (30 points) is deliberate: with too many points the
# high-degree polynomial has less room to exhibit dramatic overfitting, so
# the contrast between the three panels is less striking.
N_POINTS    = 30
NOISE_STD   = 0.4     # standard deviation of Gaussian noise added to observations
RANDOM_SEED = 13      # fixed for reproducibility across notebook runs

# x range for the training data and for the smooth plot curve. The plot
# range extends slightly beyond the data range (-0.1 to 3.1) so the fitted
# curves are visible at the edges rather than terminating abruptly.
X_MIN       = 0.0
X_MAX       = 3.0
X_PLOT_MIN  = -0.1
X_PLOT_MAX  = 3.1
N_PLOT_PTS  = 300

# Y axis limits: chosen to show the true function and reasonable fits clearly
# while clipping the degree-20 curve's extreme extrapolation excursions.
Y_MIN = -3.0
Y_MAX =  6.0

# The three polynomial degrees shown, one per panel.
DEGREES = [1, 4, 20]

PANEL_LABELS = [
    'Degree 1 — UNDERFITTING\n(high bias, low variance)',
    'Degree 4 — GOOD FIT\n(balanced)',
    'Degree 20 — OVERFITTING\n(low bias, high variance)',
]

PANEL_COLOURS = ['steelblue', 'seagreen', 'tomato']


def _true_function(x):
    """The noiseless ground-truth function the data was sampled from."""
    return np.sin(2 * x) + 0.5 * x


def show():
    """Render Figure 8: polynomial regression from underfitting to overfitting."""
    plt.close('Notebook11 Figure 8')

    # ── Generate synthetic dataset ────────────────────────────────────────────
    rng   = np.random.default_rng(RANDOM_SEED)
    x_tr  = np.linspace(X_MIN, X_MAX, N_POINTS)
    y_tr  = _true_function(x_tr) + rng.normal(0, NOISE_STD, N_POINTS)

    x_plot = np.linspace(X_PLOT_MIN, X_PLOT_MAX, N_PLOT_PTS)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5), num='Notebook11 Figure 8')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    for ax, degree, label, colour in zip(axes, DEGREES, PANEL_LABELS, PANEL_COLOURS):

        # np.polyfit fits the polynomial by minimising least squares; the
        # returned coefficients are in descending power order.
        # np.polyval evaluates those coefficients at new x values.
        # RankWarning is expected for degree 20 on 30 points (the Vandermonde
        # matrix becomes ill-conditioned), but the fit is still illustratively
        # useful, so the warning is suppressed here rather than propagating it
        # to the notebook output and confusing the reader.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RankWarning)
            coeffs = np.polyfit(x_tr, y_tr, degree)
        y_fit_plot  = np.polyval(coeffs, x_plot)
        y_fit_train = np.polyval(coeffs, x_tr)

        # Train RMSE is computed on the training points only: it measures how
        # well the polynomial has fitted the data it was given, not how well
        # it will generalise. For degree 20, train RMSE will be very low
        # because the polynomial passes through or very close to every point.
        train_rmse = np.sqrt(np.mean((y_tr - y_fit_train) ** 2))

        ax.scatter(x_tr, y_tr,
                   color='black', s=40, zorder=4, alpha=0.7, label='Observed data')
        ax.plot(x_plot, _true_function(x_plot),
                'k--', lw=1.5, alpha=0.4, label='True function')

        # np.clip prevents the degree-20 curve's extreme extrapolation values
        # from distorting the y-axis scale and making the other panels
        # unreadable.
        ax.plot(x_plot, np.clip(y_fit_plot, Y_MIN, Y_MAX),
                color=colour, lw=2.5, label=f'Degree {degree} fit')

        ax.set_title(f'{label}\nTrain RMSE = {train_rmse:.3f}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        'Figure 8: Polynomial regression — underfitting to overfitting',
        fontsize=12,
    )
    plt.tight_layout()