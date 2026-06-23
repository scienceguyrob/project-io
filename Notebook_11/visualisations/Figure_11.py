"""
Figure 11 — Ridge (L2) Regularisation on a Degree-15 Polynomial
=======================================================================
Shows how increasing the L2 (Ridge) regularisation penalty tames an
overfitted degree-15 polynomial, transitioning from a curve that wildly
oscillates through every training point to one that smoothly captures
the underlying trend.

The figure contains four panels, one per regularisation strength, arranged
from no regularisation (left) to strong regularisation (right):

  alpha=0.0001  — effectively unregularised: the polynomial oscillates
                  violently and fits the noise as well as the signal.
  alpha=0.1     — mild regularisation: oscillations are reduced but the
                  curve still chases some noise.
  alpha=10.0    — moderate regularisation: the curve is smooth and closely
                  follows the true underlying function.
  alpha=1000.0  — strong regularisation: coefficients are heavily suppressed
                  and the curve begins to underfit, losing detail.

The true underlying function (sin(2x) + 0.5x, shown as a dashed line in
each panel) is known because the data is synthetic, allowing each fit to
be compared directly against ground truth.

Ridge regularisation is implemented here using sklearn's Ridge regressor
applied to polynomial features rather than np.polyfit, because np.polyfit
does not support regularisation. The PolynomialFeatures transformer expands
the single input feature x into [1, x, x^2, ..., x^15], and Ridge then
fits a linear model over those features with the L2 penalty.

This figure is entirely self-contained and requires no external variables.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_11 import show
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


# ── Dataset parameters ───────────────────────────────────────────────────────
# Kept identical to Figure 8 so the reader is looking at the same underlying
# data and true function, making the regularisation effect directly comparable.
N_POINTS    = 30
NOISE_STD   = 0.4
RANDOM_SEED = 13
X_MIN       = 0.0
X_MAX       = 3.0
X_PLOT_MIN  = -0.1
X_PLOT_MAX  = 3.1
N_PLOT_PTS  = 300
Y_MIN       = -3.0
Y_MAX       =  6.0

# Polynomial degree: high enough to overfit badly without regularisation.
POLY_DEGREE = 15

# The four regularisation strengths to show (passed to Ridge as alpha=,
# sklearn's name for the lambda penalty weight). Larger values mean a
# stronger penalty and a smoother fit. Values are spaced logarithmically
# to span a wide range clearly.
LAMBDAS = [0.0001, 0.1, 10.0, 1000.0]

PANEL_LABELS = [
    r'$\lambda$ = 0.0001' + '\n(no regularisation)',
    r'$\lambda$ = 0.1' + '\n(mild)',
    r'$\lambda$ = 10' + '\n(moderate)',
    r'$\lambda$ = 1000' + '\n(strong)',
]

# Colour transitions from tomato (overfit) through gold to steelblue (underfit),
# giving a visual gradient that mirrors the complexity axis.
PANEL_COLOURS = ['tomato', 'goldenrod', 'seagreen', 'steelblue']


def _true_function(x):
    """The noiseless ground-truth function the data was sampled from."""
    return np.sin(2 * x) + 0.5 * x


def show():
    """Render Figure 11: Ridge regularisation taming a degree-15 polynomial."""
    plt.close('Notebook11 Figure 11')

    # ── Dataset ───────────────────────────────────────────────────────────────
    rng  = np.random.default_rng(RANDOM_SEED)
    x_tr = np.linspace(X_MIN, X_MAX, N_POINTS)
    y_tr = _true_function(x_tr) + rng.normal(0, NOISE_STD, N_POINTS)

    x_plot = np.linspace(X_PLOT_MIN, X_PLOT_MAX, N_PLOT_PTS)

    # Reshape to (n, 1) because sklearn expects a 2D feature matrix.
    X_tr   = x_tr.reshape(-1, 1)
    X_plot = x_plot.reshape(-1, 1)

    fig, axes = plt.subplots(1, 4, figsize=(11, 4.5), num='Notebook11 Figure 11')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    for ax, lam, label, colour in zip(axes, LAMBDAS, PANEL_LABELS, PANEL_COLOURS):

        # Ridge(alpha=lam) passes lambda to sklearn under its own parameter
        # name. The two are identical: alpha in sklearn = lambda in the formula.
        pipeline = Pipeline([
            ('poly',   PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge',  Ridge(alpha=lam)),
        ])

        pipeline.fit(X_tr, y_tr)
        y_fit_plot  = pipeline.predict(X_plot)
        y_fit_train = pipeline.predict(X_tr)
        train_rmse  = np.sqrt(np.mean((y_tr - y_fit_train) ** 2))

        # Coefficient magnitudes: the L2 penalty drives these toward zero as
        # lambda increases, which is what produces the smoothing effect.
        coef_norm = np.linalg.norm(pipeline.named_steps['ridge'].coef_)

        ax.scatter(x_tr, y_tr,
                   color='black', s=40, zorder=4, alpha=0.7, label='Observed data')
        ax.plot(x_plot, _true_function(x_plot),
                'k--', lw=1.5, alpha=0.4, label='True function')

        # Clip extreme values so an unregularised curve's oscillations do not
        # distort the y-axis and make the other panels unreadable.
        ax.plot(x_plot, np.clip(y_fit_plot, Y_MIN, Y_MAX),
                color=colour, lw=2.5, label=f'Ridge fit')

        ax.set_title(
            f'{label}\n'
            f'Train RMSE = {train_rmse:.3f}  |  ||w|| = {coef_norm:.1f}',
            fontsize=9,
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        'Figure 11: Ridge (L2) regularisation on a degree-15 polynomial\n'
        r'increasing $\lambda$ shrinks coefficients and smooths the fit',
        fontsize=11,
    )
    plt.tight_layout()