"""
Figure 1 — Motivation for Mixture Models: A Single Gaussian Is Not Enough
=======================================================================
Illustrates why a single Gaussian distribution is a poor model for
multi-modal data, motivating the need for mixture models.

The figure generates data from two well-separated Gaussians (a 1-D
bi-modal mixture), fits a single Gaussian to the combined data by
estimating the mean and standard deviation from all observations, and
then overlays the true mixture density for comparison.

The single Gaussian's mean lands in the gap between the two sub-groups,
a value that is not representative of either. Its spread is inflated by
the separation between the two groups, making the fitted density broad
and flat. The true mixture density (a weighted sum of the two component
Gaussians) fits the two peaks correctly.

This figure is entirely self-contained and requires no external variables.

Usage
-----
In a Jupyter notebook cell:

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


# ── Dataset parameters ───────────────────────────────────────────────────────
# Two well-separated groups in 1-D, chosen so the gap between them is
# clearly visible in the histogram and the single Gaussian's mean falls
# squarely in that gap.
N_GROUP_1   = 200
N_GROUP_2   = 150
MU_1        = 2.0     # centre of group 1
STD_1       = 0.6     # spread of group 1
MU_2        = 7.0     # centre of group 2
STD_2       = 0.9     # spread of group 2
RANDOM_SEED = 10

# x range for the density curves: extends slightly beyond the data range
# so the tails of the fitted Gaussians are visible.
X_PLOT_MIN  = -1.0
X_PLOT_MAX  = 11.0
N_PLOT_PTS  = 400

# Histogram bin count: enough to show the two peaks clearly without
# over-smoothing on 350 total samples.
N_BINS = 40

# Colours consistent with the notebook's palette.
COLOUR_HIST        = 'lightsteelblue'
COLOUR_SINGLE      = 'tomato'
COLOUR_TRUE_MIX    = 'seagreen'
COLOUR_GRID        = '#cccccc'


def _gaussian_pdf(x, mu, std):
    """Evaluate the Gaussian probability density function at each value in x."""
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)


def show():
    """Render Figure 1: single Gaussian vs true mixture density on bi-modal data."""
    plt.close('Notebook8bonus Figure 1')

    # ── Generate data ─────────────────────────────────────────────────────────
    rng    = np.random.default_rng(RANDOM_SEED)
    group1 = rng.normal(loc=MU_1, scale=STD_1, size=N_GROUP_1)
    group2 = rng.normal(loc=MU_2, scale=STD_2, size=N_GROUP_2)
    X_mix  = np.concatenate([group1, group2])

    # ── Single Gaussian fit ───────────────────────────────────────────────────
    # Fitting a single Gaussian means estimating its mean and standard
    # deviation from all observations combined, ignoring the two-group
    # structure. The resulting mean lands in the gap between the groups
    # and the inflated standard deviation produces a broad, flat curve.
    mu_single  = X_mix.mean()
    std_single = X_mix.std()

    x_plot     = np.linspace(X_PLOT_MIN, X_PLOT_MAX, N_PLOT_PTS)
    pdf_single = _gaussian_pdf(x_plot, mu_single, std_single)

    # ── True mixture density ──────────────────────────────────────────────────
    # The true density is a weighted sum of the two component Gaussians,
    # where the weights are the proportion of total samples from each group.
    # This is the density that correctly describes the data-generating process.
    w1           = N_GROUP_1 / (N_GROUP_1 + N_GROUP_2)
    w2           = N_GROUP_2 / (N_GROUP_1 + N_GROUP_2)
    pdf_true_mix = (w1 * _gaussian_pdf(x_plot, MU_1, STD_1) +
                    w2 * _gaussian_pdf(x_plot, MU_2, STD_2))

    print('Figure 1: single Gaussian fit to bi-modal data')
    print(f'  Single Gaussian: mean = {mu_single:.2f}, std = {std_single:.2f}')
    print(f'  This mean sits in the gap between the two sub-populations')
    print(f'  (group 1 centre = {MU_1}, group 2 centre = {MU_2})')
    print(f'  and is not representative of either.')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), num='Notebook8bonus Figure 1')

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # density=True normalises the histogram so its area integrates to 1,
    # making it directly comparable to the probability density curves.
    ax.hist(X_mix, bins=N_BINS, density=True,
            color=COLOUR_HIST, edgecolor='white', lw=0.5, alpha=0.8,
            label='Observed data')

    ax.plot(x_plot, pdf_single,
            color=COLOUR_SINGLE, lw=2.5,
            label=f'Single Gaussian fit  (mean={mu_single:.2f}, std={std_single:.2f})')

    # The true mixture density is shown as a dashed line to visually
    # distinguish it from the fitted single Gaussian.
    ax.plot(x_plot, pdf_true_mix,
            color=COLOUR_TRUE_MIX, lw=2.5, linestyle='--',
            label='True mixture density')

    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title(
        'A single Gaussian fails to capture bi-modal data\n'
        'the mixture density (dashed) fits both peaks correctly',
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, color=COLOUR_GRID, alpha=0.4)

    fig.suptitle(
        'Figure 1: Motivation for mixture models — '
        'single distributions are not always sufficient',
        fontsize=12,
    )
    plt.tight_layout()