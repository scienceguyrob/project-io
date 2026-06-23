"""
Figure 6 — Four Key Assumptions of Linear Regression
=====================================================

A 2×2 grid of diagnostic plots checking the four assumptions of OLS
linear regression against the training set residuals:

    Top-left:     Linearity — residuals vs fitted values
    Top-right:    Independence — residuals vs observation index
    Bottom-left:  Homoscedasticity — |residuals| vs fitted values
    Bottom-right: Normality — histogram of residuals with fitted normal curve

Usage
-----
From a Jupyter notebook cell (after fitting the model)::

    %matplotlib widget
    from visualisations.Figure_6 import show
    show(model, X_train, y_train)

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


def show(model, X_train, y_train):
    """
    Render the static Figure 6 regression assumption diagnostic plots.

    Parameters
    ----------
    model   : fitted sklearn LinearRegression object
    X_train : training feature DataFrame
    y_train : training target Series
    """

    plt.close('Notebook5 Figure 6')

    # ── Compute fitted values and residuals ───────────────────────────────────
    y_hat_tr = model.predict(X_train)
    res_tr   = y_train - y_hat_tr

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, num='Notebook5 Figure 6', figsize=(10, 9))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── 1. Linearity: residuals vs fitted values ──────────────────────────────
    # If the linearity assumption holds, residuals should be scattered randomly
    # around zero with no systematic curve or pattern.
    ax = axes[0, 0]
    ax.scatter(y_hat_tr, res_tr, color='steelblue', s=45,
               edgecolors='k', linewidth=0.3, alpha=0.8)
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
    ax.set_xlabel(r'Fitted values $\hat{y}$')   # raw string prevents \h warning
    ax.set_ylabel('Residuals')
    ax.set_title('1. Linearity\nNo pattern expected')
    ax.grid(True, alpha=0.2)

    # ── 2. Independence: residuals vs observation index ───────────────────────
    # If observations are independent, residuals plotted in order should show
    # no trend, drift, or cyclical pattern — just random scatter around zero.
    ax = axes[0, 1]
    ax.plot(res_tr.values, 'o-', color='steelblue',
            markersize=4, linewidth=0.8, alpha=0.7)
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Observation index')
    ax.set_ylabel('Residuals')
    ax.set_title('2. Independence\nNo trend along index expected')
    ax.grid(True, alpha=0.2)

    # ── 3. Homoscedasticity: |residuals| vs fitted values ────────────────────
    # The spread of residuals should be roughly constant across all fitted values.
    # A fan shape (spread increasing with fitted value) indicates heteroscedasticity.
    ax = axes[1, 0]
    ax.scatter(y_hat_tr, np.abs(res_tr), color='seagreen', s=45,
               edgecolors='k', linewidth=0.3, alpha=0.8)
    ax.axhline(np.abs(res_tr).mean(), color='red', linewidth=1.5,
               linestyle='--', label='Mean |residual|')
    ax.set_xlabel(r'Fitted values $\hat{y}$')   # raw string prevents \h warning
    ax.set_ylabel('|Residuals|')
    ax.set_title('3. Homoscedasticity\nConstant spread expected')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # ── 4. Normality: histogram of residuals with fitted normal curve ─────────
    # OLS inference assumes residuals are normally distributed. A histogram
    # close to the overlaid bell curve suggests this assumption is satisfied.
    ax = axes[1, 1]
    ax.hist(res_tr, bins=20, color='tomato', edgecolor='white',
            linewidth=0.5, alpha=0.85, density=True)

    # Overlay a normal distribution fitted to the residuals.
    # We compute the PDF manually: f(x) = exp(-0.5*((x-μ)/σ)²) / (σ√(2π))
    mu    = res_tr.mean()
    sigma = res_tr.std()
    xs    = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax.plot(
        xs,
        np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
        'r-', linewidth=2, label='Normal fit',
    )
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('4. Normality\nResiduals should follow a bell curve')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        'Figure 6: Four key assumptions of linear regression (training residuals)',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()