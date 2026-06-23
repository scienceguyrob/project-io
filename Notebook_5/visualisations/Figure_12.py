"""
Figure 12 — MLE Loss Landscape for Logistic Regression
=======================================================

Two panels visualising the negative log-likelihood (NLL) loss function
for the tumour classification logistic regression model:

Left panel:  a 1D slice of the NLL as β₁ varies (β₀ fixed at its MLE value)
             showing the convex loss curve with the minimum marked.

Right panel: the full 2D NLL surface over (β₀, β₁), showing the bowl-shaped
             convex landscape and the optimal parameter combination.

Usage
-----
From a Jupyter notebook cell (after fitting log_reg)::

    %matplotlib widget
    from visualisations.Figure_12 import show
    show(log_reg, X_tr4, y_tr4)

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


def show(log_reg, X_tr4, y_tr4):
    """
    Render the static Figure 12 MLE loss landscape.

    Parameters
    ----------
    log_reg : fitted sklearn LogisticRegression object
    X_tr4   : training feature array, shape (n, 1)
    y_tr4   : training labels, shape (n,)
    """

    plt.close('Notebook5 Figure 12')

    # ── Sigmoid and NLL functions ─────────────────────────────────────────────
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def neg_log_likelihood(b0, b1, X, y):
        """Compute negative log-likelihood for logistic regression."""
        z = b0 + b1 * X.ravel()
        p = sigmoid(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)   # avoid log(0)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    # ── Extract fitted parameters ─────────────────────────────────────────────
    b0_fixed = log_reg.intercept_[0]
    b1_mle   = log_reg.coef_[0][0]

    # ── 1D slice: vary β₁ with β₀ fixed ──────────────────────────────────────
    b1_range = np.linspace(b1_mle - 1.0, b1_mle + 1.0, 300)
    nll_vals = [neg_log_likelihood(b0_fixed, b1, X_tr4, y_tr4) for b1 in b1_range]
    min_nll  = min(nll_vals)

    # ── 2D surface: grid over (β₀, β₁) ───────────────────────────────────────
    b0_grid = np.linspace(b0_fixed - 5, b0_fixed + 5, 60)
    b1_grid = np.linspace(b1_mle - 0.5, b1_mle + 0.5, 60)
    B0g, B1g = np.meshgrid(b0_grid, b1_grid)
    NLL_surf = np.array([
        neg_log_likelihood(b0, b1, X_tr4, y_tr4)
        for b0, b1 in zip(B0g.ravel(), B1g.ravel())
    ]).reshape(B0g.shape)

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook5 Figure 12', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'bottom'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: 1D NLL curve ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(b1_range, nll_vals, color='seagreen', linewidth=2.5)

    # Mark the MLE optimum — the lowest point on the curve
    ax.axvline(b1_mle, color='red', linewidth=2, linestyle='--',
               label=f'MLE optimum:  b1 = {b1_mle:.3f}')
    ax.scatter([b1_mle], [neg_log_likelihood(b0_fixed, b1_mle, X_tr4, y_tr4)],
               color='red', s=80, zorder=5)

    # Use plain text labels to avoid LaTeX parsing errors
    ax.set_xlabel('b1  (coefficient of tumour size)', fontsize=10)
    ax.set_ylabel('Negative log-likelihood (Loss)', fontsize=10)
    ax.set_title(
        'Figure 12a: MLE loss curve\n'
        '(b0 fixed at MLE estimate, b1 varies)',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right panel: 2D NLL surface ───────────────────────────────────────────
    ax = axes[1]
    cp = ax.contourf(B0g, B1g, NLL_surf, levels=30, cmap='RdYlGn_r')
    plt.colorbar(cp, ax=ax, label='Negative log-likelihood')

    # White star at the MLE optimum — the global minimum of the surface
    ax.scatter([b0_fixed], [b1_mle], color='white', s=150, zorder=5,
               marker='*', edgecolors='black', lw=1.0,
               label=(f'MLE optimum\n'
                      f'b0={b0_fixed:.2f},  b1={b1_mle:.3f}'))

    ax.set_xlabel('b0  (intercept)', fontsize=10)
    ax.set_ylabel('b1  (slope in log-odds space)', fontsize=10)
    ax.set_title(
        'Figure 12b: 2-D MLE loss surface\n'
        '(darker green = lower loss = better fit)',
        fontsize=10,
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f'MLE optimum:  b0 = {b0_fixed:.4f},  b1 = {b1_mle:.4f}')
    print(f'Minimum negative log-likelihood = {min_nll:.3f}')