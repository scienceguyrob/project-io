"""
Figure 11 — Principal Axes on the Data Cloud
===============================================
A direct visualisation of everything covered in Sections 6-8: a 2D dataset
is plotted as a scatter, its covariance matrix is computed, and its
eigenvectors are drawn on top as arrows from the data's mean — these are
the principal axes (principal components).

The figure shows:

  - The data points, as a scatter cloud.
  - The mean of the data, marked with a cross.
  - PC1 (the eigenvector with the largest eigenvalue), drawn as a double-
    headed arrow along the direction of greatest variance, scaled by its
    eigenvalue.
  - PC2 (the eigenvector with the smallest eigenvalue), drawn the same way,
    always perpendicular to PC1.

An annotation panel shows the covariance matrix, both eigenvalues, both
eigenvectors, and the percentage of total variance each principal
component accounts for.

This figure is intentionally static — its purpose is a single clear
illustration that ties the abstract eigenvector/eigenvalue machinery of
Sections 7-8 back to a concrete dataset, rather than an interactive
exploration.

Usage
-----
In a Jupyter notebook cell:

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

import numpy as np
import matplotlib.pyplot as plt


# ── Dataset ──────────────────────────────────────────────────────────────────
# A correlated 2D dataset — same construction as the inspiration code, so
# the eigenvalues/eigenvectors used in Section 8.2's worked example
# (lambda = 1, 3 for [[2,1],[1,2]]) and the figure here tell a consistent
# visual story: most variance lies along one diagonal direction, much
# less along the perpendicular direction.
rng = np.random.default_rng(11)
X = rng.multivariate_normal([3, 3], [[3.0, 2.2], [2.2, 2.0]], 200)

COL_DATA = 'lightsteelblue'
COL_PC1  = 'tomato'
COL_PC2  = 'seagreen'


def show():
    """Render Figure 11: principal axes on a data cloud."""
    plt.close('Notebook9 Figure 11')

    fig, (ax_plot, ax_ann) = plt.subplots(
        1, 2,
        num='Notebook9 Figure 11',
        figsize=(11, 6),
        gridspec_kw={'width_ratios': [1.4, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    ax_ann.set_axis_off()

    fig.suptitle(
        'Figure 11: Principal axes on the data cloud',
        fontsize=12,
        y=0.99,
    )
    plt.subplots_adjust(wspace=0.1, top=0.80)

    # ── Covariance matrix and eigen-decomposition ────────────────────────────
    # np.cov expects rows = variables, columns = observations, hence X.T —
    # same convention as Section 6.3.
    C = np.cov(X.T)

    # np.linalg.eigh (not eig) is used because C is symmetric — eigh is
    # more numerically stable for symmetric matrices and, unlike eig,
    # guarantees real-valued results and returns eigenvalues in ascending
    # order, which we then reverse to get the largest (PC1) first.
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = eigenvalues.sum()

    # ── Scatter plot ──────────────────────────────────────────────────────────
    ax_plot.scatter(
        X[:, 0], X[:, 1],
        s=25, color=COL_DATA, edgecolors='k', lw=0.2, alpha=0.7,
        label='Data', zorder=2,
    )

    mean_point = X.mean(axis=0)
    ax_plot.scatter(
        *mean_point, s=120, color='black', marker='+', lw=2.5,
        zorder=5, label='Data mean',
    )

    # ── Principal axes ─────────────────────────────────────────────────────────
    # Each eigenvector is drawn as a double-headed arrow centred on the
    # mean, scaled by 2 standard deviations along that direction
    # (2 * sqrt(eigenvalue)) — a common convention for showing the "spread"
    # of the data along each principal axis.
    pc_colours = [COL_PC1, COL_PC2]
    pc_labels  = ['PC1', 'PC2']

    for i, (eigvec, eigval, colour, label) in enumerate(
        zip(eigenvectors.T, eigenvalues, pc_colours, pc_labels)
    ):
        scale = 2 * np.sqrt(eigval)
        start = mean_point - scale * eigvec
        end   = mean_point + scale * eigvec

        ax_plot.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(arrowstyle='<->', color=colour, lw=3),
            zorder=4,
        )

        # Position the label just beyond the arrow's positive end, offset
        # perpendicular to the arrow's direction by a small fixed amount.
        # A fixed perpendicular offset (rather than scaling along the
        # arrow itself) avoids the label overlapping the arrowhead for
        # short arrows like PC2, where eigval is small.
        perp = np.array([-eigvec[1], eigvec[0]])  # rotate eigvec by 90°
        label_pos = end + 0.3 * perp + 0.2 * eigvec
        ax_plot.text(
            label_pos[0], label_pos[1],
            f'{label}\n(eigenvalue = {eigval:.2f})',
            color=colour, fontsize=10, ha='center', va='center',
            fontweight='bold',
        )

    ax_plot.set_xlabel('Feature 1', fontsize=10)
    ax_plot.set_ylabel('Feature 2', fontsize=10)
    ax_plot.set_title(
        'PC1 (red) = direction of most variance\n'
        'PC2 (green) = direction of least variance (always perpendicular to PC1)',
        fontsize=10,
    )
    ax_plot.legend(fontsize=9, loc='lower right', framealpha=1.0, edgecolor='#cccccc')
    ax_plot.grid(True, alpha=0.2)
    ax_plot.set_aspect('equal')

    # ── Annotation panel ──────────────────────────────────────────────────────
    lines = [
        "Covariance matrix C",
        "─────────────────────────────────",
        "",
        f"  [{C[0,0]:7.3f}  {C[0,1]:7.3f}]",
        f"  [{C[1,0]:7.3f}  {C[1,1]:7.3f}]",
        "",
        "─────────────────────────────────",
        "",
        "Eigenvalues and eigenvectors",
        "(np.linalg.eigh, sorted descending)",
        "",
    ]

    for i, (eigvec, eigval, colour, label) in enumerate(
        zip(eigenvectors.T, eigenvalues, pc_colours, pc_labels)
    ):
        pct = eigval / total_variance * 100
        lines.append(f"{label}:")
        lines.append(f"  eigenvalue = {eigval:.3f}")
        lines.append(f"  eigenvector = [{eigvec[0]:.3f}, {eigvec[1]:.3f}]")
        lines.append(f"  {pct:.1f}% of total variance")
        lines.append("")

    lines.append("─────────────────────────────────")
    lines.append("")
    lines.append(f"Total variance = {total_variance:.3f}")
    lines.append("(sum of eigenvalues =")
    lines.append(" sum of diagonal of C,")
    lines.append(" i.e. Var(Feature 1) + Var(Feature 2))")

    ax_ann.text(
        0.04, 0.97, "\n".join(lines),
        transform=ax_ann.transAxes,
        fontsize=9.5, va='top', ha='left',
        family='monospace',
        linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    plt.tight_layout()
    plt.show()