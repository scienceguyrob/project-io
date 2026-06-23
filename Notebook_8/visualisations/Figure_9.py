"""
Figure 9 — k-NN Predictions for k = 1, 3, 5
=============================================
Three side-by-side panels showing how the k-NN prediction for a fixed query
point changes as k increases from 1 to 5.

Each panel shows:
  - Training points coloured by class (blue = class 0, red = class 1)
  - The query point as a gold star
  - Dashed connectors to each of the k nearest neighbours
  - A ring highlight around each selected neighbour
  - A distance table printed below each panel in place of inline labels,
    keeping the plot area clean

Usage
-----
In a Jupyter notebook cell (after the analysis cell):

    %matplotlib widget
    from visualisations.Figure_9 import show
    show(X_knn, y_knn, x_new, k_values, results)

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


# ── Colours ───────────────────────────────────────────────────────────────────
COL_CLASS = ['steelblue', 'tomato']


def show(X_knn, y_knn, x_new, k_values, results):
    """
    Render Figure 9: k-NN predictions for each value of k.

    Parameters
    ----------
    X_knn    : array (n, d)   — training feature matrix
    y_knn    : array (n,)     — training class labels
    x_new    : array (d,)     — query point
    k_values : list[int]      — values of k to display (one panel each)
    results  : dict           — precomputed results from the analysis cell;
                                keys are k values, values are dicts with
                                'pred' and 'neighbours' entries
    """
    plt.close('Notebook8 Figure 9')

    fig, axes = plt.subplots(
        1, len(k_values),
        num='Notebook8 Figure 9',
        figsize=(15, 5),
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    for ax, k in zip(axes, k_values):
        r = results[k]

        # ── Training points ───────────────────────────────────────────────────
        for cls, col in enumerate(COL_CLASS):
            mask = y_knn == cls
            ax.scatter(
                X_knn[mask, 0], X_knn[mask, 1],
                color=col, s=80, edgecolors='k', lw=0.5,
                label=f'Class {cls}', zorder=3,
            )

        # ── Query point ───────────────────────────────────────────────────────
        ax.scatter(
            *x_new, marker='*', s=280,
            color='gold', edgecolors='black', lw=1.2,
            zorder=6, label='Query',
        )

        # ── Neighbour rings ───────────────────────────────────────────────────
        # Drawn before the connector lines so rings sit behind line ends
        for _, idx in r['neighbours']:
            ax.scatter(
                *X_knn[idx], s=200,
                facecolors='none', edgecolors='black',
                lw=1.8, zorder=4,
            )

        # ── Connector lines ───────────────────────────────────────────────────
        # Distance labels are intentionally omitted from the plot area —
        # they are shown in the distance table printed below the figure
        # instead, keeping the plot clean.
        for _, idx in r['neighbours']:
            ax.plot(
                [x_new[0], X_knn[idx, 0]],
                [x_new[1], X_knn[idx, 1]],
                color='#999999', lw=1.3, linestyle='--',
                zorder=2,
            )

        # ── Panel title ───────────────────────────────────────────────────────
        pred_col = COL_CLASS[r['pred']]
        ax.set_title(
            f'k = {k}  →  predicted class {r["pred"]}',
            fontsize=10, color=pred_col, fontweight='bold',
        )

        ax.legend(fontsize=8, loc='upper left',
                  framealpha=1.0, edgecolor='#cccccc')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 9)

    plt.suptitle(
        'Figure 9: k-NN — how the choice of k changes the prediction',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # ── Distance table ────────────────────────────────────────────────────────
    # Printed below the figure so the distances are available without
    # cluttering the plot area.
    print()
    print(f'  Query point: {tuple(int(v) for v in x_new)}')
    print()
    for k in k_values:
        r = results[k]
        print(f'  k = {k}  →  predicted class {r["pred"]}')
        for dist, idx in r['neighbours']:
            pt = tuple(int(v) for v in X_knn[idx])
            print(f'      {pt}   d = {dist:.3f}')
        print()