"""
Figure 4 — Effect of Search Step Size on Threshold Search
==========================================================

Three side-by-side panels showing how the step size used during exhaustive
threshold search affects the resolution of the error landscape and the
precision of the optimal threshold found:

    Left panel:   step = 1.0  — coarse grid, few evaluations, imprecise result
    Middle panel: step = 0.25 — medium grid, moderate evaluations
    Right panel:  step = 0.05 — fine grid, many evaluations, precise result

This figure illustrates the trade-off between computational cost (number of
evaluations) and precision (how close the found threshold is to the true optimum).

Note: this figure depends on the find_theta function and the fruit dataset
(x, y) being defined in the notebook before show() is called.

Usage
-----
From a Jupyter notebook cell (after generating x, y and defining find_theta)::

    %matplotlib widget
    from visualisations.Figure_4 import show
    show(x, y, find_theta)

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


def show(x, y, find_theta):
    """
    Render the static Figure 4 step size comparison plot.

    Parameters
    ----------
    x          : array-like, feature values (fruit diameters)
    y          : array-like, true labels (0 = Apple, 1 = Orange)
    find_theta : callable, the threshold search function defined in the notebook
    """

    plt.close('Notebook4 Figure 4')

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, num='Notebook4 Figure 4', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # Three step sizes to compare — from coarse to fine
    step_sizes = [1.0, 0.25, 0.05]

    for ax, step in zip(axes, step_sizes):
        # Re-run find_theta with this step size.
        # We only need the log here — the best theta and min errors are
        # recomputed from the log for plotting purposes.
        _, _, log = find_theta(x, y, start=5.0, end=12.0, step=step)

        # Unpack the log into separate lists for the x and y axes of the plot
        thetas_s      = [t for t, e in log]
        error_rates_s = [e / len(x) for t, e in log]   # convert counts to rates

        # Find the best threshold from this log
        best_t = thetas_s[int(np.argmin(error_rates_s))]

        # Use smaller markers for dense grids so overlapping points are visible.
        # A step of 0.05 produces ~140 points; a step of 1.0 produces only 7.
        ms = 4 if step < 0.3 else 8

        ax.plot(thetas_s, error_rates_s, 'o-', color='steelblue',
                markersize=ms, linewidth=1.5)

        # Red dashed line marks the best threshold found at this step size
        ax.axvline(best_t, color='red', linestyle='--',
                   label=f'Best theta = {best_t:.2f}')

        ax.set_title(f'Step = {step}  |  {len(log)} evaluations\nbest = {best_t:.2f} cm')
        ax.set_xlabel('Theta (cm)')
        ax.set_ylabel('Error rate')
        #ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Figure 4: Effect of search step size — coarser is faster but less precise',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()