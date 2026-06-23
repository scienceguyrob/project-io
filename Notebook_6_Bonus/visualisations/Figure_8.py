"""
Figure 8 — Training Loss Decreasing Over Epochs
===================================================

Shows a single training loss curve against epoch number, illustrating
the general shape of how loss decreases as an MLP is trained: a steep
drop early on, a bend where progress starts to slow, and a long,
near-flat tail as the network converges. Three regions of the curve are
labelled directly: the large initial error when weights are still
random, the rapid early improvement as gradient descent finds a good
descent direction, and the eventual convergence where the loss changes
very little from one epoch to the next.

The curve itself is synthetic rather than the output of an actual
training run. It is generated from a decaying exponential plus a small
amount of epoch-dependent noise, chosen purely to produce a curve with
the recognisable shape of real cross-entropy training loss: steep at
first, then progressively flatter. The exact numbers are not meant to
correspond to any specific dataset or model; the point of the figure is
the shape of the curve and what each part of that shape represents, not
a particular numerical result. A fixed random seed is used so the curve
looks the same every time the figure is generated.

This is a static reference diagram rather than an interactive one. There
are no sliders, since there is no meaningful parameter for the reader to
search over here — the figure exists to give a single, clear picture of
what "the loss is decreasing" looks like, to be referred back to
whenever a real loss curve (such as those produced by
MLPClassifier.loss_curve_ in Section 11) is discussed.

The key teaching point: training loss curves for neural networks tend to
follow this same three-phase shape regardless of the specific dataset or
architecture — a large early error, a period of fast improvement, and a
long settling-in period as the model approaches a (local) minimum of the
loss function.

Usage
-----
From a Jupyter notebook cell::

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

import numpy as np
import matplotlib.pyplot as plt


def show():
    """Render the static Figure 8 training loss curve diagram."""

    plt.close('Notebook6bonus Figure 8')

    # ── Colours ──────────────────────────────────────────────────────────────
    COL_CURVE = '#4C78A8'
    COL_FINAL = '#888888'
    COL_TEXT  = '#222222'
    COL_ANNOT = '#555555'

    # ── Generate a plausible synthetic loss curve ────────────────────────────
    rng = np.random.default_rng(7)
    n_epochs = 60
    epochs = np.arange(1, n_epochs + 1)

    floor = 0.08
    initial = 0.95
    decay_rate = 0.09
    base = floor + (initial - floor) * np.exp(-decay_rate * epochs)
    noise = rng.normal(0, 1, n_epochs) * 0.015 * np.exp(-0.04 * epochs)
    loss = np.maximum(base + noise, floor * 0.95)

    final_loss = loss[-5:].mean()

    fig, ax = plt.subplots(num='Notebook6bonus Figure 8', figsize=(10, 5.6),
                            constrained_layout=True)

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    fig.suptitle('Figure 8: Training loss decreasing over epochs — the MLP '
                 'is minimising error', fontsize=12.5, color=COL_TEXT)

    ax.plot(epochs, loss, color=COL_CURVE, lw=2.3, zorder=3)
    ax.axhline(final_loss, color=COL_FINAL, lw=1.2, linestyle='--', zorder=2)
    ax.text(n_epochs - 1, final_loss + 0.03, f'Converged loss \u2248 {final_loss:.2f}',
            ha='right', fontsize=9, color=COL_FINAL)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training loss (cross-entropy)', fontsize=11)
    ax.set_xlim(0, n_epochs + 1)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.2)

    # ── Annotated regions ────────────────────────────────────────────────────
    ax.annotate('Large initial error:\nweights are random',
                xy=(epochs[1], loss[1]), xytext=(13, 0.86),
                fontsize=9.5, color=COL_ANNOT, ha='left',
                arrowprops=dict(arrowstyle='-', color='#999999', lw=0.9,
                                 connectionstyle='arc3,rad=-0.2'))

    mid_idx = 11
    ax.annotate('Rapid improvement:\ngradient descent finds\na descent direction',
                xy=(epochs[mid_idx], loss[mid_idx]), xytext=(24, 0.52),
                fontsize=9.5, color=COL_ANNOT, ha='left',
                arrowprops=dict(arrowstyle='-', color='#999999', lw=0.9,
                                 connectionstyle='arc3,rad=-0.2'))

    late_idx = 48
    ax.annotate('Convergence:\nloss changes very little',
                xy=(epochs[late_idx], loss[late_idx]), xytext=(35, 0.22),
                fontsize=9.5, color=COL_ANNOT, ha='left',
                arrowprops=dict(arrowstyle='-', color='#999999', lw=0.9,
                                 connectionstyle='arc3,rad=0.2'))