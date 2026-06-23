"""
Figure 7a — Geometric Consequences of Growing Dimensionality
============================================================

Two side-by-side panels illustrating the curse of dimensionality:

Left panel:  the volume of a unit hypersphere (radius = 1) as the number
             of dimensions increases. Volume peaks around d = 5 then
             shrinks back toward zero, showing that high-dimensional
             spheres are vanishingly small relative to the surrounding cube.

Right panel: the fraction of randomly sampled points in a unit hypercube
             that fall within distance 0.5 of the centre. In low dimensions
             this fraction is large; in high dimensions almost all points
             end up in the corners, far from the centre.

Both panels illustrate why intuitions built in 2D or 3D break down in the
high-dimensional spaces that machine learning models operate in.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_7 import show
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
from math import pi
from scipy.special import gamma


def show():
    """Render the static Figure 7 curse of dimensionality illustration."""

    plt.close('Notebook4 Figure 7a')

    # ── Hypersphere volume function ───────────────────────────────────────────
    def unit_ball_volume(d):
        """
        Volume of a unit hypersphere (radius = 1) in d dimensions.
        Formula: V_d = π^(d/2) / Γ(d/2 + 1)
        where Γ (Gamma) is the generalisation of the factorial to real numbers.
        In 2D this gives π (area of a circle); in 3D it gives 4π/3 (sphere volume).
        """
        return (pi ** (d / 2)) / gamma(d / 2 + 1)

    dims      = list(range(1, 21))   # dimensions 1 through 20
    ball_vols = [unit_ball_volume(d) for d in dims]

    # ── Simulate fraction of hypercube points near the centre ─────────────────
    # For each dimensionality d, generate 50,000 random points uniformly
    # distributed in the d-dimensional unit hypercube [0,1]^d, shift to centre
    # them on the origin, then measure how many fall within distance 0.5 of it.
    n_samples = 50_000
    fracs_sim = []

    for d in dims:
        # Draw n_samples × d uniform random values; subtract 0.5 to centre on origin
        pts = np.random.default_rng(42).uniform(0, 1, (n_samples, d)) - 0.5
        # Euclidean distance from the origin: sqrt(sum of squared coordinates)
        dists = np.sqrt((pts ** 2).sum(axis=1))
        # Fraction of points within radius 0.5
        fracs_sim.append((dists <= 0.5).mean())

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook4 Figure 7a', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: hypersphere volume ────────────────────────────────────────
    ax = axes[0]
    ax.bar(dims, ball_vols, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Number of dimensions')
    ax.set_ylabel('Volume of unit hypersphere')
    ax.set_title('Volume of the unit hypersphere as dimension grows\n'
                 '(peaks around d = 5 then shrinks back toward zero)')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Right panel: fraction of points near the centre ───────────────────────
    ax = axes[1]
    ax.plot(dims, fracs_sim, 'o-', color='tomato', linewidth=2, markersize=6,
            label='Fraction of points within radius 0.5 of centre')
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Number of dimensions')
    ax.set_ylabel('Fraction of unit hypercube')
    ax.set_title('Almost everything ends up in the corners!\n'
                 '(near-centre region shrinks to nothing)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Figure 7a: Geometric consequences of growing dimensionality',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # Print a plain-text summary so users can see the numbers directly
    print('Fraction of points within distance 0.5 of the centre:')
    for d, f in zip(dims[:10], fracs_sim[:10]):
        bar = '#' * int(f * 40)
        print(f'  d={d:2d}:  {f:.3f}  {bar}')