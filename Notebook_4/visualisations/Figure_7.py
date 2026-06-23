"""
Figure 7 — Circle in a Square and Sphere in a Cube
=====================================================

Two panels illustrating the thought experiment that motivates the curse
of dimensionality:

Left panel:  a circle inscribed in a square in 2D. The circle touches the
             edges of the square and covers roughly 78% of the square's area.
             Points are scattered randomly — those inside the circle are
             coloured blue, those in the corners are red.

Right panel: a bar chart comparing the fraction of hypercube volume covered
             by a unit hypersphere across 2D, 3D and selected higher dimensions,
             making the collapse toward zero concrete and numerical.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_7a import show
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
    """Render the static Figure 7a circle-in-square illustration."""

    plt.close('Notebook4 Figure 7')

    # ── Hypersphere volume as fraction of hypercube ───────────────────────────
    def ball_fraction(d):
        """
        Fraction of the unit hypercube [-0.5, 0.5]^d covered by the
        inscribed hypersphere of radius 0.5.
        Volume of hypersphere = π^(d/2) / Γ(d/2 + 1) × r^d
        Volume of hypercube   = 1
        """
        r  = 0.5
        return (pi ** (d / 2)) / gamma(d / 2 + 1) * (r ** d)

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook4 Figure 7a', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: circle inscribed in a square, 2D ─────────────────────────
    ax = axes[0]

    # Draw the square boundary
    square = plt.Polygon(
        [[-1,-1],[1,-1],[1,1],[-1,1]],
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(square)

    # Draw the inscribed circle (radius = 1, touching all four edges)
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='steelblue', linewidth=2)
    ax.add_patch(circle)

    # Scatter random points and colour by whether they are inside the circle
    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, (3000, 2))
    dist = np.sqrt((pts ** 2).sum(axis=1))
    inside  = dist <= 1
    outside = ~inside

    ax.scatter(pts[inside,  0], pts[inside,  1], s=4, alpha=0.4,
               color='steelblue', label=f'Inside circle  ({inside.mean():.0%})')
    ax.scatter(pts[outside, 0], pts[outside, 1], s=4, alpha=0.4,
               color='tomato',    label=f'In corners      ({outside.mean():.0%})')

    # Annotations
    ax.text(0, 0, f'{inside.mean():.0%}\nof area', ha='center', va='center',
            fontsize=13, fontweight='bold', color='steelblue')
    ax.text(0.78, 0.88, f'{outside.mean():.0%}\nin corners',
            ha='center', fontsize=10, color='tomato',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='tomato', alpha=0.8))

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.set_title('2D: circle inscribed in a square\n'
                 'The circle covers ~78% of the area', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # ── Right panel: fraction covered across dimensions ───────────────────────
    ax = axes[1]

    dims       = [2, 3, 5, 7, 10, 15, 20]
    fractions  = [ball_fraction(d) for d in dims]
    pct_labels = [f'{f*100:.2f}%' if f > 0.0001 else f'{f*100:.4f}%'
                  for f in fractions]
    colours    = ['steelblue' if d <= 3 else 'tomato' for d in dims]

    bars = ax.bar([str(d) for d in dims], [f * 100 for f in fractions],
                  color=colours, edgecolor='white', linewidth=0.5)

    # Label each bar with its percentage
    for bar, label in zip(bars, pct_labels):
        height = bar.get_height()
        y_pos  = height + 0.3 if height > 0.5 else height + 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                label, ha='center', va='bottom', fontsize=8.5)

    ax.set_xlabel('Number of dimensions')
    ax.set_ylabel('% of hypercube covered by inscribed hypersphere')
    ax.set_title('Fraction of hypercube volume covered by inscribed sphere\n'
                 '(blue = familiar dimensions; red = high dimensions)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Figure 7: As dimensions grow, almost all space migrates to the corners',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    print('Fraction of hypercube covered by inscribed hypersphere:')
    for d, f in zip(dims, fractions):
        bar_str = '#' * max(1, int(f * 50))
        print(f'  d={d:2d}:  {f*100:7.3f}%  {bar_str}')