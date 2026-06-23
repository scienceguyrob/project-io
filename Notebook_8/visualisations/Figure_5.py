"""
Figure 5 — Minkowski Unit Balls: How p Changes the Geometry of Distance
========================================================================
Visualises how the shape of the unit ball changes as the Minkowski order p
varies from 1 to infinity.

The unit ball is the set of all points at distance exactly 1 from the origin
under a given metric. Its shape is the geometric fingerprint of the metric:

  p = 1            → diamond (Manhattan / L1)
  p = 2            → circle  (Euclidean / L2)
  p = 3            → intermediate, corners filling toward square
  p → ∞ (p = 10)  → approaches square (Chebyshev / L∞)

A numerical comparison table is printed below the figure showing the
Minkowski distance from (0, 0) to (3, 4) for p = 1, 2, 3, 10, 100,
illustrating how the value converges to the Chebyshev limit (max(3, 4) = 4)
as p grows.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_5 import show
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


# ── Curve resolution ──────────────────────────────────────────────────────────
# 500 points gives a smooth curve for all p values including p=1, where the
# diamond corners need enough points to render sharply.
N_THETA = 500
THETA   = np.linspace(0, 2 * np.pi, N_THETA)

# ── p values and their colours ────────────────────────────────────────────────
# p=10 is used as a practical stand-in for p→∞ (Chebyshev): at p=10 the unit
# ball is visually indistinguishable from a square at this resolution.
P_VALS   = [1, 2, 3, 10]
COLOURS  = ['steelblue', 'tomato', 'seagreen', 'goldenrod']

# ── Numerical comparison: fixed point pair ────────────────────────────────────
# (0,0) → (3,4) is a classic Pythagorean pair: Euclidean distance = 5.
# Chebyshev limit = max(3, 4) = 4, which the printed table approaches as p grows.
A = np.array([0.0, 0.0])
B = np.array([3.0, 4.0])
P_PRINT = [1, 2, 3, 10, 100]


def _unit_ball(p):
    r"""
    Parameterise the L_p unit circle in 2D.

    For |x|^p + |y|^p = 1, a valid parameterisation is:
        x = sign(cos t) * |cos t|^(2/p)
        y = sign(sin t) * |sin t|^(2/p)

    The sign() factor preserves the correct quadrant for all t, while the
    exponent 2/p maps the cosine/sine range [0,1] onto the L_p unit circle.
    """
    x = np.sign(np.cos(THETA)) * np.abs(np.cos(THETA)) ** (2 / p)
    y = np.sign(np.sin(THETA)) * np.abs(np.sin(THETA)) ** (2 / p)
    return x, y


def _label(p):
    """Human-readable legend label matching the terminology in Section 5.3."""
    suffixes = {
        1:  ' — Manhattan ($L_1$)',
        2:  ' — Euclidean ($L_2$)',
        10: ' — approaches Chebyshev ($L_\\infty$)',
    }
    return f'$p = {p}$' + suffixes.get(p, '')


def show():
    """Render Figure 5: Minkowski unit balls for p = 1, 2, 3, 10."""
    plt.close('Notebook8 Figure 5')

    fig, ax = plt.subplots(
        num='Notebook8 Figure 5', figsize=(7, 7)
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # ── Unit ball curves ──────────────────────────────────────────────────────
    for p, col in zip(P_VALS, COLOURS):
        x, y = _unit_ball(p)
        ax.plot(x, y, color=col, lw=2.5, label=_label(p))

    # ── Axis lines through the origin ─────────────────────────────────────────
    # Drawn as dotted reference lines so the diamond vertices and square corners
    # are clearly anchored to the coordinate axes.
    ax.axhline(0, color='black', lw=0.8, linestyle=':')
    ax.axvline(0, color='black', lw=0.8, linestyle=':')

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_aspect('equal')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.set_title(
        'Unit balls under the Minkowski metric for increasing $p$\n'
        'Every point on each curve is at distance 1 from the origin',
        fontsize=10,
    )
    ax.legend(fontsize=10, framealpha=1.0, edgecolor='#cccccc')
    ax.grid(True, alpha=0.25)

    plt.suptitle(
        'Figure 5: How the Minkowski order $p$ changes the geometry of distance',
        fontsize=11, y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # ── Numerical comparison table ────────────────────────────────────────────
    # Shows how the Minkowski distance from A to B converges to the Chebyshev
    # limit (max(|3|, |4|) = 4) as p grows. Printed after the figure so the
    # learner can connect the geometry above to the numbers below.
    print('Minkowski distance from (0, 0) to (3, 4):')
    print(f'  {"p":<6}  {"distance":>10}  note')
    print('  ' + '-' * 40)
    for p in P_PRINT:
        d    = np.sum(np.abs(A - B) ** p) ** (1 / p)
        note = {1: '(Manhattan)', 2: '(Euclidean)'}.get(p, '')
        print(f'  p={p:<4}   d = {d:.4f}    {note}')
    print(f'\n  Chebyshev limit: max(3, 4) = {max(abs(B - A)):.4f}')
