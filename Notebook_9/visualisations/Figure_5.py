"""
Figure 5 — The Curse of Dimensionality Revisited: Volume and Coverage
========================================================================
Recaps the geometric core of the curse of dimensionality from Notebook 4,
Section 6.1, with a single interactive control: a slider for the number
of dimensions, d.

As d changes, three linked panels update:

  - Left panel: the volume of a unit hypersphere as a function of d,
    with the current d highlighted, showing the peak around d=5 and the
    collapse toward zero afterwards.
  - Middle panel: a live simulation — 5,000 random points are scattered
    uniformly inside the unit hypercube in d dimensions, and the fraction
    falling within distance 0.5 of the centre is shown as a bar, alongside
    the same quantity for all dimensions up to the current d.
  - Right panel: the number of training samples needed to maintain a
    fixed coverage density (5 bins per dimension, 10 samples per cell),
    on a log scale, with horizontal reference lines at 1,000, 1,000,000,
    and 1,000,000,000 samples.

An annotation panel reports the numerical values for the current d,
connecting the geometric collapse, the sparsity simulation, and the
sample-size explosion in one place.

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
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display
from scipy.special import gamma


# ── Fixed analysis parameters ───────────────────────────────────────────────
MAX_DIM       = 20      # maximum dimension shown on the sliders/plots
N_POINTS      = 5000    # points sampled for the sparsity simulation
RADIUS        = 0.5     # "near the centre" threshold for the sparsity panel
BINS_PER_DIM  = 5        # bins per dimension for the coverage calculation
SAMPLES_PER_CELL = 10     # desired samples per grid cell for coverage

DIMS = np.arange(1, MAX_DIM + 1)


def _hypersphere_volume(d, r=1.0):
    """
    Volume of a d-dimensional hypersphere of radius r.

    Uses the closed-form formula V_d(r) = (pi^(d/2) / Gamma(d/2 + 1)) * r^d.
    gamma() from scipy.special extends the factorial function to
    non-integer and half-integer arguments, which is needed because d/2
    is a half-integer for odd d.
    """
    return (np.pi ** (d / 2) / gamma(d / 2 + 1)) * r ** d


# Precompute the volume curve once — it does not depend on the slider.
VOLUMES = _hypersphere_volume(DIMS)

# Precompute the coverage-sample curve once — also independent of the slider.
# n_samples = (BINS_PER_DIM ** d) * SAMPLES_PER_CELL
COVERAGE_SAMPLES = (BINS_PER_DIM ** DIMS.astype(float)) * SAMPLES_PER_CELL

# Reference lines for the coverage panel
REFERENCE_LINES = [1e3, 1e6, 1e9]
REFERENCE_LABELS = ['1,000', '1,000,000', '1,000,000,000']


def _sparsity_fraction(d, rng, n_points=N_POINTS, radius=RADIUS):
    """
    Simulate the fraction of points near the centre of a d-dimensional
    unit hypercube.

    Points are sampled uniformly in [-0.5, 0.5]^d (a unit hypercube
    centred on the origin), and we compute what fraction fall within
    `radius` of the origin — i.e. inside a hypersphere of that radius
    centred at the cube's centre.
    """
    points = rng.uniform(-0.5, 0.5, size=(n_points, d))
    distances = np.linalg.norm(points, axis=1)
    return float((distances < radius).mean())


def show():
    """Render Figure 5: interactive curse-of-dimensionality recap."""
    plt.close('Notebook9 Figure 5')

    fig, (ax_vol, ax_sparse, ax_cover) = plt.subplots(
        1, 3,
        num='Notebook9 Figure 5',
        figsize=(12, 6),
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    fig.suptitle(
        'Figure 5: The curse of dimensionality — volume, sparsity, '
        'and the data you would need\n'
        'Drag the slider to change the number of dimensions, d',
        fontsize=11,
    )
    plt.subplots_adjust(wspace=0.35, top=0.84, bottom=0.14)

    # ── Left panel: hypersphere volume curve (static, drawn once) ───────────
    ax_vol.plot(DIMS, VOLUMES, color='steelblue', marker='o', markersize=4)
    ax_vol.set_xlabel('Dimensions (d)', fontsize=10)
    ax_vol.set_ylabel('Volume of unit hypersphere', fontsize=10)
    ax_vol.set_title('Hypersphere volume vs dimension', fontsize=10)
    ax_vol.grid(True, alpha=0.15)
    vol_marker, = ax_vol.plot([], [], 'o', color='tomato', markersize=10, zorder=5)

    # ── Middle panel: sparsity simulation (bars rebuilt on update) ──────────
    ax_sparse.set_xlabel('Dimensions (d)', fontsize=10)
    ax_sparse.set_ylabel(f'Fraction of points within {RADIUS} of centre', fontsize=10)
    ax_sparse.set_title('Simulated sparsity', fontsize=10)
    ax_sparse.set_xlim(0.5, MAX_DIM + 0.5)
    ax_sparse.set_ylim(0, 1.0)
    ax_sparse.grid(True, alpha=0.15, axis='y')

    # ── Right panel: coverage sample curve (static, drawn once) ─────────────
    ax_cover.plot(DIMS, COVERAGE_SAMPLES, color='seagreen', marker='o', markersize=4)
    ax_cover.set_yscale('log')
    ax_cover.set_xlabel('Dimensions (d)', fontsize=10)
    ax_cover.set_ylabel('Samples needed for fixed coverage', fontsize=10)
    ax_cover.set_title('Samples needed vs dimension', fontsize=10)
    ax_cover.grid(True, alpha=0.15, which='both')
    cover_marker, = ax_cover.plot([], [], 'o', color='tomato', markersize=10, zorder=5)

    for ref, label in zip(REFERENCE_LINES, REFERENCE_LABELS):
        ax_cover.axhline(ref, color='#999999', linestyle='--', linewidth=1)
        ax_cover.text(
            MAX_DIM, ref, f'  {label}',
            va='bottom', ha='right', fontsize=8, color='#666666',
        )

    # Annotation panel, placed below the three plots using fig.text
    ann_text = fig.text(
        0.5, 0.02, '', ha='center', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7',
                  edgecolor='#cccccc', alpha=1.0),
    )

    # rng is created once and reused across updates so that repeated visits
    # to the same d give visually consistent (though not identical, since
    # the rng advances) results, rather than wildly different bars each time.
    rng = np.random.default_rng(0)

    # Cache of sparsity fractions already computed for each d, so that
    # moving the slider back to a previously-visited value does not
    # require resimulating it.
    _sparsity_cache = {}

    def _draw(d):
        d = int(d)

        # ── Update hypersphere volume marker ─────────────────────────────
        vol_d = float(_hypersphere_volume(d))
        vol_marker.set_data([d], [vol_d])

        # ── Update sparsity simulation up to d ────────────────────────────
        ax_sparse.clear()
        ax_sparse.set_xlabel('Dimensions (d)', fontsize=10)
        ax_sparse.set_ylabel(f'Fraction of points within {RADIUS} of centre', fontsize=10)
        ax_sparse.set_title('Simulated sparsity', fontsize=10)
        ax_sparse.set_xlim(0.5, MAX_DIM + 0.5)
        ax_sparse.set_ylim(0, 1.0)
        ax_sparse.grid(True, alpha=0.15, axis='y')

        fractions = []
        for dim in range(1, d + 1):
            if dim not in _sparsity_cache:
                _sparsity_cache[dim] = _sparsity_fraction(dim, rng)
            fractions.append(_sparsity_cache[dim])

        bar_colours = ['steelblue'] * (d - 1) + ['tomato']
        ax_sparse.bar(range(1, d + 1), fractions, color=bar_colours, edgecolor='k', lw=0.3)

        # ── Update coverage marker ────────────────────────────────────────
        cover_d = float(COVERAGE_SAMPLES[d - 1])
        cover_marker.set_data([d], [cover_d])

        # ── Annotation text ────────────────────────────────────────────────
        current_fraction = fractions[-1]
        ann_text.set_text(
            f'd = {d}    |    '
            f'unit hypersphere volume ≈ {vol_d:.4f}    |    '
            f'fraction of points near centre ≈ {current_fraction:.4f}    |    '
            f'samples needed for fixed coverage ≈ {cover_d:,.0f}'
        )

        fig.canvas.draw_idle()

    _draw(1)

    # ── Widget ───────────────────────────────────────────────────────────────
    slider_d = widgets.IntSlider(
        value=1, min=1, max=MAX_DIM, step=1,
        description='Dimensions (d)',
        style={'description_width': '110px'},
        layout=widgets.Layout(width='400px'),
    )

    out = interactive_output(_draw, {'d': slider_d})

    controls = widgets.VBox([slider_d])
    display(controls, out)