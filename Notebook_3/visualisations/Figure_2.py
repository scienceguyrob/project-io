"""
Figure 2 — Informative vs Uninformative Features
=================================================

Two side-by-side histograms comparing a feature that separates two classes
well (body mass) against one that does not (ear length), using synthetic
polar bear data:

    Left panel:  Body mass (kg) — male and female distributions are well
                 separated, making this an informative feature for classification.

    Right panel: Ear length (cm) — male and female distributions heavily
                 overlap, making this an uninformative feature.

This figure motivates the importance of feature selection in machine learning:
not all measurements are equally useful for distinguishing between classes.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_2 import show
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
    """Render the static Figure 2 informative vs uninformative feature comparison."""

    plt.close('Notebook3 Figure 2')

    # ── Data generation ───────────────────────────────────────────────────────
    # Fixed seed so every user sees identical distributions.
    rng = np.random.default_rng(7)
    n   = 200   # 200 bears per sex

    # Body mass (kg): males substantially heavier than females —
    # the two distributions sit far apart, making this easy to classify on
    mass_male   = rng.normal(loc=135, scale=25, size=n)
    mass_female = rng.normal(loc=68,  scale=15, size=n)

    # Ear length (cm): almost identical between sexes —
    # the distributions overlap almost entirely, so this feature carries
    # very little information about which class a point belongs to
    ear_male    = rng.normal(loc=14.0, scale=1.2, size=n)
    ear_female  = rng.normal(loc=13.8, scale=1.2, size=n)

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook3 Figure 2', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Left panel: body mass (informative) ───────────────────────────────────
    ax = axes[0]

    ax.hist(mass_male,   bins=35, color='steelblue', alpha=0.6,
            edgecolor='white', lw=0.3, label='Male')
    ax.hist(mass_female, bins=35, color='tomato',    alpha=0.6,
            edgecolor='white', lw=0.3, label='Female')

    # Dashed vertical lines mark each group's mean so users can see
    # how far apart the two distributions are centred
    ax.axvline(mass_male.mean(),   color='steelblue', linewidth=2.5, linestyle='--',
               label=f'Male mean = {mass_male.mean():.0f} kg')
    ax.axvline(mass_female.mean(), color='tomato',    linewidth=2.5, linestyle='--',
               label=f'Female mean = {mass_female.mean():.0f} kg')

    ax.set_xlabel('Body mass (kg)')
    ax.set_ylabel('Count')
    ax.set_title('Body mass: well-separated distributions\nINFORMATIVE feature')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # ── Right panel: ear length (uninformative) ───────────────────────────────
    ax = axes[1]

    ax.hist(ear_male,   bins=30, color='steelblue', alpha=0.6,
            edgecolor='white', lw=0.3, label='Male')
    ax.hist(ear_female, bins=30, color='tomato',    alpha=0.6,
            edgecolor='white', lw=0.3, label='Female')

    ax.axvline(ear_male.mean(),   color='steelblue', linewidth=2.5, linestyle='--',
               label=f'Male mean = {ear_male.mean():.1f} cm')
    ax.axvline(ear_female.mean(), color='tomato',    linewidth=2.5, linestyle='--',
               label=f'Female mean = {ear_female.mean():.1f} cm')

    ax.set_xlabel('Ear length (cm)')
    ax.set_title('Ear length: heavily overlapping distributions\nUNINFORMATIVE feature')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # suptitle with rect reserves space at the top so tight_layout never
    # lets the subplots overlap or clip the title
    plt.suptitle(
        'Figure 2: Comparing an informative feature (mass) with an uninformative one (ear length)',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()