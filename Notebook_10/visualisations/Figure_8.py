"""
Figure 8 — Sampling Methods: Random vs Stratified
=======================================================================
Demonstrates why how a split is sampled matters as much as how much data
goes into each part, particularly on imbalanced datasets.

  RANDOM SAMPLING: picks samples uniformly at random for the test set.
  On an imbalanced dataset, this gives no guarantee that the minority class
  will be represented in the test set in anything like its true proportion,
  by chance, a given split could contain far more or far fewer minority
  class examples than expected.

  STRATIFIED SAMPLING: samples separately from each class, preserving the
  original class proportions in every split. This is enabled by passing
  stratify=y to train_test_split.

A 90/10 imbalanced dataset is split 30 times using each method, and the
fraction of the test set belonging to the minority class is recorded each
time. The left panel shows how this fraction varies across the 30 repeats
for each method, and the right panel summarises the spread of these values
as boxplots.

Usage
-----
In a Jupyter notebook cell:

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
from sklearn.model_selection import train_test_split


# ── Imbalanced dataset ───────────────────────────────────────────────────────
# A 90/10 imbalanced dataset, 900 points belonging to class 0 and 100
# points belonging to class 1. Only the class labels matter for this
# figure, so the features themselves are simple standard-normal noise,
# the point being made is purely about how the LABELS are sampled, not
# about classifier performance.
rng = np.random.default_rng(0)

N_TOTAL = 1000
N_CLASS1 = 100   # 10% minority class

X_imb = rng.normal(0, 1, (N_TOTAL, 2))
y_imb = np.array([0] * (N_TOTAL - N_CLASS1) + [1] * N_CLASS1)

# Each split is repeated 30 times with different random seeds, so the
# variability of each sampling method can be compared rather than relying
# on a single split.
N_REPEATS = 30


def show():
    """Render Figure 8: random vs stratified sampling on an imbalanced dataset."""
    plt.close('Notebook10 Figure 8')

    print('Original dataset class distribution:')
    print(f'  Class 0: {(y_imb == 0).sum()}  ({(y_imb == 0).mean():.1%})')
    print(f'  Class 1: {(y_imb == 1).sum()}  ({(y_imb == 1).mean():.1%})')
    print()

    # ── Repeated splits ───────────────────────────────────────────────────────
    random_minority_fracs = []
    stratified_minority_fracs = []

    for seed in range(N_REPEATS):
        # Random split: class proportions in the test set are not controlled,
        # so they can drift away from the true 10% minority rate by chance.
        _, _, _, y_te_r = train_test_split(
            X_imb, y_imb, test_size=0.25, random_state=seed,
        )
        random_minority_fracs.append((y_te_r == 1).mean())

        # Stratified split: stratify=y_imb forces the test set to contain
        # (approximately) the same proportion of each class as the full
        # dataset, regardless of the random seed.
        _, _, _, y_te_s = train_test_split(
            X_imb, y_imb, test_size=0.25, random_state=seed, stratify=y_imb,
        )
        stratified_minority_fracs.append((y_te_s == 1).mean())

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, num='Notebook10 Figure 8', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # Left panel: minority class fraction in the test set, repeat by repeat,
    # for both sampling methods. The dashed line marks the true 10%
    # minority rate that a perfect split would preserve.
    ax = axes[0]
    ax.plot(
        range(N_REPEATS), random_minority_fracs, 'o-',
        color='tomato', ms=5, lw=1.5, label='Random split',
    )
    ax.plot(
        range(N_REPEATS), stratified_minority_fracs, 's-',
        color='steelblue', ms=5, lw=1.5, label='Stratified split',
    )
    ax.axhline(0.10, color='black', lw=1.5, linestyle='--', label='True minority rate (10%)')

    ax.set_xlabel('Repeat number', fontsize=10)
    ax.set_ylabel('Minority class fraction in test set', fontsize=10)
    ax.set_title(
        'Stratified sampling keeps the class\n'
        'ratio stable across splits',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.22)

    # Right panel: boxplots summarising the spread of the minority class
    # fraction across all 30 repeats for each method. tick_labels is used
    # in place of the deprecated labels argument (Matplotlib 3.9+).
    ax = axes[1]
    ax.boxplot(
        [random_minority_fracs, stratified_minority_fracs],
        tick_labels=['Random', 'Stratified'],
        patch_artist=True,
        boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
        medianprops=dict(color='tomato', lw=2),
        whiskerprops=dict(color='steelblue'),
        capprops=dict(color='steelblue'),
    )
    ax.axhline(0.10, color='black', lw=1.5, linestyle='--', label='True rate (10%)')

    ax.set_ylabel('Minority class fraction in test set', fontsize=10)
    ax.set_title(
        'Spread of class fraction across 30 splits,\n'
        'stratified has far lower variance',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        'Figure 8: random vs stratified sampling on imbalanced data',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # ── Stability summary ────────────────────────────────────────────────────
    random_std = np.std(random_minority_fracs)
    stratified_std = np.std(stratified_minority_fracs)

    print(f'Random sampling     - minority fraction std: {random_std:.4f}')
    print(f'Stratified sampling - minority fraction std: {stratified_std:.4f}')
    print(f'Stratified is {random_std / max(stratified_std, 1e-9):.1f}x more stable')