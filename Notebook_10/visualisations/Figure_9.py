"""
Figure 9 — Visualising k-Fold Cross-Validation
=======================================================================
Draws a schematic of 5-fold cross-validation.

Each row represents one iteration (fold) of the cross-validation
procedure. Within each row, the samples are divided into the same 5
blocks, but the role of each block changes from row to row:

  - The tomato block is the TEST fold for that iteration.
  - The remaining blocks are the TRAINING folds for that iteration.

Reading down the rows, every sample appears in the tomato test block
exactly once across the 5 rows, this is the defining property of k-fold
cross-validation: every sample is used for testing exactly once, and for
training in every other iteration.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_9 import show
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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Layout parameters ────────────────────────────────────────────────────────
# n_samples_vis is the number of samples shown along each row, purely for
# illustration, it does not need to match the size of any real dataset used
# elsewhere in this notebook. k is the number of folds, fold_size is how many
# of the n_samples_vis samples fall into each fold.
N_SAMPLES_VIS = 30
K = 5
FOLD_SIZE = N_SAMPLES_VIS // K

# One colour per fold, used consistently across all rows so that, for
# example, "Fold 2's samples" are always the same colour whether they are
# currently playing the role of training data or test data.
FOLD_COLOURS = ['steelblue', 'seagreen', 'goldenrod', 'mediumpurple', 'coral']


def show():
    """Render Figure 9: a schematic of 5-fold cross-validation."""
    plt.close('Notebook10 Figure 9')

    fig, ax = plt.subplots(num='Notebook10 Figure 9', figsize=(10, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # ── Draw one row per cross-validation iteration ─────────────────────────
    # For each row (fold_idx), every sample is drawn as a small rectangle.
    # A sample's natural colour comes from FOLD_COLOURS, based on which
    # fold it belongs to, unless this row's iteration is using that fold
    # as the test fold, in which case it is drawn in tomato instead.
    for fold_idx in range(K):
        for sample_idx in range(N_SAMPLES_VIS):
            fold_of_sample = sample_idx // FOLD_SIZE
            is_test = (fold_of_sample == fold_idx)
            colour = 'tomato' if is_test else FOLD_COLOURS[fold_of_sample]

            rect = plt.Rectangle(
                (sample_idx, K - fold_idx - 1), 0.9, 0.8,
                color=colour, alpha=0.85,
            )
            ax.add_patch(rect)

            # Label the middle sample of the test block with "TEST", so each
            # row clearly shows which fold is playing the test role.
            if is_test and sample_idx == fold_idx * FOLD_SIZE + FOLD_SIZE // 2:
                ax.text(
                    sample_idx + 0.45, K - fold_idx - 0.55, 'TEST',
                    ha='center', va='center', fontsize=6,
                    color='white', fontweight='bold',
                )

    # ── Row labels ────────────────────────────────────────────────────────────
    for fold_idx in range(K):
        ax.text(
            -1.5, K - fold_idx - 0.55, f'Fold {fold_idx + 1}',
            va='center', ha='right', fontsize=10,
        )

    # ── Sample index labels along the bottom ─────────────────────────────────
    for i in range(0, N_SAMPLES_VIS, 5):
        ax.text(i + 0.45, -0.4, str(i), ha='center', fontsize=8, color='gray')

    ax.set_xlim(-2, N_SAMPLES_VIS + 0.5)
    ax.set_ylim(-0.6, K + 0.3)
    ax.axis('off')

    ax.set_title(
        'Figure 9: 5-fold cross-validation\n'
        'red is the test fold for that iteration, coloured blocks are training folds\n'
        'every sample is used for testing exactly once',
        fontsize=11,
    )

    legend_elements = [
        mpatches.Patch(facecolor='tomato', label='Test fold'),
        mpatches.Patch(facecolor='steelblue', label='Training fold (example colour)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.show()