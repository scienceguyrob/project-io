"""
Figure 3 -- ROC Curve Shapes: A Visual Gallery
===============================================
Shows five archetypal ROC curve shapes and what each one tells you about
a classifier's behaviour. The five cases are:

  Perfect    -- curve hugs the top-left corner (AUC = 1.0). Every positive
                example scores higher than every negative.

  Good       -- curve bows toward the top-left (AUC > 0.5). Most positives
                score higher than most negatives.

  Random     -- diagonal line (AUC = 0.5). Scores carry no information
                about the true class.

  Poor       -- curve bends below the diagonal (AUC < 0.5). This does not
                mean the model is worthless: the scores are simply inverted.
                Flipping the decision rule (predict positive when score is
                LOW) recovers an effective AUC of 1 - original AUC.

  Misleading -- high AUC but an extremely small positive class. A handful
                of minority-class examples drives the curve upward, making
                the model appear better than it is in practice.

All panels use synthetic data generated with fixed random seeds so the
figure is fully self-contained and identical on every run.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_3 import show
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
from sklearn.metrics import roc_curve, roc_auc_score

# -- Synthetic dataset parameters ---------------------------------------------
# All panels except Misleading use a balanced dataset of N_SYN examples.
N_SYN  = 200
N_HALF = N_SYN // 2

# Misleading panel uses a heavily imbalanced dataset to show how a tiny
# minority class can inflate the apparent AUC.
N_MISLEAD_MAJ = 190
N_MISLEAD_MIN = 10

# -- Colours for each panel ---------------------------------------------------
COL_PERFECT    = "seagreen"
COL_GOOD       = "steelblue"
COL_RANDOM     = "gray"
COL_POOR       = "tomato"
COL_MISLEADING = "goldenrod"


def _make_data():
    """
    Generate all synthetic score arrays used by each panel.
    Fixed seeds ensure the figure is identical on every run.
    Returns a list of (y_true, scores, title, colour) tuples in display order.
    """
    rng = np.random.default_rng(7)

    y_base = np.array([0] * N_HALF + [1] * N_HALF)

    # Perfect: positives score exactly 1, negatives exactly 0.
    # Tiny noise prevents roc_curve from collapsing to two points.
    scores_perfect = np.where(y_base == 1, 1.0, 0.0) + rng.normal(0, 0.001, N_SYN)

    # Good: positives draw from a higher-mean Gaussian, negatives from a
    # lower-mean Gaussian, with enough overlap to produce a realistic curve.
    scores_good = np.where(
        y_base == 1,
        np.clip(rng.normal(0.70, 0.15, N_SYN), 0, 1),
        np.clip(rng.normal(0.30, 0.15, N_SYN), 0, 1),
    )

    # Random: uniform scores carry no class signal.
    scores_random = rng.uniform(0, 1, N_SYN)

    # Poor / inverted: positives deliberately receive LOW scores and
    # negatives receive HIGH scores, producing AUC < 0.5.
    scores_poor = np.where(
        y_base == 1,
        rng.beta(1, 5, N_SYN),   # Positives: skewed toward 0
        rng.beta(5, 1, N_SYN),   # Negatives: skewed toward 1
    )

    # Misleading: separate RNG and imbalanced dataset.
    rng_m = np.random.default_rng(11)
    y_mislead = np.array([0] * N_MISLEAD_MAJ + [1] * N_MISLEAD_MIN)
    scores_mislead = np.concatenate([
        rng_m.beta(2, 5, N_MISLEAD_MAJ),  # Majority class: medium-low scores
        rng_m.beta(4, 3, N_MISLEAD_MIN),  # Minority class: slightly higher scores
    ])

    return [
        (y_base,    scores_perfect,  "Perfect\nAUC = 1.0",                COL_PERFECT),
        (y_base,    scores_good,     "Good\nAUC > 0.5",                   COL_GOOD),
        (y_base,    scores_random,   "Random guessing\nAUC = 0.5",        COL_RANDOM),
        (y_base,    scores_poor,     "Poor / Inverted\nAUC < 0.5",        COL_POOR),
        (y_mislead, scores_mislead,  "Misleading:\nHigh AUC, tiny minority", COL_MISLEADING),
    ]


def show():
    """Render Figure 3: a five-panel gallery of archetypal ROC curve shapes."""
    plt.close("Notebook12 Figure 3")

    configs = _make_data()

    fig, axes = plt.subplots(
        1, 5,
        num="Notebook12 Figure 3",
        figsize=(12, 5),
    )
    fig.canvas.header_visible   = False
    fig.canvas.toolbar_visible  = True
    fig.canvas.toolbar_position = "right"

    for ax, (yt, sc, title, col) in zip(axes, configs):
        fpr, tpr, _ = roc_curve(yt, sc)
        auc = roc_auc_score(yt, sc)

        ax.plot(fpr, tpr, color=col, lw=2.5, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.12, color=col)

        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Figure 3: ROC curve shapes -- what each one tells you",
        fontsize=12, y=0.98,
    )
    plt.tight_layout()