"""
Figure 7 — Class Imbalance: The "Always Predict Majority Class" Baseline
=======================================================================
Demonstrates why accuracy alone becomes misleading once class distributions
become uneven.

When classes are unequal, a trivial strategy, always predicting the
majority class, achieves high accuracy without the model learning anything
at all. The more imbalanced the dataset, the higher this baseline accuracy
becomes, and the more misleading a plain accuracy figure is as a measure of
whether a model has learned anything useful.

This figure shows the majority-class baseline accuracy for several minority
class fractions, and visualises what a balanced 50/50 class distribution
looks like next to a heavily imbalanced 99/1 distribution.

Usage
-----
In a Jupyter notebook cell:

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

import matplotlib.pyplot as plt


# ── Imbalance ratios to compare ─────────────────────────────────────────────
# Each value is the fraction of the dataset belonging to the minority class.
# As this fraction shrinks, the majority class makes up a larger and larger
# share of the data, so simply always predicting that class becomes an
# increasingly strong, but completely uninformative, baseline.
IMBALANCE_RATIOS = [0.5, 0.3, 0.1, 0.05, 0.01]

# Above this majority-class baseline accuracy, a plain accuracy figure is
# flagged as dangerously misleading, since a model could exceed this score
# without having learned anything.
DANGER_THRESHOLD = 0.9


def show():
    """Render Figure 7: the majority-class baseline under different class imbalances."""
    plt.close('Notebook10 Figure 7')

    # ── Baseline accuracy printout ────────────────────────────────────────────
    # For each imbalance ratio, the "always predict majority class" baseline
    # is simply 1 minus the minority fraction, since that is the proportion
    # of points that belong to the majority class and would therefore be
    # classified correctly by always guessing it.
    print('The "always predict majority class" baseline:')
    print(f'  {"Minority %":<14} {"Majority %":<14} {"Majority baseline accuracy"}')
    print('-' * 52)

    for r in IMBALANCE_RATIOS:
        majority_acc = 1 - r
        flag = '  <- DANGER: misleading!' if majority_acc > DANGER_THRESHOLD else ''
        print(f'  {r * 100:<14.0f} {(1 - r) * 100:<14.0f} {majority_acc:.1%}{flag}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, num='Notebook10 Figure 7', figsize=(11, 5))

    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'

    # Left panel: horizontal bar chart of baseline accuracies. Bars above
    # the danger threshold are coloured tomato to flag them visually,
    # everything else is seagreen.
    ax = axes[0]
    pcts = [r * 100 for r in IMBALANCE_RATIOS]
    maj_accs = [1 - r for r in IMBALANCE_RATIOS]

    ax.barh(
        [f'{p:.0f}% minority\n{100 - p:.0f}% majority' for p in pcts],
        maj_accs,
        color=['seagreen' if a < DANGER_THRESHOLD else 'tomato' for a in maj_accs],
        edgecolor='white', linewidth=0.4,
    )

    ax.axvline(DANGER_THRESHOLD, color='black', lw=1.5, linestyle='--', label='90% threshold')
    ax.set_xlabel('"Accuracy" of always predicting majority class', fontsize=10)
    ax.set_title(
        'Class imbalance: high accuracy can be meaningless,\n'
        'red bars are dangerously misleading',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.15)

    for i, v in enumerate(maj_accs):
        ax.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=10)

    # Middle panel: a balanced 50/50 class distribution, shown as a pie
    # chart for an immediate visual sense of what "balanced" looks like.
    axes[1].pie(
        [500, 500],
        colors=['steelblue', 'tomato'],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'lw': 1.5},
        labels=['Class 0', 'Class 1'],
        autopct='%1.0f%%',
        textprops={'fontsize': 10},
    )
    axes[1].set_title('Balanced\n50 / 50', fontsize=10)

    # Right panel: a heavily imbalanced 99/1 class distribution, for direct
    # visual comparison against the balanced case in the middle panel.
    axes[2].pie(
        [990, 10],
        colors=['steelblue', 'tomato'],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'lw': 1.5},
        labels=['Class 0', 'Class 1'],
        autopct='%1.0f%%',
        textprops={'fontsize': 10},
    )
    axes[2].set_title('Imbalanced\n99 / 1', fontsize=10)

    fig.suptitle(
        'Figure 7: class imbalance, accuracy can mislead when classes are uneven',
        fontsize=12, y=0.98,
    )
    plt.tight_layout()
    plt.show()