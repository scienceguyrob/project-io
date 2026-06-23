"""
Figure 24 — Interactive Feature Distribution Explorer
======================================================
Visualises the class-separated distribution of a single feature from the
Breast Cancer Wisconsin dataset. A dropdown menu selects the feature; the
plot updates live to show:

  Left panel  — KDE curves for Benign and Malignant classes, with a
                vertical line at each class mean and a shaded overlap region.

  Right panel — Box plots for each class, showing the five-number summary
                (median, IQR, whiskers, and outliers) side by side.

A Cohen's d value is computed and displayed for the selected feature,
quantifying the separation between the two class distributions.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_24 import show
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display
from sklearn.datasets import load_breast_cancer
import pandas as pd

# ── Load and prepare the dataset ─────────────────────────────────────────────
_bc      = load_breast_cancer()
_df_raw  = pd.DataFrame(_bc.data, columns=_bc.feature_names)
_df_raw['target']    = _bc.target
_df_raw['diagnosis'] = _df_raw['target'].map({1: 'Benign', 0: 'Malignant'})

# Restrict to 'mean' features — consistent with the notebook's earlier sections
MEAN_COLS = [c for c in _df_raw.columns if c.startswith('mean')]
df        = _df_raw.copy()

COL_BENIGN    = 'steelblue'
COL_MALIGNANT = 'tomato'
CLASSES       = ['Benign', 'Malignant']
COLOURS       = {'Benign': COL_BENIGN, 'Malignant': COL_MALIGNANT}

# ── Helper functions ──────────────────────────────────────────────────────────

def _cohens_d(feature):
    """
    Compute Cohen's d for a single feature.
    Returns the absolute standardised mean difference between Benign and Malignant.
    """
    a = df.loc[df['diagnosis'] == 'Benign',    feature].values
    b = df.loc[df['diagnosis'] == 'Malignant', feature].values
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    if pooled_std == 0:
        return 0.0
    return abs(a.mean() - b.mean()) / pooled_std


def _effect_label(d):
    """Map a Cohen's d value to a conventional verbal label and colour."""
    if d < 0.2:
        return 'Negligible',  '#999999'
    elif d < 0.5:
        return 'Small',       '#e67e22'
    elif d < 0.8:
        return 'Medium',      '#e67e22'
    elif d < 1.5:
        return 'Large',       'seagreen'
    else:
        return 'Very large',  'seagreen'


def _kde_values(data, x_grid):
    """Fit a Gaussian KDE to data and evaluate it on x_grid."""
    kde = gaussian_kde(data, bw_method='scott')
    return kde(x_grid)


# ── Main show() function ──────────────────────────────────────────────────────

def show():
    """Render Figure 24: Interactive Feature Distribution Explorer."""
    plt.close('Notebook7 Figure 24')

    fig = plt.figure(num='Notebook7 Figure 24', figsize=(10, 5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    gs      = GridSpec(1, 2, figure=fig, wspace=0.32)
    ax_kde  = fig.add_subplot(gs[0, 0])
    ax_box  = fig.add_subplot(gs[0, 1])

    # ── Initial feature ───────────────────────────────────────────────────────
    INIT_FEATURE = MEAN_COLS[0]

    def _get_data(feature):
        benign    = df.loc[df['diagnosis'] == 'Benign',    feature].values
        malignant = df.loc[df['diagnosis'] == 'Malignant', feature].values
        return benign, malignant

    benign_init, malignant_init = _get_data(INIT_FEATURE)
    all_vals  = np.concatenate([benign_init, malignant_init])
    x_min, x_max = all_vals.min(), all_vals.max()
    pad   = (x_max - x_min) * 0.10
    x_grid = np.linspace(x_min - pad, x_max + pad, 400)

    # ── LEFT PANEL: KDE ───────────────────────────────────────────────────────
    kde_b_vals = _kde_values(benign_init,    x_grid)
    kde_m_vals = _kde_values(malignant_init, x_grid)

    # Filled KDE curves
    fill_b = ax_kde.fill_between(x_grid, kde_b_vals, alpha=0.35, color=COL_BENIGN)
    fill_m = ax_kde.fill_between(x_grid, kde_m_vals, alpha=0.35, color=COL_MALIGNANT)

    # KDE outline curves
    (line_b,) = ax_kde.plot(x_grid, kde_b_vals, color=COL_BENIGN,    lw=2.0, label='Benign')
    (line_m,) = ax_kde.plot(x_grid, kde_m_vals, color=COL_MALIGNANT, lw=2.0, label='Malignant')

    # Class mean vertical lines
    (vline_b,) = ax_kde.plot(
        [benign_init.mean(), benign_init.mean()], [0, max(kde_b_vals)],
        color=COL_BENIGN, lw=1.5, ls='--', alpha=0.8,
    )
    (vline_m,) = ax_kde.plot(
        [malignant_init.mean(), malignant_init.mean()], [0, max(kde_m_vals)],
        color=COL_MALIGNANT, lw=1.5, ls='--', alpha=0.8,
    )

    ax_kde.set_xlabel(INIT_FEATURE, fontsize=10)
    ax_kde.set_ylabel('Density', fontsize=10)
    ax_kde.legend(fontsize=9, framealpha=1.0, edgecolor='#ccc')
    ax_kde.grid(True, alpha=0.15)

    d_init         = _cohens_d(INIT_FEATURE)
    eff_lbl, eff_col = _effect_label(d_init)

    kde_title = ax_kde.set_title(
        f"KDE by class — {INIT_FEATURE}\n"
        f"Cohen's d = {d_init:.3f}  ({eff_lbl} effect)",
        fontsize=10,
    )

    # Cohen's d annotation box on KDE panel
    d_text = ax_kde.text(
        0.97, 0.97,
        f"d = {d_init:.3f}\n{eff_lbl}",
        transform=ax_kde.transAxes,
        fontsize=9, va='top', ha='right', color=eff_col,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#ccc', alpha=1.0),
        zorder=9,
    )

    # ── RIGHT PANEL: Box plots ────────────────────────────────────────────────
    # We redraw the box plots on each update by clearing the axis,
    # because matplotlib box plot artists cannot be updated in place cleanly.
    def _draw_boxplots(feature):
        ax_box.cla()
        benign, malignant = _get_data(feature)

        bp = ax_box.boxplot(
            [benign, malignant],
            patch_artist=True,        # Filled boxes rather than outline-only
            widths=0.45,
            medianprops=dict(color='white', linewidth=2.0),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            flierprops=dict(marker='o', markersize=4, alpha=0.5,
                            markeredgewidth=0.4),
            labels=['Benign', 'Malignant'],
        )

        # Apply class colours to each box and its associated artists
        for patch, colour in zip(bp['boxes'], [COL_BENIGN, COL_MALIGNANT]):
            patch.set_facecolor(colour)
            patch.set_alpha(0.65)

        for whisker, colour in zip(bp['whiskers'], [COL_BENIGN, COL_BENIGN,
                                                     COL_MALIGNANT, COL_MALIGNANT]):
            whisker.set_color(colour)

        for cap, colour in zip(bp['caps'], [COL_BENIGN, COL_BENIGN,
                                             COL_MALIGNANT, COL_MALIGNANT]):
            cap.set_color(colour)

        for flier, colour in zip(bp['fliers'], [COL_BENIGN, COL_MALIGNANT]):
            flier.set_markerfacecolor(colour)
            flier.set_markeredgecolor(colour)

        ax_box.set_ylabel(feature, fontsize=10)
        ax_box.set_title(f'Box plot by class — {feature}', fontsize=10)
        ax_box.grid(True, alpha=0.15, axis='y')

    _draw_boxplots(INIT_FEATURE)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    suptitle = fig.suptitle(
        f'Figure 24: Feature distribution explorer — {INIT_FEATURE}',
        fontsize=11,
    )
    plt.subplots_adjust(top=0.88)

    # ── Widget ────────────────────────────────────────────────────────────────
    feature_dropdown = widgets.Dropdown(
        options=MEAN_COLS,
        value=INIT_FEATURE,
        description='Feature:',
        style={'description_width': '60px'},
        layout=widgets.Layout(width='340px'),
    )

    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_reset(_):
        feature_dropdown.value = INIT_FEATURE

    reset_btn.on_click(on_reset)
    controls = widgets.HBox([feature_dropdown, widgets.Label('  '), reset_btn])

    # ── Update function ───────────────────────────────────────────────────────
    def update(feature):
        benign, malignant = _get_data(feature)
        all_vals  = np.concatenate([benign, malignant])
        x_min, x_max = all_vals.min(), all_vals.max()
        pad   = (x_max - x_min) * 0.10
        x_grid = np.linspace(x_min - pad, x_max + pad, 400)

        # Recompute KDE values
        kde_b = _kde_values(benign,    x_grid)
        kde_m = _kde_values(malignant, x_grid)

        # Update filled regions — remove old collections and redraw
        for coll in list(ax_kde.collections):
            coll.remove()

        ax_kde.fill_between(x_grid, kde_b, alpha=0.35, color=COL_BENIGN)
        ax_kde.fill_between(x_grid, kde_m, alpha=0.35, color=COL_MALIGNANT)

        # Update KDE outline curves
        line_b.set_xdata(x_grid)
        line_b.set_ydata(kde_b)
        line_m.set_xdata(x_grid)
        line_m.set_ydata(kde_m)

        # Update mean lines
        vline_b.set_xdata([benign.mean(),    benign.mean()])
        vline_b.set_ydata([0, max(kde_b)])
        vline_m.set_xdata([malignant.mean(), malignant.mean()])
        vline_m.set_ydata([0, max(kde_m)])

        # Rescale KDE axes
        ax_kde.relim()
        ax_kde.autoscale_view()
        ax_kde.set_xlabel(feature, fontsize=10)

        # Update Cohen's d
        d          = _cohens_d(feature)
        eff_lbl, eff_col = _effect_label(d)

        kde_title.set_text(
            f"KDE by class — {feature}\n"
            f"Cohen's d = {d:.3f}  ({eff_lbl} effect)"
        )

        d_text.set_text(f"d = {d:.3f}\n{eff_lbl}")
        d_text.set_color(eff_col)
        d_text.get_bbox_patch().set_edgecolor('#ccc')

        # Redraw box plots
        _draw_boxplots(feature)

        # Update suptitle
        suptitle.set_text(
            f'Figure 24: Feature distribution explorer — {feature}'
        )

        fig.canvas.draw_idle()

    out = interactive_output(update, {'feature': feature_dropdown})
    display(controls, out)