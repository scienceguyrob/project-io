"""
Figure 28 — Interactive Lasso (L1) Feature Selection Explorer
=============================================================
Visualises how L1 regularisation strength controls the number of surviving
features and model accuracy in a Logistic Regression classifier trained on
the Breast Cancer Wisconsin dataset.

A slider controls the regularisation parameter C. As C is varied, three
panels update live:

  Left panel   — Horizontal bar chart of surviving (non-zero) feature
                 coefficients, sorted by absolute magnitude. Features
                 driven to zero by the penalty are absent from the chart.

  Centre panel — The full regularisation path: number of non-zero
                 coefficients plotted against C across the full range,
                 with a vertical marker tracking the current C value.

  Right panel  — Test accuracy plotted against C across the full range,
                 with a vertical marker tracking the current C value.

A summary annotation shows the current C value, the number of surviving
features, and the test accuracy at that penalty level.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_28 import show
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
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive_output
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Load and prepare the dataset ─────────────────────────────────────────────
_bc      = load_breast_cancer()
_df      = pd.DataFrame(_bc.data, columns=_bc.feature_names)
_df['target'] = _bc.target

ALL_FEATURES = list(_bc.feature_names)

X_all = _df[ALL_FEATURES].values
y_all = _df['target'].values

# Scale once — coefficients are only comparable across features when all
# features are on the same scale (zero mean, unit variance).
_scaler  = StandardScaler()
X_scaled = _scaler.fit_transform(X_all)

# Fixed train/test split — consistent across all C values so accuracy
# differences reflect regularisation strength alone, not split variation.
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=0)

# ── Pre-compute the regularisation path ──────────────────────────────────────
# Evaluate a dense grid of C values so the path curves are smooth.
# np.logspace(-3, 1, 60) gives 60 values evenly spaced on a log scale
# from 0.001 to 10 — appropriate because C spans several orders of magnitude.
C_GRID    = np.logspace(-3, 1, 60)
N_NONZERO = []   # Number of surviving (non-zero) features at each C
TEST_ACCS = []   # Test accuracy at each C

for c in C_GRID:
    m = LogisticRegression(penalty='l1', C=c, solver='liblinear',
                           max_iter=2000, random_state=0)
    m.fit(X_tr, y_tr)
    N_NONZERO.append((m.coef_[0] != 0).sum())
    TEST_ACCS.append(m.score(X_te, y_te))

N_NONZERO = np.array(N_NONZERO)
TEST_ACCS = np.array(TEST_ACCS)

# Discrete C values exposed to the slider — log-spaced for intuitive control.
# These are the values the user can select; the path curves use C_GRID above.
# Build C options as (label, float) tuples so SelectionSlider stores the actual
# float value rather than a string — avoids floating point string-matching errors.
_C_RAW    = np.logspace(-3, 1, 40)
C_OPTIONS = [(f'{c:.5g}', float(c)) for c in _C_RAW]

# Find the closest available C value to the desired initial value of 0.05
C_INIT    = min(C_OPTIONS, key=lambda t: abs(t[1] - 0.05))[1]

COL_SURVIVE  = 'steelblue'
COL_ZERO     = '#cccccc'
COL_PATH     = 'seagreen'
COL_ACC      = 'steelblue'
COL_MARKER   = 'tomato'


# ── Helper: fit model at a given C and return coefficient Series ──────────────
def _fit(c):
    m = LogisticRegression(penalty='l1', C=c, solver='liblinear',
                           max_iter=2000, random_state=0)
    m.fit(X_tr, y_tr)
    coef = pd.Series(m.coef_[0], index=ALL_FEATURES)
    acc  = m.score(X_te, y_te)
    return coef, acc


# ── Main show() function ──────────────────────────────────────────────────────
def show():
    """Render Figure 28: Interactive Lasso Feature Selection Explorer."""
    plt.close('Notebook7 Figure 28')

    fig, axes = plt.subplots(
        1, 3,
        num='Notebook7 Figure 28',
        figsize=(12, 6),
        gridspec_kw={'width_ratios': [1.6, 1, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    ax_coef, ax_path, ax_acc = axes

    # ── Initial fit ───────────────────────────────────────────────────────────
    coef_init, acc_init = _fit(C_INIT)
    surviving_init = coef_init[coef_init != 0].sort_values(key=abs)
    n_surv_init    = len(surviving_init)

    # ── LEFT PANEL: coefficient bar chart ────────────────────────────────────
    # Bars are drawn horizontally so feature names are readable.
    # Only surviving (non-zero) features appear; features driven to zero
    # by the penalty are absent — their absence is the selection signal.
    bar_colours = [COL_SURVIVE if v > 0 else COL_MARKER
                   for v in surviving_init.values]

    bars = ax_coef.barh(
        surviving_init.index,
        surviving_init.values,
        color=bar_colours,
        edgecolor='white', linewidth=0.4,
    )

    ax_coef.axvline(0, color='#333', linewidth=0.8)
    ax_coef.set_xlabel('Coefficient value  (scaled features)', fontsize=9)
    ax_coef.set_ylabel('')
    ax_coef.tick_params(axis='y', labelsize=7.5)
    ax_coef.grid(True, alpha=0.15, axis='x')

    coef_title = ax_coef.set_title(
        f'Surviving features at C = {C_INIT}\n'
        f'{n_surv_init} of {len(ALL_FEATURES)} features retained',
        fontsize=9,
    )

    # ── CENTRE PANEL: regularisation path (n_nonzero vs C) ───────────────────
    ax_path.semilogx(C_GRID, N_NONZERO, '-', color=COL_PATH, lw=2.0)
    ax_path.set_xlabel('C  (smaller = stronger penalty)', fontsize=9)
    ax_path.set_ylabel('Non-zero coefficients', fontsize=9)
    ax_path.set_title('Features surviving vs C', fontsize=9)
    ax_path.grid(True, alpha=0.2)

    # Vertical marker tracks the current C on the path curve
    (path_vline,) = ax_path.plot(
        [C_INIT, C_INIT],
        [0, len(ALL_FEATURES)],
        color=COL_MARKER, lw=1.8, ls='--',
    )
    # Horizontal dot showing the n_nonzero value at the current C
    path_n  = np.interp(C_INIT, C_GRID, N_NONZERO)
    (path_dot,) = ax_path.plot(
        C_INIT, path_n,
        'o', color=COL_MARKER, ms=9, zorder=5,
    )

    # ── RIGHT PANEL: accuracy path (test accuracy vs C) ──────────────────────
    ax_acc.semilogx(C_GRID, TEST_ACCS, '-', color=COL_ACC, lw=2.0)
    ax_acc.set_xlabel('C  (smaller = stronger penalty)', fontsize=9)
    ax_acc.set_ylabel('Test accuracy', fontsize=9)
    ax_acc.set_title('Test accuracy vs C', fontsize=9)
    ax_acc.set_ylim(
        max(0, min(TEST_ACCS) - 0.05),
        min(1.0, max(TEST_ACCS) + 0.02),
    )
    ax_acc.grid(True, alpha=0.2)

    # Vertical marker and dot tracking current C on the accuracy curve
    (acc_vline,) = ax_acc.plot(
        [C_INIT, C_INIT],
        [0, 1],
        color=COL_MARKER, lw=1.8, ls='--',
    )
    acc_val = np.interp(C_INIT, C_GRID, TEST_ACCS)
    (acc_dot,) = ax_acc.plot(
        C_INIT, acc_val,
        's', color=COL_MARKER, ms=9, zorder=5,
    )

    # ── Suptitle ──────────────────────────────────────────────────────────────
    suptitle = fig.suptitle(
        f'Figure 28: Lasso (L1) feature selection — '
        f'C = {C_INIT}  |  {n_surv_init} features  |  '
        f'test accuracy = {acc_init:.3f}',
        fontsize=10,
    )
    plt.subplots_adjust(top=0.88, wspace=0.35)

    # ── Widgets ───────────────────────────────────────────────────────────────
    c_slider = widgets.SelectionSlider(
        options=C_OPTIONS,    # list of (label_string, float_value) tuples
        value=C_INIT,         # float — matched exactly against the tuple values
        description='C:',
        style={'description_width': '20px'},
        layout=widgets.Layout(width='500px'),
        readout=True,
    )
    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_reset(_):
        c_slider.value = C_INIT

    reset_btn.on_click(on_reset)
    controls = widgets.HBox([c_slider, widgets.Label('  '), reset_btn])

    # ── Update function ───────────────────────────────────────────────────────
    def update(c_str):
        c = c_str   # already a float — SelectionSlider passes the tuple's value
        coef, acc = _fit(c)

        # Surviving features only — sorted by absolute coefficient magnitude
        surviving = coef[coef != 0].sort_values(key=abs)
        n_surv    = len(surviving)

        # --- Redraw coefficient bar chart ---
        # The number of surviving features changes with C so we clear and
        # redraw the left panel entirely on each update rather than trying
        # to update individual bar patches in place.
        ax_coef.cla()

        if n_surv == 0:
            ax_coef.text(0.5, 0.5, 'No features survive\nat this penalty strength',
                         transform=ax_coef.transAxes,
                         ha='center', va='center', fontsize=10, color='#999')
        else:
            b_cols = [COL_SURVIVE if v > 0 else COL_MARKER
                      for v in surviving.values]
            ax_coef.barh(surviving.index, surviving.values,
                         color=b_cols, edgecolor='white', linewidth=0.4)
            ax_coef.axvline(0, color='#333', linewidth=0.8)
            ax_coef.tick_params(axis='y', labelsize=7.5)
            ax_coef.grid(True, alpha=0.15, axis='x')

        ax_coef.set_xlabel('Coefficient value  (scaled features)', fontsize=9)
        ax_coef.set_title(
            f'Surviving features at C = {c}\n'
            f'{n_surv} of {len(ALL_FEATURES)} features retained',
            fontsize=9,
        )

        # --- Update path marker (centre panel) ---
        path_vline.set_xdata([c, c])
        path_n = np.interp(c, C_GRID, N_NONZERO)
        path_dot.set_xdata([c])
        path_dot.set_ydata([path_n])

        # --- Update accuracy marker (right panel) ---
        acc_vline.set_xdata([c, c])
        acc_val = np.interp(c, C_GRID, TEST_ACCS)
        acc_dot.set_xdata([c])
        acc_dot.set_ydata([acc_val])

        # --- Update suptitle ---
        suptitle.set_text(
            f'Figure 28: Lasso (L1) feature selection — '
            f'C = {c}  |  {n_surv} features  |  '
            f'test accuracy = {acc:.3f}'
        )

        fig.canvas.draw_idle()

    out = interactive_output(update, {'c_str': c_slider})
    display(controls, out)