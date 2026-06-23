"""
Figure 13f — Interactive SVM Maximum Margin Explorer
====================================================
Visualises what a linear Support Vector Machine is doing:

  - The decision boundary (hyperplane): w^T x + b = 0
  - The two margin boundaries at +1 and -1
  - The shaded margin region between them
  - Support vectors highlighted with a ring marker
  - A live equation panel showing ||w||, margin width, hinge loss, and total loss

The C slider controls the regularisation trade-off:
  - Small C  — wide margin, tolerates violations (soft boundary)
  - Large C  — narrow margin, insists on correct classification

A toggle button adds a misclassified point to the dataset so users can
see how C controls the model's tolerance for margin violations.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_13f import show
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
from sklearn.svm import SVC

# ── Generate clean linearly separable 2D data ────────────────────────────────
rng = np.random.default_rng(42)
N   = 30

X_pos = rng.multivariate_normal([2.5,  2.5], [[0.4, 0.1], [0.1, 0.4]], N)
X_neg = rng.multivariate_normal([-2.5, -2.5], [[0.4, 0.1], [0.1, 0.4]], N)

X_BASE = np.vstack([X_pos, X_neg])
y_BASE = np.array([1] * N + [-1] * N)

# Pool of noisy points added one at a time when the button is clicked.
# Each sits near the decision boundary on the wrong side to create
# meaningful margin violations that illustrate the effect of C.
NOISY_POOL = np.array([
    [ 0.3,  0.8],
    [-0.5,  0.2],
    [ 0.8, -0.3],
    [-0.2,  1.1],
    [ 1.0,  0.1],
])
NOISY_LABELS = np.array([-1, 1, 1, -1, -1])

COL_POS  = 'steelblue'
COL_NEG  = 'tomato'
COL_SV   = 'gold'
COL_BD   = '#222222'
COL_MARG = '#aaaaaa'


def _fit_svm(C, n_noisy):
    """
    Fit a linear SVM on the base data plus the first n_noisy points
    from NOISY_POOL. Returns the fitted model and the data used.
    """
    if n_noisy > 0:
        X = np.vstack([X_BASE, NOISY_POOL[:n_noisy]])
        y = np.concatenate([y_BASE, NOISY_LABELS[:n_noisy]])
    else:
        X = X_BASE.copy()
        y = y_BASE.copy()

    clf = SVC(kernel='linear', C=C, random_state=0)
    clf.fit(X, y)
    return clf, X, y


def _get_boundary_line(clf, x_range):
    """
    Compute the decision boundary and margin lines over x_range.
    For a linear SVM: w[0]*x + w[1]*y + b = 0  →  y = -(w[0]*x + b) / w[1]
    """
    w = clf.coef_[0]
    b = clf.intercept_[0]
    xs = np.linspace(x_range[0], x_range[1], 300)

    if abs(w[1]) < 1e-10:
        return xs, None, None, None

    # Decision boundary: w^T x + b = 0
    y_bd    = -(w[0] * xs + b) / w[1]
    # Margin boundaries: w^T x + b = +1 and -1
    y_marg1 = -(w[0] * xs + b - 1) / w[1]
    y_marg2 = -(w[0] * xs + b + 1) / w[1]

    return xs, y_bd, y_marg1, y_marg2


def _eq_string(clf, X, y, C):
    """Build the live equation annotation string."""
    w      = clf.coef_[0]
    b      = clf.intercept_[0]
    w_norm = np.linalg.norm(w)
    margin = 2.0 / w_norm if w_norm > 1e-10 else float('inf')

    # Hinge loss: sum of max(0, 1 - y_i * (w^T x_i + b))
    scores     = y * (X @ w + b)
    hinge      = float(np.sum(np.maximum(0, 1 - scores)))
    total_loss = 0.5 * w_norm**2 + C * hinge

    # Count support vectors
    n_sv = len(clf.support_)

    return (
        r'$\mathbf{w} = $' + f'[{w[0]:.3f}, {w[1]:.3f}]' + '\n'
        r'$b = $' + f'{b:.3f}' + '\n\n'
        r'$\|\mathbf{w}\| = $' + f'{w_norm:.4f}' + '\n'
        r'Margin $= \frac{2}{\|\mathbf{w}\|} = $' + f'{margin:.4f}' + '\n\n'
        r'Hinge loss $= $' + f'{hinge:.4f}' + '\n'
        r'$C \times$ hinge $= $' + f'{C * hinge:.4f}' + '\n'
        r'Total loss $= $' + f'{total_loss:.4f}' + '\n\n'
        f'Support vectors: {n_sv}'
    )


def show():
    """Render Figure 13c: Interactive SVM Maximum Margin Explorer."""
    plt.close('Notebook6 Figure 13f')
    fig, ax = plt.subplots(num='Notebook6 Figure 13f', figsize=(8, 6))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    C_INIT  = 1.0
    MAX_NOISY = len(NOISY_POOL)

    X_LIM = (-5, 5)
    Y_LIM = (-5, 5)

    # ── Initial fit ───────────────────────────────────────────────────────────
    clf, X, y = _fit_svm(C_INIT, 0)
    xs, y_bd, y_m1, y_m2 = _get_boundary_line(clf, X_LIM)

    # ── Static scatter (re-drawn on noisy toggle) ─────────────────────────────
    sc_pos = ax.scatter(X[y == 1, 0], X[y == 1, 1],
                        color=COL_POS, s=55, edgecolors='k', lw=0.5,
                        zorder=4, label='Class +1')
    sc_neg = ax.scatter(X[y == -1, 0], X[y == -1, 1],
                        color=COL_NEG, s=55, edgecolors='k', lw=0.5,
                        zorder=4, label='Class −1')

    # Noisy points — accumulate as button is clicked
    sc_noisy = ax.scatter([], [], color=COL_NEG, s=120,
                          edgecolors='red', lw=2.0, zorder=5,
                          marker='*', label='Noisy points')

    # ── Support vector rings ──────────────────────────────────────────────────
    sv_ring = ax.scatter(X[clf.support_, 0], X[clf.support_, 1],
                         facecolors='none', edgecolors=COL_SV,
                         s=160, lw=2.5, zorder=5,
                         label='Support vectors')

    # ── Decision boundary and margin lines ────────────────────────────────────
    (bd_line,)  = ax.plot(xs, y_bd, color=COL_BD, lw=2.5,
                          linestyle='-',  zorder=3,
                          label=r'Decision boundary  $\mathbf{w}^\top\mathbf{x}+b=0$')
    (m1_line,)  = ax.plot(xs, y_m1, color=COL_MARG, lw=1.5,
                          linestyle='--', zorder=3,
                          label=r'Margin  $= \pm 1$')
    (m2_line,)  = ax.plot(xs, y_m2, color=COL_MARG, lw=1.5,
                          linestyle='--', zorder=3)

    # ── Shaded margin region ──────────────────────────────────────────────────
    fill_container = [ax.fill_between(xs, y_m1, y_m2,
                                      alpha=0.10, color=COL_MARG, zorder=2)]

    # ── Equation annotation ───────────────────────────────────────────────────
    eq_text = ax.text(
        0.02, 0.97, _eq_string(clf, X, y, C_INIT),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#cccccc', alpha=0.95),
        zorder=6,
        linespacing=1.55,
        family='monospace',
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    title = ax.set_title(
        rf'Linear SVM — $C = {C_INIT:.3f}$  |  '
        rf'Margin = {2.0 / np.linalg.norm(clf.coef_[0]):.4f}  |  '
        rf'Support vectors: {len(clf.support_)}',
        fontsize=10,
    )

    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.set_xlabel(r'Feature $x_1$', fontsize=11)
    ax.set_ylabel(r'Feature $x_2$', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(True, alpha=0.15)
    ax.set_aspect('equal')

    plt.tight_layout()

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════

    C_slider = widgets.FloatLogSlider(
        value=C_INIT, base=10, min=-2, max=3, step=0.05,
        description='C',
        style={'description_width': '30px'},
        layout=widgets.Layout(width='380px'),
        readout=True,
        readout_format='.3f',
    )

    add_btn = widgets.Button(
        description='Add noisy point',
        button_style='warning',
        layout=widgets.Layout(width='160px'),
    )

    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    # Count of noisy points currently added — stored in a dict so the
    # button callback can mutate it without a nonlocal declaration.
    state = {'n_noisy': 0}

    status_lbl = widgets.Label(
        value=f'Noisy points added: 0 / {MAX_NOISY}',
        layout=widgets.Layout(width='200px'),
    )

    def on_add(_):
        if state['n_noisy'] >= MAX_NOISY:
            status_lbl.value = f'Maximum noisy points ({MAX_NOISY}) reached.'
            return
        state['n_noisy'] += 1
        status_lbl.value = f'Noisy points added: {state["n_noisy"]} / {MAX_NOISY}'
        _redraw(C_slider.value, state['n_noisy'])

    def on_reset(_):
        C_slider.value  = C_INIT
        state['n_noisy'] = 0
        status_lbl.value = f'Noisy points added: 0 / {MAX_NOISY}'
        _redraw(C_INIT, 0)

    add_btn.on_click(on_add)
    reset_btn.on_click(on_reset)

    controls = widgets.HBox([
        C_slider,
        widgets.Label('  '),
        add_btn,
        widgets.Label(' '),
        reset_btn,
        widgets.Label(' '),
        status_lbl,
    ])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # ════════════════════════════════════════════════════════════════════════
    def _redraw(C, n_noisy):
        clf, X, y = _fit_svm(C, n_noisy)
        xs, y_bd, y_m1, y_m2 = _get_boundary_line(clf, X_LIM)

        if y_bd is None:
            return

        # Update scatter positions
        sc_pos.set_offsets(X[y ==  1])
        sc_neg.set_offsets(X[y == -1])

        if n_noisy > 0:
            sc_noisy.set_offsets(NOISY_POOL[:n_noisy])
        else:
            sc_noisy.set_offsets(np.empty((0, 2)))

        # Update support vector rings
        sv_ring.set_offsets(X[clf.support_])

        # Update boundary and margin lines
        bd_line.set_xdata(xs)
        bd_line.set_ydata(y_bd)
        m1_line.set_xdata(xs)
        m1_line.set_ydata(y_m1)
        m2_line.set_xdata(xs)
        m2_line.set_ydata(y_m2)

        # Redraw margin fill
        fill_container[0].remove()
        fill_container[0] = ax.fill_between(
            xs, y_m1, y_m2, alpha=0.10, color=COL_MARG, zorder=2
        )

        # Update equation panel and title
        w_norm = np.linalg.norm(clf.coef_[0])
        margin = 2.0 / w_norm if w_norm > 1e-10 else float('inf')

        eq_text.set_text(_eq_string(clf, X, y, C))
        title.set_text(
            rf'Linear SVM — $C = {C:.3f}$  |  '
            rf'Margin = {margin:.4f}  |  '
            rf'Support vectors: {len(clf.support_)}'
        )

        fig.canvas.draw_idle()

    # Wire C slider to redraw — noisy count is managed by button callbacks above
    def _on_C_change(change):
        _redraw(change['new'], state['n_noisy'])

    C_slider.observe(_on_C_change, names='value')

    display(controls)