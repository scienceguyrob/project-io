"""
Figure 6 — Interactive Entropy Explorer
========================================
Lets users explore how entropy changes as the class proportions in a
group of data points change.

The two-class case (Apple vs Orange) is used throughout to keep the
example concrete and connected to the fruit classification problem.

A single slider controls p_orange — the proportion of oranges in the
current group. p_apple = 1 - p_orange is computed automatically.

The figure shows:
  - The full entropy curve H(p) across all values of p from 0 to 1
  - A vertical marker tracking the current p value
  - A dot on the curve showing the current entropy
  - The formula with current values fully substituted in and computed live
  - Annotations marking the zero-entropy and maximum-entropy points

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_6 import show
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

# ── Pre-compute the full entropy curve ───────────────────────────────────────
# We evaluate H(p) at 500 points from 0 to 1.
# At p=0 and p=1, entropy is exactly 0 (pure groups).
# np.errstate suppresses the log(0) warning at the endpoints —
# we handle those cases explicitly by clipping.
P_VALS = np.linspace(0, 1, 500)

def entropy_two_class(p):
    """
    Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    Handles p=0 and p=1 cleanly (returns 0 at both endpoints).

    Parameters
    ----------
    p : float or array — proportion of one class (0 <= p <= 1)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(p > 0, p * np.log2(p), 0) \
            -np.where(p < 1, (1 - p) * np.log2(1 - p), 0)
    return h

H_VALS = entropy_two_class(P_VALS)

# ── Default value ─────────────────────────────────────────────────────────────
P_INIT = 0.5


def show():
    """
    Render Figure 6: Interactive Entropy Explorer.

    Slider controls p_orange; the marker, dot, and equation update live.
    """
    plt.close('Notebook6 Figure 6')
    fig, ax = plt.subplots(num='Notebook6 Figure 6', figsize=(10, 5))

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # ── Full entropy curve (static) ───────────────────────────────────────────
    ax.plot(P_VALS, H_VALS, color='steelblue', lw=2.5, zorder=3,
            label=r'$H = -p_\mathrm{orange}\log_2 p_\mathrm{orange} '
                  r'- p_\mathrm{apple}\log_2 p_\mathrm{apple}$')

    # Shade under the curve to give it visual weight
    ax.fill_between(P_VALS, H_VALS, alpha=0.08, color='steelblue', zorder=2)

    # ── Annotations for key points ────────────────────────────────────────────
    # Mark the two zero-entropy endpoints
    for px, lbl in [(0.0, 'All Apple\nH = 0'), (1.0, 'All Orange\nH = 0')]:
        ax.annotate(lbl,
                    xy=(px, 0.0), xytext=(px + (0.08 if px == 0 else -0.08), 0.18),
                    fontsize=8.5, color='#555', ha='left' if px == 0 else 'right',
                    arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.0),
                    zorder=5)

    # Mark the maximum entropy point at p=0.5
    # Arrow points vertically down from the peak; text sits below the arrowhead.
    ax.annotate('Equal mix\nH = 1.0 bit\n(maximum uncertainty)',
                xy=(0.5, 1.0), xytext=(0.5, 0.68),
                fontsize=8.5, color='#555', ha='center',
                arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.0),
                zorder=5)

    # ── Vertical line marking current p ──────────────────────────────────────
    p_line = ax.axvline(P_INIT, color='tomato', lw=1.8,
                        linestyle='--', alpha=0.7, zorder=4,
                        label=r'Current $p_\mathrm{orange}$')

    # ── Dot on the curve at current p ────────────────────────────────────────
    h_init = entropy_two_class(P_INIT)
    (dot,) = ax.plot(P_INIT, h_init, 'o', color='tomato',
                     ms=10, zorder=6)

    # ── Horizontal drop line from dot to y-axis ───────────────────────────────
    (h_line,) = ax.plot([0, P_INIT], [h_init, h_init],
                        color='tomato', lw=1.2,
                        linestyle=':', zorder=4, alpha=0.7)

    # ── Live equation annotation ───────────────────────────────────────────────
    eq_text = ax.text(
        0.98, 0.97,
        _make_eq_string(P_INIT),
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#cccccc', alpha=0.95),
        zorder=7,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    title = ax.set_title(
        rf'Entropy: $p_\mathrm{{orange}} = {P_INIT:.2f}$, '
        rf'$p_\mathrm{{apple}} = {1-P_INIT:.2f}$, '
        rf'$H = {h_init:.4f}$ bits',
        fontsize=11,
    )

    # ── Axes settings ─────────────────────────────────────────────────────────
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel(r'$p_\mathrm{orange}$ — proportion of oranges in the group',
                  fontsize=11)
    ax.set_ylabel(r'Entropy $H$ (bits)', fontsize=11)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # ════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ════════════════════════════════════════════════════════════════════════
    p_slider = widgets.FloatSlider(
        value=P_INIT, min=0.0, max=1.0, step=0.01,
        description='p (orange)',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='420px'),
        readout=False,
    )
    p_box = widgets.BoundedFloatText(
        value=P_INIT, min=0.0, max=1.0, step=0.01,
        layout=widgets.Layout(width='80px'),
    )
    widgets.jslink((p_slider, 'value'), (p_box, 'value'))

    reset_btn = widgets.Button(
        description='Reset',
        layout=widgets.Layout(width='80px'),
    )

    def on_reset(_):
        p_slider.value = P_INIT

    reset_btn.on_click(on_reset)

    controls = widgets.HBox([
        p_slider, p_box,
        widgets.Label('  '),
        reset_btn,
    ])

    # ════════════════════════════════════════════════════════════════════════
    # UPDATE FUNCTION
    # ════════════════════════════════════════════════════════════════════════
    def update(p):
        h = entropy_two_class(p)

        # Move vertical marker
        p_line.set_xdata([p, p])

        # Move dot on curve
        dot.set_xdata([p])
        dot.set_ydata([h])

        # Move horizontal drop line
        h_line.set_xdata([0, p])
        h_line.set_ydata([h, h])

        # Update equation and title
        eq_text.set_text(_make_eq_string(p))
        title.set_text(
            rf'Entropy: $p_\mathrm{{orange}} = {p:.2f}$, '
            rf'$p_\mathrm{{apple}} = {1-p:.2f}$, '
            rf'$H = {h:.4f}$ bits'
        )

        fig.canvas.draw_idle()

    out = interactive_output(update, {'p': p_slider})
    display(controls, out)


def _make_eq_string(p):
    """
    Build the live equation string showing the entropy formula with
    current p_orange and p_apple substituted in and fully evaluated.
    """
    pa = 1.0 - p

    # Handle the p=0 and p=1 edge cases cleanly in the displayed terms
    term_orange = f'{p:.2f} x log2({p:.2f}) = {-p * np.log2(p) if p > 0 else 0.0:.4f}'  if p > 0 else f'{p:.2f} x log2({p:.2f}) = 0'
    term_apple  = f'{pa:.2f} x log2({pa:.2f}) = {-pa * np.log2(pa) if pa > 0 else 0.0:.4f}' if pa > 0 else f'{pa:.2f} x log2({pa:.2f}) = 0'
    h           = entropy_two_class(p)

    return (
        r'$H = -\sum_k p_k \log_2 p_k$'
        '\n\n'
        r'$= -(p_\mathrm{orange} \cdot \log_2 p_\mathrm{orange})$'
        '\n'
        r'$\quad -(p_\mathrm{apple} \cdot \log_2 p_\mathrm{apple})$'
        '\n\n'
        rf'$= -({term_orange})$'
        '\n'
        rf'$\quad -({term_apple})$'
        '\n\n'
        rf'$H \approx \mathbf{{{h:.4f}}}$ bits'
    )