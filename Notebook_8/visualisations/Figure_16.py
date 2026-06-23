"""
Figure 16 — Interactive Hierarchical Clustering: Cut the Dendrogram
====================================================================
Allows the user to drag a horizontal cut line across a Ward-linkage
dendrogram and see the resulting cluster assignments update in real time
on the scatter plot to the right.

As the cut line moves up and down the dendrogram:
  - The number of clusters changes (shown in the scatter plot title)
  - Each cluster is coloured distinctly on the scatter plot
  - The cut height and number of resulting clusters are displayed

This makes the connection between dendrogram structure and cluster
assignments concrete and interactive.

Interaction
-----------
Click and drag the red horizontal cut line up or down the dendrogram.
The scatter plot updates live.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_16 import show
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
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.datasets import make_blobs


# ── Dataset ───────────────────────────────────────────────────────────────────
# Four natural clusters — chosen so the dendrogram shows a clear large gap
# at the four-cluster level, making the optimal cut height obvious.
X_hc, _ = make_blobs(
    n_samples=80, centers=4, cluster_std=0.7, random_state=3
)

# ── Build the linkage matrix ──────────────────────────────────────────────────
# Ward linkage minimises the increase in WCSS at each merge — it produces
# the clearest dendrogram gaps for compact, spherical clusters like these.
Z = linkage(X_hc, method='ward')

# ── Colour palette ────────────────────────────────────────────────────────────
# Tab10 gives up to 10 visually distinct colours — enough for any
# reasonable number of clusters the cut line might produce.
CMAP = cm.get_cmap('tab10')

def _col(j):
    return CMAP(j / 10.0)


# ── Cut height limits ─────────────────────────────────────────────────────────
# The dendrogram y-axis runs from 0 to the height of the final merge.
# We add a small margin at the top so the cut line can sit above all merges
# (producing 1 cluster) without clipping.
Y_MIN  = 0.0
Y_MAX  = float(Z[-1, 2]) * 1.15
Y_INIT = float(Z[-2, 2]) * 0.85   # start just below the top merge — 2 clusters


def _labels_from_cut(cut_height):
    """
    Return cluster labels for each data point given a cut height.
    fcluster with criterion='distance' cuts the dendrogram at the given
    height and returns a 1-based label array — we subtract 1 to make
    labels 0-based for consistent colour indexing.
    """
    if cut_height <= 0:
        return np.arange(len(X_hc))   # every point its own cluster
    labels = fcluster(Z, t=cut_height, criterion='distance')
    return labels - 1   # convert to 0-based


def _n_clusters(cut_height):
    return len(np.unique(_labels_from_cut(cut_height)))


def show():
    """Render Figure 16: interactive dendrogram cut explorer."""
    plt.close('Notebook8 Figure 16')

    fig, (ax_dend, ax_scatter) = plt.subplots(
        1, 2,
        num='Notebook8 Figure 16',
        figsize=(10, 6),
        gridspec_kw={'width_ratios': [1.2, 1]},
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    state    = {'cut': Y_INIT}
    dragging = [False]

    # ── Left panel: dendrogram ────────────────────────────────────────────────
    # Draw the full Ward dendrogram once — it never changes.
    # above_threshold_color and color_threshold=0 force all branches to be
    # drawn in a neutral grey so the cluster colours on the scatter plot are
    # the primary visual encoding, not the dendrogram branch colours.
    dendrogram(
        Z, ax=ax_dend,
        color_threshold=0,
        above_threshold_color='#888888',
        link_color_func=lambda k: '#888888',
        leaf_font_size=0,
        no_labels=True,
    )

    ax_dend.set_ylim(Y_MIN, Y_MAX)
    ax_dend.set_xlabel('Data points', fontsize=10)
    ax_dend.set_ylabel('Merge distance (Ward)', fontsize=10)
    ax_dend.set_title(
        'Dendrogram — drag the red line to cut\n'
        'Large vertical gaps = natural cluster boundaries',
        fontsize=10,
    )
    ax_dend.grid(True, alpha=0.15, axis='y')

    # Draggable cut line
    cut_line = ax_dend.axhline(
        state['cut'], color='tomato', lw=2.0, ls='--', zorder=5,
    )

    # Cut height annotation on the dendrogram
    cut_ann = ax_dend.text(
        0.02, state['cut'],
        f'  cut = {state["cut"]:.2f}',
        transform=ax_dend.get_yaxis_transform(),
        fontsize=9, color='tomato', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.85),
        zorder=6,
    )

    # ── Right panel: scatter ──────────────────────────────────────────────────
    def _draw_scatter(cut_height):
        ax_scatter.clear()

        labels   = _labels_from_cut(cut_height)
        n_clust  = len(np.unique(labels))

        for j in range(n_clust):
            mask = labels == j
            ax_scatter.scatter(
                X_hc[mask, 0], X_hc[mask, 1],
                color=_col(j), s=45, alpha=0.85,
                edgecolors='k', lw=0.3,
                label=f'Cluster {j}',
                zorder=2,
            )

        ax_scatter.set_title(
            f'Cut height = {cut_height:.2f}   →   {n_clust} cluster{"s" if n_clust != 1 else ""}',
            fontsize=10,
        )
        ax_scatter.set_xlabel('Feature 1', fontsize=10)
        ax_scatter.set_ylabel('Feature 2', fontsize=10)
        ax_scatter.legend(
            fontsize=8, loc='upper left',
            framealpha=1.0, edgecolor='#cccccc',
        )
        ax_scatter.grid(True, alpha=0.2)

    _draw_scatter(state['cut'])

    fig.suptitle(
        'Figure 16: Hierarchical clustering — drag the cut line to choose k',
        fontsize=11,
    )
    plt.subplots_adjust(wspace=0.28, top=0.88)

    # ── Drag interaction ──────────────────────────────────────────────────────
    def _on_press(event):
        if event.inaxes is not ax_dend:
            return
        # Accept a click anywhere on the dendrogram panel — the cut line
        # jumps to the y-coordinate of the click, making it easy to position
        if event.ydata is not None:
            dragging[0] = True
            state['cut'] = float(np.clip(event.ydata, Y_MIN, Y_MAX))
            _update()

    def _on_release(event):
        dragging[0] = False

    def _on_motion(event):
        if not dragging[0] or event.inaxes is not ax_dend:
            return
        if event.ydata is not None:
            state['cut'] = float(np.clip(event.ydata, Y_MIN, Y_MAX))
            _update()

    def _update():
        cut = state['cut']

        # Update cut line position and annotation
        cut_line.set_ydata([cut, cut])
        cut_ann.set_position((0.02, cut))
        cut_ann.set_text(f'  cut = {cut:.2f}  ({_n_clusters(cut)} clusters)')

        # Redraw scatter
        _draw_scatter(cut)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event',  _on_motion)

    plt.show()