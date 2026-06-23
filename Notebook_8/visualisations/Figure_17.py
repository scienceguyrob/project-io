"""
Figure 17 — Interactive Hierarchical Clustering: Three Linkage Methods
=======================================================================
Shows the dendrogram and cluster assignments for Ward, Single, and Complete
linkage simultaneously. A single draggable cut line moves across all three
dendrograms proportionally — each dendrogram uses its own y-axis scale so
no tree is compressed, while the cut fraction is shared so all three panels
always show the same relative cut depth.

The three scatter plots below update together as the cut moves, so the
effect of linkage choice at any given cut depth is immediately comparable.

Interaction
-----------
Click and drag anywhere on any of the three dendrogram panels to move
the cut line. All six panels update live.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_17 import show
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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.datasets import make_blobs


# ── Dataset ───────────────────────────────────────────────────────────────────
X_hc, _ = make_blobs(
    n_samples=80, centers=4, cluster_std=0.7, random_state=3
)

# ── Colour palettes ───────────────────────────────────────────────────────────
DEND_COLOURS = {
    'Ward':     'steelblue',
    'Single':   'tomato',
    'Complete': 'seagreen',
}
PALETTE_HC = ['steelblue', 'tomato', 'seagreen', 'goldenrod',
              'mediumpurple', 'darkorange', 'pink', 'brown']


def _match_labels_to_reference(labels, ref_centroids, X):
    """
    Remap cluster indices so each cluster is assigned the colour of the
    nearest Ward centroid. Ensures colour consistency across linkage methods.
    """
    n_clust   = len(np.unique(labels))
    centroids = np.array([
        X[labels == j].mean(axis=0)
        for j in range(n_clust)
        if np.any(labels == j)
    ])
    mapping = {}
    used    = set()
    for j, c in enumerate(centroids):
        dists = np.linalg.norm(ref_centroids - c, axis=1)
        for idx in np.argsort(dists):
            if idx not in used:
                mapping[j] = int(idx)
                used.add(int(idx))
                break
        else:
            mapping[j] = j % len(ref_centroids)
    return np.array([mapping[l] for l in labels])


def _labels_from_cut(Z, cut_height):
    """Cut the linkage tree Z at cut_height and return 0-based labels."""
    if cut_height <= 0:
        return np.arange(len(X_hc))
    return fcluster(Z, t=cut_height, criterion='distance') - 1


def show():
    """Render Figure 17: interactive three-linkage comparison."""
    plt.close('Notebook8 Figure 17')

    # ── Build linkage matrices ────────────────────────────────────────────────
    Z_ward     = linkage(X_hc, method='ward')
    Z_single   = linkage(X_hc, method='single')
    Z_complete = linkage(X_hc, method='complete')

    linkage_configs = [
        ('Ward',     Z_ward),
        ('Single',   Z_single),
        ('Complete', Z_complete),
    ]

    # ── Per-tree y-axis limits ────────────────────────────────────────────────
    # Each dendrogram gets its own y scale so no tree is compressed.
    # Y_MAX[i] is the maximum merge height for tree i, with a small margin.
    Y_MAXES = [float(Z[-1, 2]) * 1.15 for _, Z in linkage_configs]

    # ── Shared cut fraction ───────────────────────────────────────────────────
    # The cut is stored as a fraction (0–1) of each tree's own y-axis range.
    # This fraction is applied to each panel independently, so dragging on
    # any panel moves all three cut lines to the same relative depth.
    # Initial fraction places the cut just below the top merge of the Ward tree.
    CUT_FRAC_INIT = float(Z_ward[-2, 2]) * 0.85 / Y_MAXES[0]

    state    = {'frac': CUT_FRAC_INIT}
    dragging = [False]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 3,
        num='Notebook8 Figure 17',
        figsize=(10, 10),
    )
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible  = False

    # ── Draw dendrograms ──────────────────────────────────────────────────────
    for i, (ax, (name, Z)) in enumerate(zip(axes[0], linkage_configs)):
        col = DEND_COLOURS[name]
        dendrogram(
            Z, ax=ax,
            color_threshold=0,
            link_color_func=lambda k, c=col: c,
            leaf_font_size=0,
            no_labels=True,
        )
        # Each panel uses its own y-axis scale
        ax.set_ylim(0.0, Y_MAXES[i])
        ax.set_title(f'{name} linkage\ndendrogram', fontsize=10)
        ax.set_xlabel('Data points', fontsize=9)
        ax.set_ylabel('Merge distance', fontsize=9)
        ax.grid(True, alpha=0.15, axis='y')

    # Cut lines — one per panel, each at the fraction of its own y-axis
    cut_lines = [
        ax.axhline(
            state['frac'] * Y_MAXES[i],
            color='black', lw=1.8, ls='--', zorder=5,
        )
        for i, ax in enumerate(axes[0])
    ]

    # Annotation on the leftmost dendrogram showing the Ward cluster count
    cut_ann = axes[0][0].text(
        0.02, state['frac'] * Y_MAXES[0],
        '',
        transform=axes[0][0].get_yaxis_transform(),
        fontsize=9, color='black', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.85),
        zorder=6,
    )

    # ── Ward reference centroids ──────────────────────────────────────────────
    def _ward_ref_centroids(cut_height_ward):
        lab = _labels_from_cut(Z_ward, cut_height_ward)
        return np.array([
            X_hc[lab == j].mean(axis=0)
            for j in range(len(np.unique(lab)))
            if np.any(lab == j)
        ])

    # ── Draw scatter panels ───────────────────────────────────────────────────
    def _draw_scatter(frac):
        # Each tree is cut at the fraction of its own maximum height
        cut_heights = [frac * ym for ym in Y_MAXES]
        ref_centroids = _ward_ref_centroids(cut_heights[0])

        for ax, (name, Z), cut_h in zip(axes[1], linkage_configs, cut_heights):
            ax.clear()
            labels  = _labels_from_cut(Z, cut_h)
            labels  = _match_labels_to_reference(labels, ref_centroids, X_hc)
            n_clust = len(np.unique(labels))

            for j in range(n_clust):
                mask = labels == j
                ax.scatter(
                    X_hc[mask, 0], X_hc[mask, 1],
                    color=PALETTE_HC[j % len(PALETTE_HC)],
                    s=40, edgecolors='k', lw=0.3, alpha=0.8,
                    label=f'Cluster {j}', zorder=2,
                )

            ax.set_title(
                f'{name} linkage\n'
                f'{n_clust} cluster{"s" if n_clust != 1 else ""}',
                fontsize=10,
            )
            ax.set_xlabel('Feature 1', fontsize=9)
            ax.set_ylabel('Feature 2', fontsize=9)
            ax.legend(fontsize=7, loc='upper left',
                      framealpha=1.0, edgecolor='#cccccc')
            ax.grid(True, alpha=0.2)

    _draw_scatter(state['frac'])

    fig.suptitle(
        'Figure 17: Hierarchical clustering — drag the cut line to compare linkage methods\n'
        'Each dendrogram uses its own scale; the cut moves proportionally across all three',
        fontsize=11, y=0.98,
    )
    plt.tight_layout()

    # ── Update ────────────────────────────────────────────────────────────────
    def _update():
        frac = state['frac']

        # Move each cut line to the correct absolute height on its own scale
        for i, line in enumerate(cut_lines):
            line.set_ydata([frac * Y_MAXES[i], frac * Y_MAXES[i]])

        # Update annotation with Ward cluster count at this cut
        ward_cut  = frac * Y_MAXES[0]
        n_ward    = len(np.unique(_labels_from_cut(Z_ward, ward_cut)))
        cut_ann.set_position((0.02, ward_cut))
        cut_ann.set_text(f'  cut = {frac:.2f}  (Ward: {n_ward} clusters)')

        _draw_scatter(frac)
        fig.canvas.draw_idle()

    # ── Drag interaction ──────────────────────────────────────────────────────
    dend_axes = list(axes[0])

    def _on_press(event):
        if event.inaxes in dend_axes and event.ydata is not None:
            i    = dend_axes.index(event.inaxes)
            frac = float(np.clip(event.ydata / Y_MAXES[i], 0.0, 1.0))
            dragging[0]   = True
            state['frac'] = frac
            _update()

    def _on_release(event):
        dragging[0] = False

    def _on_motion(event):
        if dragging[0] and event.inaxes in dend_axes and event.ydata is not None:
            i    = dend_axes.index(event.inaxes)
            frac = float(np.clip(event.ydata / Y_MAXES[i], 0.0, 1.0))
            state['frac'] = frac
            _update()

    fig.canvas.mpl_connect('button_press_event',   _on_press)
    fig.canvas.mpl_connect('button_release_event', _on_release)
    fig.canvas.mpl_connect('motion_notify_event',  _on_motion)

    plt.show()