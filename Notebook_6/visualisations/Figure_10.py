"""
Figure 10 — Interactive Decision Tree Boundary Builder
=======================================================
Scatter plot (matplotlib) on top; tree diagram (SVG) below.
Each button press adds the next best Gini split up to max depth 6.

Usage
-----
In a Jupyter notebook cell:

    %matplotlib widget
    from visualisations.Figure_10 import show
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
from IPython.display import display, HTML

# ── Data ─────────────────────────────────────────────────────────────────────
rng     = np.random.default_rng(42)
N_TOTAL = 60
X1      = rng.uniform(4.0, 11.0, N_TOTAL)
X2      = rng.uniform(0.5,  9.5, N_TOTAL)
noise   = rng.uniform(-0.45, 0.45, N_TOTAL)
y_all   = (((X2 - 0.5) / 9.0 + noise) > 0.5).astype(int)
X_all   = np.column_stack([X1, X2])

COL_APPLE  = 'steelblue'
COL_ORANGE = 'tomato'
X1_MIN, X1_MAX = X1.min() - 0.5, X1.max() + 0.5
X2_MIN, X2_MAX = X2.min() - 0.5, X2.max() + 0.5
MAX_DEPTH = 6
FEAT_LABELS = ['Diameter', 'Colour score']
FEAT_SHORT  = ['D', 'C']


# ── Gini / split logic ────────────────────────────────────────────────────────

def gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / n
    return float(1.0 - np.sum(p ** 2))


def best_split(X, y):
    best_feat, best_thresh, best_red = None, None, -1.0
    n = len(y)
    gp = gini(y)
    for feat in range(X.shape[1]):
        vals = np.sort(np.unique(X[:, feat]))
        for thresh in (vals[:-1] + vals[1:]) / 2.0:
            l = y[X[:, feat] <  thresh]
            r = y[X[:, feat] >= thresh]
            if len(l) == 0 or len(r) == 0:
                continue
            red = gp - (len(l)/n)*gini(l) - (len(r)/n)*gini(r)
            if red > best_red:
                best_red, best_feat, best_thresh = red, feat, thresh
    return best_feat, best_thresh, best_red


class Node:
    _id = 0

    def __init__(self, X, y, depth, x1_lo, x1_hi, x2_lo, x2_hi):
        self.X, self.y   = X, y
        self.depth       = depth
        self.x1_lo, self.x1_hi = x1_lo, x1_hi
        self.x2_lo, self.x2_hi = x2_lo, x2_hi
        counts           = np.bincount(y, minlength=2)
        self.pred        = int(np.argmax(counts))
        self.na          = int(counts[0])
        self.no          = int(counts[1])
        self.g           = gini(y)
        self.feat = self.thresh = self.left = self.right = None
        self.is_leaf     = True
        self.id          = Node._id
        Node._id        += 1

    def pure(self):
        return self.g == 0.0

    def do_split(self):
        if self.pure() or len(self.y) < 2 or self.depth >= MAX_DEPTH:
            return None, None
        feat, thresh, red = best_split(self.X, self.y)
        if feat is None or red <= 1e-9:
            return None, None
        self.feat, self.thresh, self.is_leaf = feat, thresh, False
        ml = self.X[:, feat] <  thresh
        mr = ~ml
        if feat == 0:
            l = Node(self.X[ml], self.y[ml], self.depth+1,
                     self.x1_lo, thresh, self.x2_lo, self.x2_hi)
            r = Node(self.X[mr], self.y[mr], self.depth+1,
                     thresh, self.x1_hi, self.x2_lo, self.x2_hi)
        else:
            l = Node(self.X[ml], self.y[ml], self.depth+1,
                     self.x1_lo, self.x1_hi, self.x2_lo, thresh)
            r = Node(self.X[mr], self.y[mr], self.depth+1,
                     self.x1_lo, self.x1_hi, thresh, self.x2_hi)
        self.left, self.right = l, r
        return l, r


def leaves(root):
    if root is None:
        return []
    if root.is_leaf:
        return [root]
    return leaves(root.left) + leaves(root.right)


def next_split_node(root):
    """Return the impure leaf with the best Gini reduction, depth < MAX_DEPTH."""
    candidates = [n for n in leaves(root)
                  if not n.pure() and len(n.y) > 1 and n.depth < MAX_DEPTH]
    if not candidates:
        return None
    best, best_red = None, -1.0
    for nd in candidates:
        _, _, red = best_split(nd.X, nd.y)
        if red > best_red:
            best_red, best = red, nd
    return best


# ── Scatter helpers ───────────────────────────────────────────────────────────

def redraw_scatter(ax, root, n_splits):
    ax.cla()
    for leaf in leaves(root):
        col = COL_APPLE if leaf.pred == 0 else COL_ORANGE
        ax.fill_between([leaf.x1_lo, leaf.x1_hi],
                        leaf.x2_lo, leaf.x2_hi,
                        alpha=0.13, color=col, zorder=1)
    _draw_splits(ax, root)
    ax.scatter(X1[y_all==0], X2[y_all==0], color=COL_APPLE,
               s=45, edgecolors='k', lw=0.4, alpha=0.85, zorder=5, label='Apple')
    ax.scatter(X1[y_all==1], X2[y_all==1], color=COL_ORANGE,
               s=45, edgecolors='k', lw=0.4, alpha=0.85, zorder=5, label='Orange')
    ax.set_xlim(X1_MIN, X1_MAX)
    ax.set_ylim(X2_MIN, X2_MAX)
    ax.set_xlabel('Diameter (cm)', fontsize=11)
    ax.set_ylabel('Colour score',  fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.15)
    ax.set_title(
        f'Decision boundaries — {n_splits} split{"s" if n_splits!=1 else ""}  '
        f'(max depth {MAX_DEPTH})',
        fontsize=10)


def _draw_splits(ax, node):
    if node is None or node.is_leaf:
        return
    if node.feat == 0:
        ax.plot([node.thresh, node.thresh], [node.x2_lo, node.x2_hi],
                color='#222', lw=1.8, ls='--', zorder=4)
    else:
        ax.plot([node.x1_lo, node.x1_hi], [node.thresh, node.thresh],
                color='#222', lw=1.8, ls='--', zorder=4)
    _draw_splits(ax, node.left)
    _draw_splits(ax, node.right)


# ── SVG tree builder ──────────────────────────────────────────────────────────
# Uses the Reingold-Tilford-inspired algorithm: assign x positions bottom-up
# so sibling subtrees never overlap, then centre parents over their children.

NODE_W  = 110   # SVG units
NODE_H  = 44
H_GAP   = 18    # horizontal gap between sibling nodes
V_GAP   = 52    # vertical gap between levels
PAD_X   = 20
PAD_Y   = 20


def _layout(node):
    """
    Compute x positions for each node via a simple bottom-up contour approach.
    Returns a dict {node_id: (x_centre, y_top)} in SVG coords.
    Also returns the total (width, height) needed.
    """
    pos = {}

    def _place(nd, depth, x_offset):
        """
        Place nd's subtree so its leftmost extent starts at x_offset.
        Returns the x_centre of nd and the width consumed by its subtree.
        """
        y = PAD_Y + depth * (NODE_H + V_GAP)
        if nd.is_leaf:
            xc = x_offset + NODE_W / 2
            pos[nd.id] = (xc, y)
            return xc, NODE_W
        # Place left child
        lc, lw = _place(nd.left,  depth+1, x_offset)
        # Place right child with a gap after the left subtree
        rc, rw = _place(nd.right, depth+1, x_offset + lw + H_GAP)
        # Centre parent over children
        xc = (lc + rc) / 2
        pos[nd.id] = (xc, y)
        return xc, lw + H_GAP + rw

    _, total_w = _place(node, 0, PAD_X)
    max_depth  = _max_depth(node)
    total_h    = PAD_Y + (max_depth + 1) * (NODE_H + V_GAP) + PAD_Y
    return pos, total_w + 2 * PAD_X, total_h


def _max_depth(node):
    if node is None or node.is_leaf:
        return 0
    return 1 + max(_max_depth(node.left), _max_depth(node.right))


def _svg_node(nd, pos):
    x, y = pos[nd.id]
    x0   = x - NODE_W / 2
    col  = '#dce8f5' if not nd.is_leaf else ('#aaccee' if nd.pred == 0 else '#f4b8a8')
    border = '#4a6fa5'
    lines  = []

    # Box
    lines.append(
        f'<rect x="{x0:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" '
        f'rx="6" fill="{col}" stroke="{border}" stroke-width="1.8"/>'
    )

    # Text — two lines
    if nd.is_leaf:
        pred_lbl = 'Apple' if nd.pred == 0 else 'Orange'
        lines.append(
            f'<text x="{x:.1f}" y="{y+16:.1f}" text-anchor="middle" '
            f'font-size="12" font-weight="bold" fill="#1a1a2e">{pred_lbl}</text>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y+32:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#444">A:{nd.na} O:{nd.no}  G={nd.g:.3f}</text>'
        )
    else:
        fname = FEAT_SHORT[nd.feat]
        lines.append(
            f'<text x="{x:.1f}" y="{y+16:.1f}" text-anchor="middle" '
            f'font-size="12" font-weight="bold" fill="#1a1a2e">'
            f'{fname} &gt;= {nd.thresh:.2f}?</text>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y+32:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#444">A:{nd.na} O:{nd.no}  G={nd.g:.3f}</text>'
        )
    return '\n'.join(lines)


def _svg_edge(parent, child, pos, label):
    px, py = pos[parent.id]
    cx, cy = pos[child.id]
    # Arrow from bottom of parent to top of child
    x1, y1 = px, py + NODE_H
    x2, y2 = cx, cy
    mx, my = (x1+x2)/2, (y1+y2)/2 - 4
    # White rectangle behind label
    lw = 30
    lines = [
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="#888" stroke-width="1.5" marker-end="url(#arr)"/>',
        f'<rect x="{mx-lw/2:.1f}" y="{my-9:.1f}" width="{lw}" height="14" '
        f'rx="3" fill="white"/>',
        f'<text x="{mx:.1f}" y="{my+2:.1f}" text-anchor="middle" '
        f'font-size="10" fill="#555" font-style="italic">{label}</text>',
    ]
    return '\n'.join(lines)


def _collect_svg(node, pos):
    parts = []
    if not node.is_leaf:
        parts.append(_svg_edge(node, node.left,  pos, 'No'))
        parts.append(_svg_edge(node, node.right, pos, 'Yes'))
        parts += _collect_svg(node.left,  pos)
        parts += _collect_svg(node.right, pos)
    parts.append(_svg_node(node, pos))
    return parts


def build_svg(root):
    if root is None:
        return '<p style="color:#aaa;font-style:italic">No splits yet.</p>'

    pos, w, h = _layout(root)
    parts = [
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{w:.0f}px;display:block;'
        f'background:white;border:1px solid #c8d8ea;border-radius:8px;'
        f'box-shadow:0 2px 6px rgba(0,0,0,0.07);">',
        # arrowhead marker
        '<defs><marker id="arr" markerWidth="8" markerHeight="8" '
        'refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L8,3 z" fill="#888"/></marker></defs>',
    ]
    parts += _collect_svg(root, pos)
    parts.append('</svg>')
    parts.append(
        '<p style="text-align:center;font-size:12px;color:#555;margin-top:4px;">'
        '<b>D</b> = Diameter (cm) &nbsp;|&nbsp; <b>C</b> = Colour score &nbsp;|&nbsp; '
        'G = Gini impurity</p>'
    )
    return '\n'.join(parts)


# ── show() ────────────────────────────────────────────────────────────────────

def show():
    """Render Figure 10: scatter (matplotlib) + tree (SVG)."""
    plt.close('Notebook6 Figure 10')
    fig, ax = plt.subplots(num='Notebook6 Figure 10', figsize=(10, 5))
    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible  = False
    fig.canvas.resizable       = True

    # Reset node id counter
    Node._id = 0

    state = {'root': None, 'n_splits': 0}

    def init():
        Node._id = 0
        root = Node(X_all, y_all, depth=0,
                    x1_lo=X1_MIN, x1_hi=X1_MAX,
                    x2_lo=X2_MIN, x2_hi=X2_MAX)
        state['root']     = root
        state['n_splits'] = 0
        return root

    root = init()
    redraw_scatter(ax, root, 0)
    plt.tight_layout()

    # SVG output widget — updates in place without redrawing the matplotlib fig
    svg_out    = widgets.HTML(value=build_svg(root))
    status_lbl = widgets.Label(
        value='Click "Add Next Split" to begin.',
        layout=widgets.Layout(width='520px'),
    )
    add_btn    = widgets.Button(description='Add Next Split',
                                button_style='success',
                                layout=widgets.Layout(width='160px'))
    reset_btn  = widgets.Button(description='Reset',
                                layout=widgets.Layout(width='90px'))

    def on_add(_):
        root = state['root']
        nd   = next_split_node(root)
        if nd is None:
            status_lbl.value = f'Tree fully grown (max depth {MAX_DEPTH}).'
            return
        l, r = nd.do_split()
        if l is None:
            status_lbl.value = 'No further useful splits available.'
            return
        state['n_splits'] += 1
        status_lbl.value = (
            f"Split {state['n_splits']}: "
            f"{FEAT_LABELS[nd.feat]} >= {nd.thresh:.2f}  |  "
            f"Left: A={l.na} O={l.no}  Right: A={r.na} O={r.no}"
        )
        redraw_scatter(ax, root, state['n_splits'])
        svg_out.value = build_svg(root)
        fig.canvas.draw_idle()

    def on_reset(_):
        root = init()
        status_lbl.value = 'Reset. Click "Add Next Split" to begin.'
        redraw_scatter(ax, root, 0)
        svg_out.value = build_svg(root)
        fig.canvas.draw_idle()

    add_btn.on_click(on_add)
    reset_btn.on_click(on_reset)

    display(
        widgets.HBox([add_btn, reset_btn, status_lbl]),
        svg_out,
    )