"""
Figure 5 — Decision Tree Structure Diagram
==========================================
A static diagram showing the anatomy of a decision tree rendered as an
inline SVG via IPython display. This avoids matplotlib coordinate issues
and gives precise control over layout and sizing.

Usage
-----
In a Jupyter notebook cell:

    from visualisations.Figure_5 import show
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

from IPython.display import display, HTML

SVG = """
<div style="font-family: Georgia, serif; padding: 16px;
            background: white; border: 1.5px solid #c8d8ea;
            border-radius: 10px; max-width: 820px; margin: auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
<svg viewBox="0 0 780 360" xmlns="http://www.w3.org/2000/svg"
     style="width:100%; display:block; margin:auto;">

  <!-- ── depth labels ── -->
  <text x="6" y="72"  font-size="13" fill="#aaa" font-style="italic">Depth 0</text>
  <text x="6" y="182" font-size="13" fill="#aaa" font-style="italic">Depth 1</text>
  <text x="6" y="292" font-size="13" fill="#aaa" font-style="italic">Depth 2</text>

  <!-- ── connecting lines (drawn first so nodes sit on top) ── -->
  <line x1="310" y1="90" x2="190" y2="162" stroke="#666" stroke-width="1.8"
        marker-end="url(#arr)"/>
  <line x1="390" y1="90" x2="530" y2="162" stroke="#666" stroke-width="1.8"
        marker-end="url(#arr)"/>
  <line x1="500" y1="202" x2="430" y2="272" stroke="#666" stroke-width="1.8"
        marker-end="url(#arr)"/>
  <line x1="570" y1="202" x2="640" y2="272" stroke="#666" stroke-width="1.8"
        marker-end="url(#arr)"/>

  <!-- ── arrowhead marker ── -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8"
            refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#666"/>
    </marker>
  </defs>

  <!-- ── ROOT NODE ── -->
  <rect x="270" y="44" width="160" height="52" rx="8"
        fill="#dce8f5" stroke="#4a6fa5" stroke-width="2"/>
  <text x="350" y="65" text-anchor="middle" font-size="13.5"
        font-weight="bold" fill="#1a1a2e">Diameter &gt;= 7.5 cm?</text>
  <text x="350" y="84" text-anchor="middle" font-size="11.5"
        fill="#444">Feature: Diameter (cm)</text>

  <!-- ── LEFT LEAF (depth 1) ── -->
  <rect x="110" y="162" width="160" height="46" rx="8"
        fill="#cce5ff" stroke="#4a6fa5" stroke-width="2"/>
  <text x="190" y="182" text-anchor="middle" font-size="13.5"
        font-weight="bold" fill="#1a1a2e">Predict: Apple</text>
  <text x="190" y="200" text-anchor="middle" font-size="11"
        fill="#444">diameter &lt; 7.5 cm</text>

  <!-- ── INTERNAL NODE (depth 1) ── -->
  <rect x="450" y="162" width="170" height="46" rx="8"
        fill="#dce8f5" stroke="#4a6fa5" stroke-width="2"/>
  <text x="535" y="182" text-anchor="middle" font-size="13.5"
        font-weight="bold" fill="#1a1a2e">Colour score &gt;= 6?</text>
  <text x="535" y="200" text-anchor="middle" font-size="11"
        fill="#444">Feature: Colour score</text>

  <!-- ── LEAF: large + low colour → Apple (depth 2) ── -->
  <rect x="350" y="272" width="160" height="46" rx="8"
        fill="#cce5ff" stroke="#4a6fa5" stroke-width="2"/>
  <text x="430" y="292" text-anchor="middle" font-size="13.5"
        font-weight="bold" fill="#1a1a2e">Predict: Apple</text>
  <text x="430" y="310" text-anchor="middle" font-size="11"
        fill="#444">large, low colour score</text>

  <!-- ── LEAF: large + high colour → Orange (depth 2) ── -->
  <rect x="560" y="272" width="165" height="46" rx="8"
        fill="#ffe0cc" stroke="#4a6fa5" stroke-width="2"/>
  <text x="642" y="292" text-anchor="middle" font-size="13.5"
        font-weight="bold" fill="#1a1a2e">Predict: Orange</text>
  <text x="642" y="310" text-anchor="middle" font-size="11"
        fill="#444">large, high colour score</text>

  <!-- ── branch labels — white rect behind each label punches through the line ── -->
  <rect x="166" y="122" width="94" height="18" rx="3" fill="white"/>
  <text x="213" y="136" text-anchor="middle" font-size="11.5"
        fill="#555" font-style="italic">No (&lt; 7.5 cm)</text>

  <rect x="414" y="122" width="116" height="18" rx="3" fill="white"/>
  <text x="472" y="136" text-anchor="middle" font-size="11.5"
        fill="#555" font-style="italic">Yes (&gt;= 7.5 cm)</text>

  <rect x="406" y="234" width="88" height="18" rx="3" fill="white"/>
  <text x="450" y="248" text-anchor="middle" font-size="11.5"
        fill="#555" font-style="italic">No (&lt; 6.0)</text>

  <rect x="576" y="234" width="96" height="18" rx="3" fill="white"/>
  <text x="624" y="248" text-anchor="middle" font-size="11.5"
        fill="#555" font-style="italic">Yes (&gt;= 6.0)</text>

  <!-- ── structural annotation labels — positioned inside nodes, not on lines ── -->
  <!-- Root: sits just above the root box -->
  <text x="350" y="34" text-anchor="middle" font-size="11.5"
        fill="#4a6fa5" font-style="italic">Root node — first split</text>

  <!-- Left leaf depth-1: sits below the box -->
  <text x="190" y="222" text-anchor="middle" font-size="11.5"
        fill="#4a6fa5" font-style="italic">Leaf node</text>

  <!-- Internal node label: sits to the right of the box, clear of connecting lines -->
  <text x="760" y="186" text-anchor="end" font-size="11.5"
        fill="#4a6fa5" font-style="italic">Internal node —</text>
  <text x="760" y="200" text-anchor="end" font-size="11.5"
        fill="#4a6fa5" font-style="italic">second split,</text>
  <text x="760" y="214" text-anchor="end" font-size="11.5"
        fill="#4a6fa5" font-style="italic">different feature</text>

  <!-- Depth-2 leaf labels: sit below their boxes -->
  <text x="430" y="332" text-anchor="middle" font-size="11.5"
        fill="#4a6fa5" font-style="italic">Leaf node</text>

  <text x="642" y="332" text-anchor="middle" font-size="11.5"
        fill="#4a6fa5" font-style="italic">Leaf node</text>

</svg>
<p style="text-align:center; font-size:13px; color:#555; margin-top:4px;">
  <b>Figure 5:</b> Anatomy of a Decision Tree —
  each internal node asks one question on one feature;
  leaf nodes give the final class prediction.
</p>
</div>
"""


def show():
    """
    Render Figure 5: Decision Tree Structure Diagram as inline SVG.

    No %matplotlib widget magic needed for this figure.

    Usage
    -----
        from visualisations.Lab_6.Figure_5 import show
        show()
    """
    display(HTML(SVG))