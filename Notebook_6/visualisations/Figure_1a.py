"""
Figure 1a — Interactive Bayes' Theorem Explorer
================================================
An interactive HTML widget injected directly into the notebook output cell.
No external dependencies — no MathJax, no CDN, no internet connection required.
Equations are rendered using Unicode and styled HTML fractions.

Users set three probabilities via sliders:
    P(B | A) — likelihood
    P(A)     — prior
    P(B | ¬A) — false positive rate (used to auto-compute P(B))

The widget shows both the general form P(A|B) and the classification form
P(Y|X), with all values substituted into the formula live.

Usage
-----
In a Jupyter notebook cell:

    from visualisations.Figure_1a import show
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

# HTML_SOURCE is a fully self-contained widget — no CDN, no external fonts,
# no MathJax. All equation rendering is done with HTML/CSS fraction layout
# and Unicode symbols so it works offline on any machine.
HTML_SOURCE = """
<div id="bayes-root">
<style>
  #bayes-root {
    font-family: Georgia, 'Times New Roman', serif;
    background: #0f1117;
    color: #e2e8f0;
    padding: 20px;
    border-radius: 10px;
    font-size: 14px;
    line-height: 1.6;
  }

  #bayes-root * { box-sizing: border-box; }

  #bayes-root h2 {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4f9cf9;
    margin-bottom: 16px;
  }

  .b-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .b-panel {
    background: #1a1d27;
    border: 1px solid #2e3348;
    border-radius: 10px;
    padding: 16px;
  }

  .b-panel-title {
    font-family: 'Courier New', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7a849e;
    margin-bottom: 14px;
  }

  /* ── Sliders ── */
  .b-slider-group { margin-bottom: 14px; }

  .b-slider-label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 5px;
  }

  .b-slider-name {
    font-family: 'Courier New', monospace;
    font-size: 13px;
  }

  .b-slider-term {
    font-size: 11px;
    color: #7a849e;
    margin-left: 6px;
    font-family: Georgia, serif;
  }

  .b-slider-val {
    font-family: 'Courier New', monospace;
    font-size: 14px;
    font-weight: 700;
    min-width: 52px;
    text-align: right;
  }

  input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    height: 5px;
    border-radius: 3px;
    outline: none;
    cursor: pointer;
  }
  input[type=range].b-blue   { background: linear-gradient(to right, #4f9cf9 var(--p,50%), #2e3348 var(--p,50%)); }
  input[type=range].b-orange { background: linear-gradient(to right, #f97b4f var(--p,50%), #2e3348 var(--p,50%)); }
  input[type=range].b-green  { background: linear-gradient(to right, #6ee7b7 var(--p,50%), #2e3348 var(--p,50%)); }

  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #e2e8f0;
    border: 2px solid #0f1117;
  }

  /* ── HTML fraction rendering ── */
  .b-frac {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: middle;
    margin: 0 3px;
    font-size: 0.95em;
  }
  .b-frac .b-num {
    border-bottom: 1.5px solid #e2e8f0;
    padding: 0 4px 2px 4px;
    text-align: center;
    white-space: nowrap;
  }
  .b-frac .b-den {
    padding: 2px 4px 0 4px;
    text-align: center;
    white-space: nowrap;
  }

  .b-eq {
    font-size: 15px;
    line-height: 2.4;
    padding: 10px 12px;
    background: #141720;
    border: 1px solid #2e3348;
    border-radius: 8px;
    margin-bottom: 10px;
    min-height: 58px;
  }

  .b-eq-label {
    font-family: 'Courier New', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a849e;
    margin-bottom: 5px;
  }

  /* ── Result ── */
  .b-result-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
  }

  .b-result-label {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #7a849e;
    white-space: nowrap;
  }

  .b-result-val {
    font-family: 'Courier New', monospace;
    font-size: 24px;
    font-weight: 700;
    color: #6ee7b7;
    min-width: 80px;
  }

  .b-bar-wrap {
    flex: 1;
    height: 8px;
    background: #2e3348;
    border-radius: 4px;
    overflow: hidden;
  }

  .b-bar {
    height: 100%;
    background: #6ee7b7;
    border-radius: 4px;
    transition: width 0.12s ease;
  }

  .b-warn {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #f97b4f;
    min-height: 16px;
    margin-top: 6px;
  }

  /* ── Mapping note ── */
  .b-mapping {
    font-size: 12px;
    color: #7a849e;
    margin-top: 12px;
    line-height: 1.9;
  }
  .b-mapping b { color: #e2e8f0; font-family: 'Courier New', monospace; }

  /* ── Reset button ── */
  .b-reset {
    margin-top: 14px;
    background: transparent;
    border: 1px solid #2e3348;
    color: #7a849e;
    font-family: Georgia, serif;
    font-size: 12px;
    padding: 5px 14px;
    border-radius: 6px;
    cursor: pointer;
  }
  .b-reset:hover { border-color: #4f9cf9; color: #4f9cf9; }

  /* italic helpers */
  i { font-style: italic; }
</style>

<h2>Bayes&#x2019; Theorem &#x2014; Interactive Explorer</h2>

<div class="b-layout">

  <!-- LEFT: controls -->
  <div class="b-panel">
    <div class="b-panel-title">Set the probabilities</div>

    <div class="b-slider-group">
      <div class="b-slider-label">
        <span class="b-slider-name">P(B &#x7C; A)<span class="b-slider-term">&#x2014; likelihood</span></span>
        <span class="b-slider-val" id="b-pba-val" style="color:#4f9cf9">0.890</span>
      </div>
      <input type="range" class="b-blue" id="b-pba" min="0.01" max="0.99" step="0.01" value="0.89"
             oninput="bUpdate()">
    </div>

    <div class="b-slider-group">
      <div class="b-slider-label">
        <span class="b-slider-name">P(A)<span class="b-slider-term">&#x2014; prior</span></span>
        <span class="b-slider-val" id="b-pa-val" style="color:#f97b4f">0.003</span>
      </div>
      <input type="range" class="b-orange" id="b-pa" min="0.001" max="0.999" step="0.001" value="0.003"
             oninput="bUpdate()">
    </div>

    <div class="b-slider-group">
      <div class="b-slider-label">
        <span class="b-slider-name">P(B &#x7C; &#xAC;A)<span class="b-slider-term">&#x2014; false positive rate</span></span>
        <span class="b-slider-val" id="b-pbac-val" style="color:#6ee7b7">0.070</span>
      </div>
      <input type="range" class="b-green" id="b-pbac" min="0.001" max="0.999" step="0.001" value="0.07"
             oninput="bUpdate()">
    </div>

    <div class="b-warn" id="b-warn"></div>
    <button class="b-reset" onclick="bReset()">Reset to defaults</button>

    <div class="b-mapping">
      <b>P(B)</b> is auto-computed via the law of total probability:<br>
      P(B) = P(B&#x7C;A)&#x22C5;P(A) + P(B&#x7C;&#xAC;A)&#x22C5;P(&#xAC;A)<br><br>
      In classification notation:<br>
      <b>A = Y</b> (class label) &nbsp;|&nbsp; <b>B = X</b> (observed features)<br>
      <b>P(A&#x7C;B)</b> becomes <b>P(Y&#x7C;X)</b> &#x2014; the posterior
    </div>
  </div>

  <!-- RIGHT: formulas -->
  <div class="b-panel">
    <div class="b-panel-title">Live calculation</div>

    <div class="b-eq-label">General form</div>
    <div class="b-eq">
      P(A &#x7C; B) =
      <span class="b-frac">
        <span class="b-num">P(B &#x7C; A) &sdot; P(A)</span>
        <span class="b-den">P(B)</span>
      </span>
    </div>

    <div class="b-eq-label">Numbers substituted in</div>
    <div class="b-eq" id="b-numeric"></div>

    <div class="b-eq-label">Classification form &nbsp;(A = Y, B = X)</div>
    <div class="b-eq" id="b-classform"></div>

    <div class="b-result-row">
      <span class="b-result-label">P(A &#x7C; B) =</span>
      <span class="b-result-val" id="b-result">&#x2014;</span>
      <div class="b-bar-wrap"><div class="b-bar" id="b-bar" style="width:0%"></div></div>
    </div>
    <div class="b-warn" id="b-result-warn"></div>
  </div>

</div>

<script>
(function() {
  function frac(num, den) {
    return '<span class="b-frac"><span class="b-num">' + num +
           '</span><span class="b-den">' + den + '</span></span>';
  }

  function track(el) {
    var pct = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty('--p', pct + '%');
  }

  document.querySelectorAll('#bayes-root input[type=range]').forEach(function(s) {
    track(s);
    s.addEventListener('input', function() { track(s); });
  });

  function fmt(v, dp) { return parseFloat(v).toFixed(dp === undefined ? 3 : dp); }

  window.bUpdate = function() {
    var pba  = parseFloat(document.getElementById('b-pba').value);
    var pa   = parseFloat(document.getElementById('b-pa').value);
    var pbac = parseFloat(document.getElementById('b-pbac').value);

    document.getElementById('b-pba-val').textContent  = fmt(pba);
    document.getElementById('b-pa-val').textContent   = fmt(pa, 3);
    document.getElementById('b-pbac-val').textContent = fmt(pbac);

    // Auto-compute P(B) via law of total probability
    var pb = pba * pa + pbac * (1 - pa);

    var numerator = pba * pa;
    var posterior = numerator / pb;

    // Numeric substitution formula
    document.getElementById('b-numeric').innerHTML =
      'P(A | B) = ' +
      frac(
        fmt(pba) + ' &sdot; ' + fmt(pa,3),
        fmt(pb, 5)
      ) +
      ' = ' +
      frac(fmt(numerator, 5), fmt(pb, 5)) +
      ' &asymp; <b style="color:#6ee7b7">' + fmt(posterior, 4) + '</b>' +
      '<br><small style="color:#7a849e">P(B) = ' + fmt(pba) + '&times;' + fmt(pa,3) +
      ' + ' + fmt(pbac) + '&times;' + fmt(1-pa,3) + ' = ' + fmt(pb,5) + '</small>';

    // Classification form
    document.getElementById('b-classform').innerHTML =
      'P(Y | <b>X</b>) = ' +
      frac(
        'P(<b>X</b> | Y) &sdot; P(Y)',
        'P(<b>X</b>)'
      ) +
      ' &asymp; <b style="color:#6ee7b7">' + fmt(posterior, 4) + '</b>';

    // Result bar
    document.getElementById('b-result').textContent = fmt(posterior, 4);
    document.getElementById('b-bar').style.width = Math.min(posterior, 1) * 100 + '%';

    var rwarn = document.getElementById('b-result-warn');
    var warn  = document.getElementById('b-warn');
    warn.textContent  = '';
    rwarn.textContent = posterior > 1
      ? '\u26a0 Result exceeds 1 \u2014 P(B) may be set too low'
      : '';
  };

  window.bReset = function() {
    document.getElementById('b-pba').value  = 0.89;
    document.getElementById('b-pa').value   = 0.003;
    document.getElementById('b-pbac').value = 0.07;
    document.querySelectorAll('#bayes-root input[type=range]').forEach(track);
    bUpdate();
  };

  bUpdate();
})();
</script>
</div>
"""


def show():
    """
    Render Figure 1a: Interactive Bayes' Theorem Explorer.

    Injects a fully self-contained HTML widget into the notebook output cell.
    No internet connection, CDN, or external dependencies required.

    Usage
    -----
        from visualisations.Lab_6.Figure_1a import show
        show()
    """
    # display(HTML(...)) renders the string directly in the notebook cell.
    # The widget is scoped inside #bayes-root so its styles do not leak
    # into the rest of the notebook.
    from IPython.display import display, HTML
    display(HTML(HTML_SOURCE))