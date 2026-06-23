"""
Figure 6c — Gradient Descent on a 3D Loss Surface
===================================================

Shows a 3D loss surface L(θ₁, θ₂) rendered as a surface plot. The gradient
descent path is drawn on top of the surface as a red line with dots at each
step, so user can see the algorithm descending from a starting point
toward the minimum.

A second panel shows the same path from above as a contour plot, which is
easier to read for precise positions.

The user can rotate the 3D surface using the toolbar to view it from
different angles — reinforcing that the surface is a genuine 3D object
being navigated by the algorithm.

The key teaching point: even with two parameters (a 3D landscape), the
algorithm applies exactly the same rule as in 1D — compute the gradient
in each direction independently, step downhill. With more parameters the
landscape has more dimensions but the rule is identical.

Usage
-----
From a Jupyter notebook cell::

    %matplotlib widget
    from visualisations.Figure_6c import show
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
from ipywidgets import interactive_output, FloatSlider, IntSlider, VBox, HBox, HTML
from IPython.display import display


def show():
    """Render the interactive Figure 6c 3D loss surface explorer."""

    plt.close('Notebook4 Figure 6c')

    # ── Loss function and gradients ───────────────────────────────────────────
    def loss_fn(t1, t2):
        # A bowl with a slight tilt so the minimum is not at the origin
        return 0.5 * t1**2 + 1.2 * t2**2 + 0.4 * t1 * t2 + 0.2 * t1 - 0.3 * t2

    def grad_t1(t1, t2):
        return 1.0 * t1 + 0.4 * t2 + 0.2

    def grad_t2(t1, t2):
        return 2.4 * t2 + 0.4 * t1 - 0.3

    # True minimum (solve the linear system grad = 0)
    A_mat = np.array([[1.0, 0.4], [0.4, 2.4]])
    b_vec = np.array([-0.2, 0.3])
    t_min = np.linalg.solve(A_mat, b_vec)

    # Grid for surface and contour
    res      = 80
    t1_vals  = np.linspace(-3.5, 3.5, res)
    t2_vals  = np.linspace(-3.5, 3.5, res)
    T1, T2   = np.meshgrid(t1_vals, t2_vals)
    Z        = loss_fn(T1, T2)

    # ── Defaults ──────────────────────────────────────────────────────────────
    DEFAULT_T1    = -3.0
    DEFAULT_T2    =  3.0
    DEFAULT_LR    =  0.15
    DEFAULT_STEPS = 25

    # ── Build the figure — 3D surface on left, contour on right ───────────────
    fig = plt.figure(num='Notebook4 Figure 6c', figsize=(10, 5))
    ax3d     = fig.add_subplot(1, 2, 1, projection='3d')
    ax_ctr   = fig.add_subplot(1, 2, 2)

    fig.canvas.toolbar_visible = True
    fig.canvas.toolbar_position = 'right'
    fig.canvas.header_visible = False
    fig.canvas.resizable = True

    # ── Draw the static loss surface ──────────────────────────────────────────
    ax3d.plot_surface(T1, T2, Z, cmap='Blues_r', alpha=0.7,
                      rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax3d.set_xlabel('θ₁', fontsize=9, labelpad=6)
    ax3d.set_ylabel('θ₂', fontsize=9, labelpad=6)
    ax3d.set_zlabel('Loss L(θ₁,θ₂)', fontsize=9, labelpad=6)
    ax3d.set_title('3D loss surface\n(rotate with toolbar)', fontsize=9)
    ax3d.tick_params(labelsize=7)

    # Mark the true minimum on the 3D surface
    ax3d.scatter(*t_min, loss_fn(*t_min), color='yellow', s=120,
                 marker='*', zorder=6, depthshade=False)

    # Mutable: 3D descent path — stored as a single line updated each call
    path3d_line, = ax3d.plot([], [], [], 'o-', color='tomato',
                              markersize=4, linewidth=1.5, zorder=5)

    # ── Draw the static contour ───────────────────────────────────────────────
    ax_ctr.contourf(T1, T2, Z, levels=40, cmap='Blues_r', alpha=0.85)
    ax_ctr.contour(T1, T2, Z, levels=15, colors='white', alpha=0.25, linewidths=0.5)
    ax_ctr.scatter(*t_min, color='yellow', s=160, marker='*', zorder=6,
                   edgecolors='k', lw=0.8, label='True minimum')

    path_ctr, = ax_ctr.plot([], [], 'o-', color='tomato', markersize=5,
                             linewidth=1.5, zorder=4, label='Descent path')
    start_ctr  = ax_ctr.scatter([], [], color='limegreen', s=120, zorder=7,
                                edgecolors='k', lw=1.0, label='Start')
    end_ctr    = ax_ctr.scatter([], [], color='black', s=120, marker='D',
                                zorder=7, edgecolors='white', lw=1.0,
                                label='Final position')

    ax_ctr.set_xlabel('θ₁  (parameter 1)', fontsize=10)
    ax_ctr.set_ylabel('θ₂  (parameter 2)', fontsize=10)
    ax_ctr.set_title('Top-down view (contour map)\ndarker = lower loss', fontsize=9)
    ax_ctr.set_xlim(-3.6, 3.6)
    ax_ctr.set_ylim(-3.6, 3.6)
    ax_ctr.legend(fontsize=8, loc='upper right')

    plt.tight_layout()

    # ── Readout widget ────────────────────────────────────────────────────────
    readout = widgets.Output()

    # ── Update function ───────────────────────────────────────────────────────
    def update(t1_start, t2_start, lr, n_steps):
        t1, t2 = t1_start, t2_start
        path_t1, path_t2, path_L = [t1], [t2], [loss_fn(t1, t2)]

        for _ in range(n_steps):
            g1 = grad_t1(t1, t2)
            g2 = grad_t2(t1, t2)
            t1 = t1 - lr * g1
            t2 = t2 - lr * g2
            path_t1.append(t1)
            path_t2.append(t2)
            path_L.append(loss_fn(t1, t2))

        path_t1 = np.array(path_t1)
        path_t2 = np.array(path_t2)
        path_L  = np.array(path_L)

        # Update 3D path
        path3d_line.set_data_3d(path_t1, path_t2, path_L)

        # Update contour path
        path_ctr.set_data(path_t1, path_t2)
        start_ctr.set_offsets([[path_t1[0],  path_t2[0]]])
        end_ctr.set_offsets([[path_t1[-1], path_t2[-1]]])

        ax3d.set_title(
            f'3D loss surface  |  start: ({t1_start:.1f}, {t2_start:.1f})\n'
            f'Rotate with toolbar to see the descent path from different angles',
            fontsize=8,
        )
        ax_ctr.set_title(
            f'Top-down view  |  lr = {lr}  |  {n_steps} steps\n'
            f'Final loss: {path_L[-1]:.4f}  (true min: {loss_fn(*t_min):.4f})',
            fontsize=9,
        )

        # Print trace
        readout.clear_output(wait=True)
        with readout:
            print(f"{'Step':<6} {'θ₁':>10} {'θ₂':>10} {'Loss':>10} "
                  f"{'∂L/∂θ₁':>10} {'∂L/∂θ₂':>10}")
            print('-' * 62)
            t1_p, t2_p = t1_start, t2_start
            for i in range(min(n_steps + 1, 12)):
                g1_p = grad_t1(t1_p, t2_p)
                g2_p = grad_t2(t1_p, t2_p)
                print(f'{i:<6} {t1_p:>10.4f} {t2_p:>10.4f} '
                      f'{loss_fn(t1_p,t2_p):>10.4f} {g1_p:>10.4f} {g2_p:>10.4f}')
                t1_p -= lr * g1_p
                t2_p -= lr * g2_p
            if n_steps >= 12:
                print(f'  ... ({n_steps - 11} more steps not shown)')

        fig.canvas.draw_idle()

    # ── Sliders ───────────────────────────────────────────────────────────────
    t1_s = FloatSlider(value=DEFAULT_T1, min=-3.5, max=3.5, step=0.1,
                        description='Start θ₁',
                        style={'description_width': '80px'},
                        layout=widgets.Layout(width='320px'),
                        continuous_update=True)
    t2_s = FloatSlider(value=DEFAULT_T2, min=-3.5, max=3.5, step=0.1,
                        description='Start θ₂',
                        style={'description_width': '80px'},
                        layout=widgets.Layout(width='320px'),
                        continuous_update=True)
    lr_s = FloatSlider(value=DEFAULT_LR, min=0.01, max=0.6, step=0.01,
                        description='Learning rate',
                        style={'description_width': '100px'},
                        layout=widgets.Layout(width='320px'),
                        continuous_update=True)
    steps_s = IntSlider(value=DEFAULT_STEPS, min=1, max=60, step=1,
                         description='Steps',
                         style={'description_width': '80px'},
                         layout=widgets.Layout(width='320px'),
                         continuous_update=True)

    t1_b    = widgets.BoundedFloatText(value=DEFAULT_T1, min=-3.5, max=3.5, step=0.1,
                                        description='', layout=widgets.Layout(width='80px'))
    t2_b    = widgets.BoundedFloatText(value=DEFAULT_T2, min=-3.5, max=3.5, step=0.1,
                                        description='', layout=widgets.Layout(width='80px'))
    lr_b    = widgets.BoundedFloatText(value=DEFAULT_LR, min=0.01, max=0.6, step=0.01,
                                        description='', layout=widgets.Layout(width='80px'))
    steps_b = widgets.BoundedIntText(value=DEFAULT_STEPS, min=1, max=60, step=1,
                                      description='', layout=widgets.Layout(width='80px'))

    for s, b in [(t1_s, t1_b), (t2_s, t2_b), (lr_s, lr_b), (steps_s, steps_b)]:
        widgets.jslink((s, 'value'), (b, 'value'))

    reset_btn = widgets.Button(description='Reset', button_style='warning',
                               layout=widgets.Layout(width='100px'))

    def on_reset(b):
        t1_s.value = DEFAULT_T1;  t2_s.value = DEFAULT_T2
        lr_s.value = DEFAULT_LR;  steps_s.value = DEFAULT_STEPS

    reset_btn.on_click(on_reset)

    sep = HTML('<hr style="margin:4px 0; border-color:#ccc">')

    controls = VBox([
        HTML('<b>Gradient descent — 2 parameters, 3D loss surface</b>'),
        HBox([t1_s, t1_b]),
        HBox([t2_s, t2_b]),
        HBox([lr_s, lr_b]),
        HBox([steps_s, steps_b]),
        sep,
        reset_btn,
    ])

    out = interactive_output(update, {
        't1_start': t1_s, 't2_start': t2_s,
        'lr': lr_s, 'n_steps': steps_s,
    })

    display(controls, out)
    display(readout)