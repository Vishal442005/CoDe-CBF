import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, Polygon, Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11.5,
    'axes.labelsize': 12.5,
    'axes.titlesize': 13.5,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 9.2,
    'figure.titlesize': 15.5,
    'mathtext.fontset': 'dejavusans'
})

fig = plt.figure(figsize=(19.0, 13.2), dpi=300)
gs = gridspec.GridSpec(3, 2, height_ratios=[0.42, 1.0, 1.0], hspace=0.34, wspace=0.22)

ax_flow = fig.add_subplot(gs[0, :])
ax_a = fig.add_subplot(gs[1, 0])
ax_b = fig.add_subplot(gs[1, 1])
ax_c = fig.add_subplot(gs[2, 0])
ax_d = fig.add_subplot(gs[2, 1])

# =============================================================================
# TOP ARCHITECTURE & CLOSED-LOOP FLOWCHART STRIP
# =============================================================================
ax_flow.set_xlim(0, 100)
ax_flow.set_ylim(0, 26)
ax_flow.axis('off')
ax_flow.set_title("CoDe-CBF End-to-End Closed-Loop Safety Architecture & Signal Flowchart",
                  fontsize=15.0, fontweight='bold', pad=6, color='#1a252f')

blocks = [
    (1.5,  7.2, 18.8, 14.5, '#eafaf1', '#1e8449', '#145a32',
     "(a) STAGE 1: PHYSICAL PLANT",
     "• NREL 3D Sonic Wind $\\mathbf{w}_i(t)$\n• Rotor Wake Tensor $\\mathbf{E}_j(\\mathbf{v}_j)$\n• Ego State $(\\mathbf{p}_i, \\mathbf{v}_i)$"),
    (27.5, 7.2, 18.8, 14.5, '#fef9e7', '#d68910', '#7e5109',
     "(b) STAGE 2: CYBER ROS 2 DDS",
     "• Variable AoI Delay $\\tau_{ij}\\Delta t$\n• Constant-Vel Extrap. $\\hat{\\mathbf{p}}_j(t)$\n• Residual Score $s_t(\\tau_{ij}, r_t)$"),
    (53.5, 7.2, 18.8, 14.5, '#ebf5fb', '#2471a3', '#154360',
     "(c) STAGE 3: ACI SAFETY ENGINE",
     "• Online Quantile $\\alpha_{t+1}$ Update\n• Dual Margin $\\hat{q}_t = q_{\\mathrm{phys}} + q_{\\mathrm{net}}$\n• A-DABS $\\eta_{ij}$ & V-CAPS $\\alpha(v_{\\mathrm{rel}})$"),
    (79.5, 7.2, 19.0, 14.5, '#fdedec', '#c0392b', '#78281f',
     "(d) STAGE 4: OSQP FILTER (50 Hz)",
     "• 4-Var Convex QP (< 0.65 ms)\n• Tangential Deadlock $\\mathbf{u}_{\\mathrm{rot},i}$\n• Penalized Slack $\\rho_{\\mathrm{slack}}\\delta_i^2$")
]

for x0, y0, w, h, fc, ec, tc, title, body in blocks:
    box = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.0,rounding_size=1.0",
                         facecolor=fc, edgecolor=ec, lw=2.2, zorder=3)
    ax_flow.add_patch(box)
    # Header bar cleanly inside top of box
    hdr_h = 4.2
    hdr = FancyBboxPatch((x0, y0 + h - hdr_h), w, hdr_h, boxstyle="round,pad=0.0,rounding_size=0.8",
                         facecolor=ec, edgecolor=ec, lw=1.0, zorder=4)
    ax_flow.add_patch(hdr)
    ax_flow.text(x0 + w/2, y0 + h - hdr_h/2, title, ha='center', va='center',
                 color='white', fontsize=10.2, fontweight='bold', zorder=5)
    ax_flow.text(x0 + 0.8, y0 + (h - hdr_h)/2, body, ha='left', va='center',
                 color=tc, fontsize=9.6, linespacing=1.45, zorder=5)

# Forward pipeline arrows between blocks (clean 7.2-unit gaps)
arrow_specs = [
    (20.6, 13.5, 27.2, 13.5, r"$\mathbf{p}_i,\mathbf{v}_i$" "\n" r"$\mathbf{w}_i(t)$"),
    (46.6, 13.5, 53.2, 13.5, r"$\hat{\mathbf{p}}_j(t)$" "\n" r"$\tau_{ij}, s_t$"),
    (72.6, 13.5, 79.2, 13.5, r"$\hat{q}_t,\eta_{ij}$" "\n" r"$\alpha(v_{\mathrm{rel}})$"),
]
for x1, y1, x2, y2, lbl in arrow_specs:
    ax_flow.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.6",
                                     color='#2c3e50', lw=2.6), zorder=6)
    ax_flow.text((x1 + x2)/2, y1 + 1.5, lbl, ha='center', va='bottom',
                 fontsize=9.3, fontweight='bold', color='#1a252f',
                 bbox=dict(boxstyle="round,pad=0.18", fc='white', ec='#bdc3c7', lw=0.9))

# Closed-loop feedback path from Stage 4 back to Stage 1
fb_x = [89.0, 89.0, 10.9, 10.9]
fb_y = [7.1, 2.2, 2.2, 7.0]
ax_flow.plot(fb_x[:3], fb_y[:3], color='#196f3d', lw=2.6, ls='-', zorder=5)
ax_flow.annotate('', xy=(10.9, 7.1), xytext=(10.9, 2.2),
                 arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7",
                                 color='#196f3d', lw=2.6), zorder=6)
ax_flow.text(50.0, 2.2,
             r"Closed-Loop Safe Actuation: Optimal Filtered Acceleration $\mathbf{u}_i^*(t)$ @ 50 Hz ($\Delta t = 20$ ms)",
             ha='center', va='center', fontsize=10.5, fontweight='bold', color='#145a32',
             bbox=dict(boxstyle="round,pad=0.3", fc='#eafaf1', ec='#1e8449', lw=1.5), zorder=7)

# High-end Quadrotor Drawing Function
def draw_detailed_quadrotor(ax, x, y, tilt_deg=0.0, scale=1.0, body_color='#2c3e50', arm_color='#34495e', rotor_color='#e74c3c'):
    rad = np.radians(tilt_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    arm_span = 0.55 * scale
    x_l, y_l = x - arm_span * cos_t, y - arm_span * sin_t
    x_r, y_r = x + arm_span * cos_t, y + arm_span * sin_t
    ax.plot([x_l, x_r], [y_l, y_r], color=arm_color, lw=3.6 * scale, solid_capstyle='round', zorder=6)
    body = Circle((x, y), 0.12 * scale, facecolor=body_color, edgecolor='#1a252f', lw=1.5, zorder=7)
    ax.add_patch(body)
    ax.plot(x, y, 'o', color='#3498db', markersize=5 * scale, zorder=8)
    disk_w, disk_h = 0.38 * scale, 0.09 * scale
    for rx, ry in [(x_l, y_l), (x_r, y_r)]:
        rotor = Ellipse((rx, ry), disk_w, disk_h, angle=tilt_deg,
                        facecolor=rotor_color, edgecolor=rotor_color, alpha=0.45, lw=1.2, zorder=8)
        ax.add_patch(rotor)
        ax.plot(rx, ry, 'o', color='#1a252f', markersize=6 * scale, zorder=9)

# -----------------------------------------------------------------------------
# PANEL (A): Physical Layer - Turbulent Wind Shear & Deflected Wake
# -----------------------------------------------------------------------------
ax_a.set_title(r"$\mathbf{(a)}$ Physical Layer: Turbulent Wind Shear & Dynamic Wake Slipstream", pad=10, color='#145a32')
ax_a.set_xlim(-0.1, 5.3)
ax_a.set_ylim(-0.5, 3.6)
ax_a.set_aspect('equal')
ax_a.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')

ax_a.plot([0.15, 0.15], [-0.35, 3.4], color='#7f8c8d', lw=3.0, zorder=4)
for ym in np.linspace(-0.2, 3.2, 6):
    ax_a.plot([0.10, 0.20], [ym, ym], color='#7f8c8d', lw=2.0)

for yw, vw in zip([2.1, 2.7], [4.0, 4.8]):
    arrow_len = vw * 0.24
    ax_a.annotate('', xy=(0.25 + arrow_len, yw), xytext=(0.25, yw),
                  arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=2.4, mutation_scale=14))
ax_a.annotate('', xy=(0.58, 0.4), xytext=(0.25, 0.4),
              arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=2.0, mutation_scale=12))

ax_a.text(0.35, 3.22, "Sonic Wind $\\mathbf{w}_i(t)$\n(Anemometer Mast)",
          color='#1f618d', fontsize=10.5, fontweight='bold', va='center')

draw_detailed_quadrotor(ax_a, 3.5, 2.4, tilt_deg=-12, scale=1.0, body_color='#c0392b', arm_color='#922b21', rotor_color='#e74c3c')
ax_a.text(3.6, 3.08, "Lead UAV $j$ ($\\mathbf{v}_j = 3.5\\,\\mathrm{m/s}$)", ha='center', va='center', fontsize=10.8,
          fontweight='bold', color='#922b21', bbox=dict(boxstyle="round,pad=0.28", fc='white', ec='#e74c3c', lw=1.4, alpha=0.96))

ax_a.annotate('', xy=(4.95, 2.4), xytext=(3.95, 2.4),
              arrowprops=dict(arrowstyle="-|>", color='#c0392b', lw=3.0, mutation_scale=18))
ax_a.text(4.48, 2.13, r"$\mathbf{v}_j\,\mathbf{e}_x$", color='#c0392b', fontsize=11.5, fontweight='bold', ha='center')

theta_w = np.radians(37.9)
wake_x0, wake_y0 = 3.5, 2.25
poly_pts = [[wake_x0, wake_y0],
            [wake_x0 - 2.0 * np.sin(theta_w) - 0.55, wake_y0 - 2.0 * np.cos(theta_w)],
            [wake_x0 - 2.0 * np.sin(theta_w) + 0.55, wake_y0 - 2.0 * np.cos(theta_w)]]
wake_cone = Polygon(poly_pts, facecolor='#fadbd8', edgecolor='#e74c3c', lw=1.6, ls='--', alpha=0.45, zorder=2)
ax_a.add_patch(wake_cone)

for dx_off, col in zip([-0.3, -0.1, 0.1, 0.3], ['#c0392b', '#d35400', '#d35400', '#c0392b']):
    ys_stream = np.linspace(2.25, 0.5, 40)
    xs_stream = wake_x0 + dx_off - 0.85 * (2.25 - ys_stream)**1.15
    ax_a.plot(xs_stream, ys_stream, color=col, ls='-', lw=1.6, alpha=0.85, zorder=3)
    mid_idx = len(ys_stream)//2
    ax_a.annotate('', xy=(xs_stream[mid_idx+2], ys_stream[mid_idx+2]), xytext=(xs_stream[mid_idx], ys_stream[mid_idx]),
                  arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=11))

ax_a.text(4.22, 1.05, "Deflected Wake Plume\n$\\theta_{\\mathrm{wake}} = \\arctan(\\|\\mathbf{v}_h\\|/v_i) = 37.9^\\circ$",
          color='#78281f', fontsize=10.0, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.28", fc='#fdf2e9', ec='#d35400', lw=1.4, alpha=0.96))

draw_detailed_quadrotor(ax_a, 1.6, 1.0, tilt_deg=6, scale=1.0, body_color='#27ae60', arm_color='#1e8449', rotor_color='#2ecc71')
ax_a.annotate("Trailing Ego\nUAV $i$",
              xy=(1.35, 1.0), xytext=(0.55, 1.48),
              ha='center', va='center', fontsize=10.8, fontweight='bold', color='#196f3d',
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.8),
              bbox=dict(boxstyle="round,pad=0.28", fc='white', ec='#27ae60', lw=1.4, alpha=0.96))

ellipse_barrier = Ellipse((1.6, 1.0), 1.35, 2.2, angle=-37.9,
                          facecolor='#a9dfbf', edgecolor='#196f3d', lw=2.6, ls='-', alpha=0.35, zorder=3)
ax_a.add_patch(ellipse_barrier)
ax_a.text(1.6, -0.26, "Dynamic Barrier Metric $\\mathbf{E}_j(\\mathbf{v}_j)$", color='#196f3d', fontsize=10.8, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.25", fc='#eafaf1', ec='#27ae60', lw=1.2, alpha=0.96))

ax_a.set_xlabel("Streamwise Distance $x$ (m)")
ax_a.set_ylabel("Altitude $z$ (m)")

# -----------------------------------------------------------------------------
# PANEL (B): Cyber Network Layer - Intermittent DDS & State Extrapolation
# -----------------------------------------------------------------------------
ax_b.set_title(r"$\mathbf{(b)}$ Cyber Layer: Intermittent ROS 2 DDS & AoI Extrapolation", pad=10, color='#7e5109')
ax_b.set_xlim(-0.2, 5.4)
ax_b.set_ylim(-0.4, 3.6)
ax_b.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')

ax_b.plot([0.1, 4.7], [2.2, 2.2], color='#34495e', lw=2.8, zorder=3)
ax_b.text(4.85, 2.2, "Time $t$", va='center', fontsize=11.5, fontweight='bold', color='#2c3e50')

packets = [
    (0.6, 'k-3', True), (1.1, 'k-2', True), (1.6, 'k-1', False), (2.1, 'k', False),
    (2.6, 'k+1', False), (3.1, 'k+2', True), (3.6, 'k+3', True), (4.1, 'k+4', True),
]
for px, plabel, psuccess in packets:
    col = '#2ecc71' if psuccess else '#e74c3c'
    mark = r'$\checkmark$' if psuccess else r'$\times$'
    rect = FancyBboxPatch((px-0.18, 2.0), 0.36, 0.4, boxstyle="round,pad=0.04",
                          facecolor=col, edgecolor='#1e8449' if psuccess else '#922b21', lw=1.5, zorder=4)
    ax_b.add_patch(rect)
    ax_b.text(px, 2.2, mark, ha='center', va='center', color='white', fontweight='bold', fontsize=11, zorder=5)
    ax_b.text(px, 2.50, plabel, ha='center', va='bottom', color='#196f3d' if psuccess else '#922b21', fontweight='bold', fontsize=10)

ax_b.axvspan(1.4, 2.8, ymin=0.52, ymax=0.92, facecolor='#fef9e7', edgecolor='#f39c12', lw=1.8, ls='--', alpha=0.75, zorder=2)
ax_b.text(2.80, 3.18, "ROS 2 DDS Burst Dropout\n($\\tau_{ij} = 3$ ticks = $60\\,\\mathrm{ms}$)",
          color='#9a7d0a', fontsize=10.0, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.25", fc='#fffcf5', ec='#f39c12', lw=1.4, alpha=0.96))

ax_b.plot(0.6, 0.7, 's', color='#7f8c8d', markersize=12, label=r'Past Stored State $\mathbf{p}_j(t-\tau)$', zorder=5)
ax_b.text(0.6, 0.35, "$\\mathbf{p}_j(t-\\tau)$\n(Last Known)", ha='center', va='top', fontsize=10.0, color='#34495e', fontweight='bold')

ax_b.annotate('', xy=(2.4, 0.7), xytext=(0.75, 0.7),
              arrowprops=dict(arrowstyle="-|>", color='#d35400', lw=2.8, mutation_scale=15))
ax_b.text(1.55, 1.00, "Extrapolation: $+ (\\tau_{ij}\\Delta t)\\mathbf{v}_j$", color='#ba4a00', fontsize=10.2, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.2", fc='#fbeee6', ec='#e59866', lw=1.0, alpha=0.9))

ax_b.plot(2.5, 0.7, 'o', color='#e67e22', markersize=12, label=r'Extrapolated State $\hat{\mathbf{p}}_j(t)$', zorder=5)
uncert_ellipse = Ellipse((2.5, 0.7), 1.4, 0.85, facecolor='#f8c471', edgecolor='#d35400', lw=2.0, ls='--', alpha=0.4, zorder=3)
ax_b.add_patch(uncert_ellipse)
ax_b.text(2.5, 0.02, "AoI Uncertainty Tube: $\\mathcal{O}(\\tau_{ij}\\Delta t)$", color='#a04000', fontsize=10.0, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.2", fc='#fef5e7', ec='#f5b041', lw=1.0, alpha=0.9))

ax_b.plot(3.15, 0.88, '*', color='#c0392b', markersize=15, label=r'True Realized State $\mathbf{p}_j^{\mathrm{true}}(t)$', zorder=6)
ax_b.annotate('', xy=(3.15, 0.88), xytext=(2.5, 0.7),
              arrowprops=dict(arrowstyle="<->", color='#c0392b', lw=1.8, ls=':'))
ax_b.text(3.42, 0.88, "$\\mathbf{p}_j^{\\mathrm{true}}$\n(Residual $r_t$)", color='#922b21', fontsize=10.5, fontweight='bold', va='center')

ax_b.set_xlabel("Flight Elapsed Timeline (s)")
ax_b.legend(loc='upper left', framealpha=0.96, fontsize=9.0)

# -----------------------------------------------------------------------------
# PANEL (C): Safety Engine - Adaptive Conformal Inference (ACI)
# -----------------------------------------------------------------------------
ax_c.set_title(r"$\mathbf{(c)}$ Safety Engine: Adaptive Conformal Inference (ACI) Margin", pad=10, color='#154360')
ax_c.set_xlim(-0.1, 5.6)
ax_c.set_ylim(0, 3.3)
ax_c.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')

scores_x = np.linspace(0.1, 2.2, 120)
scores_pdf = 2.6 * scores_x * np.exp(-1.8 * (scores_x**1.5))
ax_c.plot(scores_x, scores_pdf, color='#2980b9', lw=2.8, label=r'Score Density $s_t = \frac{r_t}{1 + \lambda_{\mathrm{aoi}}\tau\Delta t}$')
ax_c.fill_between(scores_x, 0, scores_pdf, color='#ebf5fb', alpha=0.7)

q_val = 1.35
ax_c.axvline(q_val, color='#c0392b', lw=3.0, ls='-', label=r'Adaptive Quantile $\tilde{q}_t$ ($1-\alpha_t$)')
ax_c.text(q_val + 0.08, 1.30, r"$\tilde{q}_t$", color='#922b21', fontsize=13.5, fontweight='bold')
ax_c.fill_between(scores_x[scores_x >= q_val], 0, scores_pdf[scores_x >= q_val], color='#e74c3c', alpha=0.55, label=r'Target Risk Tail ($\alpha = 5\%$)')

bar_x = 4.10
bar_w = 0.62
q_phys, q_net = 1.35, 0.85

ax_c.bar(bar_x, q_phys, width=bar_w, bottom=0.0, color='#3498db', edgecolor='#1f618d', lw=2.0, label=r'$q_{\mathrm{phys}}$ (Wind/Wake Margin)', zorder=4)
ax_c.bar(bar_x, q_net, width=bar_w, bottom=q_phys, color='#e67e22', edgecolor='#b9770e', lw=2.0, label=r'$q_{\mathrm{net}}$ (Network Delay Margin)', zorder=4)

ax_c.text(bar_x, q_phys / 2.0, "$\\mathbf{q}_{\\mathrm{phys}}$\n(Phys)", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5, zorder=5)
ax_c.text(bar_x, q_phys + q_net / 2.0, "$\\mathbf{q}_{\\mathrm{net}}$\n(Net)", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5, zorder=5)

ax_c.annotate('', xy=(bar_x + 0.45, q_phys + q_net), xytext=(bar_x + 0.45, 0.0),
              arrowprops=dict(arrowstyle="<->", color='#2c3e50', lw=2.4))
ax_c.text(bar_x + 0.56, (q_phys + q_net)/2.0, "$\\hat{q}_t = q_{\\mathrm{phys}} + q_{\\mathrm{net}}$\n(Total Margin)",
          va='center', color='#1a252f', fontweight='bold', fontsize=10.0)

ax_c.text(1.75, 2.48,
          "$\\alpha_{t+1} = \\Pi_{[\\alpha_{\\min},\\alpha_{\\max}]}\\left[\\alpha_t + \\gamma(\\alpha_{\\mathrm{target}} - \\mathrm{err}_t)\\right]$\n"
          "$\\bullet$ Provable Sub-Linear Regret: $\\frac{1}{T}\\mathcal{R}_T \\leq \\mathcal{O}(1/T)$",
          ha='center', va='center', fontsize=10.0, fontweight='bold', color='#145a32',
          bbox=dict(boxstyle="round,pad=0.35", fc="#eafaf1", ec="#27ae60", lw=1.5, alpha=0.96))

ax_c.set_xlabel("Normalized Nonconformity Metric $s$")
ax_c.set_ylabel("Probability Density")
ax_c.legend(loc='upper right', framealpha=0.96, fontsize=9.0)

# -----------------------------------------------------------------------------
# PANEL (D): Control Space OSQP Geometry & Quadratic Objective Contours
# -----------------------------------------------------------------------------
ax_d.set_title(r"$\mathbf{(d)}$ Control Layer: Convex OSQP Optimization & Hyperplane Geometry", pad=10, color='#78281f')
ax_d.set_xlim(-4.2, 4.2)
ax_d.set_ylim(-4.2, 4.2)
ax_d.set_aspect('equal')
ax_d.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')
ax_d.axhline(0, color='#7f8c8d', lw=1.0)
ax_d.axvline(0, color='#7f8c8d', lw=1.0)

a_box = 3.2
rect_act = patches.Rectangle((-a_box, -a_box), 2*a_box, 2*a_box,
                             facecolor='#f8f9f9', edgecolor='#7f8c8d', lw=2.0, ls='--', label=r'Actuator Box $\mathcal{U} = [-a_{\max}, a_{\max}]^2$')
ax_d.add_patch(rect_act)

u_nom = np.array([0.2, -1.2])
for cr in [0.6, 1.2, 1.8, 2.4]:
    ell = Ellipse((u_nom[0], u_nom[1]), 2*cr, 2*cr, color='#3498db', fill=False, lw=1.2, ls=':', alpha=0.6)
    ax_d.add_patch(ell)
ax_d.text(1.75, -0.90, r"$\frac{1}{2}\|\mathbf{u} - \mathbf{u}_{\mathrm{nom}}\|^2$",
          color='#2980b9', fontsize=9.8, fontweight='bold')

line_xs = np.linspace(-3.2, 3.2, 100)
line_ys = (0.8 - line_xs) / 0.6
valid = (line_ys >= -3.2) & (line_ys <= 3.2)
ax_d.plot(line_xs[valid], line_ys[valid], color='#c0392b', lw=3.0, label=r'Barrier Hyperplane $\mathbf{a}_{ij}^\top \mathbf{u}_i + b_{ij} = \hat{q}_t$')
ax_d.fill_between(line_xs[valid], -3.2, line_ys[valid], color='#fadbd8', hatch='//', edgecolor='#e74c3c', alpha=0.45)
ax_d.text(-2.60, -2.65, "UNSAFE REGION\n($\\mathbf{a}_{ij}^\\top \\mathbf{u}_i + b_{ij} < \\hat{q}_t$)",
          color='#922b21', fontsize=9.8, fontweight='bold', ha='center', va='center',
          bbox=dict(boxstyle="round,pad=0.28", fc="#fdedec", ec="#e74c3c", lw=1.4, alpha=0.96))

line_ys_slack = (0.8 - 0.9 - line_xs) / 0.6
valid_s = (line_ys_slack >= -3.2) & (line_ys_slack <= 3.2)
ax_d.plot(line_xs[valid_s], line_ys_slack[valid_s], color='#f39c12', lw=2.4, ls='--', label=r'Slack-Relaxed Boundary $-\delta_i^*$')

ax_d.plot(u_nom[0], u_nom[1], 'o', color='#2980b9', markersize=12, zorder=8)
ax_d.annotate("$\\mathbf{u}_{\\mathrm{nom},i}$\n(Unsafe)",
              xy=(u_nom[0], u_nom[1]), xytext=(-0.45, -1.65),
              ha='center', va='center', color='#1f618d', fontweight='bold', fontsize=10.2,
              arrowprops=dict(arrowstyle="->", color='#2980b9', lw=1.8),
              bbox=dict(boxstyle="round,pad=0.22", fc='white', ec='#2980b9', lw=1.2, alpha=0.95),
              zorder=10)

u_rot_vec = np.array([-1.2, 0.7])
ax_d.annotate('', xy=(u_nom[0] + u_rot_vec[0], u_nom[1] + u_rot_vec[1]), xytext=(u_nom[0], u_nom[1]),
              arrowprops=dict(arrowstyle="-|>", color='#8e44ad', lw=3.0, mutation_scale=16))
ax_d.text(u_nom[0] + u_rot_vec[0] - 0.25, u_nom[1] + u_rot_vec[1] * 0.5 - 0.45, "$\\mathbf{u}_{\\mathrm{rot},i}$\n(Tangential)",
          color='#5b2c6f', fontweight='bold', fontsize=10.8, ha='right')

u_safe = np.array([0.7, 0.4])
ax_d.plot(u_safe[0], u_safe[1], 's', color='#27ae60', markersize=13, zorder=9)
ax_d.text(u_safe[0] + 0.22, u_safe[1] + 0.25, "$\\mathbf{u}_i^*$ (Safe Optimal)", color='#196f3d', fontweight='bold', fontsize=11.2)

u_target = u_nom + u_rot_vec
ax_d.annotate('', xy=(u_safe[0], u_safe[1]), xytext=(u_target[0], u_target[1]),
              arrowprops=dict(arrowstyle="-|>", color='#27ae60', lw=3.2, mutation_scale=16))

ax_d.set_xlabel(r"Commanded Acceleration $u_x$ ($\mathrm{m/s^2}$)")
ax_d.set_ylabel(r"Commanded Acceleration $u_y$ ($\mathrm{m/s^2}$)")
ax_d.legend(loc='upper right', framealpha=0.96, fontsize=9.2)

fig.subplots_adjust(top=0.95, bottom=0.06, left=0.05, right=0.98, hspace=0.34, wspace=0.22)
for out_path in [
    "/home/jarvis/Drones_CoDe_CBF/figures/figure1_system_architecture_and_pipeline.png",
    "/home/jarvis/Drones_CoDe_CBF/manuscript/figures/figure1_system_architecture_and_pipeline.png",
]:
    plt.savefig(out_path, dpi=300)
plt.close()
print("Upgraded Figure 1 successfully generated!")
