import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, Polygon, Rectangle, Wedge
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

fig = plt.figure(figsize=(18.5, 11.0), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.25)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# High-end Quadrotor Drawing Function
def draw_detailed_quadrotor(ax, x, y, tilt_deg=0.0, scale=1.0, body_color='#2c3e50', arm_color='#34495e', rotor_color='#e74c3c'):
    rad = np.radians(tilt_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    arm_span = 0.55 * scale
    
    # Left rotor arm & hub
    x_l = x - arm_span * cos_t
    y_l = y - arm_span * sin_t
    # Right rotor arm & hub
    x_r = x + arm_span * cos_t
    y_r = y + arm_span * sin_t
    
    # Structural carbon fiber arm
    ax.plot([x_l, x_r], [y_l, y_r], color=arm_color, lw=3.6 * scale, solid_capstyle='round', zorder=6)
    
    # Central avionics body / chassis
    body = Circle((x, y), 0.12 * scale, facecolor=body_color, edgecolor='#1a252f', lw=1.5, zorder=7)
    ax.add_patch(body)
    # Status LED
    ax.plot(x, y, 'o', color='#3498db', markersize=5 * scale, zorder=8)
    
    # Rotor disks (ellipses indicating spinning rotors)
    disk_w, disk_h = 0.38 * scale, 0.09 * scale
    rotor_l = Ellipse((x_l, y_l), disk_w, disk_h, angle=tilt_deg,
                      facecolor=rotor_color, edgecolor=rotor_color, alpha=0.45, lw=1.2, zorder=8)
    rotor_r = Ellipse((x_r, y_r), disk_w, disk_h, angle=tilt_deg,
                      facecolor=rotor_color, edgecolor=rotor_color, alpha=0.45, lw=1.2, zorder=8)
    ax.add_patch(rotor_l)
    ax.add_patch(rotor_r)
    
    # Rotor hub motor mounts
    ax.plot(x_l, y_l, 'o', color='#1a252f', markersize=6 * scale, zorder=9)
    ax.plot(x_r, y_r, 'o', color='#1a252f', markersize=6 * scale, zorder=9)

# -----------------------------------------------------------------------------
# PANEL (A): Physical Layer - Turbulent Wind Shear & Deflected Wake
# -----------------------------------------------------------------------------
ax_a.set_title(r"$\mathbf{(a)}$ Physical Layer: Turbulent Wind Shear & Dynamic Wake Slipstream", pad=12)
ax_a.set_xlim(-0.1, 5.2)
ax_a.set_ylim(-0.4, 3.6)
ax_a.set_aspect('equal')
ax_a.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')

# Anemometer mast tower on left
ax_a.plot([0.15, 0.15], [-0.3, 3.4], color='#7f8c8d', lw=3.0, zorder=4)
for ym in np.linspace(-0.2, 3.2, 6):
    ax_a.plot([0.10, 0.20], [ym, ym], color='#7f8c8d', lw=2.0)

# Sonic anemometer wind arrows placed clearly in upper layer (no collision with Ego UAV)
for yw, vw in zip([2.1, 2.7], [4.0, 4.8]):
    arrow_len = vw * 0.24
    ax_a.annotate('', xy=(0.25 + arrow_len, yw), xytext=(0.25, yw),
                  arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=2.4, mutation_scale=14))
# Low-altitude short boundary layer arrow
ax_a.annotate('', xy=(0.58, 0.4), xytext=(0.25, 0.4),
              arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=2.0, mutation_scale=12))

ax_a.text(0.35, 3.22, r"$\mathbf{Sonic\ Wind}\ \mathbf{w}_i(t)$" "\n" r"$\mathrm{(Anemometer\ Mast)}$",
          color='#1f618d', fontsize=10.5, fontweight='bold', va='center')

# Lead UAV j at (3.5, 2.4)
draw_detailed_quadrotor(ax_a, 3.5, 2.4, tilt_deg=-12, scale=1.0, body_color='#c0392b', arm_color='#922b21', rotor_color='#e74c3c')
ax_a.text(3.6, 3.05, r"$\mathbf{Lead\ UAV}\ j\ (\mathbf{v}_j = 3.5\,\mathrm{m/s})$", ha='center', va='center', fontsize=11.0,
          fontweight='bold', color='#922b21', bbox=dict(boxstyle="round,pad=0.28", fc='white', ec='#e74c3c', lw=1.4, alpha=0.96))

# Forward velocity vector
ax_a.annotate('', xy=(4.9, 2.4), xytext=(3.95, 2.4),
              arrowprops=dict(arrowstyle="-|>", color='#c0392b', lw=3.0, mutation_scale=18))
ax_a.text(4.45, 2.15, r"$\mathbf{v}_j\ \mathbf{e}_x$", color='#c0392b', fontsize=11.5, fontweight='bold', ha='center')

# Deflected Wake Plume: realistic streamlines with velocity gradient
theta_w = np.radians(37.9)
wake_x0, wake_y0 = 3.5, 2.25

# Shaded momentum wake expansion envelope
poly_pts = [[wake_x0, wake_y0],
            [wake_x0 - 2.0 * np.sin(theta_w) - 0.55, wake_y0 - 2.0 * np.cos(theta_w)],
            [wake_x0 - 2.0 * np.sin(theta_w) + 0.55, wake_y0 - 2.0 * np.cos(theta_w)]]
wake_cone = Polygon(poly_pts, facecolor='#fadbd8', edgecolor='#e74c3c', lw=1.6, ls='--', alpha=0.45, zorder=2)
ax_a.add_patch(wake_cone)

# Curved slipstream lines with arrows
for dx_off, col in zip([-0.3, -0.1, 0.1, 0.3], ['#c0392b', '#d35400', '#d35400', '#c0392b']):
    ys_stream = np.linspace(2.25, 0.5, 40)
    xs_stream = wake_x0 + dx_off - 0.85 * (2.25 - ys_stream)**1.15
    ax_a.plot(xs_stream, ys_stream, color=col, ls='-', lw=1.6, alpha=0.85, zorder=3)
    # Streamline mid-arrow
    mid_idx = len(ys_stream)//2
    ax_a.annotate('', xy=(xs_stream[mid_idx+2], ys_stream[mid_idx+2]), xytext=(xs_stream[mid_idx], ys_stream[mid_idx]),
                  arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=11))

# Deflected Wake Callout Badge shifted clear of streamlines
ax_a.text(4.20, 1.10, r"$\mathbf{Deflected\ Wake\ Plume}$" "\n" r"$\theta_{\mathrm{wake}} = \arctan(\|\mathbf{v}_h\|/v_i) = 37.9^\circ$",
          color='#78281f', fontsize=10.0, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.28", fc='#fdf2e9', ec='#d35400', lw=1.4, alpha=0.96))

# Trailing Ego UAV i at (1.6, 1.0)
draw_detailed_quadrotor(ax_a, 1.6, 1.0, tilt_deg=6, scale=1.0, body_color='#27ae60', arm_color='#1e8449', rotor_color='#2ecc71')
ax_a.annotate(r"$\mathbf{Trailing\ Ego}$" "\n" r"$\mathbf{UAV}\ i$",
              xy=(1.35, 1.0), xytext=(0.55, 1.45),
              ha='center', va='center', fontsize=11.0, fontweight='bold', color='#196f3d',
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.8),
              bbox=dict(boxstyle="round,pad=0.28", fc='white', ec='#27ae60', lw=1.4, alpha=0.96))

# Dynamic Safety Barrier Ellipsoid around Ego UAV tilted to match wake
ellipse_barrier = Ellipse((1.6, 1.0), 1.35, 2.2, angle=-37.9,
                          facecolor='#a9dfbf', edgecolor='#196f3d', lw=2.6, ls='-', alpha=0.35, zorder=3)
ax_a.add_patch(ellipse_barrier)
ax_a.text(1.6, -0.22, r"$\mathbf{Dynamic\ Barrier\ Metric}\ \mathbf{E}_j(\mathbf{v}_j)$", color='#196f3d', fontsize=11.0, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.25", fc='#eafaf1', ec='#27ae60', lw=1.2, alpha=0.96))

ax_a.set_xlabel("Streamwise Distance $x$ (m)")
ax_a.set_ylabel("Altitude $z$ (m)")

# -----------------------------------------------------------------------------
# PANEL (B): Cyber Network Layer - Intermittent DDS & State Extrapolation
# -----------------------------------------------------------------------------
ax_b.set_title(r"$\mathbf{(b)}$ Cyber Layer: Intermittent ROS 2 DDS & AoI Extrapolation", pad=12)
ax_b.set_xlim(-0.2, 5.4)
ax_b.set_ylim(-0.4, 3.6)
ax_b.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')

# Transmission Timeline
ax_b.plot([0.1, 4.7], [2.2, 2.2], color='#34495e', lw=2.8, zorder=3)
ax_b.text(4.85, 2.2, r"$\mathrm{Time}\ t$", va='center', fontsize=11.5, fontweight='bold', color='#2c3e50')

# Packets with timestamps
packets = [
    (0.6, 'k-3', True),
    (1.1, 'k-2', True),
    (1.6, 'k-1', False),
    (2.1, 'k', False),
    (2.6, 'k+1', False),
    (3.1, 'k+2', True),
    (3.6, 'k+3', True),
    (4.1, 'k+4', True),
]

for px, plabel, psuccess in packets:
    col = '#2ecc71' if psuccess else '#e74c3c'
    mark = r'$\checkmark$' if psuccess else r'$\times$'
    rect = patches.FancyBboxPatch((px-0.18, 2.0), 0.36, 0.4, boxstyle="round,pad=0.04",
                                  facecolor=col, edgecolor='#1e8449' if psuccess else '#922b21', lw=1.5, zorder=4)
    ax_b.add_patch(rect)
    ax_b.text(px, 2.2, mark, ha='center', va='center', color='white', fontweight='bold', fontsize=11, zorder=5)
    ax_b.text(px, 2.50, plabel, ha='center', va='bottom', color='#196f3d' if psuccess else '#922b21', fontweight='bold', fontsize=10)

# Burst blackout shaded region
ax_b.axvspan(1.4, 2.8, ymin=0.52, ymax=0.92, facecolor='#fef9e7', edgecolor='#f39c12', lw=1.8, ls='--', alpha=0.75, zorder=2)
ax_b.text(2.65, 3.15, r"$\mathbf{ROS\ 2\ DDS\ Burst\ Dropout}$" "\n" r"($\tau_{ij} = 3\ \mathrm{ticks} = 60\,\mathrm{ms}$)",
          color='#b7950b', fontsize=10.0, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.25", fc='#fffcf5', ec='#f39c12', lw=1.4, alpha=0.96))

# Extrapolation visual pipeline at bottom of panel
# 1. Past Stored State
ax_b.plot(0.6, 0.7, 's', color='#7f8c8d', markersize=12, label=r'Past Stored State $\mathbf{p}_j(t-\tau)$', zorder=5)
ax_b.text(0.6, 0.35, r"$\mathbf{p}_j(t-\tau)$" "\n(Last Known)", ha='center', va='top', fontsize=10.0, color='#34495e', fontweight='bold')

# Extrapolation Velocity Arrow
ax_b.annotate('', xy=(2.4, 0.7), xytext=(0.75, 0.7),
              arrowprops=dict(arrowstyle="-|>", color='#d35400', lw=2.8, mutation_scale=15))
ax_b.text(1.55, 0.95, r"$\mathbf{Extrapolation:}\ + (\tau_{ij}\Delta t)\mathbf{v}_j$", color='#ba4a00', fontsize=10.2, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.2", fc='#fbeee6', ec='#e59866', lw=1.0, alpha=0.9))

# 2. Extrapolated State with Expanding AoI Uncertainty Ellipse
ax_b.plot(2.5, 0.7, 'o', color='#e67e22', markersize=12, label=r'Extrapolated State $\hat{\mathbf{p}}_j(t)$', zorder=5)
uncert_ellipse = Ellipse((2.5, 0.7), 1.4, 0.85, facecolor='#f8c471', edgecolor='#d35400', lw=2.0, ls='--', alpha=0.4, zorder=3)
ax_b.add_patch(uncert_ellipse)
ax_b.text(2.5, 0.05, r"$\mathbf{AoI\ Uncertainty\ Tube:}\ \mathcal{O}(\tau_{ij}\Delta t)$", color='#a04000', fontsize=10.0, fontweight='bold', ha='center',
          bbox=dict(boxstyle="round,pad=0.2", fc='#fef5e7', ec='#f5b041', lw=1.0, alpha=0.9))

# 3. True Realized State & Residual Error
ax_b.plot(3.1, 0.85, '*', color='#c0392b', markersize=15, label=r'True Realized State $\mathbf{p}_j^{\mathrm{true}}(t)$', zorder=6)
ax_b.annotate('', xy=(3.1, 0.85), xytext=(2.5, 0.7),
              arrowprops=dict(arrowstyle="<->", color='#c0392b', lw=1.8, ls=':'))
ax_b.text(3.35, 0.85, r"$\mathbf{p}_j^{\mathrm{true}}$" "\n" r"(Residual $r_t$)", color='#922b21', fontsize=10.5, fontweight='bold', va='center')

ax_b.set_xlabel("Flight Elapsed Timeline (s)")
ax_b.legend(loc='upper left', framealpha=0.96, fontsize=9.0)

# -----------------------------------------------------------------------------
# PANEL (C): Safety Engine - Adaptive Conformal Inference (ACI)
# -----------------------------------------------------------------------------
ax_c.set_title(r"$\mathbf{(c)}$ Safety Engine: Adaptive Conformal Inference (ACI) Margin", pad=12)
ax_c.set_xlim(-0.1, 5.4)
ax_c.set_ylim(0, 3.2)
ax_c.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')

# Nonconformity score density curve
scores_x = np.linspace(0.1, 2.2, 120)
scores_pdf = 2.6 * scores_x * np.exp(-1.8 * (scores_x**1.5))
ax_c.plot(scores_x, scores_pdf, color='#2980b9', lw=2.8, label=r'Score Density $s_t = \frac{r_t}{1 + \lambda_{\mathrm{aoi}}\tau\Delta t}$')
ax_c.fill_between(scores_x, 0, scores_pdf, color='#ebf5fb', alpha=0.7)

# Quantile cutoff line
q_val = 1.35
ax_c.axvline(q_val, color='#c0392b', lw=3.0, ls='-', label=r'Adaptive Quantile $\tilde{q}_t$ ($1-\alpha_t$)')
ax_c.text(q_val + 0.08, 1.45, r"$\tilde{q}_t$", color='#922b21', fontsize=13.0, fontweight='bold')
ax_c.fill_between(scores_x[scores_x >= q_val], 0, scores_pdf[scores_x >= q_val], color='#e74c3c', alpha=0.55, label=r'Target Risk Tail ($\alpha = 5\%$)')

# Dual-Rate Stacked Margin Bar
bar_x = 4.10
bar_w = 0.60
q_phys = 1.35
q_net = 0.85

ax_c.bar(bar_x, q_phys, width=bar_w, bottom=0.0, color='#3498db', edgecolor='#1f618d', lw=2.0, label=r'$q_{\mathrm{phys}}$ (Wind/Wake Margin)', zorder=4)
ax_c.bar(bar_x, q_net, width=bar_w, bottom=q_phys, color='#e67e22', edgecolor='#b9770e', lw=2.0, label=r'$q_{\mathrm{net}}$ (Network Delay Margin)', zorder=4)

ax_c.text(bar_x, q_phys / 2.0, r"$\mathbf{q}_{\mathrm{phys}}$" "\n(Phys)", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5, zorder=5)
ax_c.text(bar_x, q_phys + q_net / 2.0, r"$\mathbf{q}_{\mathrm{net}}$" "\n(Net)", ha='center', va='center', color='white', fontweight='bold', fontsize=10.5, zorder=5)

# Total margin bracket arrow
ax_c.annotate('', xy=(bar_x + 0.45, q_phys + q_net), xytext=(bar_x + 0.45, 0.0),
              arrowprops=dict(arrowstyle="<->", color='#2c3e50', lw=2.4))
ax_c.text(bar_x + 0.58, (q_phys + q_net)/2.0, r"$\hat{q}_t = q_{\mathrm{phys}} + q_{\mathrm{net}}$" "\n(Total Margin)",
          va='center', color='#1a252f', fontweight='bold', fontsize=10.0)

# Online Adaptation Equation Box centered in wide open space at (1.60, 2.30)
ax_c.text(1.60, 2.30, r"$\alpha_{t+1} = \Pi_{[\alpha_{\min},\alpha_{\max}]}\left[\alpha_t + \gamma(\alpha_{\mathrm{target}} - \mathrm{err}_t)\right]$" "\n"
                     r"$\bullet\ \mathbf{Provable\ Sub-Linear\ Regret:}\ \frac{1}{T}\mathcal{R}_T \leq \mathcal{O}(1/T)$",
          ha='center', va='center', fontsize=10.2, color='#145a32',
          bbox=dict(boxstyle="round,pad=0.32", fc="#eafaf1", ec="#27ae60", lw=1.5, alpha=0.96))

ax_c.set_xlabel("Normalized Nonconformity Metric $s$")
ax_c.set_ylabel("Probability Density")
ax_c.legend(loc='upper right', framealpha=0.96, fontsize=9.2)

# -----------------------------------------------------------------------------
# PANEL (D): Control Space OSQP Geometry & Quadratic Objective Contours
# -----------------------------------------------------------------------------
ax_d.set_title(r"$\mathbf{(d)}$ Control Layer: Convex OSQP Optimization & Hyperplane Geometry", pad=12)
ax_d.set_xlim(-4.2, 4.2)
ax_d.set_ylim(-4.2, 4.2)
ax_d.set_aspect('equal')
ax_d.grid(True, linestyle='--', alpha=0.35, color='#bdc3c7')
ax_d.axhline(0, color='#7f8c8d', lw=1.0)
ax_d.axvline(0, color='#7f8c8d', lw=1.0)

# Actuator acceleration limits box
a_box = 3.2
rect_act = patches.Rectangle((-a_box, -a_box), 2*a_box, 2*a_box,
                             facecolor='#f8f9f9', edgecolor='#7f8c8d', lw=2.0, ls='--', label=r'Actuator Box $\mathcal{U} = [-a_{\max}, a_{\max}]^2$')
ax_d.add_patch(rect_act)

# Nominal Desired Acceleration
u_nom = np.array([0.2, -1.2])

# Concentric Elliptical Contours of Quadratic Objective: 0.5 * ||u - u_nom||^2
contour_radii = [0.6, 1.2, 1.8, 2.4]
for cr in contour_radii:
    ell = Ellipse((u_nom[0], u_nom[1]), 2*cr, 2*cr, color='#3498db', fill=False, lw=1.2, ls=':', alpha=0.6)
    ax_d.add_patch(ell)
ax_d.text(1.75, -0.90, r"$\frac{1}{2}\|\mathbf{u} - \mathbf{u}_{\mathrm{nom}}\|^2$",
          color='#2980b9', fontsize=9.5, fontweight='bold')

# Barrier hyperplane: a_ij^T u_i + b_ij >= q_hat
line_xs = np.linspace(-3.2, 3.2, 100)
line_ys = (0.8 - line_xs) / 0.6
valid = (line_ys >= -3.2) & (line_ys <= 3.2)
ax_d.plot(line_xs[valid], line_ys[valid], color='#c0392b', lw=3.0, label=r'Barrier Hyperplane $\mathbf{a}_{ij}^\top \mathbf{u}_i + b_{ij} = \hat{q}_t$')

# Shaded unsafe half-space with diagonal hatch
ax_d.fill_between(line_xs[valid], -3.2, line_ys[valid], color='#fadbd8', hatch='//', edgecolor='#e74c3c', alpha=0.45)
ax_d.text(-2.65, -2.65, r"$\mathbf{UNSAFE\ REGION}$" "\n" r"($\mathbf{a}_{ij}^\top \mathbf{u}_i + b_{ij} < \hat{q}_t$)",
          color='#922b21', fontsize=9.8, fontweight='bold', ha='center', va='center',
          bbox=dict(boxstyle="round,pad=0.28", fc="#fdedec", ec="#e74c3c", lw=1.4, alpha=0.96))

# Relaxed hyperplane with slack delta_i*
line_ys_slack = (0.8 - 0.9 - line_xs) / 0.6
valid_s = (line_ys_slack >= -3.2) & (line_ys_slack <= 3.2)
ax_d.plot(line_xs[valid_s], line_ys_slack[valid_s], color='#f39c12', lw=2.4, ls='--', label=r'Slack-Relaxed Boundary $-\delta_i^*$')

# 1. Nominal desired control u_nom with clean pointer arrow in open pocket
ax_d.plot(u_nom[0], u_nom[1], 'o', color='#2980b9', markersize=12, zorder=8)
ax_d.annotate(r"$\mathbf{u}_{\mathrm{nom},i}$" "\n(Unsafe)",
              xy=(u_nom[0], u_nom[1]), xytext=(-0.45, -1.60),
              ha='center', va='center', color='#1f618d', fontweight='bold', fontsize=10.2,
              arrowprops=dict(arrowstyle="->", color='#2980b9', lw=1.8),
              bbox=dict(boxstyle="round,pad=0.22", fc='white', ec='#2980b9', lw=1.2, alpha=0.95),
              zorder=10)

# 2. Tangential rotational deadlock breaking vector u_rot
u_rot_vec = np.array([-1.2, 0.7])
ax_d.annotate('', xy=(u_nom[0] + u_rot_vec[0], u_nom[1] + u_rot_vec[1]), xytext=(u_nom[0], u_nom[1]),
              arrowprops=dict(arrowstyle="-|>", color='#8e44ad', lw=3.0, mutation_scale=16))
ax_d.text(u_nom[0] + u_rot_vec[0] - 0.25, u_nom[1] + u_rot_vec[1] * 0.5 - 0.45, r"$\mathbf{u}_{\mathrm{rot},i}$" "\n(Tangential)",
          color='#5b2c6f', fontweight='bold', fontsize=11.0, ha='right')

# 3. Filtered optimal control u_safe on the boundary
u_safe = np.array([0.7, 0.4])
ax_d.plot(u_safe[0], u_safe[1], 's', color='#27ae60', markersize=13, zorder=9)
ax_d.text(u_safe[0] + 0.22, u_safe[1] + 0.25, r"$\mathbf{u}_i^*\ \mathrm{(Safe\ Optimal)}$", color='#196f3d', fontweight='bold', fontsize=11.5)

# Projection arrow from u_target to u_safe
u_target = u_nom + u_rot_vec
ax_d.annotate('', xy=(u_safe[0], u_safe[1]), xytext=(u_target[0], u_target[1]),
              arrowprops=dict(arrowstyle="-|>", color='#27ae60', lw=3.2, mutation_scale=16))

ax_d.set_xlabel(r"Commanded Acceleration $u_x$ ($\mathrm{m/s^2}$)")
ax_d.set_ylabel(r"Commanded Acceleration $u_y$ ($\mathrm{m/s^2}$)")
ax_d.legend(loc='upper right', framealpha=0.96, fontsize=9.2)

fig.subplots_adjust(top=0.94, bottom=0.08, left=0.06, right=0.98, hspace=0.36, wspace=0.25)
output_path = "/home/jarvis/Drones_CoDe_CBF/figures/figure1_system_architecture_and_pipeline.png"
plt.savefig(output_path, dpi=300)
plt.close()
print("Final Figure 1 successfully generated at:", output_path)
