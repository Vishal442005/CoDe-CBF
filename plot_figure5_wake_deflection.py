import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, Arc, Rectangle, Polygon
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12.0,
    'axes.labelsize': 13.5,
    'axes.titlesize': 14.8,
    'xtick.labelsize': 11.2,
    'ytick.labelsize': 11.2,
    'legend.fontsize': 10.0,
    'figure.titlesize': 17.5,
    'mathtext.fontset': 'dejavusans'
})

fig = plt.figure(figsize=(17.0, 10.8), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.36, wspace=0.26)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# Parameters
r_safe = 0.8
kappa_z = 2.0
v_induced = 4.5
v_forward = 3.5

# Function to draw detailed authentic quadrotor
def draw_drone(ax, x, y, tilt_deg=0.0, scale=1.0, body_color='#2c3e50', arm_color='#34495e', rotor_color='#e74c3c'):
    rad = np.radians(tilt_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    
    # Body fuselage
    w_b, h_b = 0.32 * scale, 0.10 * scale
    body_pts = np.array([
        [-w_b/2, -h_b/2], [w_b/2, -h_b/2], [w_b/2, h_b/2], [-w_b/2, h_b/2]
    ])
    R_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    body_rot = (R_mat @ body_pts.T).T + np.array([x, y])
    ax.add_patch(Polygon(body_rot, facecolor=body_color, edgecolor='#1a252f', lw=1.5, zorder=8))
    
    # Central avionics dome
    ax.add_patch(Circle((x - 0.02*sin_t, y + 0.02*cos_t), 0.06*scale, facecolor='#3498db', edgecolor='#1b4f72', lw=1.2, zorder=9))
    
    # Arms and rotors
    arm_len = 0.44 * scale
    for side in [-1, 1]:
        arm_x = side * arm_len * cos_t
        arm_y = side * arm_len * sin_t
        ax.plot([x, x + arm_x], [y, y + arm_y], color=arm_color, lw=3.2*scale, solid_capstyle='round', zorder=7)
        
        # Motor pod
        pod_center = np.array([x + arm_x, y + arm_y])
        ax.plot([pod_center[0] - 0.03*sin_t, pod_center[0] + 0.03*sin_t],
                [pod_center[1] + 0.03*cos_t, pod_center[1] - 0.03*cos_t], color='#1a252f', lw=2.2, zorder=8)
        
        # Rotor blur disk
        ax.add_patch(Ellipse((pod_center[0], pod_center[1]), 0.28*scale, 0.07*scale, angle=tilt_deg,
                             facecolor=rotor_color, edgecolor='#922b21', lw=1.2, alpha=0.82, zorder=9))

# -----------------------------------------------------------------------------
# Panel (a): Static Downwash Ellipsoid (Hover: v_j = 0)
# -----------------------------------------------------------------------------
ax_a.set_title(r"$\mathbf{(a)}$ Hover Condition: Static Vertical Downwash ($\mathbf{v}_j = \mathbf{0}$)", pad=12)

# Isotropic circle
circle_iso = Circle((0, 0), r_safe, fill=False, edgecolor='#7f8c8d', linestyle='--', lw=2.0, label=r'Isotropic Sphere ($r_{\mathrm{safe}} = 0.8\,\mathrm{m}$)')
ax_a.add_patch(circle_iso)

# Static Downwash Ellipsoid
ellipse_static = Ellipse((0, 0), width=2*r_safe, height=2*kappa_z*r_safe, angle=0,
                         facecolor='#e74c3c', alpha=0.22, edgecolor='#c0392b', lw=2.4,
                         label=r'Static Ellipsoid ($\kappa_z = 2.0$)')
ax_a.add_patch(ellipse_static)

# Downwash stream arrows
for x_pos in [-0.35, -0.18, 0.0, 0.18, 0.35]:
    ax_a.annotate('', xy=(x_pos, -1.5), xytext=(x_pos, -0.1),
                 arrowprops=dict(arrowstyle="-|>", color='#c0392b', lw=2.0, ls=':', mutation_scale=12))

ax_a.text(0.48, -0.85, r"$\mathbf{Downwash}\ \mathbf{w}_{\mathrm{down}}$" "\n" r"($v_{\mathrm{ind}} = 4.5\,\mathrm{m/s}$)",
          color='#c0392b', fontsize=10.5, fontweight='bold', ha='left')

# Draw detailed drone at (0, 0)
draw_drone(ax_a, 0, 0, tilt_deg=0.0, scale=0.9)
ax_a.text(0.0, 0.26, r"$\mathbf{Lead\ UAV\ } j\ (\mathbf{v}_j = \mathbf{0})$", ha='center', va='bottom',
          fontsize=10.5, fontweight='bold', color='#1a252f',
          bbox=dict(boxstyle="round,pad=0.25", fc="#fdfefe", ec="#2c3e50", lw=1.2, alpha=0.95))

ax_a.set_xlim(-2.0, 2.0)
ax_a.set_ylim(-2.1, 1.75)
ax_a.set_xlabel(r"Relative Lateral Position $\Delta x_{ij}$ (m)")
ax_a.set_ylabel(r"Relative Vertical Position $\Delta z_{ij}$ (m)")
ax_a.grid(True, linestyle='--', alpha=0.45)
ax_a.legend(loc='upper right', framealpha=0.95, fontsize=10.0)
ax_a.set_aspect('equal')

# -----------------------------------------------------------------------------
# Panel (b): Dynamic Tilt-Aware Wake Deflection (Forward Flight: v_j = 3.5 m/s)
# -----------------------------------------------------------------------------
theta_wake_rad = np.arctan2(v_forward, v_induced)
theta_wake_deg = np.degrees(theta_wake_rad) # ~37.9 deg

ax_b.set_title(r"$\mathbf{(b)}$ Forward Flight: Dynamic Tilt-Aware Wake ($\mathbf{v}_j = 3.5\,\mathrm{m/s}\ \mathbf{e}_x$)", pad=12)

# Static Ellipsoid for comparison (dashed ghost outline)
ellipse_stat_ghost = Ellipse((0, 0), width=2*r_safe, height=2*kappa_z*r_safe, angle=0,
                             facecolor='none', edgecolor='#95a5a6', linestyle=':', lw=2.0, label='Static Ellipsoid (Unaligned)')
ax_b.add_patch(ellipse_stat_ghost)

# Dynamic Deflected Ellipsoid
ellipse_dynamic = Ellipse((0, 0), width=2*r_safe, height=2*kappa_z*r_safe, angle=-theta_wake_deg,
                          facecolor='#27ae60', alpha=0.25, edgecolor='#1e8449', lw=2.6,
                          label=r'Dynamic Wake Tensor $\mathbf{E}_j(\mathbf{v}_j)$')
ax_b.add_patch(ellipse_dynamic)

# Curved streamline arrows deflecting backwards
y_s = np.linspace(-0.1, -1.6, 60)
for x0 in [-0.25, -0.05, 0.15]:
    xs = x0 - (v_forward / v_induced) * np.abs(y_s)**1.08
    ax_b.plot(xs, y_s, color='#1e8449', ls='--', lw=1.8, alpha=0.85)
    ax_b.annotate('', xy=(xs[-1], y_s[-1]), xytext=(xs[-5], y_s[-5]),
                  arrowprops=dict(arrowstyle="-|>", color='#1e8449', lw=1.8, mutation_scale=12))

# Wake axis vector
wake_len = 1.65
n_wake_x = -wake_len * np.sin(theta_wake_rad)
n_wake_z = -wake_len * np.cos(theta_wake_rad)
ax_b.annotate('', xy=(n_wake_x, n_wake_z), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color='#d35400', lw=2.6, ls='-', mutation_scale=14))

# Deflection Sector Arc
arc_radius = 0.75
arc = Arc((0, 0), 2*arc_radius, 2*arc_radius, angle=0, theta1=270 - theta_wake_deg, theta2=270,
          color='#d35400', lw=2.0, ls='-')
ax_b.add_patch(arc)
ax_b.text(-0.36, -0.72, rf"$\theta_{{\mathrm{{wake}}}} = {theta_wake_deg:.1f}^\circ$",
          color='#ba4a00', fontweight='bold', fontsize=10.0, ha='center',
          bbox=dict(boxstyle="round,pad=0.22", fc="#fdf2e9", ec="#e67e22", lw=1.1, alpha=0.96))

# Tilted Drone J at (0, 0)
draw_drone(ax_b, 0, 0, tilt_deg=-12.0, scale=0.9)

# Forward Velocity Vector
ax_b.annotate('', xy=(1.45, 0.0), xytext=(0.2, 0.0),
             arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=2.8, mutation_scale=14))
ax_b.text(0.75, 0.22, r"$\mathbf{v}_j = 3.5\,\mathrm{m/s}$", color='#1f618d', fontweight='bold', fontsize=10.5)

ax_b.text(-1.42, -0.25, r"$\mathbf{Wake\ Axis}$" "\n" r"$\mathbf{n}_{\mathrm{wake}}(\mathbf{v}_j)$",
          color='#d35400', fontweight='bold', fontsize=10.2, ha='center',
          bbox=dict(boxstyle="round,pad=0.25", fc="#fdf2e9", ec="#d35400", lw=1.1, alpha=0.95))

ax_b.set_xlim(-2.0, 2.0)
ax_b.set_ylim(-2.1, 1.75)
ax_b.set_xlabel(r"Relative Lateral Position $\Delta x_{ij}$ (m)")
ax_b.set_ylabel(r"Relative Vertical Position $\Delta z_{ij}$ (m)")
ax_b.grid(True, linestyle='--', alpha=0.45)
ax_b.legend(loc='upper right', framealpha=0.95, fontsize=10.0)
ax_b.set_aspect('equal')

# -----------------------------------------------------------------------------
# Panel (c): Wake Deflection Angle & Eccentricity vs Flight Speed
# -----------------------------------------------------------------------------
ax_c.set_title(r"$\mathbf{(c)}$ Rotor Wake Deflection Angle $\theta_{\mathrm{wake}}$ vs Horizontal Speed $\|\mathbf{v}_h\|$", pad=12)
speeds = np.linspace(0, 8.0, 150)
theta_deg_curves = np.degrees(np.arctan2(speeds, v_induced))
theta_deg_clipped = np.minimum(theta_deg_curves, 45.0)

# Shaded operating regimes
ax_c.axvspan(0.0, 4.5, color='#ebf5fb', alpha=0.5, label='Linear Slipstream Inflow Regime')
ax_c.axvspan(4.5, 8.0, color='#fef5e7', alpha=0.5, label=r'Saturated Envelope Regime ($\theta \leq 45^\circ$)')

ax_c.plot(speeds, theta_deg_curves, color='#2980b9', lw=2.6, label=r'Theoretical Slipstream: $\arctan(\|\mathbf{v}_h\| / v_{\mathrm{ind}})$')
ax_c.plot(speeds, theta_deg_clipped, color='#c0392b', lw=2.4, ls='--', label=r'CoDe-CBF Saturated Angle ($\theta_{\max} = 45^\circ$)')

# Current operating point
ax_c.axvline(3.5, color='#27ae60', ls=':', lw=2.0)
ax_c.axhline(theta_wake_deg, color='#27ae60', ls=':', lw=2.0)
ax_c.plot(3.5, theta_wake_deg, 'o', color='#27ae60', markersize=10, zorder=6)

ax_c.annotate(rf"Tested Speed: $3.5\,\mathrm{{m/s}}$" "\n" rf"Deflection $\theta = {theta_wake_deg:.1f}^\circ$",
              xy=(3.5, theta_wake_deg), xytext=(4.2, theta_wake_deg - 10),
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.8),
              fontsize=10.2, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.28", fc="#eafaf1", ec="#27ae60", lw=1.2, alpha=0.96))

ax_c.set_xlabel(r"UAV Horizontal Forward Airspeed $\|\mathbf{v}_h\|$ (m/s)")
ax_c.set_ylabel(r"Wake Deflection Angle $\theta_{\mathrm{wake}}$ (deg)")
ax_c.set_xlim(0, 8.0)
ax_c.set_ylim(0, 58)
ax_c.grid(True, linestyle='--', alpha=0.45)
ax_c.legend(loc='lower right', framealpha=0.95, fontsize=9.6)

# -----------------------------------------------------------------------------
# Panel (d): Safety Barrier Margin h_ij along Trailing Path
# -----------------------------------------------------------------------------
ax_d.set_title(r"$\mathbf{(d)}$ Barrier Evaluation $h_{ij}$ for Under-Passing UAV $i$ at $\Delta z = -1.2\,\mathrm{m}$", pad=12)

x_eval = np.linspace(-2.5, 2.0, 250)
z_eval = -1.2

# 1. Isotropic
h_iso = (x_eval**2 + z_eval**2) - r_safe**2

# 2. Static Ellipsoid: E = diag(1, 1, 1/kappa_z^2)
h_static = (x_eval**2 + (z_eval**2) / (kappa_z**2)) - r_safe**2

# 3. Dynamic Tensor
n_wake = np.array([-np.sin(theta_wake_rad), 0.0, -np.cos(theta_wake_rad)])
alpha_e = (1.0 / (kappa_z**2)) - 1.0
E_dyn = np.eye(3) + alpha_e * np.outer(n_wake, n_wake)

h_dynamic = []
for x in x_eval:
    dp = np.array([x, 0.0, z_eval])
    h_val = float(dp.T @ E_dyn @ dp) - r_safe**2
    h_dynamic.append(h_val)
h_dynamic = np.array(h_dynamic)

ax_d.plot(x_eval, h_iso, label=r'Isotropic Sphere (Dangerously Over-Optimistic)', color='#7f8c8d', lw=2.0, ls=':')
ax_d.plot(x_eval, h_static, label=r'Static Ellipsoid (Misses Downstream Plume)', color='#c0392b', lw=2.4, ls='--')
ax_d.plot(x_eval, h_dynamic, label=r'CoDe-CBF Dynamic Wake $\mathbf{E}_j(\mathbf{v}_j)$ (Ground-Truth)', color='#27ae60', lw=2.8)

ax_d.axhline(0.0, color='black', lw=1.6, ls='-', label='Safety Boundary ($h_{ij} = 0$)')
ax_d.fill_between(x_eval, -1.2, 0.0, color='#e74c3c', alpha=0.10)
ax_d.text(-2.3, -0.65, r'$\mathbf{UNSAFE\ REGION}$' '\n' r'($h_{ij} < 0$)', color='#c0392b', fontweight='bold', fontsize=10.5)

# Dangerous slipstream hazard region hatched
hazard_mask = (h_dynamic < h_static) & (x_eval < 0.2)
ax_d.fill_between(x_eval, h_static, h_dynamic, where=hazard_mask,
                 color='#f39c12', hatch='//', edgecolor='#d35400', alpha=0.35,
                 label='Dangerous Wake Deficit Missed by Static')

# Annotate the wake deficit safely inside Panel (d)
ax_d.annotate(r"$\mathbf{Critical\ Wake\ Deficit}$" "\n"
              r"Static $h_{ij} > 0$ (Falsely Safe)" "\n"
              r"Dynamic $h_{ij} < 0$ (Real Inflow Hazard)",
              xy=(-0.55, -0.1), xytext=(-1.75, 2.2),
              arrowprops=dict(arrowstyle="->", color='#d35400', lw=2.0),
              fontsize=9.6, fontweight='bold', color='#ba4a00',
              bbox=dict(boxstyle="round,pad=0.28", fc="#fdf2e9", ec="#e67e22", lw=1.2, alpha=0.96))

ax_d.set_xlabel(r"Relative Horizontal Position $\Delta x_{ij} = x_i - x_j$ (m)")
ax_d.set_ylabel(r"Barrier Function Value $h_{ij}(\mathbf{p}_i, \mathbf{p}_j)$")
ax_d.set_xlim(-2.5, 2.0)
ax_d.set_ylim(-1.2, 4.8)
ax_d.grid(True, linestyle='--', alpha=0.45)
ax_d.legend(loc='upper right', framealpha=0.95, fontsize=9.2)

fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.965, hspace=0.36, wspace=0.26)
output_path = "/home/jarvis/Drones_CoDe_CBF/figures/figure5_wake_deflection_geometry.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Figure 3 generated successfully at: {output_path}")
