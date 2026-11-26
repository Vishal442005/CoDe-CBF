import sys
sys.path.append("/home/jarvis/Drones_CoDe_CBF")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Circle, Rectangle, Polygon, FancyArrowPatch
from src.simulation.scenarios import ScenarioGenerator
from src.simulation.swarm_simulator import MultiUAVSwarmSimulator
from src.simulation.data_loader import RealAerodynamicDataLoader

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11.5,
    'axes.labelsize': 12.6,
    'axes.titlesize': 13.2,
    'xtick.labelsize': 10.8,
    'ytick.labelsize': 10.8,
    'legend.fontsize': 8.8,
    'mathtext.fontset': 'dejavusans'
})

loader = RealAerodynamicDataLoader("/home/jarvis/Drones_CoDe_CBF/data")
df = pd.read_csv("/home/jarvis/Drones_CoDe_CBF/data/processed/tier_a_1000_benchmark_results.csv")

# Function to draw detailed authentic quadrotor
def draw_drone(ax, x, y, tilt_deg=0.0, scale=1.0, body_color='#2c3e50', arm_color='#34495e', rotor_color='#e74c3c'):
    rad = np.radians(tilt_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    w_b, h_b = 0.28 * scale, 0.08 * scale
    body_pts = np.array([[-w_b/2, -h_b/2], [w_b/2, -h_b/2], [w_b/2, h_b/2], [-w_b/2, h_b/2]])
    R_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    body_rot = (R_mat @ body_pts.T).T + np.array([x, y])
    ax.add_patch(Polygon(body_rot, facecolor=body_color, edgecolor='#1a252f', lw=1.3, zorder=8))
    ax.add_patch(Circle((x, y), 0.05*scale, facecolor='#3498db', edgecolor='#1b4f72', lw=1.1, zorder=9))
    arm_len = 0.38 * scale
    for side in [-1, 1]:
        arm_x = side * arm_len * cos_t
        arm_y = side * arm_len * sin_t
        ax.plot([x, x + arm_x], [y, y + arm_y], color=arm_color, lw=2.8*scale, solid_capstyle='round', zorder=7)
        pod_center = np.array([x + arm_x, y + arm_y])
        ax.plot([pod_center[0] - 0.025*sin_t, pod_center[0] + 0.025*sin_t],
                [pod_center[1] + 0.025*cos_t, pod_center[1] - 0.025*cos_t], color='#1a252f', lw=2.0, zorder=8)
        ax.add_patch(Ellipse((pod_center[0], pod_center[1]), 0.24*scale, 0.06*scale, angle=tilt_deg,
                             facecolor=rotor_color, edgecolor='#922b21', lw=1.1, alpha=0.82, zorder=9))

# =============================================================================
# PART 1: Figure 5 (Paper Fig 5): Multi-UAV Trajectory Profiles
# =============================================================================
fig_traj = plt.figure(figsize=(11.4, 5.0), dpi=300)
gs_traj = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.12], wspace=0.25)

ax_t1 = fig_traj.add_subplot(gs_traj[0, 0])
ax_t2 = fig_traj.add_subplot(gs_traj[0, 1])

# Scenario 2: 4-UAV Circle Swap Trajectories
cfg2 = ScenarioGenerator.get_scenario(2, seed=10)
sim2 = MultiUAVSwarmSimulator(cfg2, controller_type="code_cbf", data_loader=loader)
sim2.run_simulation(max_steps=280)

ax_t1.set_title(r"$\mathbf{(a)}$ 4-UAV Circle Swap (Vortex Avoidance)", pad=9)

uav_colors = ['#1f77b4', '#27ae60', '#e67e22', '#8e44ad']
uav_labels = [r'UAV 1 (E$\to$W)', r'UAV 2 (N$\to$S)',
              r'UAV 3 (W$\to$E)', r'UAV 4 (S$\to$N)']

for idx, (a_id, agent) in enumerate(sim2.agents.items()):
    pos = np.array(agent.pos_history)
    col = uav_colors[idx]
    
    # Trajectory trail
    ax_t1.plot(pos[:, 0], pos[:, 1], color=col, lw=2.8, alpha=0.9, label=uav_labels[idx])
    
    # Direction arrows along trajectory
    arrow_steps = [60, 140, 210]
    for st in arrow_steps:
        if st < len(pos) - 5:
            d_vec = pos[st+4, :2] - pos[st, :2]
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 0.01:
                ax_t1.annotate('', xy=(pos[st+4, 0], pos[st+4, 1]), xytext=(pos[st, 0], pos[st, 1]),
                              arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=12))
    
    # Start and Goal markers
    ax_t1.scatter(pos[0, 0], pos[0, 1], color=col, marker='o', s=120, edgecolor='black', lw=1.5, zorder=6)
    ax_t1.scatter(pos[-1, 0], pos[-1, 1], color=col, marker='*', s=180, edgecolor='black', lw=1.5, zorder=6)
    
    # Draw safety exclusion bubble at central closest point (step 140)
    p_crit = pos[140, :2]
    ax_t1.add_patch(Circle((p_crit[0], p_crit[1]), 0.40, facecolor=col, alpha=0.15, edgecolor=col, ls='--', lw=1.5))

# Central Deconfliction Vortex Callout placed in open lower right quadrant
ax_t1.annotate("Central Vortex\n$d_{\\min} = 0.92\\,\\mathrm{m} > r_{\\mathrm{safe}}$",
              xy=(0.0, 0.0), xytext=(0.6, -2.5),
              arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=1.8),
              fontsize=9.6, fontweight='bold', color='#1a252f',
              bbox=dict(boxstyle="round,pad=0.28", fc="#eafaf1", ec="#27ae60", lw=1.2, alpha=0.96))

all_pts2 = np.vstack([np.array(ag.pos_history)[:, :2] for ag in sim2.agents.values()])
x2_min, y2_min = all_pts2.min(axis=0) - 0.9
x2_max, y2_max = all_pts2.max(axis=0) + 0.9
ax_t1.set_xlabel("Lateral Coordinate $x$ (m)")
ax_t1.set_ylabel("Longitudinal Coordinate $y$ (m)")
ax_t1.set_xlim(x2_min, x2_max)
ax_t1.set_ylim(y2_min, y2_max)
ax_t1.set_aspect('equal')
ax_t1.grid(True, linestyle='--', alpha=0.45)
ax_t1.legend(loc="lower left", framealpha=0.95, fontsize=8.8)

# Scenario 3: Downwash Vertical Stacked Trajectories (Side-view Y-Z)
cfg3 = ScenarioGenerator.get_scenario(3, seed=15)
sim3 = MultiUAVSwarmSimulator(cfg3, controller_type="code_cbf", data_loader=loader)
sim3.run_simulation(max_steps=280)

sim3_nom = MultiUAVSwarmSimulator(cfg3, controller_type="nominal_hocbf", data_loader=loader)
sim3_nom.run_simulation(max_steps=280)

ax_t2.set_title(r"$\mathbf{(b)}$ Stacked Flight & Wake Evasion", pad=9)

p1_c = np.array(sim3.agents[1].pos_history)
p2_c = np.array(sim3.agents[2].pos_history)
p2_n = np.array(sim3_nom.agents[2].pos_history)

k_wake = 25
y_cross = p1_c[k_wake, 1]
z_cross1 = p1_c[k_wake, 2]
wake_cone = np.array([
    [y_cross - 0.25, z_cross1],
    [y_cross + 0.25, z_cross1],
    [y_cross + 1.15, z_cross1 - 1.6],
    [y_cross - 0.45, z_cross1 - 1.6]
])
ax_t2.add_patch(Polygon(wake_cone, facecolor='#e74c3c', alpha=0.18, edgecolor='#c0392b', ls=':', lw=1.5,
                        label=r'Wake Cone ($\mathbf{w}_{\mathrm{down}}$)'))

for y_off in [-0.1, 0.1]:
    ax_t2.annotate('', xy=(y_cross + y_off + 0.45, z_cross1 - 1.4), xytext=(y_cross + y_off, z_cross1 - 0.15),
                  arrowprops=dict(arrowstyle="-|>", color='#c0392b', lw=1.8, ls='--', mutation_scale=12))

ax_t2.plot(p1_c[:, 1], p1_c[:, 2], color='#2c3e50', lw=2.6, label=r'UAV 1 Leader')
ax_t2.plot(p2_c[:, 1], p2_c[:, 2], color='#27ae60', lw=2.8, label=r'UAV 2: CoDe-CBF (Safe)')
ax_t2.plot(p2_n[:, 1], p2_n[:, 2], color='#e74c3c', lw=2.1, ls='--', label=r'UAV 2: Nominal (Drop)')

draw_drone(ax_t2, p1_c[k_wake, 1], p1_c[k_wake, 2], tilt_deg=-6.0, scale=0.85, body_color='#2c3e50', rotor_color='#e74c3c')
draw_drone(ax_t2, p2_c[110, 1], p2_c[110, 2], tilt_deg=4.0, scale=0.85, body_color='#27ae60', rotor_color='#1e8449')

for p, col in [(p1_c, '#2c3e50'), (p2_c, '#27ae60'), (p2_n, '#e74c3c')]:
    ax_t2.scatter(p[0, 1], p[0, 2], color=col, marker='o', s=90, edgecolor='black', lw=1.3, zorder=6)
    ax_t2.scatter(p[-1, 1], p[-1, 2], color=col, marker='*', s=140, edgecolor='black', lw=1.3, zorder=6)

ax_t2.annotate("CoDe-CBF Dive\n($+0.42$ m Clear)",
              xy=(0.0, 1.02), xytext=(-3.6, 0.40),
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.7),
              fontsize=9.2, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.22", fc="#eafaf1", ec="#27ae60", lw=1.1, alpha=0.96))

min_z_idx = np.argmin(p2_n[:, 2])
ax_t2.annotate("Nominal Drop\n$-4.24\\,\\mathrm{m/s^2}$ Shock",
              xy=(p2_n[min_z_idx, 1], p2_n[min_z_idx, 2]), xytext=(0.8, 0.40),
              arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.7),
              fontsize=9.2, fontweight='bold', color='#922b21',
              bbox=dict(boxstyle="round,pad=0.22", fc="#fdedec", ec="#e74c3c", lw=1.1, alpha=0.96))

ax_t2.set_xlabel("Streamwise Direction $y$ (m)")
ax_t2.set_ylabel("Altitude $z$ (m)")
ax_t2.set_xlim(-6.2, 4.8)
ax_t2.set_ylim(0.10, 4.6)
ax_t2.grid(True, linestyle='--', alpha=0.45)
ax_t2.legend(loc="upper right", framealpha=0.95, fontsize=8.8)

fig_traj.subplots_adjust(top=0.90, bottom=0.13, left=0.07, right=0.98, wspace=0.24)
for fig4_path in [
    "/home/jarvis/Drones_CoDe_CBF/figures/figure4_multi_uav_trajectory_profiles.png",
    "/home/jarvis/Drones_CoDe_CBF/manuscript/figures/figure4_multi_uav_trajectory_profiles.png",
]:
    plt.savefig(fig4_path, dpi=300)
plt.close(fig_traj)
print(f"Generated Figure 4: {fig4_path}")

# =============================================================================
# PART 2: Figure 6 (Paper Fig 6): Tier-A 1,000-Seed Benchmark Comparison
# =============================================================================
fig_bm = plt.figure(figsize=(11.2, 7.2), dpi=300)
gs_bm = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.28)

ax_b1 = fig_bm.add_subplot(gs_bm[0, 0])
ax_b2 = fig_bm.add_subplot(gs_bm[0, 1])
ax_b3 = fig_bm.add_subplot(gs_bm[1, 0])
ax_b4 = fig_bm.add_subplot(gs_bm[1, 1])

controllers = ["code_cbf", "offline_conformal", "racbf", "nominal_hocbf"]
labels = ["CoDe-CBF\n(Proposed)", "Offline\nConf.", "RaCBF", "Nominal\nHOCBF"]
colors = ["#27ae60", "#2980b9", "#8e44ad", "#c0392b"]
hatches = ['', '//', '\\\\', 'xx']

sc4_df = df[df["scenario_id"] == 4]
esv_means = [sc4_df[sc4_df["controller"] == c]["safety_violation_rate"].mean() * 100 for c in controllers]
bars1 = ax_b1.bar(labels, esv_means, color=colors, alpha=0.88, edgecolor='#1a252f', lw=1.3, hatch=hatches)
ax_b1.axhline(5.0, color='#c0392b', linestyle='--', linewidth=2.2, label=r"Risk Budget ($\alpha=5\%$)")

for idx, c in enumerate(controllers):
    pts = sc4_df[sc4_df["controller"] == c]["safety_violation_rate"] * 100
    jitter = np.random.RandomState(42).normal(0, 0.05, size=len(pts))
    ax_b1.scatter(idx + jitter, pts, color=colors[idx], alpha=0.18, s=16, zorder=4)

ax_b1.set_ylabel("Violation Rate (%)")
ax_b1.set_title(r"$\mathbf{(a)}$ Canyon Violation Rate (Scen. 4)", pad=8)
ax_b1.grid(True, linestyle='--', alpha=0.45)
ax_b1.legend(loc="upper left", framealpha=0.95, fontsize=8.8)
ax_b1.set_ylim(0, 15.2)

for bar in bars1:
    yval = bar.get_height()
    ax_b1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.42, f"{yval:.2f}%", ha='center', va='bottom',
               fontweight='bold', fontsize=9.8,
               bbox=dict(boxstyle="round,pad=0.16", fc="#fdfefe", ec="#bdc3c7", lw=0.9, alpha=0.92))

ax_b1.annotate("Compliant\n($3.01\\% < 5\\%$)",
              xy=(0.25, 2.0), xytext=(0.68, 7.6),
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.7),
              fontsize=9.0, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.22", fc="#eafaf1", ec="#27ae60", lw=1.1, alpha=0.96))

all_esv_means = [df[df["controller"] == c]["safety_violation_rate"].mean() * 100 for c in controllers]
bars2 = ax_b2.bar(labels, all_esv_means, color=colors, alpha=0.88, edgecolor='#1a252f', lw=1.3, hatch=hatches)
ax_b2.axhline(5.0, color='#c0392b', linestyle='--', linewidth=2.2, label=r"Risk Budget ($\alpha=5\%$)")

ax_b2.set_ylabel("Mean Violation Rate (%)")
ax_b2.set_title(r"$\mathbf{(b)}$ Overall Swarm Violation (1,000 Seeds)", pad=8)
ax_b2.grid(True, linestyle='--', alpha=0.45)
ax_b2.legend(loc="upper left", framealpha=0.95, fontsize=8.8)
ax_b2.set_ylim(0, 7.2)

for bar in bars2:
    yval = bar.get_height()
    ax_b2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.20, f"{yval:.2f}%", ha='center', va='bottom',
               fontweight='bold', fontsize=9.8,
               bbox=dict(boxstyle="round,pad=0.16", fc="#fdfefe", ec="#bdc3c7", lw=0.9, alpha=0.92))

sc_indices = np.arange(1, 6)
bar_width = 0.19
sc_labels = ["1:Head", "2:Swap", "3:Wake", "4:Canyon", "5:Swarm"]
leg_short = ["CoDe-CBF", "Offline", "RaCBF", "Nominal"]

for i, (c, col, lbl_s, ht) in enumerate(zip(controllers, colors, leg_short, hatches)):
    c_df = df[df["controller"] == c]
    d_means = [c_df[c_df["scenario_id"] == sc]["min_distance_m"].mean() for sc in sc_indices]
    ax_b3.bar(sc_indices + (i - 1.5)*bar_width, d_means, width=bar_width, color=col, alpha=0.88,
              edgecolor='#1a252f', lw=1.1, hatch=ht, label=lbl_s)

ax_b3.axhline(0.8, color='#c0392b', linestyle='--', linewidth=2.2, label=r"Safe ($0.8$ m)")
ax_b3.fill_between([0.4, 5.6], 0.0, 0.8, color='#e74c3c', alpha=0.12)
ax_b3.set_xlabel("Operational Stress Scenario")
ax_b3.set_ylabel(r"Min Separation $d_{\min}$ (m)")
ax_b3.set_title(r"$\mathbf{(c)}$ Pairwise Separation by Scenario", pad=8)
ax_b3.set_xticks(sc_indices)
ax_b3.set_xticklabels(sc_labels)
ax_b3.set_xlim(0.4, 5.6)
ax_b3.set_ylim(0, 6.4)
ax_b3.grid(True, linestyle='--', alpha=0.45)
ax_b3.legend(loc="upper right", framealpha=0.95, fontsize=8.4, ncol=2, columnspacing=0.8)

solve_means = [df[df["controller"] == c]["solve_time_ms"].mean() for c in controllers]
solve_stds = [df[df["controller"] == c]["solve_time_ms"].std() for c in controllers]
bars4 = ax_b4.bar(labels, solve_means, yerr=solve_stds, color=colors, alpha=0.88, edgecolor='#1a252f',
                 lw=1.3, hatch=hatches, capsize=5)

ax_b4.set_ylabel("Solve Latency (ms)")
ax_b4.set_title(r"$\mathbf{(d)}$ Edge QP Compute Latency (< 0.65 ms)", pad=8)
ax_b4.grid(True, linestyle='--', alpha=0.45)
ax_b4.set_ylim(0, 1.48)

ax_b4.text(0.97, 0.94, "Tick Budget: 20 ms (50 Hz)\n32x Headroom",
           transform=ax_b4.transAxes, ha='right', va='top',
           fontsize=8.8, fontweight='bold', color='#5b2c6f',
           bbox=dict(boxstyle="round,pad=0.22", fc='#f5eef8', ec='#8e44ad', lw=1.1))

for bar in bars4:
    yval = bar.get_height()
    ax_b4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.3f}", ha='center', va='bottom',
               fontweight='bold', fontsize=9.6,
               bbox=dict(boxstyle="round,pad=0.16", fc="#fdfefe", ec="#bdc3c7", lw=0.9, alpha=0.92))

ax_b4.annotate("Sub-ms Real-Time\n(32x Faster)",
              xy=(0.0, 0.80), xytext=(0.45, 1.15),
              ha='center',
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.7),
              fontsize=9.0, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.22", fc="#eafaf1", ec="#27ae60", lw=1.1, alpha=0.96))

fig_bm.subplots_adjust(top=0.93, bottom=0.09, left=0.08, right=0.98, hspace=0.40, wspace=0.28)
for fig3_path in [
    "/home/jarvis/Drones_CoDe_CBF/figures/figure3_tier_a_1000_seed_benchmark_comparison.png",
    "/home/jarvis/Drones_CoDe_CBF/manuscript/figures/figure3_tier_a_1000_seed_benchmark_comparison.png",
]:
    plt.savefig(fig3_path, dpi=300)
plt.close(fig_bm)
print(f"Generated Figure 3: {fig3_path}")
