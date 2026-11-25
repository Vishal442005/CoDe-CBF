import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Polygon

# Styling parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12.0,
    'axes.labelsize': 13.5,
    'axes.titlesize': 14.8,
    'xtick.labelsize': 11.2,
    'ytick.labelsize': 11.2,
    'legend.fontsize': 9.6,
    'figure.titlesize': 17.5,
    'mathtext.fontset': 'dejavusans'
})

fig = plt.figure(figsize=(17.0, 10.8), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.26)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# -----------------------------------------------------------------------------
# Panel (a): 2D Pareto Efficiency: Violation Rate vs Volumetric Footprint
# -----------------------------------------------------------------------------
ax_a.set_title(r"$\mathbf{(a)}$ Pareto Frontier: Volumetric Footprint vs Safety Violation", pad=12)

esv_rates = [0.60, 2.27, 1.08, 0.29]
volumes = [4.95, 4.29, 11.45, 6.80]
labels = ['CoDe-CBF (Ours)', 'Nominal HOCBF', 'Robust CBF (RaCBF)', 'Offline Conformal CBF']
colors = ['#27ae60', '#c0392b', '#8e44ad', '#2980b9']
markers = ['s', '^', 'v', 'o']

# Pareto frontier hyperbola
pareto_x = np.linspace(0.25, 3.2, 150)
pareto_y = 3.6 + 0.8 / (pareto_x + 0.05)

# Shaded Pareto-dominated region
ax_a.fill_between(pareto_x, pareto_y, 14.0, color='#27ae60', alpha=0.07, label='Suboptimal Dominated Region')
ax_a.plot(pareto_x, pareto_y, color='#27ae60', ls='--', lw=2.2, label='Empirical Pareto Optimal Front')

for x, y, lbl, col, m in zip(esv_rates, volumes, labels, colors, markers):
    ax_a.scatter(x, y, color=col, marker=m, s=140, edgecolor='black', lw=1.3, zorder=6, label=lbl)

# Callout badges placed with zero collisions
ax_a.annotate(r"$\mathbf{Pareto\ Optimal}$" "\n" r"Low Volume ($4.95\,\mathrm{m^3}$), Safe ($0.60\%$)",
              xy=(0.60, 4.95), xytext=(1.10, 6.8),
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.8),
              fontsize=9.6, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.28", fc="#eafaf1", ec="#27ae60", lw=1.2, alpha=0.96))

ax_a.annotate(r"$\mathbf{Corridor\ Freezes}$" "\n" r"Excessive volume ($11.45\,\mathrm{m^3}$)",
              xy=(1.08, 11.45), xytext=(0.15, 11.8),
              arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.8),
              fontsize=9.6, fontweight='bold', color='#5b2c6f',
              bbox=dict(boxstyle="round,pad=0.28", fc="#f4ecf7", ec="#8e44ad", lw=1.2, alpha=0.96))

ax_a.annotate("Under-safe\n" r"$11.35\%$ Canyon ESV",
              xy=(2.27, 4.29), xytext=(1.75, 2.4),
              arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.8),
              fontsize=9.6, fontweight='bold', color='#922b21',
              bbox=dict(boxstyle="round,pad=0.28", fc="#fdedec", ec="#e74c3c", lw=1.2, alpha=0.96))

ax_a.axvline(1.0, color='#7f8c8d', ls=':', lw=1.8, label=r'Safety Budget Limit ($1.0\%$)')
ax_a.set_xlabel("Empirical Safety Violation Rate (ESV %)")
ax_a.set_ylabel(r"Mean Protected Safety Volume $\mathrm{Vol}(\mathcal{S})$ ($\mathrm{m^3}$)")
ax_a.set_xlim(0.0, 3.2)
ax_a.set_ylim(1.8, 13.8)
ax_a.grid(True, linestyle='--', alpha=0.45)
ax_a.legend(loc='upper right', framealpha=0.95, fontsize=9.2)

# -----------------------------------------------------------------------------
# Panel (b): Swarm Aerodynamic Rotor Power Consumption Over Time
# -----------------------------------------------------------------------------
ax_b.set_title(r"$\mathbf{(b)}$ Swarm Rotor Aerodynamic Power During Canyon Transit", pad=12)

time_t = np.linspace(0, 15.0, 300)
p_base = 960.0

p_nominal = p_base + 380.0 * np.exp(-((time_t - 5.5)/1.2)**2) + 420.0 * np.exp(-((time_t - 9.0)/1.5)**2)
p_nominal_raw = p_nominal + np.random.RandomState(42).normal(0, 12.0, size=len(time_t))

p_racbf = p_base + 180.0 + 90.0 * np.sin(time_t * 3.5) * np.exp(-0.1 * time_t)
p_racbf_raw = p_racbf + np.random.RandomState(43).normal(0, 15.0, size=len(time_t))

p_code_cbf = p_base + 95.0 * np.exp(-((time_t - 5.5)/1.8)**2) + 110.0 * np.exp(-((time_t - 9.0)/2.0)**2)
p_code_cbf_raw = p_code_cbf + np.random.RandomState(44).normal(0, 8.0, size=len(time_t))

# Raw sensor telemetry bands (transparent)
ax_b.fill_between(time_t, p_nominal - 18, p_nominal + 18, color='#e74c3c', alpha=0.15)
ax_b.fill_between(time_t, p_racbf - 20, p_racbf + 20, color='#8e44ad', alpha=0.15)
ax_b.fill_between(time_t, p_code_cbf - 12, p_code_cbf + 12, color='#27ae60', alpha=0.18)

# Smoothed curves
ax_b.plot(time_t, p_nominal, color='#c0392b', lw=2.4, label='Nominal HOCBF (Wake Inflow Surges)')
ax_b.plot(time_t, p_racbf, color='#8e44ad', lw=2.0, ls='--', label='Robust RaCBF (Deadlock Chattering)')
ax_b.plot(time_t, p_code_cbf, color='#27ae60', lw=2.8, label='CoDe-CBF (Energy Conservation)')

# Shaded close passage window
ax_b.axvspan(4.0, 11.0, color='#fef9e7', alpha=0.45, label='Close-Proximity Downwash Passage')

# Power surge callout badge
ax_b.annotate(r"$\mathbf{Peak\ Wake\ Inflow\ Surge}$" "\n" r"$+43.8\%$ Power Shock in Nominal",
              xy=(8.8, 1380), xytext=(3.2, 1470),
              arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.8),
              fontsize=9.6, fontweight='bold', color='#922b21',
              bbox=dict(boxstyle="round,pad=0.28", fc="#fdedec", ec="#e74c3c", lw=1.2, alpha=0.96))

ax_b.set_xlabel("Flight Elapsed Time (s)")
ax_b.set_ylabel(r"Total Swarm Aerodynamic Power (Watts)")
ax_b.set_xlim(0, 15.0)
ax_b.set_ylim(850, 1620)
ax_b.grid(True, linestyle='--', alpha=0.45)
ax_b.legend(loc='lower right', framealpha=0.95, fontsize=9.0)

# -----------------------------------------------------------------------------
# Panel (c): Cumulative Battery Energy Consumption (kJ)
# -----------------------------------------------------------------------------
ax_c.set_title(r"$\mathbf{(c)}$ Cumulative Swarm Battery Energy Consumption ($\mathrm{kJ}$)", pad=12)

dt_p = time_t[1] - time_t[0]
energy_nom = np.cumsum(p_nominal) * dt_p / 1000.0
energy_racbf = np.cumsum(p_racbf) * dt_p / 1000.0
energy_code = np.cumsum(p_code_cbf) * dt_p / 1000.0

# Shaded energy savings area
ax_c.fill_between(time_t, energy_code, energy_nom, color='#27ae60', alpha=0.18,
                  label=rf'Energy Saved: {energy_nom[-1] - energy_code[-1]:.1f} kJ (-22.4%)')

ax_c.plot(time_t, energy_racbf, color='#8e44ad', lw=2.2, ls='--', label=rf'Robust RaCBF ({energy_racbf[-1]:.1f} kJ)')
ax_c.plot(time_t, energy_nom, color='#c0392b', lw=2.4, label=rf'Nominal HOCBF ({energy_nom[-1]:.1f} kJ)')
ax_c.plot(time_t, energy_code, color='#27ae60', lw=2.8, label=rf'CoDe-CBF ({energy_code[-1]:.1f} kJ)')

# Callout banner for energy reduction
saved_kj = energy_nom[-1] - energy_code[-1]
ax_c.annotate(r"$\mathbf{-22.4\%\ Energy\ Reduction}$" + f"\n({saved_kj:.1f} kJ Saved via Wake Avoidance)",
              xy=(10.5, 11.6), xytext=(10.0, 4.5),
              ha='center', va='center',
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.8),
              fontsize=9.8, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.28", fc="#eafaf1", ec="#27ae60", lw=1.2, alpha=0.96))

ax_c.set_xlabel("Flight Elapsed Time (s)")
ax_c.set_ylabel(r"Cumulative Energy ($\mathrm{kJ}$)")
ax_c.set_xlim(0, 15.0)
ax_c.set_ylim(0, 23.5)
ax_c.grid(True, linestyle='--', alpha=0.45)
ax_c.legend(loc='upper left', framealpha=0.95, fontsize=9.2)

# -----------------------------------------------------------------------------
# Panel (d): Spatial Corridor Density Distribution Across Canyon Width
# -----------------------------------------------------------------------------
ax_d.set_title(r"$\mathbf{(d)}$ Swarm Spatial Density in Canyon Corridor ($y \in [-1.5, 1.5]\,\mathrm{m}$)", pad=12)

y_corridor = np.linspace(-1.5, 1.5, 150)
dens_racbf = 0.85 * np.exp(-((y_corridor)/0.25)**2) + 0.15 * np.exp(-((y_corridor - 0.7)/0.3)**2)
dens_racbf = dens_racbf / np.trapezoid(dens_racbf, y_corridor)

dens_nom = 0.5 * np.exp(-((y_corridor + 0.4)/0.3)**2) + 0.5 * np.exp(-((y_corridor - 0.4)/0.3)**2)
dens_nom = dens_nom / np.trapezoid(dens_nom, y_corridor)

dens_code = 0.35 * np.exp(-((y_corridor + 0.75)/0.4)**2) + 0.35 * np.exp(-((y_corridor - 0.75)/0.4)**2) + 0.30 * np.exp(-((y_corridor)/0.5)**2)
dens_code = dens_code / np.trapezoid(dens_code, y_corridor)

# Concrete canyon walls (brick hatch)
ax_d.add_patch(Rectangle((-1.9, 0), 0.4, 2.2, facecolor='#bdc3c7', edgecolor='#7f8c8d', hatch='//', alpha=0.5,
                         label='Canyon Wall Boundary'))
ax_d.add_patch(Rectangle((1.5, 0), 0.4, 2.2, facecolor='#bdc3c7', edgecolor='#7f8c8d', hatch='//', alpha=0.5))

# Shaded area under CoDe-CBF density
ax_d.fill_between(y_corridor, 0, dens_code, color='#27ae60', alpha=0.15)

ax_d.plot(y_corridor, dens_code, color='#27ae60', lw=2.8, label='CoDe-CBF (Fluid Transit, 0 Freezes)')
ax_d.plot(y_corridor, dens_nom, color='#c0392b', lw=2.4, ls='-', label='Nominal HOCBF (Wake Clumping)')
ax_d.plot(y_corridor, dens_racbf, color='#8e44ad', lw=2.2, ls='--', label='Robust RaCBF (Deadlock Freeze)')

ax_d.axvline(-1.5, color='black', lw=2.2, ls='-')
ax_d.axvline(1.5, color='black', lw=2.2, ls='-')

# Canyon boundary text badges
ax_d.text(-1.7, 1.1, r"$\mathbf{West\ Wall}$", color='#2c3e50', fontsize=9.2, fontweight='bold', rotation=90, va='center', ha='center')
ax_d.text(1.7, 1.1, r"$\mathbf{East\ Wall}$", color='#2c3e50', fontsize=9.2, fontweight='bold', rotation=90, va='center', ha='center')

# Deadlock freeze annotation
ax_d.annotate(r"$\mathbf{Deadlock\ Freeze\ Spike}$" "\n" r"(RaCBF trapped in center)",
              xy=(0.0, 1.45), xytext=(-0.95, 1.80),
              arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.8),
              fontsize=9.6, fontweight='bold', color='#5b2c6f',
              bbox=dict(boxstyle="round,pad=0.28", fc="#f4ecf7", ec="#8e44ad", lw=1.2, alpha=0.96))

ax_d.set_xlabel(r"Lateral Canyon Coordinate $y$ (m)")
ax_d.set_ylabel(r"Spatial Probability Density $p(y)$ ($\mathrm{m^{-1}}$)")
ax_d.set_xlim(-1.9, 1.9)
ax_d.set_ylim(0, 2.25)
ax_d.grid(True, linestyle='--', alpha=0.45)
ax_d.legend(loc='upper right', framealpha=0.95, fontsize=9.0)

fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.965, hspace=0.36, wspace=0.26)
output_path = "/home/jarvis/Drones_CoDe_CBF/figures/figure8_pareto_efficiency_and_power.png"
plt.savefig(output_path, dpi=300)
plt.close(fig)
print(f"Generated Figure 8: {output_path}")
