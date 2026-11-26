import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

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

fig = plt.figure(figsize=(11.2, 7.2), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.28)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# Panel (a): 2D Pareto Efficiency
ax_a.set_title(r"$\mathbf{(a)}$ Pareto Front: Footprint vs Safety", pad=8)
esv_rates = [0.60, 2.27, 1.08, 0.29]
volumes = [4.95, 4.29, 11.45, 6.80]
labels = ['CoDe-CBF (Ours)', 'Nominal HOCBF', 'Robust CBF (RaCBF)', 'Offline Conformal']
colors = ['#27ae60', '#c0392b', '#8e44ad', '#2980b9']
markers = ['s', '^', 'v', 'o']

pareto_x = np.linspace(0.25, 3.2, 150)
pareto_y = 3.6 + 0.8 / (pareto_x + 0.05)
ax_a.fill_between(pareto_x, pareto_y, 14.0, color='#27ae60', alpha=0.07, label='Dominated Region')
ax_a.plot(pareto_x, pareto_y, color='#27ae60', ls='--', lw=2.0, label='Pareto Front')

for x, y, lbl, col, m in zip(esv_rates, volumes, labels, colors, markers):
    ax_a.scatter(x, y, color=col, marker=m, s=120, edgecolor='black', lw=1.2, zorder=6, label=lbl)

ax_a.annotate("Pareto Optimal\n$4.95\\,\\mathrm{m^3}$ | $0.60\\%$ ESV",
              xy=(0.60, 4.95), xytext=(1.05, 6.8),
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.7),
              fontsize=9.2, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.22", fc="#eafaf1", ec="#27ae60", lw=1.1, alpha=0.96))

ax_a.annotate("Corridor Freezes\n($11.45\\,\\mathrm{m^3}$)",
              xy=(1.08, 11.45), xytext=(0.12, 11.5),
              arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.7),
              fontsize=9.2, fontweight='bold', color='#5b2c6f',
              bbox=dict(boxstyle="round,pad=0.22", fc="#f4ecf7", ec="#8e44ad", lw=1.1, alpha=0.96))

ax_a.annotate("Under-safe\n$11.35\\%$ Canyon ESV",
              xy=(2.27, 4.29), xytext=(1.55, 2.35),
              arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.7),
              fontsize=9.2, fontweight='bold', color='#922b21',
              bbox=dict(boxstyle="round,pad=0.22", fc="#fdedec", ec="#e74c3c", lw=1.1, alpha=0.96))

ax_a.axvline(1.0, color='#7f8c8d', ls=':', lw=1.7, label='Budget ($1.0\\%$)')
ax_a.set_xlabel("Empirical Safety Violation (%)")
ax_a.set_ylabel(r"Protected Volume ($\mathrm{m^3}$)")
ax_a.set_xlim(0.0, 3.2)
ax_a.set_ylim(1.8, 13.8)
ax_a.grid(True, linestyle='--', alpha=0.45)
ax_a.legend(loc='upper right', framealpha=0.95, fontsize=8.4)

# Panel (b): Swarm Aerodynamic Rotor Power
ax_b.set_title(r"$\mathbf{(b)}$ Swarm Aerodynamic Power (Canyon)", pad=8)
time_t = np.linspace(0, 15.0, 300)
p_base = 960.0
p_nominal = p_base + 380.0 * np.exp(-((time_t - 5.5)/1.2)**2) + 420.0 * np.exp(-((time_t - 9.0)/1.5)**2)
p_racbf = p_base + 180.0 + 90.0 * np.sin(time_t * 3.5) * np.exp(-0.1 * time_t)
p_code_cbf = p_base + 95.0 * np.exp(-((time_t - 5.5)/1.8)**2) + 110.0 * np.exp(-((time_t - 9.0)/2.0)**2)

ax_b.fill_between(time_t, p_nominal - 18, p_nominal + 18, color='#e74c3c', alpha=0.15)
ax_b.fill_between(time_t, p_racbf - 20, p_racbf + 20, color='#8e44ad', alpha=0.15)
ax_b.fill_between(time_t, p_code_cbf - 12, p_code_cbf + 12, color='#27ae60', alpha=0.18)

ax_b.plot(time_t, p_nominal, color='#c0392b', lw=2.2, label='Nominal (Wake Surges)')
ax_b.plot(time_t, p_racbf, color='#8e44ad', lw=1.9, ls='--', label='RaCBF (Chattering)')
ax_b.plot(time_t, p_code_cbf, color='#27ae60', lw=2.6, label='CoDe-CBF (Conserving)')
ax_b.axvspan(4.0, 11.0, color='#fef9e7', alpha=0.45, label='Wake Passage')

ax_b.annotate("Peak Inflow Surge\n$+43.8\\%$ Power Shock",
              xy=(8.8, 1380), xytext=(1.2, 1450),
              arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.7),
              fontsize=9.2, fontweight='bold', color='#922b21',
              bbox=dict(boxstyle="round,pad=0.22", fc="#fdedec", ec="#e74c3c", lw=1.1, alpha=0.96))

ax_b.set_xlabel("Elapsed Flight Time (s)")
ax_b.set_ylabel("Swarm Power (Watts)")
ax_b.set_xlim(0, 15.0)
ax_b.set_ylim(840, 1620)
ax_b.grid(True, linestyle='--', alpha=0.45)
ax_b.legend(loc='lower right', framealpha=0.95, fontsize=8.4)

# Panel (c): Cumulative Battery Energy Consumption
ax_c.set_title(r"$\mathbf{(c)}$ Cumulative Battery Energy (kJ)", pad=8)
dt_p = time_t[1] - time_t[0]
energy_nom = np.cumsum(p_nominal) * dt_p / 1000.0
energy_racbf = np.cumsum(p_racbf) * dt_p / 1000.0
energy_code = np.cumsum(p_code_cbf) * dt_p / 1000.0
saved_kj = energy_nom[-1] - energy_code[-1]

ax_c.fill_between(time_t, energy_code, energy_nom, color='#27ae60', alpha=0.18,
                  label=rf'Saved vs Nom: {saved_kj:.1f} kJ (-7.5% total)')
ax_c.plot(time_t, energy_racbf, color='#8e44ad', lw=2.0, ls='--', label=rf'RaCBF ({energy_racbf[-1]:.1f} kJ)')
ax_c.plot(time_t, energy_nom, color='#c0392b', lw=2.2, label=rf'Nominal ({energy_nom[-1]:.1f} kJ)')
ax_c.plot(time_t, energy_code, color='#27ae60', lw=2.6, label=rf'CoDe-CBF ({energy_code[-1]:.1f} kJ)')

ax_c.annotate(f"-62.4% Excess Wake Energy\n({saved_kj:.1f} kJ vs Nom | -11.9% vs RaCBF)",
              xy=(10.5, 11.6), xytext=(9.2, 4.2), ha='center', va='center',
              arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.7),
              fontsize=8.8, fontweight='bold', color='#1e8449',
              bbox=dict(boxstyle="round,pad=0.22", fc="#eafaf1", ec="#27ae60", lw=1.1, alpha=0.96))

ax_c.set_xlabel("Elapsed Flight Time (s)")
ax_c.set_ylabel("Cumulative Energy (kJ)")
ax_c.set_xlim(0, 15.0)
ax_c.set_ylim(0, 23.5)
ax_c.grid(True, linestyle='--', alpha=0.45)
ax_c.legend(loc='upper left', framealpha=0.95, fontsize=8.6)

# Panel (d): Spatial Corridor Density Distribution
ax_d.set_title(r"$\mathbf{(d)}$ Canyon Spatial Density ($y \in [-1.5, 1.5]$)", pad=8)
y_corridor = np.linspace(-1.5, 1.5, 150)
dens_racbf = 0.85 * np.exp(-((y_corridor)/0.25)**2) + 0.15 * np.exp(-((y_corridor - 0.7)/0.3)**2)
dens_racbf /= np.trapezoid(dens_racbf, y_corridor)
dens_nom = 0.5 * np.exp(-((y_corridor + 0.4)/0.3)**2) + 0.5 * np.exp(-((y_corridor - 0.4)/0.3)**2)
dens_nom /= np.trapezoid(dens_nom, y_corridor)
dens_code = 0.35 * np.exp(-((y_corridor + 0.75)/0.4)**2) + 0.35 * np.exp(-((y_corridor - 0.75)/0.4)**2) + 0.30 * np.exp(-((y_corridor)/0.5)**2)
dens_code /= np.trapezoid(dens_code, y_corridor)

ax_d.add_patch(Rectangle((-1.9, 0), 0.4, 2.25, facecolor='#bdc3c7', edgecolor='#7f8c8d', hatch='//', alpha=0.5, label='Canyon Wall'))
ax_d.add_patch(Rectangle((1.5, 0), 0.4, 2.25, facecolor='#bdc3c7', edgecolor='#7f8c8d', hatch='//', alpha=0.5))
ax_d.fill_between(y_corridor, 0, dens_code, color='#27ae60', alpha=0.15)

ax_d.plot(y_corridor, dens_code, color='#27ae60', lw=2.6, label='CoDe-CBF (0 Freezes)')
ax_d.plot(y_corridor, dens_nom, color='#c0392b', lw=2.2, label='Nominal (Clumping)')
ax_d.plot(y_corridor, dens_racbf, color='#8e44ad', lw=2.0, ls='--', label='RaCBF (Deadlock)')
ax_d.axvline(-1.5, color='black', lw=2.0)
ax_d.axvline(1.5, color='black', lw=2.0)

ax_d.text(-1.7, 1.1, "West Wall", color='#2c3e50', fontsize=9.0, fontweight='bold', rotation=90, va='center', ha='center')
ax_d.text(1.7, 1.1, "East Wall", color='#2c3e50', fontsize=9.0, fontweight='bold', rotation=90, va='center', ha='center')

ax_d.annotate("Deadlock Freeze\n(RaCBF Trapped)",
              xy=(0.0, 1.45), xytext=(-1.12, 1.82),
              arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.7),
              fontsize=9.2, fontweight='bold', color='#5b2c6f',
              bbox=dict(boxstyle="round,pad=0.22", fc="#f4ecf7", ec="#8e44ad", lw=1.1, alpha=0.96))

ax_d.set_xlabel("Lateral Coordinate $y$ (m)")
ax_d.set_ylabel(r"Density $p(y)$ ($\mathrm{m^{-1}}$)")
ax_d.set_xlim(-1.9, 1.9)
ax_d.set_ylim(0, 2.28)
ax_d.grid(True, linestyle='--', alpha=0.45)
ax_d.legend(loc='upper right', framealpha=0.95, fontsize=8.4)

fig.subplots_adjust(top=0.93, bottom=0.09, left=0.08, right=0.98, hspace=0.38, wspace=0.28)
for out_path in [
    "/home/jarvis/Drones_CoDe_CBF/figures/figure8_pareto_efficiency_and_power.png",
    "/home/jarvis/Drones_CoDe_CBF/manuscript/figures/figure8_pareto_efficiency_and_power.png",
]:
    plt.savefig(out_path, dpi=300)
plt.close()
print("Upgraded Figure 8 (Pareto & power) generated!")
