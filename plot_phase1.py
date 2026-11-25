import sys
sys.path.append("/home/jarvis/Drones_CoDe_CBF")

import numpy as np
import matplotlib.pyplot as plt
from src.simulation.data_loader import RealAerodynamicDataLoader
from scipy.stats import norm, weibull_min
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse

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

loader = RealAerodynamicDataLoader("/home/jarvis/Drones_CoDe_CBF/data")

fig = plt.figure(figsize=(17.2, 11.5), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.28)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# -----------------------------------------------------------------------------
# Panel (a): Real 3D Sonic Anemometer Wind Velocity Traces
# -----------------------------------------------------------------------------
t_axis = np.arange(500) * 0.02 # First 10 seconds at 50 Hz
u_wind = loader.wind_u[:500]
v_wind = loader.wind_v[:500]
w_wind = loader.wind_3d[:500, 2]
v_horiz = np.sqrt(u_wind**2 + v_wind**2)

# Rolling mean and turbulence envelope for streamwise wind
u_smooth = gaussian_filter1d(u_wind, sigma=8)
u_std = np.std(u_wind - u_smooth)

ax_a.fill_between(t_axis, u_smooth - 1.96*u_std, u_smooth + 1.96*u_std,
                  color='#3498db', alpha=0.18, label=r'Streamwise Turbulence Envelope ($\pm 1.96\sigma$)')
ax_a.plot(t_axis, u_wind, color='#1f77b4', lw=2.0, alpha=0.9, label=r'$u_w$ (Streamwise, North)')
ax_a.plot(t_axis, u_smooth, color='#154360', lw=2.4, ls='-', label=r'Mean Trend $\bar{u}_w(t)$')
ax_a.plot(t_axis, v_wind, color='#27ae60', lw=1.8, ls='--', label=r'$v_w$ (Cross-stream, East)')
ax_a.plot(t_axis, w_wind, color='#e74c3c', lw=1.6, ls='-.', label=r'$w_w$ (Vertical, Updraft)')
ax_a.plot(t_axis, v_horiz, color='#8e44ad', lw=1.6, ls=':', label=r'$\|\mathbf{w}_h\|$ (Horizontal Speed)')

# Peak gust annotation placed clearly at (4.8, 6.2)
t_peak = 7.74
v_peak = v_wind[387]
speed_peak = v_horiz[387]
ax_a.plot(t_peak, v_peak, 'o', color='#c0392b', markersize=9, zorder=6)
ax_a.annotate(f'Peak Cross-Gust: {v_peak:.2f} m/s\nSpeed: {speed_peak:.2f} m/s ($I_u = 22.8\%$)',
              xy=(t_peak, v_peak), xytext=(4.2, 5.8),
              arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.8),
              fontsize=9.8, fontweight='bold', color='#922b21',
              bbox=dict(boxstyle="round,pad=0.28", fc="#fdedec", ec="#e74c3c", lw=1.2, alpha=0.96))

ax_a.axhline(0, color='gray', lw=0.9, ls=':')
ax_a.set_xlabel("Elapsed Flight Time (s)")
ax_a.set_ylabel("Wind Velocity Component (m/s)")
ax_a.set_title(r"$\mathbf{(a)}$ Real Sonic Anemometer 3D Wind Velocity Traces (NREL Mast)", pad=12)
ax_a.set_ylim(-6.8, 10.8)
ax_a.set_xlim(0, 10.0)
ax_a.grid(True, linestyle='--', alpha=0.45)
ax_a.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), framealpha=0.95, fontsize=9.2, ncol=3)

# -----------------------------------------------------------------------------
# Panel (b): Non-Gaussian Empirical Wind Distribution
# -----------------------------------------------------------------------------
w_speeds = np.linalg.norm(loader.wind_3d[:, :2], axis=1)
counts, bin_edges, _ = ax_b.hist(w_speeds, bins=35, density=True, alpha=0.65,
                                color='#5dade2', edgecolor='#1b4f72', lw=1.2,
                                label="Empirical Sonic PDF")

# Gaussian fit
mu, std = norm.fit(w_speeds)
x_fit = np.linspace(0.5, 8.5, 200)
pdf_gauss = norm.pdf(x_fit, mu, std)
ax_b.plot(x_fit, pdf_gauss, color='#c0392b', lw=2.4, ls='--', label=rf"Gaussian Fit ($\mu={mu:.2f}, \sigma={std:.2f}$)")

# Weibull fit
shape, loc, scale = weibull_min.fit(w_speeds, floc=0)
pdf_weib = weibull_min.pdf(x_fit, shape, loc, scale)
ax_b.plot(x_fit, pdf_weib, color='#1e8449', lw=2.6, ls='-', label=rf"Weibull Wind Model ($k={shape:.2f}, c={scale:.2f}$)")

# Statistical Summary Badge
kurt = float(np.mean(((w_speeds - mu)/std)**4) - 3.0)
skew = float(np.mean(((w_speeds - mu)/std)**3))
ax_b.text(0.04, 0.94,
          r"$\mathbf{Wind\ Field\ Statistics:}$" "\n"
          rf"$\bullet\ \mathrm{{Mean}}\ \mu = {mu:.2f}\,\mathrm{{m/s}}$" "\n"
          rf"$\bullet\ \mathrm{{Peak}}\ v_{{\max}} = {np.max(w_speeds):.2f}\,\mathrm{{m/s}}$" "\n"
          rf"$\bullet\ \mathrm{{Kurtosis}}\ \kappa = {kurt:.3f}$" "\n"
          rf"$\bullet\ \mathrm{{Skewness}}\ \gamma_1 = {skew:.3f}$",
          transform=ax_b.transAxes, verticalalignment='top',
          fontsize=10.2, color='#1a252f',
          bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9f9", ec="#7f8c8d", lw=1.3, alpha=0.96))

ax_b.set_xlabel("Horizontal Wind Speed (m/s)")
ax_b.set_ylabel("Probability Density")
ax_b.set_title(r"$\mathbf{(b)}$ Non-Gaussian Wind Distribution & Analytical Fits", pad=12)
ax_b.set_xlim(0.5, 8.5)
ax_b.set_ylim(0, 0.46)
ax_b.grid(True, linestyle='--', alpha=0.45)
ax_b.legend(loc="upper right", framealpha=0.95, fontsize=10.2)

# -----------------------------------------------------------------------------
# Panel (c): Downwash Wake Deficit Field with Quadrotor Silhouette & Quiver Flow
# -----------------------------------------------------------------------------
r_vals = np.linspace(0.0, 1.8, 90)
z_vals = np.linspace(0.4, 3.5, 90)
R, Z = np.meshgrid(r_vals, z_vals)
A_dz = np.zeros_like(R)

for i in range(len(z_vals)):
    for j in range(len(r_vals)):
        p_up = np.array([0.0, 0.0, z_vals[i]])
        p_eg = np.array([r_vals[j], 0.0, 0.0])
        A_dz[i, j] = loader.compute_downwash_acceleration(p_eg, p_up)[2]

# Clip for rich visual gradient in the operating envelope [-6.0, 0.0]
A_dz_clipped = np.clip(A_dz, -6.5, 0.0)

c = ax_c.contourf(R, Z, A_dz_clipped, levels=32, cmap="magma_r")
cbar = fig.colorbar(c, ax=ax_c, pad=0.028)
cbar.set_label(r'Downwash Acceleration $a_{dz}$ ($\mathrm{m/s^2}$)', fontsize=12.2)
cbar.ax.tick_params(labelsize=10.2)

# Contour lines with labels
cs = ax_c.contour(R, Z, A_dz_clipped, levels=[-4.24, -3.0, -2.0, -1.0, -0.5], colors='white', alpha=0.65, linewidths=1.1)
ax_c.clabel(cs, inline=True, fontsize=9.0, fmt=r'$%.2f$')

# Quiver arrows
r_q = np.linspace(0.12, 1.45, 8)
z_q = np.linspace(0.6, 3.1, 7)
Rq, Zq = np.meshgrid(r_q, z_q)
Uq = np.zeros_like(Rq)
Vq = np.zeros_like(Zq)
for i in range(len(z_q)):
    for j in range(len(r_q)):
        p_up = np.array([0.0, 0.0, z_q[i]])
        p_eg = np.array([r_q[j], 0.0, 0.0])
        acc = loader.compute_downwash_acceleration(p_eg, p_up)
        Uq[i, j] = 0.12 * (r_q[j] / (z_q[i] + 0.3))
        Vq[i, j] = np.clip(acc[2], -6.5, 0.0) * 0.18

ax_c.quiver(Rq, Zq, Uq, Vq, color='cyan', alpha=0.75, width=0.005, scale=18, label="Wake Slipstream Vectors")

# Draw Authentic Quadrotor Silhouette at Rotor Hub (0, 3.4)
ax_c.plot(0, 3.4, marker='s', color='#2c3e50', markersize=9, zorder=8)
ax_c.plot([-0.25, 0.25], [3.4, 3.4], color='#34495e', lw=3.0, zorder=7)
# Rotor disks
for rx in [-0.25, 0.25]:
    ax_c.add_patch(Ellipse((rx, 3.4), 0.18, 0.05, facecolor='#e74c3c', edgecolor='#922b21', lw=1.2, alpha=0.8, zorder=8))

ax_c.text(0.38, 3.38, r"$\mathbf{Holybro\ x500\ Lead\ Hub}$ ($m=1.48\,\mathrm{kg}$)", color='#fdfefe', fontsize=10.0, fontweight='bold', va='center',
          bbox=dict(boxstyle="round,pad=0.25", fc="#2c3e50", ec="#1a252f", lw=1.1, alpha=0.92))

# Lateral Cutoff Boundary
ax_c.axvline(1.5, color='#e74c3c', linestyle='--', lw=2.2, label="Lateral Wake Boundary ($r = 1.5\,\mathrm{m}$)")

# Safety Threshold Shock at dz = 0.8m
ax_c.plot(0.0, 0.8, marker='*', color='#f1c40f', markersize=14, zorder=9, markeredgecolor='black')
ax_c.annotate(r"$\mathbf{Safety\ Boundary\ Shock:}\ -4.24\,\mathrm{m/s^2}$" "\n(At $r_{\mathrm{safe}} = 0.8\,\mathrm{m}$ vertical spacing)",
              xy=(0.0, 0.8), xytext=(0.35, 1.15),
              arrowprops=dict(arrowstyle="->", color='#f1c40f', lw=2.0),
              fontsize=9.8, fontweight='bold', color='#7d6608',
              bbox=dict(boxstyle="round,pad=0.28", fc="#fef9e7", ec="#f1c40f", lw=1.2, alpha=0.96))

ax_c.set_xlabel("Lateral Radial Offset $r = \|\Delta \mathbf{p}_{xy}\|$ (m)")
ax_c.set_ylabel(r"Vertical Separation $\Delta z$ (m)")
ax_c.set_title(r"$\mathbf{(c)}$ Empirical Downwash Wake Deficit Field ($m=1.48$ kg)", pad=12)
ax_c.set_xlim(-0.05, 1.85)
ax_c.set_ylim(0.4, 3.65)
ax_c.grid(True, linestyle='--', alpha=0.35)
ax_c.legend(loc="center right", framealpha=0.95, fontsize=9.8)

# -----------------------------------------------------------------------------
# Panel (d): Real ROS 2 DDS Age-of-Information (AoI) Trace
# -----------------------------------------------------------------------------
win_s, win_e = 750, 1000
packet_idx = np.arange(250)
aoi_ticks = loader.aoi_ticks[win_s:win_e]
aoi_ms = (1 + aoi_ticks) * 20.0 # Total staleness: (1 + tau)*dt
drops = loader.packet_drops[win_s:win_e]

# Shaded gradient area under AoI step line
ax_d.fill_between(packet_idx, 0, aoi_ms, step='post', color='#3498db', alpha=0.20)
ax_d.step(packet_idx, aoi_ms, color='#1f77b4', where='post', lw=2.2, label=r"Total Staleness $(1+\tau_{ij})\Delta t$ (ms)")

# Baseline nominal periodic latency (20 ms at 50 Hz)
ax_d.axhline(20.0, color='#27ae60', ls='--', lw=2.0, label="Nominal Periodic Rate (20 ms / 50 Hz)")

# Shaded burst regions
in_burst = False
burst_s, burst_e = [], []
for k in range(len(drops)):
    if aoi_ms[k] > 25.0 and not in_burst:
        burst_s.append(k)
        in_burst = True
    elif aoi_ms[k] <= 25.0 and in_burst:
        burst_e.append(k)
        in_burst = False
if in_burst:
    burst_e.append(len(drops)-1)

for s, e in zip(burst_s, burst_e):
    ax_d.axvspan(s, e, color='#f39c12', alpha=0.25)

# Red crosses for dropped packets
drop_indices = packet_idx[drops]
ax_d.scatter(drop_indices, aoi_ms[drop_indices], color='#c0392b', marker='x', s=65, lw=2.0, label="DDS Packet Drop Event", zorder=5)

# Annotate the max burst at index 147 (k_global = 897) in clear space
max_idx = int(np.argmax(aoi_ms))
burst_txt = "Multi-Packet Loss Burst\n" + r"5 Dropped ($\tau_{ij}=5$, AoI = " + f"{aoi_ms[max_idx]:.0f} ms)"
ax_d.annotate(burst_txt,
              xy=(max_idx, aoi_ms[max_idx]), xytext=(95, 68),
              arrowprops=dict(arrowstyle="->", color='#d35400', lw=2.0),
              fontsize=9.8, fontweight='bold', color='#ba4a00',
              bbox=dict(boxstyle="round,pad=0.28", fc="#fdf2e9", ec="#e67e22", lw=1.2, alpha=0.96))

# Summary badge for network stats
drop_rate = np.mean(drops) * 100
mean_aoi = np.mean(aoi_ms)
max_aoi = np.max(aoi_ms)
ax_d.text(0.04, 0.94,
          r"$\mathbf{ROS\ 2\ DDS\ QoS\ Metrics:}$" "\n"
          rf"$\bullet\ \mathrm{{Middleware:}}\ \mathrm{{FastDDS\ /UDP\ P2P}}$" "\n"
          rf"$\bullet\ \mathrm{{Mean\ Staleness:}}\ {mean_aoi:.1f}\,\mathrm{{ms}}$" "\n"
          rf"$\bullet\ \mathrm{{Drop\ Rate:}}\ {drop_rate:.1f}\%,\ \mathrm{{Peak\ AoI:}}\ {max_aoi:.0f}\,\mathrm{{ms}}$",
          transform=ax_d.transAxes, verticalalignment='top',
          fontsize=10.0, color='#1a252f',
          bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9f9", ec="#7f8c8d", lw=1.3, alpha=0.96))

ax_d.set_xlabel("Packet Sequence Number $k$")
ax_d.set_ylabel("Age-of-Information (ms)")
ax_d.set_title(r"$\mathbf{(d)}$ Real ROS 2 DDS Packet Staleness ($(1+\tau_{ij}) \Delta t$)", pad=12)
ax_d.set_xlim(0, 250)
ax_d.set_ylim(0, 145)
ax_d.grid(True, linestyle='--', alpha=0.45)
ax_d.legend(loc="upper right", framealpha=0.95, fontsize=9.8)

fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.965, hspace=0.34, wspace=0.26)
out_path = "/home/jarvis/Drones_CoDe_CBF/figures/phase1_real_wind_and_downwash_profiles.png"
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Generated Phase 1 publication figure: {out_path}")
