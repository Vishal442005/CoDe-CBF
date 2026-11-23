import sys
sys.path.append("/home/jarvis/Drones_CoDe_CBF")

import numpy as np
import matplotlib.pyplot as plt
from src.code_cbf.adaptive_conformal import AdaptiveConformalTracker
from src.code_cbf.deadlock_breaker import DeadlockBreaker
from src.code_cbf.cbf_qp_filter import CoDeCBFFilter

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13.5,
    'axes.labelsize': 14.5,
    'axes.titlesize': 16.0,
    'xtick.labelsize': 12.5,
    'ytick.labelsize': 12.5,
    'legend.fontsize': 11.5,
    'figure.titlesize': 18.0,
    'mathtext.fontset': 'dejavusans'
})

fig, axs = plt.subplots(2, 2, figsize=(16.5, 11.2), dpi=300)

# 1. ACI Lemma 1 Bounded Parameter Trajectory
tracker = AdaptiveConformalTracker(alpha_target=0.05, gamma=0.008, alpha_min=0.01, alpha_max=0.20)
np.random.seed(42)
for t in range(450):
    if t < 150:
        res = 0.08 if np.random.rand() > 0.05 else 0.4
    elif t < 300: # Severe shock
        res = 0.50 if np.random.rand() > 0.30 else 0.1
    else: # Quiet recovery
        res = 0.02
    q_hat, _ = tracker.get_effective_margin(aoi_ticks=1, dt=0.02)
    tracker.update(raw_residual=res, aoi_ticks=1, q_hat=q_hat)

alpha_traj = tracker.alpha_history
axs[0, 0].plot(alpha_traj, 'b-', linewidth=2.4, label=r"Adaptive Coverage ($\alpha_t$)")
axs[0, 0].axhline(0.05, color='green', linestyle='--', lw=2.2, label=r"Target $\alpha_{\mathrm{target}} = 0.05$")
axs[0, 0].axhline(0.20, color='red', linestyle=':', lw=2.2, label=r"Upper Bound $\alpha_{\max} = 0.20$")
axs[0, 0].axhline(0.01, color='purple', linestyle=':', lw=2.2, label=r"Lower Bound $\alpha_{\min} = 0.01$")
axs[0, 0].set_xlabel("Control Ticks ($t$)")
axs[0, 0].set_ylabel(r"Miscoverage Level $\alpha_t$")
axs[0, 0].set_title(r"$\mathbf{(a)}$ Lemma 1: ACI Parameter Invariance within $[0.01, 0.20]$", pad=12)
axs[0, 0].set_ylim(0.0, 0.23)
axs[0, 0].grid(True, linestyle=':', alpha=0.5)
axs[0, 0].legend(loc="upper right", framealpha=0.94, fontsize=12.0)

# 2. AoI Dynamic Margin Inflation Curves
aoi_ms_range = np.linspace(0, 400, 100)
lambda_vals = [0.5, 1.0, 1.5, 2.0]
q_tilde_base = 0.20

for lam in lambda_vals:
    q_hat_curve = (1.0 + lam * (aoi_ms_range / 1000.0)) * q_tilde_base
    axs[0, 1].plot(aoi_ms_range, q_hat_curve, linewidth=2.4, label=rf"$\lambda_{{\mathrm{{aoi}}}} = {lam}$")

axs[0, 1].set_xlabel("Communication Age-of-Information (ms)")
axs[0, 1].set_ylabel(r"Effective Margin $\hat{q}_t$")
axs[0, 1].set_title(r"$\mathbf{(b)}$ Cyber-Physical Nonconformity Inflation vs. Packet Delay", pad=12)
axs[0, 1].set_ylim(0.18, 0.42)
axs[0, 1].grid(True, linestyle=':', alpha=0.5)
axs[0, 1].legend(loc="lower right", framealpha=0.94, fontsize=12.0)

# 3. Tangential Deadlock Resolution Vectors
breaker = DeadlockBreaker(epsilon_rot=0.8)
dx_vals = np.linspace(-2.0, 2.0, 15)
dy_vals = np.linspace(-2.0, 2.0, 15)
X, Y = np.meshgrid(dx_vals, dy_vals)
U_rot = np.zeros_like(X)
V_rot = np.zeros_like(Y)

for i in range(len(dy_vals)):
    for j in range(len(dx_vals)):
        p_rel = np.array([X[i, j], Y[i, j], 0.0])
        u_rot = breaker.compute_rotational_perturbation(
            p_i=p_rel,
            v_i=np.array([0.1, 0.0, 0.0]),
            p_goal_i=np.array([5.0, 0.0, 0.0]),
            neighbors=[{'pos': np.zeros(3), 'vel': np.array([-0.1, 0.0, 0.0])}]
        )
        U_rot[i, j] = u_rot[0]
        V_rot[i, j] = u_rot[1]

axs[1, 0].quiver(X, Y, U_rot, V_rot, color='darkorange', alpha=0.9, scale=16.0)
axs[1, 0].plot(0, 0, 'ro', markersize=12, label="Neighbor Conflict Center")
axs[1, 0].set_xlabel("Relative X Distance (m)")
axs[1, 0].set_ylabel("Relative Y Distance (m)")
axs[1, 0].set_title(r"$\mathbf{(c)}$ Tangential Rotational Perturbation Field ($\mathbf{u}_{\mathrm{rot}}$)", pad=12)
axs[1, 0].grid(True, linestyle=':', alpha=0.5)
axs[1, 0].legend(loc="upper right", framealpha=0.94, fontsize=12.0)
axs[1, 0].axis('equal')

# 4. OSQP 1,000-solve Latency Benchmark Histogram
cbf = CoDeCBFFilter()
np.random.seed(42)
latencies = []
for _ in range(1000):
    p_i = np.random.randn(3)
    v_i = np.random.randn(3) * 0.5
    u_nom = np.array([1.0, 0.0, 0.0])
    p_goal = np.array([10.0, 0.0, 0.0])
    neighbors = [{'id': 2, 'pos': p_i + np.array([0.9, 0.1, 0.0]), 'vel': np.zeros(3), 'aoi_ticks': 2}]
    _, _, t_ms, _ = cbf.filter_control(p_i, v_i, u_nom, p_goal, neighbors)
    latencies.append(t_ms)

axs[1, 1].hist(latencies, bins=35, color='forestgreen', edgecolor='black', alpha=0.75, label="OSQP Solve Latencies")
mean_lat = np.mean(latencies)
p95_lat = np.percentile(latencies, 95)
axs[1, 1].axvline(mean_lat, color='blue', linestyle='--', lw=2.4, label=f"Mean = {mean_lat:.3f} ms")
axs[1, 1].axvline(p95_lat, color='red', linestyle=':', lw=2.4, label=f"95th Pct = {p95_lat:.3f} ms")
axs[1, 1].axvline(20.0, color='purple', linestyle='-', lw=2.2, label="50Hz Loop Deadline (20 ms)")
axs[1, 1].set_xlabel("Solve Latency (ms)")
axs[1, 1].set_ylabel("Frequency Count")
axs[1, 1].set_title(r"$\mathbf{(d)}$ Edge Compute Latency Distribution (< 0.65 ms)", pad=12)
axs[1, 1].grid(True, linestyle=':', alpha=0.5)
axs[1, 1].legend(loc="upper right", framealpha=0.94, fontsize=12.0)

fig.subplots_adjust(top=0.94, bottom=0.09, left=0.07, right=0.96, hspace=0.36, wspace=0.28)
fig2_path = "/home/jarvis/Drones_CoDe_CBF/figures/phase2_cbf_verification_and_solver_performance.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Generated Phase 2 verification figure: {fig2_path}")
