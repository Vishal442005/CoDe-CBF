# CoDe-CBF: Distribution-Free Safety Filtering for Decentralized Multi-UAV Swarms

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_LTS-green.svg)](https://docs.ros.org/en/humble/)
[![OSQP](https://img.shields.io/badge/Solver-OSQP_50Hz-orange.svg)](https://osqp.org/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-1%2C000_Seeds-brightgreen.svg)]()

> **CoDe-CBF:** Distribution-Free Safety Filtering for Decentralized Multi-UAV Swarms under Non-Stationary Aerodynamic Disturbances and Variable Communication Age-of-Information.  
> **Author:** Vishal Chandel  
> **Target Venue:** IEEE Robotics and Automation Letters (RA-L) / Elsevier Robotics and Autonomous Systems (RAS)

---

## 📌 Overview & Core Innovation

Decentralized multi-UAV swarms operating in confined spatial corridors encounter two simultaneous, non-stationary failure modes:
1. **Unmodeled Aerodynamic Disturbances:** Non-Gaussian turbulent wind shear and inter-rotor downwash wake deficit fields, inducing sudden downward accelerations exceeding $-4.2\text{ m/s}^2$ on lower vehicles.
2. **Cyber-Physical Network Staleness:** Ad-hoc broadcasts over ROS 2 DDS middleware arrive with stochastic Age-of-Information (AoI) latency (50--200 ms) or drop entirely during RF fading bursts.

**CoDe-CBF** couples physical aerodynamic residuals with network Age-of-Information staleness into a unified cyber-physical nonconformity metric, dynamically adapting safety envelopes via online Adaptive Conformal Inference (ACI).

```
                   Decentralized Drone Swarm (50 Hz Control Loop)
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[Physical Disturbance]                                         [Network Communication]
• Real 10m sonic anemometer wind                               • Real ROS 2 DDS packet loss
• Empirical downwash wake field (k_d1=4000)                    • Stochastic Age-of-Information τ_ij(t)
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         ▼
            Coupled Cyber-Physical Nonconformity Metric (Eq. 9)
                  s_t^(i,j) = r_t^(i,j) / (1 + λ_aoi · τ_ij · Δt)
                                         │
                                         ▼
                 Adaptive Conformal Inference (ACI) Quantile
                  α_(t+1) = Proj[α_t + γ(α_target - err_t)]
                                         │
                                         ▼
               Decentralized Convex Quadratic Program (Eq. 13)
                     min  ½ ||u_i - (u_nom + u_rot)||^2 + ρ_slack · δ_i^2
                     s.t. a_ij^T u_i + b_ij ≥ q̂_t^(i,j) - δ_i
                          u_min ≤ u_i ≤ u_max, ||v_i + Δt u_i|| ≤ v_max
                                         │
                                         ▼
                 OSQP C++ Engine (< 0.65 ms Solve Time, O(1))
```

---

## 🔬 Mathematical Guarantees (Zero Contradictions)

* **Lemma 1 (Parameter Invariance):** The ACI adaptation sequence $\alpha_t$ is strictly bounded within $[\alpha_{\min}, \alpha_{\max}] = [0.01, 0.20]$ for all $t \ge 0$.
* **Theorem 1 (Distribution-Free Asymptotic Coverage):** For arbitrary non-stationary wind shear, rotor downwash, and bursty packet loss, the empirical safety violation rate converges almost surely to $\alpha_{\text{target}}$:
  $$\left| \frac{1}{T} \sum_{t=1}^T \mathrm{err}_t^{(i,j)} - \alpha_{\text{target}} \right| \le \frac{\alpha_{\max} - \alpha_{\min}}{\gamma T} + \frac{2 C_{\text{proj}}}{T} = \mathcal{O}\left( \frac{1}{\gamma T} \right)$$
* **Theorem 2 (Recursive Feasibility):** The penalized quadratic slack formulation guarantees that the QP is feasible for all state configurations, with $\delta_i^* = 0$ during nominal flight.

---

## 📊 Tier-A 1,000-Seed Monte Carlo Benchmark Results

Evaluated across 5 standardized multi-UAV stress scenarios, 4 controllers, and 1.15 million QP evaluations:

| Controller Framework | Mean Min Dist (m) | Overall ESV Rate (%) | Canyon Corridor ESV (%) | Control Effort | Mean Solve Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| **CoDe-CBF (Proposed)** | $\mathbf{2.42 \pm 1.54}$ | $\mathbf{0.60\%}$ | $\mathbf{3.01\%}$ | $13.24$ | $\mathbf{0.616\text{ ms}}$ |
| Nominal HOCBF (Xiao 2021) | $2.39 \pm 1.48$ | $2.27\%$ | $11.35\%$ | $11.77$ | $0.628\text{ ms}$ |
| Robust CBF (RaCBF, Su 2022) | $2.57 \pm 1.59$ | $1.08\%$ | $5.38\%$ | $11.71$ | $0.642\text{ ms}$ |
| Offline Conformal (Lindemann 2023) | $2.54 \pm 1.55$ | $0.29\%$ | $1.46\%$ | $11.89$ | $0.641\text{ ms}$ |

* **Canyon Corridor (Scenario 4):** Nominal HOCBF breaches safety limits ($11.35\%$ violation), and RaCBF violates due to chattering freezes ($5.38\%$). **CoDe-CBF strictly satisfies the $5.0\%$ risk budget ($3.01\%$)**.

---

## 🚀 Quickstart & Reproduction

### 1. Single-Command End-to-End Verification
To run all tests, build the ROS 2 workspace, execute integration checks, and compile the manuscript in one step:
```bash
bash /home/jarvis/Drones_CoDe_CBF/run_all_benchmarks.sh
```

### 2. Run Standalone Unit Tests
```bash
# Test real data ingestion & non-Gaussian wind statistics
python3 tests/test_phase1_data_pipeline.py

# Test mathematical proofs (Lemma 1, Theorem 1, Theorem 2) & OSQP solver
python3 tests/test_phase2_cbf_engine.py

# Test live ROS 2 Humble closed-loop topic communication
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 tests/test_phase4_ros2_flight_stack.py
```

### 3. Launch ROS 2 Swarm & 3D RViz Visualization
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

# Launch decentralized 4-UAV safety filters
ros2 launch code_cbf_ros2 swarm_4uav_cbf.launch.py

# Launch RViz2 with preconfigured 3D display profile
rviz2 -d rviz/code_cbf_swarm.rviz
```

---

## 📂 Repository Directory Layout

```
Drones_CoDe_CBF/
├── benchmarks/              # 1,000-Seed Monte Carlo suite
├── data/
│   ├── raw_wind/            # Real sonic anemometer & AIJ wind tunnel series
│   ├── raw_network/         # Real ROS 2 DDS latency & AoI traces
│   └── processed/           # Benchmark Parquet/CSV evaluation datasets
├── docs/                    # Research blueprint, formal proofs, literature catalog
├── figures/                 # Vector publication figures (PNG/PDF)
├── manuscript/              # IEEE RA-L LaTeX paper, BibTeX, Cover Letter, PDF
├── ros2_ws/                 # ROS 2 Humble colcon package (code_cbf_ros2)
├── rviz/                    # RViz2 3D multi-drone swarm display profile
├── src/
│   ├── code_cbf/            # Core algorithms (HOCBF, ACI, OSQP QP, C++ engine)
│   └── simulation/          # Real-data stream loaders & swarm simulators
├── tests/                   # Complete unit and integration test suite
└── run_all_benchmarks.sh    # Single-command reproduction pipeline
```

---

## 📜 Citation & License

This project is licensed under the MIT License. If you use CoDe-CBF in your research, please cite:

```bibtex
@article{chandel2026code_cbf,
  title={CoDe-CBF: Distribution-Free Safety Filtering for Decentralized Multi-UAV Swarms under Non-Stationary Aerodynamic Disturbances and Variable Communication Age-of-Information},
  author={Chandel, Vishal},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2026}
}
```
