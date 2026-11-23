# CoDe-CBF v1.0.0: Initial Research & Reproducibility Release

This release contains the complete research artifact and reproducibility package for **CoDe-CBF: Conformalized Delay-Aware Control Barrier Functions for Decentralized Multi-UAV Swarms under Non-Stationary Aerodynamic Disturbances and Intermittent Communication**.

## 📦 What is Included in this Release

1. **Complete Source Code:**
   - Online Adaptive Conformal Inference (ACI) engine (`src/code_cbf/adaptive_conformal.py`).
   - Dynamic tilt-aware rotor slipstream metric tensor (`src/code_cbf/barrier_functions.py`).
   - High-performance OSQP solver integration and C++ header (`src/code_cbf/cpp/code_cbf_solver.hpp`).
   - Tangential rotational deadlock resolver (`src/code_cbf/deadlock_breaker.py`).

2. **ROS 2 Humble Stack:**
   - Full ROS 2 package (`ros2_ws/src/code_cbf_ros2`) with multi-UAV safety filter nodes and launch scripts.
   - RViz 3D display configuration (`rviz/code_cbf_swarm.rviz`).

3. **Empirical Datasets (100% Real-World Data):**
   - Real 10m sonic anemometer 3D wind turbulence records (`data/raw_wind/`).
   - Empirical Holybro x500 rotor downwash acceleration measurements (`data/raw_downwash/`).
   - Recorded ROS 2 DDS communication latency and AoI drop traces (`data/raw_network/`).

4. **1,000-Seed Monte Carlo Benchmark Suite:**
   - 5 standardized stress scenarios: Canyon Corridor Transit, Inverted Crossing, Rotor Enclosure, 90-Degree Intersection, and Sudden Wind Gust Shocks.
   - Comparative evaluation against Nominal HOCBF, Robust CBF (RaCBF), and Offline Conformal CBF.

5. **Publication Materials:**
   - IEEE Transactions/Letters format LaTeX source and compiled PDF manuscript (`manuscript/code_cbf_manuscript.pdf`).
   - 8 high-resolution publication figures (300 DPI, zero overlap).
   - Pre-submission reviewer rebuttal guide and submission cover letter.

6. **Release Assets Attached:**
   - `CoDe_CBF_Full_Submission_Package.zip`: Complete standalone reproduction package.
   - `code_cbf_manuscript.pdf`: Complete compiled IEEE manuscript.
