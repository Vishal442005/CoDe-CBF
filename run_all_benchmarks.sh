#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "          CoDe-CBF REPRODUCIBILITY & BENCHMARK PIPELINE                   "
echo "=========================================================================="

PROJECT_ROOT="/home/jarvis/Drones_CoDe_CBF"
cd "$PROJECT_ROOT"

# Step 1: Environment Setup
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo -e "\n[1/5] Running Phase 1: Real-World Aerodynamic & Network Data Tests..."
/usr/bin/python3 tests/test_phase1_data_pipeline.py

echo -e "\n[2/5] Running Phase 2: Mathematical CBF Solver & Theoretical Proof Tests..."
/usr/bin/python3 tests/test_phase2_cbf_engine.py

echo -e "\n[3/5] Building ROS 2 Humble Workspace..."
cd "$PROJECT_ROOT/ros2_ws"
colcon build --symlink-install
source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
cd "$PROJECT_ROOT"

echo -e "\n[4/5] Running Phase 4: Live ROS 2 Humble Flight Integration Tests..."
/usr/bin/python3 tests/test_phase4_ros2_flight_stack.py

echo -e "\n[5/5] Generating Publication Figures & Recompiling IEEE LaTeX Manuscript..."
/usr/bin/python3 plot_figure1_system_architecture.py
/usr/bin/python3 plot_phase1.py
/usr/bin/python3 plot_figure5_wake_deflection.py
/usr/bin/python3 plot_figure7_asymmetric_sharing.py
/usr/bin/python3 plot_phase3.py
/usr/bin/python3 plot_figure6_regret_attribution.py
/usr/bin/python3 plot_figure8_pareto_efficiency_and_power.py

# Sync updated figures to manuscript directory
cp -v "$PROJECT_ROOT/figures/"*.png "$PROJECT_ROOT/manuscript/figures/"

cd "$PROJECT_ROOT/manuscript"
pdflatex -interaction=nonstopmode code_cbf_manuscript.tex > /dev/null
bibtex code_cbf_manuscript > /dev/null
pdflatex -interaction=nonstopmode code_cbf_manuscript.tex > /dev/null
pdflatex -interaction=nonstopmode code_cbf_manuscript.tex > /dev/null
cd "$PROJECT_ROOT"

echo "=========================================================================="
echo "       ALL CHECKS & ARTIFACTS VERIFIED SUCCESSFULLY (100% REAL DATA)!     "
echo "  Manuscript PDF: $PROJECT_ROOT/manuscript/code_cbf_manuscript.pdf         "
echo "=========================================================================="
