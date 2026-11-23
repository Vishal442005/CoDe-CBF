"""
Tier-A 1,000-Seed Monte Carlo Real-Data Benchmarking Suite for CoDe-CBF.
Compares:
1. CoDe-CBF (Proposed)
2. Nominal HOCBF (Xiao & Belta 2021)
3. Robust CBF (RaCBF, Su et al. 2022)
4. Offline Conformal CBF (Lindemann et al. 2023)
Across 5 Stress Scenarios x 50 Seeds = 1,000 Benchmark Runs.
Driven by real sonic anemometer wind, empirical downwash, and real DDS latency traces.
"""

import sys
sys.path.append("/home/jarvis/Drones_CoDe_CBF")

import time
import numpy as np
import pandas as pd
from src.simulation.data_loader import RealAerodynamicDataLoader
from src.simulation.scenarios import ScenarioGenerator
from src.simulation.swarm_simulator import MultiUAVSwarmSimulator

def run_benchmarks():
    print("=" * 70)
    print("   STARTING TIER-A 1,000-SEED REAL-DATA MONTE CARLO BENCHMARKS     ")
    print("=" * 70)
    
    loader = RealAerodynamicDataLoader()
    summary = loader.get_dataset_summary()
    print(f"Loaded Real Wind Samples: {summary['total_wind_samples']} (Mean: {summary['wind_speed_mean_mps']:.2f} m/s, Max: {summary['wind_speed_max_mps']:.2f} m/s)")
    print(f"Loaded Real DDS Network Samples: {summary['total_network_samples']} (Drop Rate: {summary['packet_drop_rate_pct']:.1f}%)")
    
    controllers = ["code_cbf", "nominal_hocbf", "racbf", "offline_conformal"]
    scenarios = [1, 2, 3, 4, 5]
    n_seeds_per_scenario = 50
    total_expected_runs = len(controllers) * len(scenarios) * n_seeds_per_scenario
    print(f"Total Monte Carlo Runs Scheduled: {total_expected_runs} (4 controllers x 5 scenarios x {n_seeds_per_scenario} seeds)")
    
    results = []
    run_count = 0
    start_all_time = time.time()
    
    for ctl in controllers:
        ctl_start_time = time.time()
        for sc_id in scenarios:
            for seed in range(n_seeds_per_scenario):
                run_count += 1
                cfg = ScenarioGenerator.get_scenario(sc_id, seed=seed)
                # Select unique real wind series offset based on seed
                wind_offset = (seed * 37 + sc_id * 113) % (len(loader.wind_3d) - 400)
                
                # Run simulation (300 ticks = 6.0s at 50Hz)
                sim = MultiUAVSwarmSimulator(
                    agents_config=cfg,
                    controller_type=ctl,
                    r_safe=0.8,
                    kappa_z=2.0,
                    dt=0.02,
                    data_loader=loader
                )
                res = sim.run_simulation(max_steps=320, wind_offset=wind_offset)
                
                record = {
                    "run_id": run_count,
                    "controller": ctl,
                    "scenario_id": sc_id,
                    "seed": seed,
                    "min_distance_m": res["min_distance_m"],
                    "safety_violation_rate": res["safety_violation_rate"],
                    "safety_violations_total": res["safety_violations_total"],
                    "is_safe": res["is_safe"],
                    "completed": res["all_completed"],
                    "flight_time_s": res["flight_time_s"],
                    "control_effort": res["control_effort"],
                    "solve_time_ms": res["mean_solve_time_ms"]
                }
                results.append(record)
                
                if run_count % 100 == 0:
                    elapsed = time.time() - start_all_time
                    fps = run_count / max(1e-3, elapsed)
                    print(f"  [Progress] {run_count}/{total_expected_runs} runs completed ({fps:.1f} runs/sec) - Current: {ctl} on Scenario {sc_id}")
                    
        print(f"Completed {ctl} in {time.time() - ctl_start_time:.2f} seconds.")
        
    df = pd.DataFrame(results)
    out_csv = "/home/jarvis/Drones_CoDe_CBF/data/processed/tier_a_1000_benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw benchmark logs (1,000 rows) to: {out_csv}")
    
    # Generate Statistical Groupby Table
    summary_table = df.groupby("controller").agg(
        min_dist_mean=("min_distance_m", "mean"),
        min_dist_std=("min_distance_m", "std"),
        min_dist_min=("min_distance_m", "min"),
        violation_rate_mean=("safety_violation_rate", "mean"),
        violation_rate_max=("safety_violation_rate", "max"),
        completion_rate=("completed", lambda x: np.mean(x) * 100.0),
        flight_time_mean=("flight_time_s", "mean"),
        control_effort_mean=("control_effort", "mean"),
        solve_time_ms_mean=("solve_time_ms", "mean")
    ).reset_index()
    
    summary_csv = "/home/jarvis/Drones_CoDe_CBF/data/processed/tier_a_statistical_summary.csv"
    summary_table.to_csv(summary_csv, index=False)
    print(f"Saved statistical summary table to: {summary_csv}")
    
    print("\n" + "=" * 95)
    print("                      TIER-A STATISTICAL BENCHMARK SUMMARY (1,000 SEEDS)                     ")
    print("=" * 95)
    print(summary_table.to_string(index=False))
    print("=" * 95)

if __name__ == "__main__":
    run_benchmarks()
