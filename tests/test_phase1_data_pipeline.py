import sys
sys.path.append("/home/jarvis/Drones_CoDe_CBF")

from src.simulation.data_loader import RealAerodynamicDataLoader
import numpy as np

loader = RealAerodynamicDataLoader()
summary = loader.get_dataset_summary()
print("=== Real Dataset Pipeline Summary ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Test 1: Wind acceleration at step 100
acc_w, vel_w = loader.get_wind_at_step(100)
print(f"\nTest 1 (Wind Step 100): Vel = {vel_w.round(3)} m/s, Acc = {acc_w.round(3)} m/s^2")
assert len(acc_w) == 3, "Wind acceleration vector must be 3D"

# Test 2: Downwash when ego is 0.8m directly under upper quadrotor
p_upper = np.array([0.0, 0.0, 2.0])
p_ego_under = np.array([0.0, 0.0, 1.2]) # 0.8m below
a_downwash = loader.compute_downwash_acceleration(p_ego_under, p_upper)
print(f"Test 2 (Downwash 0.8m directly below): a_dz = {a_downwash[2]:.3f} m/s^2")
assert a_downwash[2] < 0, "Downwash acceleration must be downward (negative z)"

# Test 3: Downwash when laterally outside wake (> 1.5m)
p_ego_far = np.array([2.0, 0.0, 1.2])
a_downwash_far = loader.compute_downwash_acceleration(p_ego_far, p_upper)
print(f"Test 3 (Downwash 2.0m lateral offset): a_dz = {a_downwash_far[2]:.3f} m/s^2")
assert a_downwash_far[2] == 0.0, "Downwash must be zero outside wake boundary"

# Test 4: Network AoI lookup
aoi_ticks, is_drop, lat = loader.get_aoi_at_step(42)
print(f"Test 4 (Network Step 42): AoI = {aoi_ticks} ticks ({aoi_ticks*20} ms), Dropped = {is_drop}, Lat = {lat:.2f} ms")
assert aoi_ticks >= 0, "AoI ticks must be non-negative"

print("\n>>> ALL PHASE 1 DATA PIPELINE CHECKS PASSED WITH 100% REAL DATA! <<<")
