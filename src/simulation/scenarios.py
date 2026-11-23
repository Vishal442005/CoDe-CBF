"""
Standardized Benchmark Multi-UAV Stress Scenarios for CoDe-CBF Validation:
1. Scenario 1: Head-on 2-UAV crossing (Deadlock & symmetry stress).
2. Scenario 2: 4-UAV Antipodal circle swap (Multi-agent intersection bottleneck).
3. Scenario 3: Vertical downwash stacked crossing (Wake deficit & vertical clearance).
4. Scenario 4: Narrow canyon corridor traversal (Constricted space & conservatism).
5. Scenario 5: 6-UAV Swarm cluster traversal.
"""

import numpy as np

class ScenarioGenerator:
    @staticmethod
    def get_scenario(scenario_id, seed=0):
        np.random.seed(seed)
        
        if scenario_id == 1:
            # Scenario 1: Head-on 2-UAV crossing
            return [
                {'id': 1, 'start': [-3.5, 0.0, 1.5], 'goal': [3.5, 0.0, 1.5]},
                {'id': 2, 'start': [ 3.5, 0.0, 1.5], 'goal': [-3.5, 0.0, 1.5]}
            ]
            
        elif scenario_id == 2:
            # Scenario 2: 4-UAV Antipodal circle swap (R = 3.0m)
            R = 3.0
            angles = [0.0, np.pi/2, np.pi, 3*np.pi/2]
            configs = []
            for i, ang in enumerate(angles):
                start = [R * np.cos(ang), R * np.sin(ang), 1.5]
                # Opposite position
                opp_ang = ang + np.pi
                goal = [R * np.cos(opp_ang), R * np.sin(opp_ang), 1.5]
                configs.append({'id': i + 1, 'start': start, 'goal': goal})
            return configs
            
        elif scenario_id == 3:
            # Scenario 3: Vertical downwash stacked crossing
            # UAV 1 (Upper) at z = 2.2m, UAV 2 (Lower) at z = 1.4m (Delta z = 0.8m inside wake)
            return [
                {'id': 1, 'start': [0.0, -3.0, 2.2], 'goal': [0.0, 3.0, 2.2]},
                {'id': 2, 'start': [0.0, -2.2, 1.4], 'goal': [0.0, 3.8, 1.4]} # Lower drone follows slightly ahead
            ]
            
        elif scenario_id == 4:
            # Scenario 4: Narrow canyon corridor (Y constrained to [-1.0, 1.0])
            return [
                {'id': 1, 'start': [-4.0, -0.4, 1.5], 'goal': [4.0, -0.4, 1.5]},
                {'id': 2, 'start': [-4.0,  0.4, 1.5], 'goal': [4.0,  0.4, 1.5]},
                {'id': 3, 'start': [ 4.0, -0.3, 1.5], 'goal': [-4.0, -0.3, 1.5]},
                {'id': 4, 'start': [ 4.0,  0.3, 1.5], 'goal': [-4.0,  0.3, 1.5]}
            ]
            
        elif scenario_id == 5:
            # Scenario 5: 6-UAV Swarm cluster traversal
            R = 3.5
            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            configs = []
            for i, ang in enumerate(angles):
                start = [R * np.cos(ang), R * np.sin(ang), 1.2 + 0.15 * (i % 3)]
                opp_ang = ang + np.pi + np.random.uniform(-0.1, 0.1)
                goal = [R * np.cos(opp_ang), R * np.sin(opp_ang), 1.2 + 0.15 * ((i+1) % 3)]
                configs.append({'id': i + 1, 'start': start, 'goal': goal})
            return configs
            
        else:
            raise ValueError(f"Unknown scenario ID: {scenario_id}")
