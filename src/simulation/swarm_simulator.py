"""
Multi-UAV Swarm Simulation Engine with Real-World Aerodynamic & DDS Delays.
"""

import numpy as np
from .quadrotor_agent import QuadrotorAgent
from .data_loader import RealAerodynamicDataLoader
from src.code_cbf.cbf_qp_filter import CoDeCBFFilter
from src.code_cbf.baselines import NominalHOCBFFilter, RobustCBFFilter, OfflineConformalCBFFilter

class MultiUAVSwarmSimulator:
    def __init__(
        self,
        agents_config,           # List of dicts: [{'id': 1, 'start': [x,y,z], 'goal': [gx,gy,gz]}, ...]
        controller_type="code_cbf",
        r_safe=0.8,
        kappa_z=2.0,
        dt=0.02,
        data_loader=None
    ):
        self.dt = dt
        self.r_safe = r_safe
        self.kappa_z = kappa_z
        self.controller_type = controller_type
        self.data_loader = data_loader if data_loader is not None else RealAerodynamicDataLoader()
        
        # Instantiate agents and controller instances
        self.agents = {}
        self.controllers = {}
        
        for cfg in agents_config:
            a_id = cfg['id']
            # Instantiate controller per agent (decentralized architecture)
            if controller_type == "code_cbf":
                ctl = CoDeCBFFilter(r_safe=r_safe, kappa_z=kappa_z, dt=dt)
            elif controller_type == "nominal_hocbf":
                ctl = NominalHOCBFFilter(r_safe=r_safe, dt=dt)
            elif controller_type == "racbf":
                ctl = RobustCBFFilter(r_safe=r_safe, dt=dt)
            elif controller_type == "offline_conformal":
                ctl = OfflineConformalCBFFilter(r_safe=r_safe, dt=dt)
            elif controller_type == "unfiltered":
                ctl = None
            else:
                raise ValueError(f"Unknown controller: {controller_type}")
                
            self.controllers[a_id] = ctl
            agent = QuadrotorAgent(
                agent_id=a_id,
                init_pos=cfg['start'],
                init_vel=cfg.get('vel', [0.0, 0.0, 0.0]),
                goal_pos=cfg['goal'],
                controller_type=controller_type,
                controller_instance=ctl,
                dt=dt
            )
            self.agents[a_id] = agent
            
        # Odometry history buffer for delayed lookups: map a_id -> list of (pos, vel)
        self.odom_history = {a_id: [(agent.pos.copy(), agent.vel.copy())] for a_id, agent in self.agents.items()}
        
        # Diagnostic logging
        self.min_pairwise_distance = float('inf')
        self.safety_violation_count = 0
        self.total_pairwise_checks = 0
        self.time_history = []
        self.dist_matrix_history = []

    def run_simulation(self, max_steps=600, wind_offset=0):
        """
        Executes swarm simulation over max_steps (default 12 seconds at 50Hz).
        wind_offset: starting index into real sonic anemometer wind series.
        """
        for step in range(max_steps):
            t_curr = step * self.dt
            self.time_history.append(t_curr)
            
            # 1. Get real 3D wind disturbance from sonic anemometer dataset
            wind_acc, wind_vel = self.data_loader.get_wind_at_step(wind_offset + step)
            
            # 2. Get real DDS Age-of-Information (AoI)
            aoi_ticks, is_dropped, lat_ms = self.data_loader.get_aoi_at_step(step)
            
            # 3. Gather delayed neighbor states for each agent
            agent_commands = {}
            for a_id, agent in self.agents.items():
                neighbors_data = []
                for other_id, other_agent in self.agents.items():
                    if other_id == a_id:
                        continue
                        
                    # Retrieve delayed state from history based on real AoI
                    hist = self.odom_history[other_id]
                    hist_idx = max(0, len(hist) - 1 - aoi_ticks)
                    p_delayed, v_delayed = hist[hist_idx]
                    
                    neighbors_data.append({
                        'id': other_id,
                        'pos': p_delayed,
                        'vel': v_delayed,
                        'aoi_ticks': aoi_ticks
                    })
                    
                # 4. Compute empirical downwash from all upper quadrotors acting on this agent
                downwash_acc = np.zeros(3)
                for other_id, other_agent in self.agents.items():
                    if other_id == a_id:
                        continue
                    dw = self.data_loader.compute_downwash_acceleration(agent.pos, other_agent.pos)
                    downwash_acc += dw
                    
                # Execute agent control & integration step
                u_cmd, delta = agent.step(neighbors_data, wind_acc, downwash_acc, t_curr)
                agent_commands[a_id] = u_cmd
                
            # 5. Record evolution for online ACI quantile updating
            current_positions = {a_id: a.pos.copy() for a_id, a in self.agents.items()}
            for a_id, agent in self.agents.items():
                if self.controller_type == "code_cbf" and self.controllers[a_id] is not None:
                    neighbors_pos = {o_id: p for o_id, p in current_positions.items() if o_id != a_id}
                    self.controllers[a_id].record_step_evolution(agent.pos, neighbors_pos)
                    
            # 6. Append new odometry to history
            for a_id, agent in self.agents.items():
                self.odom_history[a_id].append((agent.pos.copy(), agent.vel.copy()))
                
            # 7. Evaluate True Safety Metrics (Pairwise distances)
            agent_ids = list(self.agents.keys())
            step_min_dist = float('inf')
            
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    id_i, id_j = agent_ids[i], agent_ids[j]
                    p_i = self.agents[id_i].pos
                    p_j = self.agents[id_j].pos
                    dist = float(np.linalg.norm(p_i - p_j))
                    
                    # Ellipsoidal downwash-aware barrier distance
                    dp = p_i - p_j
                    h_val = float(dp[0]**2 + dp[1]**2 + (dp[2]**2)/(self.kappa_z**2) - self.r_safe**2)
                    
                    self.total_pairwise_checks += 1
                    if h_val < 0 or dist < self.r_safe:
                        self.safety_violation_count += 1
                        
                    step_min_dist = min(step_min_dist, dist)
                    self.min_pairwise_distance = min(self.min_pairwise_distance, dist)
                    
            for agent in self.agents.values():
                agent.min_dist_history.append(step_min_dist)
                
            # Early termination if all agents reached goal
            if all(a.reached_goal for a in self.agents.values()):
                break
                
        # Compute summary metrics
        total_checks = max(1, self.total_pairwise_checks)
        esv_rate = self.safety_violation_count / total_checks
        
        # Flight completion metrics
        all_completed = all(a.reached_goal for a in self.agents.values())
        max_flight_time = max((a.completion_time if a.completion_time is not None else max_steps * self.dt) for a in self.agents.values())
        
        # Mean control energy
        all_energies = [np.mean([np.linalg.norm(u)**2 for u in a.u_history]) for a in self.agents.values()]
        mean_control_effort = float(np.mean(all_energies))
        
        # Mean solve time
        all_times = [t for a in self.agents.values() for t in a.solve_times]
        mean_solve_ms = float(np.mean(all_times)) if all_times else 0.0
        
        return {
            "controller": self.controller_type,
            "min_distance_m": float(self.min_pairwise_distance),
            "safety_violation_rate": float(esv_rate),
            "safety_violations_total": int(self.safety_violation_count),
            "is_safe": esv_rate <= 0.05 and self.min_pairwise_distance >= self.r_safe,
            "all_completed": bool(all_completed),
            "flight_time_s": float(max_flight_time),
            "control_effort": mean_control_effort,
            "mean_solve_time_ms": mean_solve_ms
        }
