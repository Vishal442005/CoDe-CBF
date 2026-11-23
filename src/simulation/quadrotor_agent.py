"""
Quadrotor Agent Model for Decentralized Multi-UAV Swarm Simulation.
Grounded in Holybro x500 Physical Parameters (1.4844 kg, 10-inch props).
"""

import numpy as np

class QuadrotorAgent:
    def __init__(
        self,
        agent_id,
        init_pos,
        init_vel,
        goal_pos,
        controller_type="code_cbf",
        controller_instance=None,
        mass=1.4844,             # kg
        u_max=6.0,               # m/s^2 max acceleration
        v_max=4.0,               # m/s max velocity
        dt=0.02                  # 50 Hz control loop
    ):
        self.agent_id = agent_id
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.controller_type = controller_type
        self.controller = controller_instance
        self.mass = mass
        self.u_max = u_max
        self.v_max = v_max
        self.dt = dt
        
        # Nominal guidance gains (PD position tracking setpoint generator)
        self.kp = 2.5
        self.kd = 2.0
        
        # Trajectory history
        self.pos_history = [self.pos.copy()]
        self.vel_history = [self.vel.copy()]
        self.u_history = []
        self.solve_times = []
        self.slack_history = []
        self.min_dist_history = []
        self.reached_goal = False
        self.completion_time = None

    def compute_nominal_control(self):
        """PD tracking control towards goal position."""
        dp = self.goal - self.pos
        dist_to_goal = np.linalg.norm(dp)
        if dist_to_goal < 0.25:
            self.reached_goal = True
            
        u_nom = self.kp * dp - self.kd * self.vel
        # Clip to acceleration bounds
        norm_u = np.linalg.norm(u_nom)
        if norm_u > self.u_max:
            u_nom = (u_nom / norm_u) * self.u_max
        return u_nom

    def step(self, neighbors_data, wind_acc, downwash_acc, current_time):
        """
        Calculates safe control setpoint, integrates dynamics under real wind & downwash.
        neighbors_data: list of dicts [{'id': j, 'pos': p_j_delayed, 'vel': v_j_delayed, 'aoi_ticks': tau}, ...]
        """
        u_nom = self.compute_nominal_control()
        
        # Apply Safety Filter
        if self.controller is not None:
            if self.controller_type == "code_cbf":
                u_cmd, delta_val, t_solve, telem = self.controller.filter_control(
                    self.pos, self.vel, u_nom, self.goal, neighbors_data
                )
            else: # Nominal, RaCBF, or Offline Conformal
                u_cmd, delta_val, t_solve = self.controller.filter_control(
                    self.pos, self.vel, u_nom, neighbors_data
                )
        else:
            u_cmd = u_nom
            delta_val = 0.0
            t_solve = 0.0
            
        self.u_history.append(u_cmd)
        self.solve_times.append(t_solve)
        self.slack_history.append(delta_val)
        
        # True physics integration under real disturbances:
        # a_total = u_cmd + w_wind(t) + sum(d_ij)
        a_total = u_cmd + wind_acc + downwash_acc
        
        # Discrete double integrator step: Eq. (1)
        self.pos += self.dt * self.vel + 0.5 * (self.dt**2) * a_total
        self.vel += self.dt * a_total
        
        # Velocity limit saturation
        speed = np.linalg.norm(self.vel)
        if speed > self.v_max:
            self.vel = (self.vel / speed) * self.v_max
            
        self.pos_history.append(self.pos.copy())
        self.vel_history.append(self.vel.copy())
        
        if not self.reached_goal and np.linalg.norm(self.goal - self.pos) < 0.3:
            self.reached_goal = True
            self.completion_time = current_time
            
        return u_cmd, delta_val
