"""
Tangential Perturbation Engine for Deadlock Freezes in Symmetric Multi-UAV Encounters.
Breaks head-on stagnation without violating convex QP feasibility.
"""

import numpy as np

class DeadlockBreaker:
    def __init__(
        self,
        epsilon_rot=1.2,         # Rotational acceleration magnitude (m/s^2)
        v_dead=0.35,             # Deadlock velocity threshold (m/s)
        d_conflict=2.0,          # Conflict radius threshold (m)
        d_goal_tol=0.4           # Distance to goal below which perturbation is disabled
    ):
        self.epsilon_rot = epsilon_rot
        self.v_dead = v_dead
        self.d_conflict = d_conflict
        self.d_goal_tol = d_goal_tol
        
        # 90-degree yaw rotation matrix R_z(pi/2)
        self.R_z = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 0.0]
        ])

    def compute_rotational_perturbation(self, p_i, v_i, p_goal_i, neighbors):
        """
        Calculates orthogonal perturbation acceleration u_rot,i.
        neighbors: list of dicts [{'pos': p_j, 'vel': v_j}, ...]
        """
        dist_to_goal = np.linalg.norm(p_i - p_goal_i)
        if dist_to_goal < self.d_goal_tol:
            return np.zeros(3)
            
        current_speed = np.linalg.norm(v_i)
        u_rot_total = np.zeros(3)
        
        for n in neighbors:
            p_j = n['pos']
            v_j = n.get('vel', np.zeros(3))
            
            dp = p_i - p_j
            dist = np.linalg.norm(dp)
            
            if dist < self.d_conflict:
                # Check for head-on velocity symmetry
                unit_dp = dp / max(1e-6, dist)
                
                # Check opposing motion: dot(v_i, p_j - p_i) > 0 and dot(v_j, p_i - p_j) > 0
                is_head_on = False
                if current_speed > 0.05:
                    v_dir = v_i / current_speed
                    if np.dot(v_dir, -unit_dp) > 0.6: # Heading directly towards neighbor
                        is_head_on = True
                        
                is_stagnant = current_speed < self.v_dead
                
                if is_head_on or is_stagnant:
                    # Tangential unit vector in horizontal plane
                    tangent = self.R_z @ unit_dp
                    # Weight perturbation inversely with distance
                    weight = (self.d_conflict - dist) / self.d_conflict
                    u_rot_total += self.epsilon_rot * weight * tangent
                    
        norm_rot = np.linalg.norm(u_rot_total)
        if norm_rot > self.epsilon_rot:
            u_rot_total = (u_rot_total / norm_rot) * self.epsilon_rot
            
        return u_rot_total
