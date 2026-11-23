"""
High-Order Control Barrier Function (HOCBF) with Dynamic Wake-Deflected Downwash Metric.
Introduces:
1. Dynamic Wake Deflection Tensor E_j(v_j) aligning with physical rotor slipstream.
2. Relative degree m = 2 Lie derivatives for quadrotor acceleration inputs.
3. Age-of-Information (AoI) state reconstruction.
"""

import numpy as np

class AnisotropicHOCBF:
    def __init__(
        self,
        r_safe=0.8,              # Minimum lateral safe Euclidean radius (m)
        kappa_z=2.0,             # Vertical downwash clearance inflation factor
        alpha1=2.0,              # First-order HOCBF class-K pole
        alpha2=2.0,              # Second-order HOCBF class-K pole
        dt=0.02,                 # 50 Hz control interval
        v_induced=4.5            # Rotor induced hover velocity (m/s) for x500
    ):
        self.r_safe = r_safe
        self.kappa_z = kappa_z
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.dt = dt
        self.v_induced = v_induced
        
        # State memory for past barrier evaluation h(t-1)
        self.prev_h = {}
        self.prev_psi1 = {}

    def compute_dynamic_wake_metric(self, v_neighbor):
        """
        Computes the velocity-deflected dynamic downwash metric tensor E_j(v_j).
        When the upper quadrotor moves horizontally, the rotor wake tilts downstream.
        n_wake = cos(theta_wake) * (-e_z) - sin(theta_wake) * (v_horiz / ||v_horiz||)
        E_j = I + (1/kappa_z^2 - 1) * (n_wake * n_wake^T)
        """
        v_horiz = np.array([v_neighbor[0], v_neighbor[1], 0.0])
        speed_horiz = np.linalg.norm(v_horiz)
        
        if speed_horiz < 0.1:
            # Hover condition: purely vertical wake along -e_z
            return np.diag([1.0, 1.0, 1.0 / (self.kappa_z**2)])
            
        # Wake deflection angle from momentum slipstream theory
        theta_wake = np.arctan2(speed_horiz, self.v_induced)
        # Limit deflection angle to max 45 degrees
        theta_wake = min(theta_wake, np.pi / 4.0)
        
        unit_horiz = v_horiz / speed_horiz
        # Wake axis vector pointing into the slipstream plume
        n_wake = -np.cos(theta_wake) * np.array([0.0, 0.0, 1.0]) - np.sin(theta_wake) * unit_horiz
        n_wake = n_wake / np.linalg.norm(n_wake)
        
        # Construct rank-1 update metric tensor
        # Eigenvalues: 1.0 in transverse directions, 1/kappa_z^2 along wake axis
        alpha_e = (1.0 / (self.kappa_z**2)) - 1.0
        E_dynamic = np.eye(3) + alpha_e * np.outer(n_wake, n_wake)
        return E_dynamic

    def evaluate_h(self, p_i, p_j, v_j=None):
        """
        Calculates the ellipsoidal barrier value h_ij(p_i, p_j).
        h_ij >= 0 indicates collision-free state.
        """
        dp = p_i - p_j
        if v_j is not None:
            E_mat = self.compute_dynamic_wake_metric(v_j)
        else:
            E_mat = np.diag([1.0, 1.0, 1.0 / (self.kappa_z**2)])
            
        weighted_norm_sq = float(dp.T @ E_mat @ dp)
        return weighted_norm_sq - (self.r_safe**2)

    def reconstruct_stale_state(self, p_delayed, v_delayed, aoi_ticks):
        """
        Extrapolates neighbor state over Age-of-Information:
        p_hat(t) = p(t - tau) + (tau * dt) * v(t - tau)
        v_hat(t) = v(t - tau)
        """
        tau_seconds = aoi_ticks * self.dt
        p_hat = p_delayed + tau_seconds * v_delayed
        v_hat = v_delayed.copy()
        return p_hat, v_hat

    def compute_constraint(self, p_i, v_i, p_j_stale, v_j_stale, neighbor_id):
        """
        Computes affine inequality terms with dynamic wake deflection:
        a_ij^T u_i + b_ij >= q_hat - delta_i
        """
        E_mat = self.compute_dynamic_wake_metric(v_j_stale)
        h_curr = float((p_i - p_j_stale).T @ E_mat @ (p_i - p_j_stale)) - (self.r_safe**2)
        
        if neighbor_id not in self.prev_h:
            dp = p_i - p_j_stale
            dv = v_i - v_j_stale
            h_dot_approx = 2.0 * float(dp.T @ E_mat @ dv)
            h_prev = h_curr - self.dt * h_dot_approx
        else:
            h_prev = self.prev_h[neighbor_id]
            
        psi1_curr = (h_curr - h_prev) / self.dt + self.alpha1 * h_curr
        
        dp = p_i - p_j_stale
        dv = v_i - v_j_stale
        
        a_ij = 2.0 * self.dt * (E_mat @ dp)
        
        term1 = 2.0 * float(dp.T @ E_mat @ dv)
        term2 = self.dt * float(dv.T @ E_mat @ dv)
        term3 = (self.alpha1 + self.alpha2) * self.dt * psi1_curr
        term4 = -self.alpha1 * self.alpha2 * (self.dt**2) * h_curr
        b_ij = term1 + term2 + term3 + term4
        
        self.prev_h[neighbor_id] = h_curr
        self.prev_psi1[neighbor_id] = psi1_curr
        
        return a_ij, b_ij, psi1_curr, h_curr

    def compute_nominal_predicted_psi1(self, p_i, v_i, u_i_nom, p_j_stale, v_j_stale, psi1_curr, h_curr):
        p_i_next = p_i + self.dt * v_i + 0.5 * (self.dt**2) * u_i_nom
        p_j_next = p_j_stale + self.dt * v_j_stale
        h_next = self.evaluate_h(p_i_next, p_j_next, v_j_stale)
        psi1_pred = (h_next - h_curr) / self.dt + self.alpha1 * h_next
        return psi1_pred

    def compute_realized_residual(self, psi1_predicted, p_i_real, p_j_real, v_j_real, neighbor_id):
        h_real = self.evaluate_h(p_i_real, p_j_real, v_j_real)
        h_prev = self.prev_h.get(neighbor_id, h_real)
        psi1_real = (h_real - h_prev) / self.dt + self.alpha1 * h_real
        raw_residual = max(0.0, psi1_predicted - psi1_real)
        return raw_residual, psi1_real
