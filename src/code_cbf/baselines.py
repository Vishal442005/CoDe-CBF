"""
Mandatory Benchmark Baseline Safety Filters for CoDe-CBF Comparative Evaluation:
1. Nominal HOCBF (Xiao & Belta 2021) - Assumes perfect model, zero disturbance margin.
2. Robust CBF (RaCBF / Su et al. 2022) - Subtracts worst-case disturbance bound (causes deadlocks).
3. Offline Conformal CBF (Lindemann et al. 2023) - Fixed static quantile without online ACI or AoI.
"""

import time
import numpy as np
import scipy.sparse as sp
import osqp

from .barrier_functions import AnisotropicHOCBF

class NominalHOCBFFilter:
    """Nominal High-Order CBF (Xiao & Belta, IEEE TAC 2021) without disturbance compensation."""
    def __init__(self, u_max=6.0, v_max=4.0, dt=0.02, r_safe=0.8, rho_slack=2000.0):
        self.u_max = u_max
        self.v_max = v_max
        self.dt = dt
        self.rho_slack = rho_slack
        # Uses standard isotropic Euclidean distance (kappa_z = 1.0)
        self.hocbf = AnisotropicHOCBF(r_safe=r_safe, kappa_z=1.0, dt=dt)

    def filter_control(self, p_i, v_i, u_nom, neighbors):
        t_start = time.perf_counter()
        P = sp.diags([1.0, 1.0, 1.0, 2.0 * self.rho_slack], format='csc')
        q = np.array([-u_nom[0], -u_nom[1], -u_nom[2], 0.0])
        
        row_A = []
        l_bounds = []
        u_bounds = []
        
        for n in neighbors:
            p_j = n['pos']
            v_j = n.get('vel', np.zeros(3))
            a_ij, b_ij, _, _ = self.hocbf.compute_constraint(p_i, v_i, p_j, v_j, n['id'])
            # Nominal constraint: a_ij^T u + delta >= -b_ij (zero safety margin)
            row_A.append([a_ij[0], a_ij[1], a_ij[2], 1.0])
            l_bounds.append(-b_ij)
            u_bounds.append(np.inf)
            
        # Box limits
        for k in range(3):
            row_k = [0.0, 0.0, 0.0, 0.0]
            row_k[k] = 1.0
            row_A.append(row_k)
            l_bounds.append(-self.u_max)
            u_bounds.append(self.u_max)
            
        # Slack non-negative
        row_A.append([0.0, 0.0, 0.0, 1.0])
        l_bounds.append(0.0)
        u_bounds.append(np.inf)
        
        prob = osqp.OSQP()
        prob.setup(P, q, sp.csc_matrix(row_A), np.array(l_bounds), np.array(u_bounds), verbose=False)
        res = prob.solve()
        t_ms = (time.perf_counter() - t_start) * 1000.0
        
        u_safe = res.x[:3] if res.info.status_val in [1, 2] else u_nom
        delta = max(0.0, float(res.x[3])) if res.info.status_val in [1, 2] else 0.0
        return u_safe, delta, t_ms


class RobustCBFFilter:
    """Robust CBF (RaCBF) subtracting worst-case disturbance bound bar_w * ||a_ij||."""
    def __init__(self, u_max=6.0, v_max=4.0, dt=0.02, r_safe=0.8, w_bar=3.5, rho_slack=2000.0):
        self.u_max = u_max
        self.v_max = v_max
        self.dt = dt
        self.w_bar = w_bar # Worst-case bound (causes severe conservatism)
        self.rho_slack = rho_slack
        self.hocbf = AnisotropicHOCBF(r_safe=r_safe, kappa_z=1.0, dt=dt)

    def filter_control(self, p_i, v_i, u_nom, neighbors):
        t_start = time.perf_counter()
        P = sp.diags([1.0, 1.0, 1.0, 2.0 * self.rho_slack], format='csc')
        q = np.array([-u_nom[0], -u_nom[1], -u_nom[2], 0.0])
        
        row_A = []
        l_bounds = []
        u_bounds = []
        
        for n in neighbors:
            p_j = n['pos']
            v_j = n.get('vel', np.zeros(3))
            a_ij, b_ij, _, _ = self.hocbf.compute_constraint(p_i, v_i, p_j, v_j, n['id'])
            # RaCBF constraint: a_ij^T u + delta >= w_bar * ||a_ij||_2 - b_ij
            margin = self.w_bar * np.linalg.norm(a_ij)
            row_A.append([a_ij[0], a_ij[1], a_ij[2], 1.0])
            l_bounds.append(margin - b_ij)
            u_bounds.append(np.inf)
            
        for k in range(3):
            row_k = [0.0, 0.0, 0.0, 0.0]
            row_k[k] = 1.0
            row_A.append(row_k)
            l_bounds.append(-self.u_max)
            u_bounds.append(self.u_max)
            
        row_A.append([0.0, 0.0, 0.0, 1.0])
        l_bounds.append(0.0)
        u_bounds.append(np.inf)
        
        prob = osqp.OSQP()
        prob.setup(P, q, sp.csc_matrix(row_A), np.array(l_bounds), np.array(u_bounds), verbose=False)
        res = prob.solve()
        t_ms = (time.perf_counter() - t_start) * 1000.0
        
        u_safe = res.x[:3] if res.info.status_val in [1, 2] else u_nom
        delta = max(0.0, float(res.x[3])) if res.info.status_val in [1, 2] else 0.0
        return u_safe, delta, t_ms


class OfflineConformalCBFFilter:
    """Offline Split-Conformal CBF (Lindemann et al., IEEE RA-L 2023) using static quantile."""
    def __init__(self, u_max=6.0, v_max=4.0, dt=0.02, r_safe=0.8, q_static=0.45, rho_slack=2000.0):
        self.u_max = u_max
        self.v_max = v_max
        self.dt = dt
        self.q_static = q_static # Static offline quantile, no online adaptation, no AoI
        self.rho_slack = rho_slack
        self.hocbf = AnisotropicHOCBF(r_safe=r_safe, kappa_z=1.0, dt=dt)

    def filter_control(self, p_i, v_i, u_nom, neighbors):
        t_start = time.perf_counter()
        P = sp.diags([1.0, 1.0, 1.0, 2.0 * self.rho_slack], format='csc')
        q = np.array([-u_nom[0], -u_nom[1], -u_nom[2], 0.0])
        
        row_A = []
        l_bounds = []
        u_bounds = []
        
        for n in neighbors:
            p_j = n['pos']
            v_j = n.get('vel', np.zeros(3))
            a_ij, b_ij, _, _ = self.hocbf.compute_constraint(p_i, v_i, p_j, v_j, n['id'])
            # Static offline margin constraint
            row_A.append([a_ij[0], a_ij[1], a_ij[2], 1.0])
            l_bounds.append(self.q_static - b_ij)
            u_bounds.append(np.inf)
            
        for k in range(3):
            row_k = [0.0, 0.0, 0.0, 0.0]
            row_k[k] = 1.0
            row_A.append(row_k)
            l_bounds.append(-self.u_max)
            u_bounds.append(self.u_max)
            
        row_A.append([0.0, 0.0, 0.0, 1.0])
        l_bounds.append(0.0)
        u_bounds.append(np.inf)
        
        prob = osqp.OSQP()
        prob.setup(P, q, sp.csc_matrix(row_A), np.array(l_bounds), np.array(u_bounds), verbose=False)
        res = prob.solve()
        t_ms = (time.perf_counter() - t_start) * 1000.0
        
        u_safe = res.x[:3] if res.info.status_val in [1, 2] else u_nom
        delta = max(0.0, float(res.x[3])) if res.info.status_val in [1, 2] else 0.0
        return u_safe, delta, t_ms
