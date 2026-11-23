"""
Enhanced Adaptive Conformal Inference (ACI) Engine for CoDe-CBF.
Features:
1. Online quantile tracking under non-stationary aerodynamic disturbances and bursty packet loss.
2. Finite-Horizon Conformal Cumulative Regret Tracking (Theorem 3).
3. Dual-Rate Margin Decomposition: Physical Atmospheric vs. Network Staleness Attribution.
4. Strictly bounded parameter projection (Lemma 1).
"""

import numpy as np

class AdaptiveConformalTracker:
    def __init__(
        self,
        alpha_target=0.05,       # 95% target safety coverage
        gamma=0.005,             # ACI adaptation step size
        alpha_min=0.01,          # Lower projection bound
        alpha_max=0.20,          # Upper projection bound
        window_size=100,         # Sliding window length W (2.0s at 50Hz)
        decay_beta=0.98,         # Exponential decay factor beta in (0, 1]
        lambda_aoi=1.5           # Age-of-Information scaling parameter
    ):
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.window_size = window_size
        self.decay_beta = decay_beta
        self.lambda_aoi = lambda_aoi
        
        # State variables
        self.alpha_t = alpha_target
        self.score_history = []
        self.error_history = []
        self.alpha_history = [alpha_target]
        self.effective_margin_history = []
        
        # Regret & Attribution Tracking (Novelty Multipliers)
        self.cumulative_regret = 0.0
        self.regret_history = [0.0]
        self.attribution_phys_history = []
        self.attribution_net_history = []
        
        # Precompute exponential weights for sliding window
        self.weights = np.array([self.decay_beta**(self.window_size - 1 - k) for k in range(self.window_size)])

    def get_effective_margin(self, aoi_ticks, dt=0.02):
        """
        Computes the effective safety margin q_hat_t and returns
        fine-grained attribution:
        q_phys = q_tilde_t
        q_net  = lambda_aoi * (tau * dt) * q_tilde_t
        q_hat  = q_phys + q_net
        """
        if len(self.score_history) < 5:
            q_tilde = 0.15
        else:
            scores = np.array(self.score_history[-self.window_size:])
            n = len(scores)
            w = self.weights[-n:] / np.sum(self.weights[-n:])
            
            sort_idx = np.argsort(scores)
            sorted_scores = scores[sort_idx]
            cum_w = np.cumsum(w[sort_idx])
            
            target_quantile = 1.0 - self.alpha_t
            idx = np.searchsorted(cum_w, target_quantile)
            idx = min(idx, n - 1)
            q_tilde = float(sorted_scores[idx])
            
        q_phys = q_tilde
        q_net = self.lambda_aoi * (aoi_ticks * dt) * q_tilde
        q_hat = q_phys + q_net
        
        self.effective_margin_history.append(q_hat)
        self.attribution_phys_history.append(q_phys)
        self.attribution_net_history.append(q_net)
        return q_hat, q_tilde

    def update(self, raw_residual, aoi_ticks, q_hat, dt=0.02):
        """
        Updates score history, binary miscoverage indicator err_t,
        cumulative regret R_t, and online ACI stochastic gradient update.
        """
        aoi_factor = 1.0 + self.lambda_aoi * (aoi_ticks * dt)
        norm_score = raw_residual / max(1e-6, aoi_factor)
        self.score_history.append(norm_score)
        
        err_t = 1.0 if raw_residual > q_hat else 0.0
        self.error_history.append(err_t)
        
        # Cumulative Conformal Regret update: R_t = sum_{k=1}^t (err_k - alpha_target)
        step_regret = err_t - self.alpha_target
        self.cumulative_regret += step_regret
        self.regret_history.append(self.cumulative_regret)
        
        # ACI parameter update with projection: Lemma 1
        unprojected_alpha = self.alpha_t + self.gamma * (self.alpha_target - err_t)
        self.alpha_t = float(np.clip(unprojected_alpha, self.alpha_min, self.alpha_max))
        self.alpha_history.append(self.alpha_t)
        
        return err_t, self.alpha_t

    def get_empirical_coverage(self):
        if not self.error_history:
            return 1.0
        return 1.0 - float(np.mean(self.error_history))

    def get_coverage_gap(self):
        if not self.error_history:
            return 0.0
        emp_violation = float(np.mean(self.error_history))
        return abs(emp_violation - self.alpha_target)
