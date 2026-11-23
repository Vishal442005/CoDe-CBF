#pragma once

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace code_cbf {

struct NeighborState {
    int id;
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    int aoi_ticks;
};

struct FilterResult {
    Eigen::Vector3d u_safe;
    double delta_slack;
    double solve_time_ms;
    double effective_margin;
    bool is_optimal;
};

class AdaptiveConformalInference {
public:
    AdaptiveConformalInference(
        double alpha_target = 0.05,
        double gamma = 0.005,
        double alpha_min = 0.01,
        double alpha_max = 0.20,
        int window_size = 100,
        double decay_beta = 0.98,
        double lambda_aoi = 1.5
    ) : alpha_target_(alpha_target), gamma_(gamma),
        alpha_min_(alpha_min), alpha_max_(alpha_max),
        window_size_(window_size), decay_beta_(decay_beta),
        lambda_aoi_(lambda_aoi), alpha_t_(alpha_target) {}

    double getEffectiveMargin(int aoi_ticks, double dt) {
        double q_tilde = 0.15;
        if (scores_.size() >= 5) {
            std::vector<double> sorted_scores(scores_.begin(), scores_.end());
            std::sort(sorted_scores.begin(), sorted_scores.end());
            int idx = static_cast<int>(std::floor((1.0 - alpha_t_) * sorted_scores.size()));
            idx = std::clamp(idx, 0, static_cast<int>(sorted_scores.size()) - 1);
            q_tilde = sorted_scores[idx];
        }
        double aoi_factor = 1.0 + lambda_aoi_ * (aoi_ticks * dt);
        return aoi_factor * q_tilde;
    }

    void update(double raw_residual, int aoi_ticks, double q_hat, double dt) {
        double aoi_factor = 1.0 + lambda_aoi_ * (aoi_ticks * dt);
        double s_t = raw_residual / std::max(1e-6, aoi_factor);
        scores_.push_back(s_t);
        if (scores_.size() > static_cast<size_t>(window_size_)) {
            scores_.pop_front();
        }

        double err_t = (raw_residual > q_hat) ? 1.0 : 0.0;
        double unprojected = alpha_t_ + gamma_ * (alpha_target_ - err_t);
        alpha_t_ = std::clamp(unprojected, alpha_min_, alpha_max_);
    }

    double getAlpha() const { return alpha_t_; }

private:
    double alpha_target_;
    double gamma_;
    double alpha_min_;
    double alpha_max_;
    int window_size_;
    double decay_beta_;
    double lambda_aoi_;
    double alpha_t_;
    std::deque<double> scores_;
};

class CoDeCBFEngine {
public:
    CoDeCBFEngine(
        double u_max = 6.0,
        double v_max = 4.0,
        double dt = 0.02,
        double r_safe = 0.8,
        double kappa_z = 2.0,
        double rho_slack = 2000.0
    ) : u_max_(u_max), v_max_(v_max), dt_(dt),
        r_safe_(r_safe), kappa_z_(kappa_z), rho_slack_(rho_slack) {
        E_ = Eigen::DiagonalMatrix<double, 3>(1.0, 1.0, 1.0 / (kappa_z * kappa_z));
    }

    FilterResult solveFilter(
        const Eigen::Vector3d& p_i,
        const Eigen::Vector3d& v_i,
        const Eigen::Vector3d& u_nom,
        const std::vector<NeighborState>& neighbors
    );

private:
    double u_max_;
    double v_max_;
    double dt_;
    double r_safe_;
    double kappa_z_;
    double rho_slack_;
    Eigen::DiagonalMatrix<double, 3> E_;
    std::vector<AdaptiveConformalInference> aci_trackers_;
};

} // namespace code_cbf
