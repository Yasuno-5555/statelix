/**
 * @file cox.cpp
 * @brief Cox Proportional Hazards using LBFGSOptimizer
 * 
 * Optimized using statelix's unified optimization framework.
 * Maximizes partial log-likelihood using L-BFGS.
 */
#include "cox.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include "../optimization/objective.h"
#include "../optimization/lbfgs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace statelix {

// Normal CDF for p-values
static double cox_normal_cdf(double value) {
   return 0.5 * erfc(-value * M_SQRT1_2);
}

static double cox_calculate_p_value(double z) {
    return 2.0 * (1.0 - cox_normal_cdf(std::abs(z)));
}

/**
 * @brief Cox Partial Log-Likelihood Objective
 * 
 * Minimizes: -partial_log_likelihood(beta)
 * Uses Breslow approximation for ties.
 */
class CoxObjective : public EfficientObjective {
public:
    const Eigen::MatrixXd& X;
    const Eigen::VectorXd& time;
    const Eigen::VectorXi& status;
    std::vector<std::pair<int, int>> sorted_indices;  // (original_idx, is_event)
    
    CoxObjective(const Eigen::MatrixXd& X_, const Eigen::VectorXd& time_, 
                 const Eigen::VectorXi& status_)
        : X(X_), time(time_), status(status_) {
        // Sort by time
        int n = time.size();
        std::vector<std::tuple<double, int, int>> obs(n);
        for (int i = 0; i < n; ++i) {
            obs[i] = {time(i), status(i), i};
        }
        std::sort(obs.begin(), obs.end());
        
        sorted_indices.resize(n);
        for (int i = 0; i < n; ++i) {
            sorted_indices[i] = {std::get<2>(obs[i]), std::get<1>(obs[i])};
        }
    }
    
    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& beta) const override {
        int n = X.rows();
        int p = X.cols();
        
        Eigen::VectorXd exp_xb = (X * beta).array().exp();
        
        // Risk set accumulators (start with all, subtract as we go)
        double S0 = exp_xb.sum();
        Eigen::VectorXd S1 = Eigen::VectorXd::Zero(p);
        for (int i = 0; i < n; ++i) {
            S1 += exp_xb(i) * X.row(i).transpose();
        }
        
        double neg_log_lik = 0.0;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(p);
        
        // Process in sorted order
        int i = 0;
        while (i < n) {
            double current_time = time(sorted_indices[i].first);
            
            // Find all events at this time
            std::vector<int> event_indices;
            std::vector<int> all_at_time;
            int j = i;
            while (j < n && std::abs(time(sorted_indices[j].first) - current_time) < 1e-9) {
                int orig_idx = sorted_indices[j].first;
                if (sorted_indices[j].second == 1) {
                    event_indices.push_back(orig_idx);
                }
                all_at_time.push_back(orig_idx);
                ++j;
            }
            
            // Breslow: use same risk set for all events at this time
            if (!event_indices.empty() && S0 > 1e-12) {
                int d = event_indices.size();
                Eigen::VectorXd x_bar = S1 / S0;
                
                // Log-likelihood contribution
                for (int idx : event_indices) {
                    neg_log_lik -= X.row(idx).dot(beta);
                }
                neg_log_lik += d * std::log(S0);
                
                // Gradient contribution
                Eigen::VectorXd sum_x_events = Eigen::VectorXd::Zero(p);
                for (int idx : event_indices) {
                    sum_x_events += X.row(idx).transpose();
                }
                grad -= sum_x_events;
                grad += d * x_bar;
            }
            
            // Remove from risk set
            for (int idx : all_at_time) {
                double t_val = exp_xb(idx);
                S0 -= t_val;
                S1 -= t_val * X.row(idx).transpose();
            }
            if (S0 < 0) S0 = 0;
            
            i = j;
        }
        
        return {neg_log_lik, grad};
    }
    
    int dimension() const override { return X.cols(); }
};

CoxResult CoxPH::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& time, 
                     const Eigen::VectorXi& status) {
    int n = X.rows();
    int p = X.cols();
    
    // Setup optimization
    CoxObjective objective(X, time, status);
    
    LBFGSOptimizer optimizer;
    optimizer.max_iter = max_iter;
    optimizer.epsilon = tol;
    
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
    
    OptimizerResult opt_result = optimizer.minimize(objective, beta0);
    
    CoxResult result;
    result.coef = opt_result.x;
    result.log_likelihood = -opt_result.min_value;
    result.iterations = opt_result.iterations;
    result.converged = opt_result.converged;
    
    // Compute Hessian for standard errors (numerical approximation)
    double eps = 1e-5;
    Eigen::MatrixXd hessian(p, p);
    for (int i = 0; i < p; ++i) {
        for (int j = i; j < p; ++j) {
            Eigen::VectorXd beta_pp = result.coef, beta_pm = result.coef;
            Eigen::VectorXd beta_mp = result.coef, beta_mm = result.coef;
            beta_pp(i) += eps; beta_pp(j) += eps;
            beta_pm(i) += eps; beta_pm(j) -= eps;
            beta_mp(i) -= eps; beta_mp(j) += eps;
            beta_mm(i) -= eps; beta_mm(j) -= eps;
            
            double f_pp = objective.value(beta_pp);
            double f_pm = objective.value(beta_pm);
            double f_mp = objective.value(beta_mp);
            double f_mm = objective.value(beta_mm);
            
            hessian(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps);
            hessian(j, i) = hessian(i, j);
        }
    }
    
    // Covariance = Hessian^{-1}
    result.covariance = hessian.ldlt().solve(Eigen::MatrixXd::Identity(p, p));
    result.std_error = result.covariance.diagonal().cwiseSqrt();
    
    // Z-scores and p-values
    result.z_score.resize(p);
    result.p_values.resize(p);
    for (int k = 0; k < p; ++k) {
        if (result.std_error(k) > 1e-12) {
            result.z_score(k) = result.coef(k) / result.std_error(k);
            result.p_values(k) = cox_calculate_p_value(result.z_score(k));
        } else {
            result.z_score(k) = 0.0;
            result.p_values(k) = 1.0;
        }
    }
    
    return result;
}

} // namespace statelix
