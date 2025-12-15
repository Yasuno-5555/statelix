#include "cox.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>

// Define M_PI if not defined (MSVC sometimes doesn't)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace statelix {

// Helper: Normal distribution CDF for p-values
double normal_cdf(double value) {
   return 0.5 * erfc(-value * M_SQRT1_2);
}

double CoxPH::calculate_p_value(double z) {
    // Two-tailed p-value
    return 2.0 * (1.0 - normal_cdf(std::abs(z)));
}

struct Observation {
    int id;
    double time;
    int status; // 1 = event, 0 = censored
};

CoxResult CoxPH::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& time, const Eigen::VectorXi& status) {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    
    // 1. Sort data by time ascending
    // User requested: "Sort Ascending (death early first implies handle ties correctly)"
    // Handling ties efficiently requires grouping.
    // For the loop logic "Start with everyone, remove as we go":
    // Sort such that we process strictly in time order.
    // Order between ties at same time: Usually doesn't matter for the "Group" approach
    // because we identify the block of same-time events.
    
    std::vector<Observation> obs(n);
    for(int i=0; i<n; ++i) {
        obs[i] = {i, time[i], status[i]};
    }
    
    std::sort(obs.begin(), obs.end(), [](const Observation& a, const Observation& b) {
        return a.time < b.time;
    });

    bool converged = false;
    int iter_count = 0;
    
    // Final statistics
    Eigen::MatrixXd info_matrix;
    double current_log_lik = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        iter_count = iter + 1;
        Eigen::VectorXd theta = (X * coef).array().exp();
        
        // Gradient (Score) and Information Matrix (Negative Hessian)
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd info = Eigen::MatrixXd::Zero(p, p);
        double log_lik = 0.0;

        // Initialize Risk Set Accumulators with EVERYONE (Ascending/Subtractive approach)
        // S0 = sum(theta)
        // S1 = sum(theta * x)
        // S2 = sum(theta * x * x')
        double S0 = 0.0;
        Eigen::VectorXd S1 = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd S2 = Eigen::MatrixXd::Zero(p, p);

        // Pre-calculate sums for stability (or just loop)
        // Since we modify sums inside the loop, we start with total sum.
        for(int i=0; i<n; ++i) {
            int original_idx = obs[i].id;
            double t_val = theta(original_idx);
            Eigen::VectorXd x_val = X.row(original_idx);
            
            S0 += t_val;
            S1 += t_val * x_val;
            S2 += t_val * (x_val * x_val.transpose());
        }

        // Iterate through unique times
        int i = 0;
        while(i < n) {
            double current_time = obs[i].time;
            
            // Identify the block of events/censoring at this time
            int j = i;
            std::vector<int> death_indices;
            std::vector<int> all_indices_at_t;

            while(j < n && std::abs(obs[j].time - current_time) < 1e-9) {
                if (obs[j].status == 1) {
                    death_indices.push_back(obs[j].id);
                }
                all_indices_at_t.push_back(obs[j].id);
                j++;
            }

            // Breslow Approximation:
            // For all deaths at time t, use the same Risk Set stats (Before removal)
            double deaths = (double)death_indices.size();
            
            if (deaths > 0 && S0 > 1e-12) {
                // S0, S1, S2 currently hold the sums for risk set R(current_time)
                // (Everyone who survived at least up to current_time)
                
                // Add contributions for each death
                Eigen::VectorXd x_bar = S1 / S0;
                
                // Log-Likelihood: sum(X_i * beta) - d * log(S0)
                for(int d_idx : death_indices) {
                    log_lik += X.row(d_idx).dot(coef);
                }
                log_lik -= deaths * std::log(S0);

                // Gradient: sum(X_i - x_bar)
                //         = sum(X_i) - d * x_bar
                Eigen::VectorXd sum_x_deaths = Eigen::VectorXd::Zero(p);
                for(int d_idx : death_indices) {
                    sum_x_deaths += X.row(d_idx);
                }
                grad += sum_x_deaths - (deaths * x_bar);

                // Information Matrix (Neg Hessian):
                // I = sum [ (S2/S0) - x_bar * x_bar^T ] * deaths?
                // Actually, d * Var_w(X)
                // Var_w(X) = E[XX'] - E[X]E[X]' = (S2/S0) - (S1/S0)(S1/S0)'
                Eigen::MatrixXd weighted_cov = (S2 / S0) - (x_bar * x_bar.transpose());
                info += deaths * weighted_cov;
            }

            // Remove everyone at this time from the risk set accumulators
            // (Both deaths and censored at this time are removed for the NEXT time point)
            for(int idx_to_remove : all_indices_at_t) {
                double t_val = theta(idx_to_remove);
                Eigen::VectorXd x_val = X.row(idx_to_remove);
                
                S0 -= t_val;
                S1 -= t_val * x_val;
                S2 -= t_val * (x_val * x_val.transpose());
            }

            // Numerical safety check
            if (S0 < 0) S0 = 0;

            // Advance index
            i = j;
        }
        
        current_log_lik = log_lik;
        info_matrix = info;

        // Solve for step direction: I * step = grad -> step = I^-1 * grad
        // Use LDLT for Robust Cholesky (Information matrix should be positive definite)
        Eigen::VectorXd step = info.ldlt().solve(grad);
        
        // Check convergence on the gradient norm or step size
        // User complained about step.norm(), suggested grad.norm() or likelihood
        if (grad.norm() < tol) {
            converged = true;
            break;
        }

        coef += step;
    }

    CoxResult result;
    result.coef = coef;
    result.log_likelihood = current_log_lik;
    result.iterations = iter_count;
    result.converged = converged;

    // Calculate Covariance and Standard Errors using Final Information Matrix
    // Covariance = Info^-1
    // Note: info_matrix is already computed at the last step's beta
    
    // We can just invert it or solve identity
    // Use SelfAdjointEigenSolver or LU if inverse needed explicitly
    result.covariance = info_matrix.inverse(); // For small p this is fine. For large p, maybe expensive.
    
    result.std_error = result.covariance.diagonal().cwiseSqrt();
    
    // Z-scores and P-values
    result.z_score = Eigen::VectorXd(p);
    result.p_values = Eigen::VectorXd(p);
    
    for(int k=0; k<p; ++k) {
        if(result.std_error(k) > 1e-12) {
            result.z_score(k) = result.coef(k) / result.std_error(k);
            result.p_values(k) = calculate_p_value(result.z_score(k));
        } else {
            result.z_score(k) = 0.0;
            result.p_values(k) = 1.0; 
        }
    }

    return result;
}

} // namespace statelix
