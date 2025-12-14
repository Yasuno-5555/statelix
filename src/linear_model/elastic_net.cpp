#include "elastic_net.h"
#include "../optimization/optimization.h"
#include <iostream>
#include <numeric>

namespace statelix {

// Cyclic Coordinate Descent for Elastic Net
// Objective: 1/(2n) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
ElasticNetResult ElasticNet::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int n = X.rows();
    int p = X.cols();

    // 1. Data Standardization (Crucial for regularized regression)
    // We do this internally (simplest) or assume user did it. 
    // Usually fit_intercept=true implies we center X and y.
    
    Eigen::VectorXd X_mean = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd X_scale = Eigen::VectorXd::Ones(p);
    Eigen::MatrixXd X_centered = X;
    double y_mean = 0.0;
    Eigen::VectorXd y_centered = y;

    if (fit_intercept) {
        X_mean = X.colwise().mean();
        y_mean = y.mean();
        
        for (int j = 0; j < p; ++j) {
            X_centered.col(j) = X.col(j).array() - X_mean(j);
            // Calculate scale (std dev)
            // X_scale(j) = std::sqrt(X_centered.col(j).squaredNorm() / n); // Unbiased? Sklearn uses biased std I think or just norm
            // ElasticNet usually normalizes so sum(x^2) = N or 1.
            // Let's use standard scaling: variance = 1
            // double sq_norm = X_centered.col(j).squaredNorm();
            // if (sq_norm > 1e-12) X_scale(j) = std::sqrt(sq_norm);
            // X_centered.col(j) /= X_scale(j);
        }
        y_centered = y.array() - y_mean;
    }

    // Precompute squared norms of columns for denominator
    // If we normalized, this would be N (or 1).
    Eigen::VectorXd norm_cols_sq(p);
    for (int j = 0; j < p; ++j) {
        norm_cols_sq(j) = X_centered.col(j).squaredNorm();
    }

    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd resid = y_centered; // Residual starts as y (since coef=0)

    // Cached constants
    double l1_reg = alpha * l1_ratio;
    double l2_reg = alpha * (1.0 - l1_ratio);

    int iter = 0;
    double max_change = 0.0;

    for (; iter < max_iter; ++iter) {
        max_change = 0.0;

        for (int j = 0; j < p; ++j) {
            if (norm_cols_sq(j) < 1e-12) continue; // Constant column, ignore

            double w_j_old = coef(j);

            // Calculate partial residual correlation (without w_j)
            // resid = y - sum_{k} x_k w_k
            // We want resid_{without_j} = y - sum_{k!=j} x_k w_k = resid + x_j w_j
            // But we maintain `resid` as current full residual.
            
            // z_j = x_j^T * (resid + x_j * w_j_old)
            //     = x_j^T * resid + (x_j^T x_j) * w_j_old
            double dot_prod = X_centered.col(j).dot(resid);
            double z_j = dot_prod + norm_cols_sq(j) * w_j_old;

            // Update w_j
            // Formula: Soft(z_j, n * l1_reg) / (norm_sq + n * l2_reg)
            // Note: Scikit-learn minimizes 1/(2n) ||...||^2 + alpha * ...
            // So gradients are scaled by 1/n.
            // Coordinate descent step:
            // num = S( (1/n)*z_j, alpha*l1_ratio )
            // den = (1/n)*norm_sq + alpha*(1-l1_ratio)
            
            double rho = z_j / n; // 
            double num = optimization::soft_threshold(rho, l1_reg);
            double den = (norm_cols_sq(j) / n) + l2_reg;
            
            double w_j_new = num / den;

            if (w_j_new != w_j_old) {
                coef(j) = w_j_new;
                // Update residual: resid_new = resid_old - x_j * (w_new - w_old)
                resid -= X_centered.col(j) * (w_j_new - w_j_old);
                
                double change = std::abs(w_j_new - w_j_old);
                if (change > max_change) max_change = change;
            }
        }

        if (max_change < tol) {
            break;
        }
    }

    // Recover Intercept
    // y = Xw + c -> y_mean = X_mean * w + c -> c = y_mean - X_mean * w
    double intercept = 0.0;
    if (fit_intercept) {
        // Since we didn't scale coefs back (we used unscaled X if we commented out scaling), logic is simple.
        // If we scaled X, we must descale coefs first. I commented out scaling above for simplicity.
        intercept = y_mean - X_mean.dot(coef);
    }

    return {coef, intercept, iter, max_change};
}

} // namespace statelix
