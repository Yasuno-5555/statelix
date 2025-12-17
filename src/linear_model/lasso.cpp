/**
 * @file lasso.cpp
 * @brief Lasso Regression using ProximalGradient (FISTA)
 * 
 * Optimized using statelix's unified optimization framework.
 */
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "../optimization/objective.h"
#include "../optimization/penalizer.h"
#include "../optimization/lbfgs.h"

namespace statelix {

struct LassoResult {
    Eigen::VectorXd coef;    // original-scale coefficients (length p)
    double intercept;        // intercept (original scale)
    int iterations;
    bool converged;
    double objective_value;  // final objective value
};

/**
 * @brief Least Squares Objective for Proximal Gradient
 * 
 * Minimizes: 0.5 * ||y - X*beta||^2
 */
class LeastSquaresObjective : public EfficientObjective {
public:
    const Eigen::MatrixXd& X;
    const Eigen::VectorXd& y;
    
    LeastSquaresObjective(const Eigen::MatrixXd& X_, const Eigen::VectorXd& y_)
        : X(X_), y(y_) {}
    
    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& beta) const override {
        Eigen::VectorXd residual = y - X * beta;
        double val = 0.5 * residual.squaredNorm();
        Eigen::VectorXd grad = -X.transpose() * residual;
        return {val, grad};
    }
    
    int dimension() const override { return X.cols(); }
};

/**
 * @brief Fit Lasso using Proximal Gradient (FISTA)
 * 
 * Uses O(1/k²) accelerated proximal gradient method.
 */
LassoResult fit_lasso_proximal(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    double lambda,
    int max_iter = 1000,
    double tol = 1e-6,
    bool standardize = true
) {
    int n = X.rows();
    int p = X.cols();
    
    // Standardization
    Eigen::VectorXd x_mean = X.colwise().mean();
    Eigen::VectorXd x_std(p);
    Eigen::MatrixXd Xs = X;
    
    for (int j = 0; j < p; ++j) {
        Eigen::VectorXd col = X.col(j);
        double mean = x_mean(j);
        Eigen::VectorXd centered = col.array() - mean;
        double sd = std::sqrt((centered.array().square().sum()) / (n - 1.0));
        if (sd < 1e-12) sd = 1.0;
        x_std(j) = sd;
        if (standardize) {
            Xs.col(j) = centered / sd;
        }
    }
    
    double y_mean = y.mean();
    Eigen::VectorXd y_center = y.array() - y_mean;
    
    // Setup optimization
    LeastSquaresObjective objective(Xs, y_center);
    L1Penalty l1_penalty(lambda);
    
    ProximalGradient solver;
    solver.max_iter = max_iter;
    solver.tol = tol;
    solver.use_fista = true;  // O(1/k²) acceleration
    
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
    
    OptimizerResult opt_result = solver.minimize(objective, l1_penalty, beta0);
    
    // Convert back to original scale
    Eigen::VectorXd coef_orig(p);
    if (standardize) {
        for (int j = 0; j < p; ++j) {
            coef_orig(j) = opt_result.x(j) / x_std(j);
        }
    } else {
        coef_orig = opt_result.x;
    }
    
    double intercept = y_mean - x_mean.dot(coef_orig);
    
    LassoResult res;
    res.coef = coef_orig;
    res.intercept = intercept;
    res.iterations = opt_result.iterations;
    res.converged = opt_result.converged;
    res.objective_value = opt_result.min_value + l1_penalty.penalty(opt_result.x);
    return res;
}

// Keep legacy interface for backward compatibility
LassoResult fit_lasso_cd(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    double lambda,
    int max_iter = 1000,
    double tol = 1e-6,
    bool standardize = true
) {
    // Delegate to new optimized implementation
    return fit_lasso_proximal(X, y, lambda, max_iter, tol, standardize);
}

} // namespace statelix
