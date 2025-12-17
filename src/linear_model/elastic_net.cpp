/**
 * @file elastic_net.cpp  
 * @brief Elastic Net Regression using ProximalGradient (FISTA)
 * 
 * Optimized using statelix's unified optimization framework.
 */
#include "elastic_net.h"
#include "../optimization/objective.h"
#include "../optimization/penalizer.h"
#include "../optimization/lbfgs.h"
#include <Eigen/Dense>
#include <cmath>

namespace statelix {

/**
 * @brief Least Squares Objective for Proximal Gradient
 */
class ElasticNetObjective : public EfficientObjective {
public:
    const Eigen::MatrixXd& X;
    const Eigen::VectorXd& y;
    
    ElasticNetObjective(const Eigen::MatrixXd& X_, const Eigen::VectorXd& y_)
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
 * @brief Fit Elastic Net using ProximalGradient (FISTA)
 */
ElasticNetResult ElasticNet::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int n = X.rows();
    int p = X.cols();

    // 1. Data Standardization
    Eigen::VectorXd X_mean = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd X_centered = X;
    double y_mean = 0.0;
    Eigen::VectorXd y_centered = y;

    if (fit_intercept) {
        X_mean = X.colwise().mean();
        y_mean = y.mean();
        
        for (int j = 0; j < p; ++j) {
            X_centered.col(j) = X.col(j).array() - X_mean(j);
        }
        y_centered = y.array() - y_mean;
    }

    // Setup optimization with ElasticNet penalty
    ElasticNetObjective objective(X_centered, y_centered);
    
    // Scale penalties for 1/n normalization (sklearn convention)
    double l1_reg = alpha * l1_ratio;
    double l2_reg = alpha * (1.0 - l1_ratio);
    ElasticNetPenalty penalty(l1_reg * n, l2_reg * n);  // Scale by n
    
    ProximalGradient solver;
    solver.max_iter = max_iter;
    solver.tol = tol;
    solver.use_fista = true;  // O(1/k²) acceleration
    
    Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
    
    OptimizerResult opt_result = solver.minimize(objective, penalty, beta0);

    // Recover Intercept
    double intercept = 0.0;
    if (fit_intercept) {
        intercept = y_mean - X_mean.dot(opt_result.x);
    }

    return {opt_result.x, intercept, opt_result.iterations, 
            opt_result.converged ? 0.0 : 1.0};  // max_change approximation
}

} // namespace statelix
