
#include "ols.h"
#include "solver.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace statelix {

Eigen::VectorXd fit_ols_qr(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() < X.cols()) throw std::runtime_error("Size error: obs < vars");
    return X.colPivHouseholderQr().solve(y); 
}

OLSResult fit_ols_full(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    bool fit_intercept,
    double conf_level
) {
    OLSResult result;
    int n = X.rows();
    int p = X.cols();
    result.n_obs = n;
    
    // Build design matrix with intercept
    Eigen::MatrixXd X_design;
    if (fit_intercept) {
        X_design.resize(n, p + 1);
        X_design.col(0) = Eigen::VectorXd::Ones(n);
        X_design.rightCols(p) = X;
    } else {
        X_design = X;
    }
    
    // Use WeightedSolver for numerical stability (unit weights for OLS)
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
    WeightedDesignMatrix wdm(X_design, weights);
    WeightedSolver solver(SolverStrategy::AUTO);
    
    Eigen::VectorXd beta = solver.solve(wdm, y);
    result.n_params = beta.size();
    
    if (fit_intercept) {
        result.intercept = beta(0);
        result.coef = beta.tail(p);
    } else {
        result.intercept = 0.0;
        result.coef = beta;
    }
    
    // Stats Calculation
    Eigen::VectorXd y_pred = X_design * beta;
    Eigen::VectorXd residuals = y - y_pred;
    result.residuals = residuals;
    result.fitted_values = y_pred;
    
    double sse = residuals.squaredNorm();
    double y_mean = y.mean();
    double sst = (y.array() - y_mean).square().sum();
    
    // R-squared
    if (sst > 1e-12) {
        result.r_squared = 1.0 - (sse / sst);
        result.adj_r_squared = 1.0 - (1.0 - result.r_squared) * (n - 1) / (n - result.n_params);
    } else {
        result.r_squared = (sse < 1e-12) ? 1.0 : 0.0;
        result.adj_r_squared = result.r_squared;
    }
    
    // Residual standard error
    int df = n - result.n_params;
    result.residual_std_error = (df > 0) ? std::sqrt(sse / df) : 0.0;
    
    // Variance-Covariance matrix via WeightedSolver
    try {
        Eigen::MatrixXd vcov_raw = solver.variance_covariance();
        result.vcov = result.residual_std_error * result.residual_std_error * vcov_raw;
        
        // Standard errors
        result.std_errors.resize(result.n_params);
        for (int j = 0; j < result.n_params; ++j) {
            result.std_errors(j) = std::sqrt(result.vcov(j, j));
        }
        
        // t-values
        result.t_values.resize(result.n_params);
        for (int j = 0; j < result.n_params; ++j) {
            result.t_values(j) = beta(j) / std::max(result.std_errors(j), 1e-10);
        }
    } catch (...) {
        // Fallback if vcov fails (rank deficient)
        result.std_errors = Eigen::VectorXd::Zero(result.n_params);
        result.t_values = Eigen::VectorXd::Zero(result.n_params);
    }
    
    // Log-Likelihood, AIC, BIC
    if (sse > 1e-12) {
        double log_sse_n = std::log(sse / n);
        result.log_likelihood = -0.5 * n * (std::log(2 * M_PI) + log_sse_n + 1);
        result.aic = 2 * result.n_params - 2 * result.log_likelihood;
        result.bic = result.n_params * std::log(n) - 2 * result.log_likelihood;
    } else {
        result.log_likelihood = 1e9;
        result.aic = -1e9;
        result.bic = -1e9;
    }
    
    // F-statistic
    if (df > 0 && result.n_params > 1) {
        double msr = (sst - sse) / (result.n_params - 1);
        double mse = sse / df;
        result.f_statistic = msr / std::max(mse, 1e-10);
    }
    
    return result;
}

Eigen::VectorXd predict_ols(
    const OLSResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept
) {
    if (fit_intercept) {
        return (X_new * result.coef).array() + result.intercept;
    }
    return X_new * result.coef;
}

PredictionInterval predict_with_interval(
    const OLSResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept,
    double conf_level
) {
    PredictionInterval pi;
    pi.predictions = predict_ols(result, X_new, fit_intercept);
    
    // Simplified: use residual_std_error * z for prediction interval
    // More accurate would use t-distribution and leverage
    double z = 1.96; // Approximation for 95%
    if (conf_level > 0.99) z = 2.576;
    else if (conf_level > 0.95) z = 1.96;
    else if (conf_level > 0.90) z = 1.645;
    
    double se = result.residual_std_error;
    pi.lower_bound = pi.predictions.array() - z * se;
    pi.upper_bound = pi.predictions.array() + z * se;
    
    return pi;
}

} // namespace statelix
