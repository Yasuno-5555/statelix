/**
 * @file negative_binomial.cpp
 * @brief Negative Binomial Regression using GLMSolver (IRLS)
 * 
 * Refactored to use statelix's unified GLM framework.
 */
#include "negative_binomial.h"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include "../glm/glm_solver.h"
#include "../glm/glm_base.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace statelix {

// Normal CDF
static double nb_norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// θ estimation from moment method
static double estimate_theta_from_residuals(const VectorXd& y, const VectorXd& mu) {
    int n = y.size();
    double mean_y = y.mean();
    double var_y = (y.array() - mean_y).square().sum() / (n - 1.0);
    double mean_mu = mu.mean();
    
    double theta = mean_mu * mean_mu / std::max(1e-6, var_y - mean_mu);
    if (theta <= 0 || std::isnan(theta) || std::isinf(theta)) {
        theta = 1.0;
    }
    return std::clamp(theta, 0.01, 1000.0);
}

// Negative Binomial Regression using GLMSolver
NegativeBinomialResult fit_negative_binomial(
    const MatrixXd& X,
    const VectorXd& y,
    bool fit_intercept,
    const VectorXd& offset,
    double theta_init,
    int max_iter,
    double tol,
    double conf_level
) {
    int n = X.rows();
    int p_original = X.cols();
    (void)offset;  // Not supported in current GLMSolver
    
    // Use GLMSolver with NegativeBinomialFamily
    auto nb_family = std::make_unique<NegativeBinomialFamily>(theta_init);
    
    DenseGLMSolver solver;
    solver.family = std::move(nb_family);
    solver.link = std::make_unique<LogLink>();
    solver.fit_intercept = fit_intercept;
    solver.max_iter = max_iter;
    solver.tol = tol;
    solver.conf_level = conf_level;
    
    GLMResult glm_result = solver.fit(X, y);
    
    // Convert to NegativeBinomialResult
    NegativeBinomialResult result;
    result.n_obs = glm_result.n_obs;
    result.n_params = glm_result.n_params + 1;  // +1 for theta
    result.coef = glm_result.coef;
    result.intercept = glm_result.intercept;
    result.std_errors = glm_result.std_errors;
    result.fitted_values = glm_result.fitted_values;
    result.linear_predictors = glm_result.linear_predictors;
    result.deviance_residuals = glm_result.deviance_residuals;
    result.pearson_residuals = glm_result.pearson_residuals;
    result.deviance = glm_result.deviance;
    result.null_deviance = glm_result.null_deviance;
    result.log_likelihood = glm_result.log_likelihood;
    result.pseudo_r_squared = glm_result.pseudo_r_squared;
    result.aic = glm_result.aic;
    result.bic = glm_result.bic;
    result.vcov = glm_result.vcov;
    result.iterations = glm_result.iterations;
    result.converged = glm_result.converged;
    
    // Re-estimate theta from fitted values
    result.theta = estimate_theta_from_residuals(y, glm_result.fitted_values);
    
    // z-values and p-values
    int p = result.coef.size();
    result.z_values.resize(p);
    result.p_values.resize(p);
    result.conf_int.resize(p, 2);
    
    double z_crit = 1.96;
    if (conf_level > 0.99) z_crit = 2.576;
    else if (conf_level < 0.95) z_crit = 1.645;
    
    for (int i = 0; i < p; ++i) {
        double se = (result.std_errors.size() > i) ? result.std_errors(i) : 1.0;
        result.z_values(i) = result.coef(i) / std::max(se, 1e-10);
        double z_abs = std::abs(result.z_values(i));
        result.p_values(i) = 2.0 * (1.0 - nb_norm_cdf(z_abs));
        result.conf_int(i, 0) = result.coef(i) - z_crit * se;
        result.conf_int(i, 1) = result.coef(i) + z_crit * se;
    }
    
    return result;
}

// Prediction function
VectorXd predict_negative_binomial(
    const NegativeBinomialResult& result,
    const MatrixXd& X_new,
    bool fit_intercept,
    const VectorXd& offset,
    bool return_log
) {
    int n_new = X_new.rows();
    
    VectorXd offset_vec;
    if (offset.size() == 0) {
        offset_vec = VectorXd::Zero(n_new);
    } else {
        offset_vec = offset;
    }
    
    VectorXd eta;
    if (fit_intercept) {
        eta = VectorXd::Constant(n_new, result.intercept);
        eta += X_new * result.coef + offset_vec;
    } else {
        eta = X_new * result.coef + offset_vec;
    }
    
    if (return_log) {
        return eta;
    } else {
        return eta.array().exp();
    }
}

} // namespace statelix
