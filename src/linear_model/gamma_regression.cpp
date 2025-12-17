/**
 * @file gamma_regression.cpp
 * @brief Gamma Regression using GLMSolver (IRLS)
 * 
 * Refactored to use statelix's unified GLM framework.
 */
#include "gamma_regression.h"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include "../glm/glm_solver.h"
#include "../glm/glm_base.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace statelix {

// Normal CDF for z-tests
static double gamma_norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Helper to create link function based on enum
static std::unique_ptr<LinkFunction> make_gamma_link(GammaLink link) {
    switch (link) {
        case GammaLink::LOG:
            return std::make_unique<LogLink>();
        case GammaLink::INVERSE:
            return std::make_unique<InverseLink>();
        case GammaLink::IDENTITY:
            return std::make_unique<IdentityLink>();
        default:
            return std::make_unique<LogLink>();
    }
}

// Gamma Regression using GLMSolver
GammaResult fit_gamma(
    const MatrixXd& X,
    const VectorXd& y,
    GammaLink link,
    bool fit_intercept,
    int max_iter,
    double tol,
    double conf_level
) {
    int n = X.rows();
    int p_original = X.cols();
    
    // Validate y > 0
    for (int i = 0; i < n; ++i) {
        if (y(i) <= 0) {
            throw std::runtime_error("Gamma regression requires all y > 0");
        }
    }
    
    // Use GLMSolver with GammaFamily
    DenseGLMSolver solver;
    solver.family = std::make_unique<GammaFamily>();
    solver.link = make_gamma_link(link);
    solver.fit_intercept = fit_intercept;
    solver.max_iter = max_iter;
    solver.tol = tol;
    solver.conf_level = conf_level;
    
    GLMResult glm_result = solver.fit(X, y);
    
    // Convert to GammaResult
    GammaResult result;
    result.n_obs = glm_result.n_obs;
    result.n_params = glm_result.n_params;
    result.link = link;
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
    
    // Estimate phi (dispersion) from Pearson residuals
    int df = n - result.n_params;
    double pearson_sum = result.pearson_residuals.squaredNorm();
    result.phi = (df > 0) ? pearson_sum / df : 1.0;
    
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
        result.p_values(i) = 2.0 * (1.0 - gamma_norm_cdf(z_abs));
        result.conf_int(i, 0) = result.coef(i) - z_crit * se;
        result.conf_int(i, 1) = result.coef(i) + z_crit * se;
    }
    
    return result;
}

// Prediction function
VectorXd predict_gamma(
    const GammaResult& result,
    const MatrixXd& X_new,
    bool fit_intercept
) {
    int n_new = X_new.rows();
    
    VectorXd eta;
    if (fit_intercept) {
        eta = VectorXd::Constant(n_new, result.intercept);
        eta += X_new * result.coef;
    } else {
        eta = X_new * result.coef;
    }
    
    auto link = make_gamma_link(result.link);
    return link->inverse(eta);
}

} // namespace statelix
