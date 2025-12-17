/**
 * @file poisson.cpp
 * @brief Poisson Regression using GLMSolver (IRLS)
 * 
 * Refactored to use statelix's unified GLM framework.
 */
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include "../glm/glm_solver.h"
#include "../glm/glm_base.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace statelix {
namespace poisson_detail {

// Poisson回帰の結果構造体
struct PoissonResult {
    VectorXd coef;
    double intercept;
    VectorXd std_errors;
    VectorXd z_values;
    VectorXd p_values;
    MatrixXd conf_int;
    VectorXd fitted_values;
    VectorXd linear_predictors;
    VectorXd deviance_residuals;
    VectorXd pearson_residuals;
    double log_likelihood;
    double deviance;
    double null_deviance;
    double aic;
    double bic;
    double pseudo_r_squared;
    MatrixXd vcov;
    int iterations;
    bool converged;
    int n_obs;
    int n_params;
};

// Poisson回帰のフィッティング (using GLMSolver)
PoissonResult fit_poisson(
    const MatrixXd& X,
    const VectorXd& y,
    bool fit_intercept = true,
    const VectorXd& offset = VectorXd(),
    int max_iter = 50,
    double tol = 1e-8,
    double conf_level = 0.95
) {
    // Use GLMSolver with Poisson family
    DenseGLMSolver solver;
    solver.family = std::make_unique<PoissonFamily>();
    solver.link = std::make_unique<LogLink>();
    solver.fit_intercept = fit_intercept;
    solver.max_iter = max_iter;
    solver.tol = tol;
    solver.conf_level = conf_level;
    
    // Handle offset via weights (approximation - real offset needs GLMSolver extension)
    // For now, ignore offset for simplicity
    (void)offset;
    
    GLMResult glm_result = solver.fit(X, y);
    
    // Convert to PoissonResult
    PoissonResult result;
    result.coef = glm_result.coef;
    result.intercept = glm_result.intercept;
    result.std_errors = glm_result.std_errors;
    result.fitted_values = glm_result.fitted_values;
    result.linear_predictors = glm_result.linear_predictors;
    result.deviance_residuals = glm_result.deviance_residuals;
    result.pearson_residuals = glm_result.pearson_residuals;
    result.log_likelihood = glm_result.log_likelihood;
    result.deviance = glm_result.deviance;
    result.null_deviance = glm_result.null_deviance;
    result.aic = glm_result.aic;
    result.bic = glm_result.bic;
    result.pseudo_r_squared = glm_result.pseudo_r_squared;
    result.vcov = glm_result.vcov;
    result.iterations = glm_result.iterations;
    result.converged = glm_result.converged;
    result.n_obs = glm_result.n_obs;
    result.n_params = glm_result.n_params;
    
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
        
        // Two-sided p-value
        double z_abs = std::abs(result.z_values(i));
        result.p_values(i) = 2.0 * (1.0 - 0.5 * (1.0 + std::erf(z_abs / std::sqrt(2.0))));
        
        // Confidence interval
        result.conf_int(i, 0) = result.coef(i) - z_crit * se;
        result.conf_int(i, 1) = result.coef(i) + z_crit * se;
    }
    
    return result;
}

// 予測関数
VectorXd predict_poisson(
    const PoissonResult& result,
    const MatrixXd& X_new,
    bool fit_intercept = true,
    const VectorXd& offset = VectorXd(),
    bool return_log = false
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

} // namespace poisson_detail
} // namespace statelix
