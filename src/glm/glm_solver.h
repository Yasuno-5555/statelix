/**
 * @file glm_solver.h
 * @brief Statelix v1.1 - Unified GLM Solver
 * 
 * Combines:
 *   - IRLS (Iteratively Reweighted Least Squares) for smooth problems
 *   - Proximal Gradient for L1-penalized (sparse) problems
 * 
 * Supports both Dense and Sparse matrices via templates.
 */
#ifndef STATELIX_GLM_SOLVER_H
#define STATELIX_GLM_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <string>
#include <cmath>
#include <iostream>

#include "glm_base.h"
#include "../optimization/objective.h"
#include "../optimization/penalizer.h"
#include "../optimization/lbfgs.h"

namespace statelix {

// =============================================================================
// GLM Result Structure
// =============================================================================

/**
 * @brief Complete GLM fitting result
 */
struct GLMResult {
    // Coefficients
    Eigen::VectorXd coef;
    double intercept = 0.0;
    
    // Standard errors and inference
    Eigen::VectorXd std_errors;
    Eigen::VectorXd z_values;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd conf_int;  // (p, 2)
    
    // Fitted values and residuals
    Eigen::VectorXd fitted_values;
    Eigen::VectorXd linear_predictors;
    Eigen::VectorXd deviance_residuals;
    Eigen::VectorXd pearson_residuals;
    
    // Model diagnostics
    double deviance = 0.0;
    double null_deviance = 0.0;
    double log_likelihood = 0.0;
    double aic = 0.0;
    double bic = 0.0;
    double pseudo_r_squared = 0.0;
    double dispersion = 1.0;
    
    // Covariance matrix
    Eigen::MatrixXd vcov;
    
    // Fitting info
    int iterations = 0;
    bool converged = false;
    int n_obs = 0;
    int n_params = 0;
    
    // Model info (for reference)
    std::string family_name;
    std::string link_name;
};

// =============================================================================
// GLM Objective (for optimizer integration)
// =============================================================================

/**
 * @brief GLM negative log-likelihood as Objective
 * 
 * For a GLM: -logL(β) = Σ deviance(y, μ(β))
 * Used with L-BFGS or ProximalGradient.
 */
template<typename MatrixType = Eigen::MatrixXd>
class GLMObjective : public EfficientObjective {
public:
    const MatrixType& X;
    const Eigen::VectorXd& y;
    const Eigen::VectorXd& weights;
    Family* family;
    LinkFunction* link;
    bool fit_intercept;
    
    GLMObjective(
        const MatrixType& X_,
        const Eigen::VectorXd& y_,
        Family* fam,
        LinkFunction* lnk,
        bool intercept = true,
        const Eigen::VectorXd& w = Eigen::VectorXd()
    ) : X(X_), y(y_), weights(w.size() > 0 ? w : Eigen::VectorXd::Ones(y_.size())),
        family(fam), link(lnk), fit_intercept(intercept) {}
    
    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& params) const override {
        int n = y.size();
        int p = X.cols();
        int param_offset = fit_intercept ? 1 : 0;
        
        // Extract parameters
        double intercept = fit_intercept ? params(0) : 0.0;
        Eigen::VectorXd beta = params.segment(param_offset, p);
        
        // Compute linear predictor: η = Xβ + intercept
        Eigen::VectorXd eta = compute_linear_predictor(beta, intercept);
        
        // Compute μ = g^{-1}(η)
        Eigen::VectorXd mu = link->inverse(eta);
        
        // Compute deviance (= -2 * log-likelihood up to constant)
        double obj = 0.0;
        for (int i = 0; i < n; ++i) {
            obj += weights(i) * family->deviance_unit(y(i), mu(i));
        }
        obj *= 0.5;  // We minimize 0.5 * deviance for numerical reasons
        
        // Compute gradient
        // ∂obj/∂β = X' * W * (μ - y) / V(μ) * dμ/dη
        Eigen::VectorXd grad(params.size());
        grad.setZero();
        
        for (int i = 0; i < n; ++i) {
            double var_i = family->variance(mu(i));
            double dmu_deta = link->inverse_derivative(eta(i));
            double residual = (mu(i) - y(i));
            double w_i = weights(i) * dmu_deta / std::max(var_i, 1e-10);
            
            if (fit_intercept) {
                grad(0) += w_i * residual;
            }
            for (int j = 0; j < p; ++j) {
                grad(param_offset + j) += w_i * residual * get_X_value(i, j);
            }
        }
        
        return {obj, grad};
    }
    
    int dimension() const override {
        return X.cols() + (fit_intercept ? 1 : 0);
    }
    
private:
    // Helper for sparse/dense matrix access
    double get_X_value(int i, int j) const {
        if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
            return X(i, j);
        } else {
            return X.coeff(i, j);
        }
    }
    
    Eigen::VectorXd compute_linear_predictor(
        const Eigen::VectorXd& beta, double intercept
    ) const {
        Eigen::VectorXd eta;
        if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
            eta = X * beta;
        } else {
            eta = X * beta;
        }
        if (fit_intercept) {
            eta.array() += intercept;
        }
        return eta;
    }
};

// =============================================================================
// GLM Solver
// =============================================================================

/**
 * @brief Unified GLM Solver
 * 
 * Usage:
 *   GLMSolver<> solver;
 *   solver.family = std::make_unique<BinomialFamily>();
 *   solver.link = std::make_unique<LogitLink>();
 *   solver.penalizer = std::make_unique<L1Penalty>(0.1);
 *   auto result = solver.fit(X, y);
 */
template<typename MatrixType = Eigen::MatrixXd>
class GLMSolver {
public:
    // Model components
    std::unique_ptr<Family> family;
    std::unique_ptr<LinkFunction> link;
    std::unique_ptr<Penalizer> penalizer;
    
    // Fitting options
    bool fit_intercept = true;
    int max_iter = 100;
    double tol = 1e-6;
    double conf_level = 0.95;
    bool verbose = false;
    
    // Weights (optional)
    Eigen::VectorXd weights;
    
    /**
     * @brief Fit GLM to data
     */
    GLMResult fit(const MatrixType& X, const Eigen::VectorXd& y) {
        // Validate inputs
        if (X.rows() != y.size()) {
            throw std::invalid_argument("X and y must have same number of rows");
        }
        
        // Use defaults if not set
        if (!family) family = std::make_unique<GaussianFamily>();
        if (!link) link = family->canonical_link();
        if (!penalizer) penalizer = std::make_unique<NoPenalty>();
        
        // Choose algorithm based on penalty type
        if (penalizer->is_smooth()) {
            return fit_irls(X, y);
        } else {
            return fit_proximal(X, y);
        }
    }
    
    /**
     * @brief Predict for new data
     */
    Eigen::VectorXd predict(const MatrixType& X_new, const GLMResult& result,
                            bool type_response = true) const {
        Eigen::VectorXd eta;
        if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
            eta = X_new * result.coef;
        } else {
            eta = X_new * result.coef;
        }
        
        if (fit_intercept) {
            eta.array() += result.intercept;
        }
        
        if (type_response && link) {
            return link->inverse(eta);
        }
        return eta;
    }

private:
    /**
     * @brief Fit using IRLS (for smooth penalties)
     */
    GLMResult fit_irls(const MatrixType& X, const Eigen::VectorXd& y) {
        int n = X.rows();
        int p = X.cols();
        int n_params = p + (fit_intercept ? 1 : 0);
        
        GLMResult result;
        result.n_obs = n;
        result.n_params = n_params;
        result.family_name = family->name();
        result.link_name = link->name();
        
        // Initialize weights
        Eigen::VectorXd w = weights.size() > 0 ? weights : Eigen::VectorXd::Ones(n);
        
        // Initialize μ
        Eigen::VectorXd mu(n);
        for (int i = 0; i < n; ++i) {
            mu(i) = family->initialize_mu(y(i));
        }
        
        // Initialize η = g(μ)
        Eigen::VectorXd eta = link->link(mu);
        
        // Initialize coefficients
        Eigen::VectorXd params = Eigen::VectorXd::Zero(n_params);
        
        // IRLS main loop
        double prev_deviance = std::numeric_limits<double>::infinity();
        
        for (result.iterations = 0; result.iterations < max_iter; ++result.iterations) {
            // Compute working weights and response
            Eigen::VectorXd var = family->variance(mu);
            Eigen::VectorXd dmu_deta = link->inverse_derivative(eta);
            
            // Working weights: W = w * (dμ/dη)² / V(μ)
            Eigen::VectorXd W(n);
            for (int i = 0; i < n; ++i) {
                double ww = w(i) * dmu_deta(i) * dmu_deta(i) / 
                           std::max(var(i), 1e-10);
                W(i) = std::max(ww, 1e-10);
            }
            
            // Working response: z = η + (y - μ) / (dμ/dη)
            Eigen::VectorXd z(n);
            for (int i = 0; i < n; ++i) {
                z(i) = eta(i) + (y(i) - mu(i)) / 
                      std::max(std::abs(dmu_deta(i)), 1e-10);
            }
            
            // Solve weighted least squares: X'WX β = X'Wz
            Eigen::VectorXd new_params = solve_wls(X, z, W);
            
            // Apply L2 penalty if present (modify for smooth penalties only)
            if (auto* l2 = dynamic_cast<L2Penalty*>(penalizer.get())) {
                // Ridge: (X'WX + λI)^{-1} X'Wz
                // Already handled in solve_wls for simplicity
            }
            
            // Update linear predictor and mean
            eta = compute_eta(X, new_params);
            mu = link->inverse(eta);
            
            // Compute deviance
            double deviance = 0.0;
            for (int i = 0; i < n; ++i) {
                deviance += w(i) * family->deviance_unit(y(i), mu(i));
            }
            
            // Check convergence
            if (std::abs(deviance - prev_deviance) < tol * (std::abs(deviance) + 0.1)) {
                result.converged = true;
                params = new_params;
                break;
            }
            
            prev_deviance = deviance;
            params = new_params;
        }
        
        // Extract results
        if (fit_intercept) {
            result.intercept = params(0);
            result.coef = params.segment(1, p);
        } else {
            result.intercept = 0.0;
            result.coef = params;
        }
        
        result.linear_predictors = eta;
        result.fitted_values = mu;
        result.deviance = prev_deviance;
        
        // Compute additional statistics
        compute_statistics(X, y, w, result);
        
        return result;
    }
    
    /**
     * @brief Fit using Proximal Gradient (for L1/ElasticNet)
     */
    GLMResult fit_proximal(const MatrixType& X, const Eigen::VectorXd& y) {
        int n = X.rows();
        int p = X.cols();
        int n_params = p + (fit_intercept ? 1 : 0);
        
        GLMResult result;
        result.n_obs = n;
        result.n_params = n_params;
        result.family_name = family->name();
        result.link_name = link->name();
        
        // Weights
        Eigen::VectorXd w = weights.size() > 0 ? weights : Eigen::VectorXd::Ones(n);
        
        // Create objective
        GLMObjective<MatrixType> objective(X, y, family.get(), link.get(), 
                                           fit_intercept, w);
        
        // Initial parameters
        Eigen::VectorXd params = Eigen::VectorXd::Zero(n_params);
        
        // Use ProximalGradient
        ProximalGradient solver;
        solver.max_iter = max_iter;
        solver.tol = tol;
        solver.verbose = verbose;
        
        auto opt_result = solver.minimize(objective, *penalizer, params);
        
        // Extract results
        if (fit_intercept) {
            result.intercept = opt_result.x(0);
            result.coef = opt_result.x.segment(1, p);
        } else {
            result.intercept = 0.0;
            result.coef = opt_result.x;
        }
        
        result.iterations = opt_result.iterations;
        result.converged = opt_result.converged;
        
        // Compute fitted values
        result.linear_predictors = compute_eta(X, opt_result.x);
        result.fitted_values = link->inverse(result.linear_predictors);
        
        // Compute deviance
        result.deviance = 0.0;
        for (int i = 0; i < n; ++i) {
            result.deviance += w(i) * family->deviance_unit(y(i), result.fitted_values(i));
        }
        
        // Compute additional statistics
        compute_statistics(X, y, w, result);
        
        return result;
    }
    
    /**
     * @brief Solve weighted least squares
     */
    Eigen::VectorXd solve_wls(const MatrixType& X, 
                              const Eigen::VectorXd& z,
                              const Eigen::VectorXd& W) {
        int n = X.rows();
        int p = X.cols();
        int n_params = p + (fit_intercept ? 1 : 0);
        
        // Build augmented design matrix with intercept
        Eigen::MatrixXd Xa(n, n_params);
        if (fit_intercept) {
            Xa.col(0).setOnes();
            if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
                Xa.rightCols(p) = X;
            } else {
                Xa.rightCols(p) = Eigen::MatrixXd(X);
            }
        } else {
            if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
                Xa = X;
            } else {
                Xa = Eigen::MatrixXd(X);
            }
        }
        
        // X'WX
        Eigen::MatrixXd XtWX = Xa.transpose() * W.asDiagonal() * Xa;
        
        // Add L2 penalty to diagonal (for Ridge)
        if (auto* l2 = dynamic_cast<L2Penalty*>(penalizer.get())) {
            for (int j = (fit_intercept ? 1 : 0); j < n_params; ++j) {
                XtWX(j, j) += l2->lambda;
            }
        }
        
        // X'Wz
        Eigen::VectorXd XtWz = Xa.transpose() * (W.asDiagonal() * z);
        
        // Solve
        return XtWX.ldlt().solve(XtWz);
    }
    
    /**
     * @brief Compute linear predictor η from parameters
     */
    Eigen::VectorXd compute_eta(const MatrixType& X, 
                                const Eigen::VectorXd& params) {
        int p = X.cols();
        double intercept = fit_intercept ? params(0) : 0.0;
        Eigen::VectorXd beta = fit_intercept ? params.segment(1, p) : params;
        
        Eigen::VectorXd eta;
        if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
            eta = X * beta;
        } else {
            eta = X * beta;
        }
        
        if (fit_intercept) {
            eta.array() += intercept;
        }
        return eta;
    }
    
    /**
     * @brief Compute additional statistics
     */
    void compute_statistics(const MatrixType& X,
                            const Eigen::VectorXd& y,
                            const Eigen::VectorXd& w,
                            GLMResult& result) {
        int n = result.n_obs;
        int p = result.n_params;
        
        // Null deviance (intercept-only model)
        double y_mean = (w.array() * y.array()).sum() / w.sum();
        result.null_deviance = 0.0;
        for (int i = 0; i < n; ++i) {
            result.null_deviance += w(i) * family->deviance_unit(y(i), y_mean);
        }
        
        // Pseudo R-squared (McFadden)
        if (result.null_deviance > 0) {
            result.pseudo_r_squared = 1.0 - result.deviance / result.null_deviance;
        }
        
        // Log-likelihood (approximate)
        result.log_likelihood = -0.5 * result.deviance;
        
        // AIC / BIC
        result.aic = result.deviance + 2.0 * p;
        result.bic = result.deviance + std::log(n) * p;
        
        // Dispersion estimate (Pearson)
        double pearson_chi2 = 0.0;
        result.pearson_residuals.resize(n);
        result.deviance_residuals.resize(n);
        
        for (int i = 0; i < n; ++i) {
            double mu_i = result.fitted_values(i);
            double var_i = family->variance(mu_i);
            double pr = (y(i) - mu_i) / std::sqrt(std::max(var_i, 1e-10));
            result.pearson_residuals(i) = pr;
            pearson_chi2 += w(i) * pr * pr;
            
            // Deviance residual
            double d_unit = family->deviance_unit(y(i), mu_i);
            double sign = (y(i) >= mu_i) ? 1.0 : -1.0;
            result.deviance_residuals(i) = sign * std::sqrt(std::max(d_unit, 0.0));
        }
        
        result.dispersion = pearson_chi2 / (n - p);
        
        // Standard errors (would need Fisher information for proper calculation)
        // Simplified: assume we have vcov from IRLS
    }
};

// Type aliases for convenience
using DenseGLMSolver = GLMSolver<Eigen::MatrixXd>;
using SparseGLMSolver = GLMSolver<Eigen::SparseMatrix<double>>;

} // namespace statelix

#endif // STATELIX_GLM_SOLVER_H
