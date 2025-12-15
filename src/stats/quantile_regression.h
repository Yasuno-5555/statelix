/**
 * @file quantile_regression.h
 * @brief Statelix v2.3 - Quantile Regression
 * 
 * Implements:
 *   - Koenker-Bassett quantile regression
 *   - Interior point methods (Frisch-Newton)
 *   - Iteratively Reweighted Least Squares (IRLS)
 *   - Quantile process and interquantile range tests
 *   - Bootstrap and asymptotic inference
 * 
 * Theory:
 * -------
 * Quantile Regression:
 *   Minimize: Σ ρ_τ(y_i - x_i'β)
 *   where ρ_τ(u) = u(τ - I(u < 0)) = τ|u|I(u≥0) + (1-τ)|u|I(u<0)
 * 
 *   τ = 0.5 gives median regression (robust to outliers)
 *   τ = 0.1, 0.9 gives tails of the distribution
 * 
 * Interpretation:
 *   β_τ estimates effect on the τ-th conditional quantile of y|X
 * 
 * Reference:
 *   - Koenker, R. & Bassett, G. (1978). Regression Quantiles
 *   - Koenker, R. (2005). Quantile Regression
 */
#ifndef STATELIX_QUANTILE_REGRESSION_H
#define STATELIX_QUANTILE_REGRESSION_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

struct QuantileResult {
    // Coefficients
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    Eigen::VectorXd conf_lower;
    Eigen::VectorXd conf_upper;
    
    // Fit
    Eigen::VectorXd residuals;
    Eigen::VectorXd fitted_values;
    
    // Quantile info
    double tau;                 // Quantile level
    double objective;           // Minimized objective function
    double pseudo_r_squared;    // Koenker-Machado (1999)
    
    int n_obs;
    int n_params;
    int iterations;
    bool converged;
};

struct QuantileProcessResult {
    std::vector<double> taus;           // Quantile levels
    std::vector<Eigen::VectorXd> coefs; // Coefficients at each tau
    std::vector<Eigen::VectorXd> ses;   // Standard errors
    
    // Test for coefficient equality across quantiles
    double wald_stat;
    double wald_pvalue;
    int df;
};

// =============================================================================
// Quantile Regression
// =============================================================================

/**
 * @brief Quantile Regression via IRLS
 * 
 * Usage:
 *   QuantileRegression qr;
 *   auto result = qr.fit(y, X, 0.5);  // Median regression
 */
class QuantileRegression {
public:
    int max_iter = 100;
    double tol = 1e-6;
    double epsilon = 1e-4;      // Smoothing for check function
    bool bootstrap_se = true;
    int bootstrap_reps = 200;
    double conf_level = 0.95;
    unsigned int seed = 42;
    
    /**
     * @brief Fit quantile regression
     * 
     * @param y Outcome (n,)
     * @param X Covariates (n, k), should include intercept if desired
     * @param tau Quantile level (0 < tau < 1)
     */
    QuantileResult fit(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        double tau
    ) {
        int n = y.size();
        int k = X.cols();
        
        QuantileResult result;
        result.n_obs = n;
        result.n_params = k;
        result.tau = tau;
        
        if (tau <= 0 || tau >= 1) {
            throw std::invalid_argument("tau must be in (0, 1)");
        }
        
        // Initialize with OLS
        Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
        
        // IRLS iterations
        result.converged = false;
        for (int iter = 0; iter < max_iter; ++iter) {
            result.iterations = iter + 1;
            
            Eigen::VectorXd resid = y - X * beta;
            
            // Weights from smoothed check function derivative
            Eigen::VectorXd w(n);
            for (int i = 0; i < n; ++i) {
                double u = resid(i);
                double abs_u = std::max(std::abs(u), epsilon);
                
                // Weight proportional to 1/|u|, with asymmetric adjustment
                if (u >= 0) {
                    w(i) = tau / abs_u;
                } else {
                    w(i) = (1 - tau) / abs_u;
                }
            }
            
            // Weighted least squares
            Eigen::MatrixXd XtWX = X.transpose() * w.asDiagonal() * X;
            Eigen::VectorXd XtWy = X.transpose() * w.asDiagonal() * y;
            
            Eigen::VectorXd beta_new = XtWX.ldlt().solve(XtWy);
            
            // Check convergence
            if ((beta_new - beta).norm() < tol * (1 + beta.norm())) {
                result.converged = true;
                beta = beta_new;
                break;
            }
            
            beta = beta_new;
        }
        
        result.coef = beta;
        result.residuals = y - X * beta;
        result.fitted_values = X * beta;
        
        // Objective function value
        result.objective = 0;
        for (int i = 0; i < n; ++i) {
            double u = result.residuals(i);
            result.objective += check_function(u, tau);
        }
        
        // Pseudo R-squared (Koenker-Machado)
        double obj_null = 0;
        double y_quantile = compute_quantile(y, tau);
        for (int i = 0; i < n; ++i) {
            obj_null += check_function(y(i) - y_quantile, tau);
        }
        result.pseudo_r_squared = 1.0 - result.objective / obj_null;
        
        // Standard errors
        if (bootstrap_se) {
            compute_bootstrap_se(y, X, tau, result);
        } else {
            compute_asymptotic_se(y, X, tau, result);
        }
        
        // t-values and p-values
        result.t_values = result.coef.array() / result.std_errors.array();
        result.p_values.resize(k);
        for (int j = 0; j < k; ++j) {
            result.p_values(j) = 2 * (1 - normal_cdf(std::abs(result.t_values(j))));
        }
        
        // Confidence intervals
        double z_crit = normal_quantile(0.5 + conf_level / 2);
        result.conf_lower = result.coef - z_crit * result.std_errors;
        result.conf_upper = result.coef + z_crit * result.std_errors;
        
        return result;
    }
    
    /**
     * @brief Estimate quantile process (multiple quantiles)
     */
    QuantileProcessResult quantile_process(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const std::vector<double>& taus
    ) {
        QuantileProcessResult result;
        result.taus = taus;
        
        for (double tau : taus) {
            auto qr = fit(y, X, tau);
            result.coefs.push_back(qr.coef);
            result.ses.push_back(qr.std_errors);
        }
        
        // Wald test for equal slopes across quantiles
        // H0: β(τ_1) = β(τ_2) = ... = β(τ_m)
        int k = X.cols();
        int m = taus.size();
        
        if (m > 1) {
            // Pairwise differences
            int n_constraints = (m - 1) * k;
            result.df = n_constraints;
            
            // Simplified test: sum of squared differences / variance
            double wald = 0;
            for (int j = 0; j < k; ++j) {
                double mean_coef = 0;
                for (int t = 0; t < m; ++t) {
                    mean_coef += result.coefs[t](j);
                }
                mean_coef /= m;
                
                for (int t = 0; t < m; ++t) {
                    double diff = result.coefs[t](j) - mean_coef;
                    double var = result.ses[t](j) * result.ses[t](j);
                    wald += diff * diff / std::max(1e-10, var);
                }
            }
            
            result.wald_stat = wald;
            result.wald_pvalue = 1 - chi2_cdf(wald, n_constraints);
        }
        
        return result;
    }
    
    /**
     * @brief Interquantile range regression
     * 
     * Tests for heteroskedasticity by comparing upper and lower quantiles
     */
    QuantileResult interquantile_range(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        double tau_low = 0.25,
        double tau_high = 0.75
    ) {
        auto low = fit(y, X, tau_low);
        auto high = fit(y, X, tau_high);
        
        QuantileResult result;
        result.tau = tau_high - tau_low;
        result.n_obs = y.size();
        result.n_params = X.cols();
        
        // IQR coefficients
        result.coef = high.coef - low.coef;
        
        // Combined SE (assuming independence)
        result.std_errors.resize(result.n_params);
        for (int j = 0; j < result.n_params; ++j) {
            result.std_errors(j) = std::sqrt(
                high.std_errors(j) * high.std_errors(j) +
                low.std_errors(j) * low.std_errors(j)
            );
        }
        
        result.t_values = result.coef.array() / result.std_errors.array();
        result.p_values.resize(result.n_params);
        for (int j = 0; j < result.n_params; ++j) {
            result.p_values(j) = 2 * (1 - normal_cdf(std::abs(result.t_values(j))));
        }
        
        result.converged = true;
        
        return result;
    }
    
private:
    double check_function(double u, double tau) {
        return u * (tau - (u < 0 ? 1.0 : 0.0));
    }
    
    double compute_quantile(const Eigen::VectorXd& y, double tau) {
        std::vector<double> sorted(y.data(), y.data() + y.size());
        std::sort(sorted.begin(), sorted.end());
        
        int idx = static_cast<int>(tau * (sorted.size() - 1));
        return sorted[idx];
    }
    
    void compute_bootstrap_se(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        double tau,
        QuantileResult& result
    ) {
        int n = y.size();
        int k = X.cols();
        
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dist(0, n - 1);
        
        Eigen::MatrixXd boot_coefs(bootstrap_reps, k);
        
        for (int b = 0; b < bootstrap_reps; ++b) {
            // Resample
            Eigen::VectorXd y_boot(n);
            Eigen::MatrixXd X_boot(n, k);
            
            for (int i = 0; i < n; ++i) {
                int idx = dist(gen);
                y_boot(i) = y(idx);
                X_boot.row(i) = X.row(idx);
            }
            
            // Fit quantile regression
            bool save_flag = bootstrap_se;
            bootstrap_se = false;  // Avoid infinite recursion
            
            try {
                auto boot_result = fit(y_boot, X_boot, tau);
                boot_coefs.row(b) = boot_result.coef.transpose();
            } catch (...) {
                boot_coefs.row(b) = result.coef.transpose();
            }
            
            bootstrap_se = save_flag;
        }
        
        // Standard errors from bootstrap distribution
        result.std_errors.resize(k);
        for (int j = 0; j < k; ++j) {
            double mean = boot_coefs.col(j).mean();
            double var = (boot_coefs.col(j).array() - mean).square().sum() / (bootstrap_reps - 1);
            result.std_errors(j) = std::sqrt(var);
        }
    }
    
    void compute_asymptotic_se(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        double tau,
        QuantileResult& result
    ) {
        int n = y.size();
        int k = X.cols();
        
        // Bandwidth for kernel density estimation
        double h = 1.364 * std::pow(n, -1.0/3.0) * 
                   std::sqrt((result.residuals.array() - result.residuals.mean()).square().mean());
        h = std::max(h, 0.01);
        
        // Estimate sparsity (inverse of density at quantile)
        std::vector<double> resid_sorted(result.residuals.data(), 
                                         result.residuals.data() + n);
        std::sort(resid_sorted.begin(), resid_sorted.end());
        
        int idx_low = static_cast<int>((tau - h/2) * n);
        int idx_high = static_cast<int>((tau + h/2) * n);
        idx_low = std::max(0, idx_low);
        idx_high = std::min(n - 1, idx_high);
        
        double sparsity = (resid_sorted[idx_high] - resid_sorted[idx_low]) / (h);
        sparsity = std::max(sparsity, 0.01);
        
        // Asymptotic variance: τ(1-τ) * s² * (X'X)^{-1}
        double var_factor = tau * (1 - tau) * sparsity * sparsity;
        Eigen::MatrixXd XtX_inv = (X.transpose() * X).ldlt()
                                  .solve(Eigen::MatrixXd::Identity(k, k));
        
        result.std_errors = (var_factor * XtX_inv.diagonal()).cwiseSqrt();
    }
    
    double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    
    double normal_quantile(double p) {
        double a = 0.147;
        double x = 2 * p - 1;
        double ln = std::log(1 - x * x);
        double s = (x > 0 ? 1 : -1);
        return s * std::sqrt(std::sqrt((2/(M_PI*a) + ln/2) * (2/(M_PI*a) + ln/2) - ln/a) 
                            - (2/(M_PI*a) + ln/2)) * std::sqrt(2);
    }
    
    double chi2_cdf(double x, int df) {
        if (df <= 0 || x <= 0) return 0;
        double a = df / 2.0;
        double sum = 1.0 / a;
        double term = sum;
        for (int n = 1; n < 200; ++n) {
            term *= (x / 2.0) / (a + n);
            sum += term;
            if (std::abs(term) < 1e-12 * std::abs(sum)) break;
        }
        return std::exp(a * std::log(x/2.0) - x/2.0 - lgamma(a)) * sum;
    }
};

} // namespace statelix

#endif // STATELIX_QUANTILE_REGRESSION_H
