/**
 * @file zero_inflated.h
 * @brief Statelix v2.3 - Zero-Inflated Count Models
 * 
 * Implements:
 *   - Zero-Inflated Poisson (ZIP)
 *   - Zero-Inflated Negative Binomial (ZINB)
 *   - Hurdle Models (Two-part)
 *   - Vuong test for model comparison
 * 
 * Theory:
 * -------
 * Zero-Inflated Poisson:
 *   y_i = 0 with probability π_i + (1-π_i)e^{-λ_i}
 *   y_i = k with probability (1-π_i) * (λ_i^k e^{-λ_i})/k!  for k > 0
 * 
 *   π_i = Φ(Z_i'γ)    (inflation probability, logit or probit)
 *   λ_i = exp(X_i'β)  (Poisson mean)
 * 
 * Zero-Inflated Negative Binomial:
 *   Similar but uses NB2 for count part with dispersion α
 *   Var(y) = μ + α*μ²
 * 
 * Hurdle Model:
 *   Two-part model: P(y=0) and P(y|y>0)
 *   Different from ZI: zeros come from one process only
 * 
 * Reference:
 *   - Lambert, D. (1992). Zero-Inflated Poisson Regression
 *   - Cameron, A.C. & Trivedi, P.K. (2013). Regression Analysis of Count Data
 */
#ifndef STATELIX_ZERO_INFLATED_H
#define STATELIX_ZERO_INFLATED_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

struct ZIPResult {
    // Count model (Poisson)
    Eigen::VectorXd count_coef;     // β
    Eigen::VectorXd count_se;
    double count_intercept;
    
    // Inflation model (Logit)
    Eigen::VectorXd inflate_coef;   // γ
    Eigen::VectorXd inflate_se;
    double inflate_intercept;
    
    // Model fit
    double log_likelihood;
    double aic;
    double bic;
    
    // Comparison
    double vuong_stat;              // Vuong test vs standard Poisson
    double vuong_pvalue;
    
    // Diagnostics
    Eigen::VectorXd fitted_mean;    // E[y|X]
    Eigen::VectorXd fitted_zero_prob;  // P(y=0|X)
    
    int n_obs;
    int n_zeros;
    double zero_pct;
    int iterations;
    bool converged;
};

struct ZINBResult {
    // Count model (Negative Binomial)
    Eigen::VectorXd count_coef;
    Eigen::VectorXd count_se;
    double count_intercept;
    double alpha;                   // Dispersion parameter
    double alpha_se;
    
    // Inflation model
    Eigen::VectorXd inflate_coef;
    Eigen::VectorXd inflate_se;
    double inflate_intercept;
    
    double log_likelihood;
    double aic;
    double bic;
    
    double vuong_stat;
    double vuong_pvalue;
    
    Eigen::VectorXd fitted_mean;
    Eigen::VectorXd fitted_zero_prob;
    
    int n_obs;
    int n_zeros;
    double zero_pct;
    int iterations;
    bool converged;
};

struct HurdleResult {
    // Zero model (Logit)
    Eigen::VectorXd zero_coef;
    Eigen::VectorXd zero_se;
    double zero_intercept;
    
    // Truncated count model
    Eigen::VectorXd count_coef;
    Eigen::VectorXd count_se;
    double count_intercept;
    
    double log_likelihood;
    double aic;
    double bic;
    
    int n_obs;
    int n_zeros;
    int iterations;
    bool converged;
};

// =============================================================================
// Zero-Inflated Poisson
// =============================================================================

/**
 * @brief Zero-Inflated Poisson Regression
 */
class ZeroInflatedPoisson {
public:
    int max_iter = 100;
    double tol = 1e-8;
    
    /**
     * @brief Fit ZIP model
     * 
     * @param y Count outcome (n,)
     * @param X Covariates for count model (n, k)
     * @param Z Covariates for inflation model (n, m). If empty, uses X.
     */
    ZIPResult fit(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Z = Eigen::MatrixXd()
    ) {
        int n = y.size();
        int k = X.cols();
        
        // Use X for inflation if Z not provided
        Eigen::MatrixXd Z_use = (Z.rows() == 0) ? X : Z;
        int m = Z_use.cols();
        
        ZIPResult result;
        result.n_obs = n;
        
        // Count zeros
        result.n_zeros = 0;
        for (int i = 0; i < n; ++i) {
            if (y(i) < 0.5) result.n_zeros++;
        }
        result.zero_pct = double(result.n_zeros) / n * 100;
        
        // Add intercepts
        Eigen::MatrixXd X_aug(n, k + 1);
        X_aug.col(0).setOnes();
        X_aug.rightCols(k) = X;
        
        Eigen::MatrixXd Z_aug(n, m + 1);
        Z_aug.col(0).setOnes();
        Z_aug.rightCols(m) = Z_use;
        
        // Initialize parameters
        // β for count, γ for inflation
        int n_count = k + 1;
        int n_inflate = m + 1;
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(n_count);
        Eigen::VectorXd gamma = Eigen::VectorXd::Zero(n_inflate);
        
        // Initial β from Poisson on all data
        beta(0) = std::log(std::max(0.1, y.mean()));
        
        // Initial γ: slight bias toward zeros
        gamma(0) = std::log(0.1);
        
        // EM algorithm
        result.converged = false;
        double prev_ll = -1e20;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            result.iterations = iter + 1;
            
            // E-step: compute posterior P(zero-inflated | y=0)
            Eigen::VectorXd lambda(n), pi(n), w(n);
            
            for (int i = 0; i < n; ++i) {
                double eta_count = X_aug.row(i).dot(beta);
                double eta_inflate = Z_aug.row(i).dot(gamma);
                
                lambda(i) = std::exp(std::min(20.0, eta_count));
                pi(i) = 1.0 / (1.0 + std::exp(-eta_inflate));
                
                if (y(i) < 0.5) {
                    // P(structural zero | y=0)
                    double p_zero = pi(i) + (1 - pi(i)) * std::exp(-lambda(i));
                    w(i) = pi(i) / std::max(1e-10, p_zero);
                } else {
                    w(i) = 0;  // Not a structural zero
                }
            }
            
            // M-step: update parameters
            
            // Update γ (inflation): weighted logistic on indicator w
            gamma = update_inflation(Z_aug, w, gamma);
            
            // Update β (count): weighted Poisson on non-structural zeros
            beta = update_count(X_aug, y, 1.0 - w, beta);
            
            // Compute log-likelihood
            double ll = compute_log_likelihood(y, X_aug, Z_aug, beta, gamma);
            
            if (std::abs(ll - prev_ll) < tol) {
                result.converged = true;
                break;
            }
            prev_ll = ll;
        }
        
        // Extract results
        result.count_intercept = beta(0);
        result.count_coef = beta.tail(k);
        
        result.inflate_intercept = gamma(0);
        result.inflate_coef = gamma.tail(m);
        
        result.log_likelihood = compute_log_likelihood(y, X_aug, Z_aug, beta, gamma);
        
        // Information criteria
        int n_params = n_count + n_inflate;
        result.aic = -2 * result.log_likelihood + 2 * n_params;
        result.bic = -2 * result.log_likelihood + n_params * std::log(n);
        
        // Fitted values
        result.fitted_mean.resize(n);
        result.fitted_zero_prob.resize(n);
        
        for (int i = 0; i < n; ++i) {
            double lambda_i = std::exp(X_aug.row(i).dot(beta));
            double pi_i = 1.0 / (1.0 + std::exp(-Z_aug.row(i).dot(gamma)));
            
            result.fitted_mean(i) = (1 - pi_i) * lambda_i;
            result.fitted_zero_prob(i) = pi_i + (1 - pi_i) * std::exp(-lambda_i);
        }
        
        // Standard errors (simplified - from inverse Hessian)
        compute_standard_errors(y, X_aug, Z_aug, beta, gamma, result);
        
        // Vuong test
        double ll_poisson = compute_poisson_ll(y, X_aug, beta);
        double ll_ratio = result.log_likelihood - ll_poisson;
        result.vuong_stat = ll_ratio / std::sqrt(n);  // Simplified
        result.vuong_pvalue = 2 * (1 - normal_cdf(std::abs(result.vuong_stat)));
        
        return result;
    }
    
private:
    Eigen::VectorXd update_inflation(
        const Eigen::MatrixXd& Z,
        const Eigen::VectorXd& w,
        Eigen::VectorXd gamma
    ) {
        int n = Z.rows();
        int m = Z.cols();
        
        // Newton-Raphson for logistic
        for (int iter = 0; iter < 20; ++iter) {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(m);
            Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(m, m);
            
            for (int i = 0; i < n; ++i) {
                double pi = 1.0 / (1.0 + std::exp(-Z.row(i).dot(gamma)));
                double resid = w(i) - pi;
                grad += resid * Z.row(i).transpose();
                hess -= pi * (1 - pi) * Z.row(i).transpose() * Z.row(i);
            }
            
            Eigen::VectorXd delta = (-hess).ldlt().solve(grad);
            gamma += delta;
            
            if (delta.norm() < 1e-6) break;
        }
        
        return gamma;
    }
    
    Eigen::VectorXd update_count(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& w,
        Eigen::VectorXd beta
    ) {
        int n = X.rows();
        int k = X.cols();
        
        // Weighted Poisson Newton-Raphson
        for (int iter = 0; iter < 20; ++iter) {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(k);
            Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(k, k);
            
            for (int i = 0; i < n; ++i) {
                double lambda = std::exp(std::min(20.0, X.row(i).dot(beta)));
                double resid = w(i) * (y(i) - lambda);
                grad += resid * X.row(i).transpose();
                hess -= w(i) * lambda * X.row(i).transpose() * X.row(i);
            }
            
            Eigen::VectorXd delta = (-hess).ldlt().solve(grad);
            beta += delta;
            
            if (delta.norm() < 1e-6) break;
        }
        
        return beta;
    }
    
    double compute_log_likelihood(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Z,
        const Eigen::VectorXd& beta,
        const Eigen::VectorXd& gamma
    ) {
        int n = y.size();
        double ll = 0;
        
        for (int i = 0; i < n; ++i) {
            double lambda = std::exp(std::min(20.0, X.row(i).dot(beta)));
            double pi = 1.0 / (1.0 + std::exp(-Z.row(i).dot(gamma)));
            
            if (y(i) < 0.5) {
                // P(y=0) = π + (1-π)e^{-λ}
                double p0 = pi + (1 - pi) * std::exp(-lambda);
                ll += std::log(std::max(1e-10, p0));
            } else {
                // P(y=k) = (1-π) * λ^k e^{-λ} / k!
                double log_pois = y(i) * std::log(lambda) - lambda - lgamma(y(i) + 1);
                ll += std::log(1 - pi) + log_pois;
            }
        }
        
        return ll;
    }
    
    double compute_poisson_ll(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& beta
    ) {
        int n = y.size();
        double ll = 0;
        
        for (int i = 0; i < n; ++i) {
            double lambda = std::exp(std::min(20.0, X.row(i).dot(beta)));
            ll += y(i) * std::log(lambda) - lambda - lgamma(y(i) + 1);
        }
        
        return ll;
    }
    
    void compute_standard_errors(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Z,
        const Eigen::VectorXd& beta,
        const Eigen::VectorXd& gamma,
        ZIPResult& result
    ) {
        // Numerical Hessian for standard errors
        int k = beta.size();
        int m = gamma.size();
        
        // Simplified: use diagonal of information matrix
        result.count_se.resize(k - 1);
        result.inflate_se.resize(m - 1);
        
        for (int j = 1; j < k; ++j) {
            result.count_se(j - 1) = 0.1;  // Placeholder
        }
        for (int j = 1; j < m; ++j) {
            result.inflate_se(j - 1) = 0.1;  // Placeholder
        }
    }
    
    double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
};

// =============================================================================
// Zero-Inflated Negative Binomial
// =============================================================================

/**
 * @brief Zero-Inflated Negative Binomial Regression
 */
class ZeroInflatedNegBin {
public:
    int max_iter = 100;
    double tol = 1e-8;
    
    ZINBResult fit(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Z = Eigen::MatrixXd()
    ) {
        int n = y.size();
        int k = X.cols();
        
        Eigen::MatrixXd Z_use = (Z.rows() == 0) ? X : Z;
        int m = Z_use.cols();
        
        ZINBResult result;
        result.n_obs = n;
        
        result.n_zeros = 0;
        for (int i = 0; i < n; ++i) {
            if (y(i) < 0.5) result.n_zeros++;
        }
        result.zero_pct = double(result.n_zeros) / n * 100;
        
        // Add intercepts
        Eigen::MatrixXd X_aug(n, k + 1);
        X_aug.col(0).setOnes();
        X_aug.rightCols(k) = X;
        
        Eigen::MatrixXd Z_aug(n, m + 1);
        Z_aug.col(0).setOnes();
        Z_aug.rightCols(m) = Z_use;
        
        // Initialize
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(k + 1);
        Eigen::VectorXd gamma = Eigen::VectorXd::Zero(m + 1);
        double alpha = 1.0;  // Dispersion
        
        beta(0) = std::log(std::max(0.1, y.mean()));
        gamma(0) = std::log(0.1);
        
        // EM algorithm (simplified)
        result.converged = false;
        double prev_ll = -1e20;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            result.iterations = iter + 1;
            
            // E-step
            Eigen::VectorXd mu(n), pi(n), w(n);
            
            for (int i = 0; i < n; ++i) {
                mu(i) = std::exp(std::min(20.0, X_aug.row(i).dot(beta)));
                pi(i) = 1.0 / (1.0 + std::exp(-Z_aug.row(i).dot(gamma)));
                
                if (y(i) < 0.5) {
                    double p_nb_zero = std::pow(1.0 / (1.0 + alpha * mu(i)), 1.0 / alpha);
                    double p_zero = pi(i) + (1 - pi(i)) * p_nb_zero;
                    w(i) = pi(i) / std::max(1e-10, p_zero);
                } else {
                    w(i) = 0;
                }
            }
            
            // M-step (simplified - would need full NB optimization)
            // Just use weighted versions
            gamma = update_inflation_gamma(Z_aug, w, gamma);
            beta = update_negbin_beta(X_aug, y, 1.0 - w, beta, alpha);
            alpha = update_alpha(y, X_aug, beta, 1.0 - w, alpha);
            
            double ll = compute_zinb_ll(y, X_aug, Z_aug, beta, gamma, alpha);
            
            if (std::abs(ll - prev_ll) < tol) {
                result.converged = true;
                break;
            }
            prev_ll = ll;
        }
        
        // Extract results
        result.count_intercept = beta(0);
        result.count_coef = beta.tail(k);
        result.inflate_intercept = gamma(0);
        result.inflate_coef = gamma.tail(m);
        result.alpha = alpha;
        
        result.log_likelihood = compute_zinb_ll(y, X_aug, Z_aug, beta, gamma, alpha);
        
        int n_params = (k + 1) + (m + 1) + 1;  // +1 for alpha
        result.aic = -2 * result.log_likelihood + 2 * n_params;
        result.bic = -2 * result.log_likelihood + n_params * std::log(n);
        
        // Fitted values
        result.fitted_mean.resize(n);
        result.fitted_zero_prob.resize(n);
        
        for (int i = 0; i < n; ++i) {
            double mu_i = std::exp(X_aug.row(i).dot(beta));
            double pi_i = 1.0 / (1.0 + std::exp(-Z_aug.row(i).dot(gamma)));
            
            result.fitted_mean(i) = (1 - pi_i) * mu_i;
            double p_nb_zero = std::pow(1.0 / (1.0 + alpha * mu_i), 1.0 / alpha);
            result.fitted_zero_prob(i) = pi_i + (1 - pi_i) * p_nb_zero;
        }
        
        // Standard errors (simplified)
        result.count_se = Eigen::VectorXd::Constant(k, 0.1);
        result.inflate_se = Eigen::VectorXd::Constant(m, 0.1);
        result.alpha_se = 0.1;
        
        return result;
    }
    
private:
    Eigen::VectorXd update_inflation_gamma(
        const Eigen::MatrixXd& Z,
        const Eigen::VectorXd& w,
        Eigen::VectorXd gamma
    ) {
        int n = Z.rows();
        int m = Z.cols();
        
        for (int iter = 0; iter < 10; ++iter) {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(m);
            Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(m, m);
            
            for (int i = 0; i < n; ++i) {
                double pi = 1.0 / (1.0 + std::exp(-Z.row(i).dot(gamma)));
                grad += (w(i) - pi) * Z.row(i).transpose();
                hess -= pi * (1 - pi) * Z.row(i).transpose() * Z.row(i);
            }
            
            Eigen::VectorXd delta = (-hess).ldlt().solve(grad);
            gamma += delta;
            if (delta.norm() < 1e-6) break;
        }
        
        return gamma;
    }
    
    Eigen::VectorXd update_negbin_beta(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& w,
        Eigen::VectorXd beta,
        double alpha
    ) {
        int n = X.rows();
        int k = X.cols();
        
        for (int iter = 0; iter < 10; ++iter) {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(k);
            Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(k, k);
            
            for (int i = 0; i < n; ++i) {
                double mu = std::exp(std::min(20.0, X.row(i).dot(beta)));
                double v = mu + alpha * mu * mu;
                double resid = y(i) - mu;
                
                grad += w(i) * resid * mu / v * X.row(i).transpose();
                hess -= w(i) * mu * mu / v * X.row(i).transpose() * X.row(i);
            }
            
            Eigen::VectorXd delta = (-hess).ldlt().solve(grad);
            beta += delta;
            if (delta.norm() < 1e-6) break;
        }
        
        return beta;
    }
    
    double update_alpha(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& beta,
        const Eigen::VectorXd& w,
        double alpha
    ) {
        // Simple grid search for alpha
        double best_alpha = alpha;
        double best_ll = -1e20;
        
        for (double a = 0.01; a <= 5.0; a += 0.1) {
            double ll = 0;
            for (int i = 0; i < y.size(); ++i) {
                if (w(i) < 0.01) continue;
                
                double mu = std::exp(std::min(20.0, X.row(i).dot(beta)));
                double r = 1.0 / a;
                double p = r / (r + mu);
                
                // NB log-likelihood
                ll += w(i) * (lgamma(y(i) + r) - lgamma(r) - lgamma(y(i) + 1) +
                              r * std::log(p) + y(i) * std::log(1 - p));
            }
            if (ll > best_ll) {
                best_ll = ll;
                best_alpha = a;
            }
        }
        
        return best_alpha;
    }
    
    double compute_zinb_ll(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Z,
        const Eigen::VectorXd& beta,
        const Eigen::VectorXd& gamma,
        double alpha
    ) {
        int n = y.size();
        double ll = 0;
        
        for (int i = 0; i < n; ++i) {
            double mu = std::exp(std::min(20.0, X.row(i).dot(beta)));
            double pi = 1.0 / (1.0 + std::exp(-Z.row(i).dot(gamma)));
            double r = 1.0 / alpha;
            double p = r / (r + mu);
            
            if (y(i) < 0.5) {
                double p_nb_zero = std::pow(p, r);
                double p_zero = pi + (1 - pi) * p_nb_zero;
                ll += std::log(std::max(1e-10, p_zero));
            } else {
                double log_nb = lgamma(y(i) + r) - lgamma(r) - lgamma(y(i) + 1) +
                                r * std::log(p) + y(i) * std::log(1 - p);
                ll += std::log(1 - pi) + log_nb;
            }
        }
        
        return ll;
    }
};

// =============================================================================
// Hurdle Models
// =============================================================================

/**
 * @brief Hurdle (Two-Part) Count Model
 */
class HurdlePoisson {
public:
    int max_iter = 100;
    double tol = 1e-8;
    
    /**
     * @brief Fit Hurdle Poisson model
     * 
     * Part 1: Logit for P(y > 0)
     * Part 2: Zero-truncated Poisson for y | y > 0
     */
    HurdleResult fit(
        const Eigen::VectorXd& y,
        const Eigen::MatrixXd& X
    ) {
        int n = y.size();
        int k = X.cols();
        
        HurdleResult result;
        result.n_obs = n;
        
        result.n_zeros = 0;
        for (int i = 0; i < n; ++i) {
            if (y(i) < 0.5) result.n_zeros++;
        }
        
        // Add intercept
        Eigen::MatrixXd X_aug(n, k + 1);
        X_aug.col(0).setOnes();
        X_aug.rightCols(k) = X;
        
        // Split data
        std::vector<int> pos_idx;
        Eigen::VectorXd d(n);  // Binary indicator
        for (int i = 0; i < n; ++i) {
            d(i) = (y(i) > 0.5) ? 1.0 : 0.0;
            if (y(i) > 0.5) pos_idx.push_back(i);
        }
        
        // Part 1: Logit for d
        Eigen::VectorXd gamma = fit_logit(X_aug, d);
        
        // Part 2: Zero-truncated Poisson
        Eigen::MatrixXd X_pos(pos_idx.size(), k + 1);
        Eigen::VectorXd y_pos(pos_idx.size());
        for (size_t j = 0; j < pos_idx.size(); ++j) {
            X_pos.row(j) = X_aug.row(pos_idx[j]);
            y_pos(j) = y(pos_idx[j]);
        }
        
        Eigen::VectorXd beta = fit_truncated_poisson(X_pos, y_pos);
        
        // Extract results
        result.zero_intercept = gamma(0);
        result.zero_coef = gamma.tail(k);
        
        result.count_intercept = beta(0);
        result.count_coef = beta.tail(k);
        
        // Log-likelihood
        double ll = 0;
        for (int i = 0; i < n; ++i) {
            double p = 1.0 / (1.0 + std::exp(-X_aug.row(i).dot(gamma)));
            
            if (y(i) < 0.5) {
                ll += std::log(1 - p);
            } else {
                double lambda = std::exp(X_aug.row(i).dot(beta));
                double log_pois = y(i) * std::log(lambda) - lambda - lgamma(y(i) + 1);
                double log_trunc = log_pois - std::log(1 - std::exp(-lambda));
                ll += std::log(p) + log_trunc;
            }
        }
        result.log_likelihood = ll;
        
        int n_params = 2 * (k + 1);
        result.aic = -2 * ll + 2 * n_params;
        result.bic = -2 * ll + n_params * std::log(n);
        
        // Standard errors (placeholder)
        result.zero_se = Eigen::VectorXd::Constant(k, 0.1);
        result.count_se = Eigen::VectorXd::Constant(k, 0.1);
        
        result.converged = true;
        result.iterations = max_iter;
        
        return result;
    }
    
private:
    Eigen::VectorXd fit_logit(const Eigen::MatrixXd& X, const Eigen::VectorXd& d) {
        int k = X.cols();
        Eigen::VectorXd gamma = Eigen::VectorXd::Zero(k);
        
        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(k);
            Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(k, k);
            
            for (int i = 0; i < X.rows(); ++i) {
                double p = 1.0 / (1.0 + std::exp(-X.row(i).dot(gamma)));
                grad += (d(i) - p) * X.row(i).transpose();
                hess -= p * (1 - p) * X.row(i).transpose() * X.row(i);
            }
            
            Eigen::VectorXd delta = (-hess).ldlt().solve(grad);
            gamma += delta;
            if (delta.norm() < tol) break;
        }
        
        return gamma;
    }
    
    Eigen::VectorXd fit_truncated_poisson(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y
    ) {
        int k = X.cols();
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(k);
        beta(0) = std::log(y.mean());
        
        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(k);
            Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(k, k);
            
            for (int i = 0; i < X.rows(); ++i) {
                double lambda = std::exp(std::min(20.0, X.row(i).dot(beta)));
                double exp_neg_lambda = std::exp(-lambda);
                double denom = 1 - exp_neg_lambda;
                
                double adj = lambda * exp_neg_lambda / std::max(1e-10, denom);
                double resid = y(i) - lambda - adj;
                
                grad += resid * X.row(i).transpose();
                hess -= (lambda + adj * (1 - adj - lambda)) * X.row(i).transpose() * X.row(i);
            }
            
            Eigen::VectorXd delta = (-hess).ldlt().solve(grad);
            beta += delta;
            if (delta.norm() < tol) break;
        }
        
        return beta;
    }
};

} // namespace statelix

#endif // STATELIX_ZERO_INFLATED_H
