/**
 * @file map.h
 * @brief Statelix v1.1 - Maximum A Posteriori (MAP) Estimation
 * 
 * MAP = argmax_θ P(θ|D) = argmax_θ [log P(D|θ) + log P(θ)]
 *     = argmin_θ [-log P(D|θ) - log P(θ)]
 *     = argmin_θ [NegLogLikelihood + NegLogPrior]
 * 
 * Connection to Penalized Regression:
 * -----------------------------------
 * - Gaussian prior N(0, σ²) → L2 penalty (Ridge)
 * - Laplace prior          → L1 penalty (Lasso)
 * - Improper flat prior    → MLE (no penalty)
 * 
 * This module leverages P0's optimization infrastructure:
 * - MAPObjective wraps likelihood + prior as RegularizedObjective
 * - Uses LBFGSOptimizer (smooth priors) or ProximalGradient (Laplace)
 * 
 * Reference: Murphy, K. (2012). Machine Learning: A Probabilistic Perspective
 */
#ifndef STATELIX_MAP_H
#define STATELIX_MAP_H

#include <Eigen/Dense>
#include <memory>
#include <cmath>
#include <stdexcept>
#include "../optimization/objective.h"
#include "../optimization/penalizer.h"
#include "../optimization/lbfgs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace statelix {

// =============================================================================
// Prior Distributions (as negative log-probability objectives)
// =============================================================================

/**
 * @brief Abstract base class for prior distributions
 * 
 * Provides negative log-prior and gradient for optimization.
 * Also provides sampling for posterior predictive checks.
 */
class Prior : public Objective {
public:
    virtual ~Prior() = default;
    
    /**
     * @brief Convert to equivalent Penalizer (if possible)
     * @return Penalizer or nullptr if no equivalent exists
     */
    virtual std::unique_ptr<Penalizer> as_penalizer() const { return nullptr; }
    
    /**
     * @brief Check if prior is proper (integrates to 1)
     */
    virtual bool is_proper() const { return true; }
    
    /**
     * @brief Sample from the prior
     */
    virtual Eigen::VectorXd sample(int dim, std::mt19937& rng) const {
        (void)dim; (void)rng;
        throw std::runtime_error("Sampling not implemented for this prior");
    }
};

/**
 * @brief Improper flat prior: P(θ) ∝ 1
 * 
 * Results in MLE estimation (no regularization).
 * -log P(θ) = constant = 0 (up to normalization)
 */
class FlatPrior : public Prior {
public:
    double value(const Eigen::VectorXd&) const override { return 0.0; }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        return Eigen::VectorXd::Zero(x.size());
    }
    
    std::unique_ptr<Penalizer> as_penalizer() const override {
        return std::make_unique<NoPenalty>();
    }
    
    bool is_proper() const override { return false; }
};

/**
 * @brief Gaussian (Normal) prior: θ ~ N(μ, Σ)
 * 
 * For isotropic case: θ_i ~ N(μ, τ²)
 * -log P(θ) = (1/2τ²) Σ(θ_i - μ)² + const
 *           = (1/2τ²) ||θ - μ||² (ignoring constant)
 * 
 * Equivalent to L2 penalty with λ = 1/τ²
 */
class GaussianPrior : public Prior {
public:
    Eigen::VectorXd mean;       // Prior mean (default: 0)
    double precision;           // 1/τ² (inverse variance)
    
    explicit GaussianPrior(double prec = 1.0) : precision(prec) {}
    
    GaussianPrior(const Eigen::VectorXd& mu, double prec)
        : mean(mu), precision(prec) {}
    
    /**
     * @brief Construct from variance (more intuitive)
     */
    static GaussianPrior from_variance(double var) {
        return GaussianPrior(1.0 / var);
    }
    
    double value(const Eigen::VectorXd& x) const override {
        if (mean.size() > 0) {
            return 0.5 * precision * (x - mean).squaredNorm();
        }
        return 0.5 * precision * x.squaredNorm();
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        if (mean.size() > 0) {
            return precision * (x - mean);
        }
        return precision * x;
    }
    
    std::unique_ptr<Penalizer> as_penalizer() const override {
        // Only works for zero-mean prior
        if (mean.size() == 0 || mean.isZero()) {
            return std::make_unique<L2Penalty>(precision);
        }
        return nullptr;  // Non-zero mean requires custom handling
    }
    
    bool is_proper() const override { return precision > 0; }
    
    Eigen::VectorXd sample(int dim, std::mt19937& rng) const override {
        std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(precision));
        Eigen::VectorXd x(dim);
        for (int i = 0; i < dim; ++i) {
            x(i) = dist(rng);
        }
        if (mean.size() > 0) {
            x += mean;
        }
        return x;
    }
};

/**
 * @brief Laplace prior: θ_i ~ Laplace(μ, b)
 * 
 * -log P(θ) = (1/b) Σ|θ_i - μ| + const
 * 
 * Equivalent to L1 penalty with λ = 1/b
 * Promotes sparsity in MAP estimate.
 */
class LaplacePrior : public Prior {
public:
    Eigen::VectorXd location;   // Prior location (default: 0)
    double scale;               // b (scale parameter)
    
    explicit LaplacePrior(double sc = 1.0) : scale(sc) {}
    
    LaplacePrior(const Eigen::VectorXd& loc, double sc)
        : location(loc), scale(sc) {}
    
    double value(const Eigen::VectorXd& x) const override {
        if (location.size() > 0) {
            return (x - location).lpNorm<1>() / scale;
        }
        return x.lpNorm<1>() / scale;
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd g(x.size());
        Eigen::VectorXd diff = (location.size() > 0) ? (x - location) : x;
        for (int i = 0; i < x.size(); ++i) {
            if (diff(i) > 0) g(i) = 1.0 / scale;
            else if (diff(i) < 0) g(i) = -1.0 / scale;
            else g(i) = 0.0;
        }
        return g;
    }
    
    std::unique_ptr<Penalizer> as_penalizer() const override {
        if (location.size() == 0 || location.isZero()) {
            return std::make_unique<L1Penalty>(1.0 / scale);
        }
        return nullptr;
    }
    
    bool is_proper() const override { return scale > 0; }
};

/**
 * @brief Horseshoe prior (for heavy sparsity)
 * 
 * θ_i | λ_i, τ ~ N(0, λ_i² τ²)
 * λ_i ~ C⁺(0, 1) (half-Cauchy)
 * 
 * This is a hierarchical prior - MAP estimation marginalizes over λ.
 * Results in robust shrinkage with less bias on large coefficients.
 * 
 * Note: Full Horseshoe requires MCMC for λ. This is a simplified version
 * using the marginal prior.
 */
class HorseshoePrior : public Prior {
public:
    double tau;  // Global shrinkage
    
    explicit HorseshoePrior(double t = 1.0) : tau(t) {}
    
    // Marginal -log prior (approximation)
    double value(const Eigen::VectorXd& x) const override {
        double val = 0.0;
        for (int i = 0; i < x.size(); ++i) {
            // log(1 + (x/τ)²) approximation to marginal
            val += std::log1p((x(i) / tau) * (x(i) / tau));
        }
        return val;
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd g(x.size());
        for (int i = 0; i < x.size(); ++i) {
            double xi_scaled = x(i) / tau;
            g(i) = (2.0 * xi_scaled / tau) / (1.0 + xi_scaled * xi_scaled);
        }
        return g;
    }
    
    bool is_proper() const override { return true; }
};

// =============================================================================
// MAP Estimation Result
// =============================================================================

struct MAPResult {
    Eigen::VectorXd mode;           // MAP estimate (posterior mode)
    double neg_log_posterior;       // -log P(θ|D) at mode
    double neg_log_likelihood;      // -log P(D|θ) at mode
    double neg_log_prior;           // -log P(θ) at mode
    
    // Uncertainty (Laplace approximation)
    Eigen::MatrixXd hessian;        // Hessian at mode
    Eigen::MatrixXd posterior_cov;  // (Hessian)^{-1} ≈ posterior covariance
    Eigen::VectorXd posterior_std;  // Diagonal of cov
    
    // Optimization info
    int iterations;
    bool converged;
    
    // Model comparison
    double log_marginal_likelihood; // Laplace approximation to log P(D)
    double bic;                     // Bayesian Information Criterion
};

// =============================================================================
// MAP Estimator
// =============================================================================

/**
 * @brief Maximum A Posteriori Estimator
 * 
 * Finds the mode of the posterior distribution:
 *   θ_MAP = argmax P(θ|D) = argmax [P(D|θ) P(θ)]
 *         = argmin [-log P(D|θ) - log P(θ)]
 * 
 * Usage:
 *   MAPEstimator map;
 *   map.prior = std::make_unique<GaussianPrior>(1.0);
 *   auto result = map.estimate(likelihood, x0);
 */
class MAPEstimator {
public:
    std::unique_ptr<Prior> prior;
    
    int max_iter = 100;
    double tol = 1e-6;
    bool compute_uncertainty = true;  // Laplace approximation
    bool verbose = false;
    
    /**
     * @brief Estimate MAP from likelihood objective
     * 
     * @param likelihood Negative log-likelihood as Objective
     * @param x0 Initial parameter values
     * @return MAPResult containing mode and uncertainty
     */
    MAPResult estimate(
        Objective& likelihood,
        const Eigen::VectorXd& x0
    ) {
        if (!prior) {
            prior = std::make_unique<FlatPrior>();
        }
        
        MAPResult result;
        
        // Strategy: Use Penalizer interface if prior supports it
        // (more efficient for L1/L2 priors)
        auto penalizer = prior->as_penalizer();
        
        OptimizerResult opt_result;
        
        if (penalizer) {
            // Use existing optimization infrastructure
            opt_result = statelix::minimize(likelihood, x0, penalizer.get(), max_iter);
        } else {
            // Create composite objective: likelihood + prior
            CompositeObjective composite(likelihood, *prior);
            LBFGSOptimizer optimizer;
            optimizer.max_iter = max_iter;
            optimizer.epsilon = tol;
            opt_result = optimizer.minimize(composite, x0);
        }
        
        result.mode = opt_result.x;
        result.iterations = opt_result.iterations;
        result.converged = opt_result.converged;
        
        // Compute objective values at mode
        result.neg_log_likelihood = likelihood.value(result.mode);
        result.neg_log_prior = prior->value(result.mode);
        result.neg_log_posterior = result.neg_log_likelihood + result.neg_log_prior;
        
        // Laplace approximation for uncertainty
        if (compute_uncertainty) {
            compute_laplace_approximation(likelihood, result);
        }
        
        return result;
    }
    
    /**
     * @brief Convenience: estimate with likelihood function
     */
    template<typename LikelihoodFunc>
    MAPResult estimate(
        LikelihoodFunc neg_log_lik,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad_neg_log_lik,
        const Eigen::VectorXd& x0
    ) {
        LambdaObjective lik_obj(neg_log_lik, grad_neg_log_lik);
        return estimate(lik_obj, x0);
    }

private:
    // Composite objective for prior + likelihood
    class CompositeObjective : public Objective {
    public:
        Objective& likelihood;
        Prior& prior;
        
        CompositeObjective(Objective& lik, Prior& p) 
            : likelihood(lik), prior(p) {}
        
        double value(const Eigen::VectorXd& x) const override {
            return likelihood.value(x) + prior.value(x);
        }
        
        Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
            return likelihood.gradient(x) + prior.gradient(x);
        }
    };
    
    // Lambda-based objective wrapper
    class LambdaObjective : public Objective {
    public:
        std::function<double(const Eigen::VectorXd&)> value_fn;
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad_fn;
        
        LambdaObjective(
            std::function<double(const Eigen::VectorXd&)> v,
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> g
        ) : value_fn(v), grad_fn(g) {}
        
        double value(const Eigen::VectorXd& x) const override {
            return value_fn(x);
        }
        
        Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
            return grad_fn(x);
        }
    };
    
    // Laplace approximation for posterior uncertainty
    void compute_laplace_approximation(Objective& likelihood, MAPResult& result) {
        int d = result.mode.size();
        
        // Compute Hessian via finite differences (or use provided if available)
        result.hessian = compute_hessian(likelihood, *prior, result.mode);
        
        // Posterior covariance ≈ H^{-1}
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(result.hessian);
        if (solver.info() == Eigen::Success) {
            Eigen::VectorXd eigenvalues = solver.eigenvalues();
            
            // Check for positive definiteness
            if (eigenvalues.minCoeff() > 1e-10) {
                result.posterior_cov = solver.eigenvectors() * 
                    eigenvalues.cwiseInverse().asDiagonal() *
                    solver.eigenvectors().transpose();
                result.posterior_std = result.posterior_cov.diagonal().cwiseSqrt();
                
                // Log marginal likelihood (Laplace approximation)
                // log P(D) ≈ log P(D|θ_MAP) + log P(θ_MAP) + (d/2)log(2π) - (1/2)log|H|
                double log_det_H = eigenvalues.array().log().sum();
                result.log_marginal_likelihood = -result.neg_log_posterior +
                    0.5 * d * std::log(2 * M_PI) - 0.5 * log_det_H;
            } else {
                // Hessian not positive definite
                result.posterior_cov = Eigen::MatrixXd::Zero(d, d);
                result.posterior_std = Eigen::VectorXd::Zero(d);
                result.log_marginal_likelihood = std::numeric_limits<double>::quiet_NaN();
            }
        }
        
        // BIC (uses only likelihood part)
        // BIC = -2 log P(D|θ_MAP) + d log(n)
        // Note: n not available here, so we approximate
        result.bic = 2 * result.neg_log_likelihood + d * std::log(100.0); // Placeholder n=100
    }
    
    // Numerical Hessian computation
    Eigen::MatrixXd compute_hessian(
        Objective& likelihood,
        Prior& prior,
        const Eigen::VectorXd& x
    ) {
        int d = x.size();
        Eigen::MatrixXd H(d, d);
        double eps = 1e-5;
        
        for (int i = 0; i < d; ++i) {
            for (int j = i; j < d; ++j) {
                Eigen::VectorXd x_pp = x, x_pm = x, x_mp = x, x_mm = x;
                x_pp(i) += eps; x_pp(j) += eps;
                x_pm(i) += eps; x_pm(j) -= eps;
                x_mp(i) -= eps; x_mp(j) += eps;
                x_mm(i) -= eps; x_mm(j) -= eps;
                
                auto f = [&](const Eigen::VectorXd& v) {
                    return likelihood.value(v) + prior.value(v);
                };
                
                H(i, j) = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps);
                H(j, i) = H(i, j);
            }
        }
        
        return H;
    }
};

// =============================================================================
// Convenience functions
// =============================================================================

/**
 * @brief Quick MAP estimation with Gaussian prior
 */
inline MAPResult map_gaussian(
    Objective& likelihood,
    const Eigen::VectorXd& x0,
    double prior_precision = 1.0
) {
    MAPEstimator estimator;
    estimator.prior = std::make_unique<GaussianPrior>(prior_precision);
    return estimator.estimate(likelihood, x0);
}

/**
 * @brief Quick MAP estimation with Laplace prior (sparse)
 */
inline MAPResult map_laplace(
    Objective& likelihood,
    const Eigen::VectorXd& x0,
    double prior_scale = 1.0
) {
    MAPEstimator estimator;
    estimator.prior = std::make_unique<LaplacePrior>(prior_scale);
    return estimator.estimate(likelihood, x0);
}

/**
 * @brief MLE (flat prior)
 */
inline MAPResult mle(
    Objective& likelihood,
    const Eigen::VectorXd& x0
) {
    MAPEstimator estimator;
    estimator.prior = std::make_unique<FlatPrior>();
    return estimator.estimate(likelihood, x0);
}

} // namespace statelix

#endif // STATELIX_MAP_H
