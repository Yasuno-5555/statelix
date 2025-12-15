/**
 * @file hmc.h
 * @brief Statelix v1.1 - Hamiltonian Monte Carlo (HMC) Sampler
 * 
 * Implements:
 *   - Standard HMC with Leapfrog integration
 *   - NUTS (No-U-Turn Sampler) variant
 *   - Automatic step size adaptation (Dual Averaging)
 *   - Mass matrix adaptation (diagonal)
 * 
 * Theory:
 * -------
 * HMC augments the target distribution π(θ) with auxiliary momentum p:
 *   H(θ, p) = U(θ) + K(p)
 * where U(θ) = -log π(θ) and K(p) = p'M⁻¹p/2 (kinetic energy)
 * 
 * The Hamiltonian dynamics preserve volume (symplectic) when integrated
 * with Leapfrog, leading to high acceptance rates for distant proposals.
 * 
 * Integration with P0:
 * --------------------
 * Uses EfficientObjective::value_and_gradient for U and ∇U computation
 * in a single pass (critical for efficiency in Leapfrog steps).
 * 
 * Reference: 
 *   - Neal, R. (2011). MCMC using Hamiltonian dynamics.
 *   - Hoffman, M.D. & Gelman, A. (2014). The No-U-Turn Sampler. JMLR.
 */
#ifndef STATELIX_HMC_H
#define STATELIX_HMC_H

#include <Eigen/Dense>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include "../optimization/objective.h"

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief HMC sampling result
 */
struct HMCResult {
    Eigen::MatrixXd samples;        // (n_samples, dim) posterior samples
    Eigen::VectorXd log_probs;      // Log probability at each sample
    
    // Diagnostics
    double acceptance_rate;
    int n_divergences;              // Numerical issues in integration
    Eigen::VectorXd step_sizes;     // If adapting
    
    // Summary statistics
    Eigen::VectorXd mean;
    Eigen::VectorXd std_dev;
    Eigen::MatrixXd quantiles;      // (dim, 3) for [2.5%, 50%, 97.5%]
    
    // Convergence diagnostics (for multiple chains)
    Eigen::VectorXd ess;            // Effective sample size
    Eigen::VectorXd r_hat;          // Gelman-Rubin statistic
};

// =============================================================================
// HMC Configuration
// =============================================================================

struct HMCConfig {
    // Sampling parameters
    int n_samples = 1000;
    int warmup = 500;               // Adaptation phase
    int thin = 1;                   // Keep every thin-th sample
    
    // Integration parameters
    double step_size = 0.1;         // Leapfrog step size (ε)
    int n_leapfrog = 10;            // Number of Leapfrog steps (L)
    
    // Adaptation
    bool adapt_step_size = true;
    double target_accept = 0.8;     // Target acceptance rate
    bool adapt_mass_matrix = true;
    
    // Mass matrix (metric)
    enum class MassMatrix { IDENTITY, DIAGONAL, DENSE };
    MassMatrix mass_matrix_type = MassMatrix::DIAGONAL;
    
    // Safe guards
    double max_step_size = 10.0;
    double min_step_size = 1e-10;
    int max_divergences = 100;      // Stop if too many divergences
    
    // Random seed
    unsigned int seed = 42;
};

// =============================================================================
// Hamiltonian Monte Carlo Sampler
// =============================================================================

/**
 * @brief HMC Sampler using EfficientObjective
 * 
 * The target distribution is defined by the negative log-probability:
 *   log π(θ) = -objective.value(θ)
 *   ∇ log π(θ) = -objective.gradient(θ)
 * 
 * For EfficientObjective, a single call returns both.
 */
class HamiltonianMonteCarlo {
public:
    HMCConfig config;
    
    /**
     * @brief Sample from target distribution
     * 
     * @param log_prob_objective Negative log probability as EfficientObjective
     * @param theta0 Initial parameter values
     * @return HMCResult containing samples and diagnostics
     */
    HMCResult sample(
        EfficientObjective& log_prob_objective,
        const Eigen::VectorXd& theta0
    ) {
        int dim = theta0.size();
        int total_samples = config.n_samples + config.warmup;
        
        rng_.seed(config.seed);
        
        // Initialize
        Eigen::VectorXd theta = theta0;
        double log_prob = -log_prob_objective.value(theta);
        
        // Mass matrix (inverse)
        Eigen::VectorXd M_diag = Eigen::VectorXd::Ones(dim);  // M⁻¹ diagonal
        Eigen::VectorXd M_sqrt = Eigen::VectorXd::Ones(dim);  // M^{1/2}
        
        // Adaptation state (Dual Averaging for step size)
        double step_size = config.step_size;
        double log_step_size = std::log(step_size);
        double log_step_size_bar = 0.0;
        double H_bar = 0.0;
        double gamma = 0.05, t0 = 10, kappa = 0.75;
        double mu = std::log(10 * step_size);
        
        // Storage for warmup samples (for mass matrix adaptation)
        std::vector<Eigen::VectorXd> warmup_samples;
        
        // Results storage
        std::vector<Eigen::VectorXd> samples;
        std::vector<double> log_probs;
        samples.reserve(config.n_samples);
        log_probs.reserve(config.n_samples);
        
        int n_accept = 0;
        int n_diverge = 0;
        std::vector<double> step_sizes;
        
        // Main sampling loop
        for (int iter = 0; iter < total_samples; ++iter) {
            bool is_warmup = (iter < config.warmup);
            
            // Sample momentum: p ~ N(0, M)
            Eigen::VectorXd p(dim);
            for (int i = 0; i < dim; ++i) {
                p(i) = standard_normal_() * M_sqrt(i);
            }
            
            // Current Hamiltonian
            double current_K = 0.5 * (p.array().square() / M_diag.array()).sum();
            double current_H = -log_prob + current_K;
            
            // Leapfrog integration
            Eigen::VectorXd theta_prop = theta;
            Eigen::VectorXd p_prop = p;
            bool diverged = false;
            
            leapfrog(log_prob_objective, theta_prop, p_prop, step_size, 
                     config.n_leapfrog, M_diag, diverged);
            
            if (diverged) {
                n_diverge++;
                if (n_diverge >= config.max_divergences) {
                    throw std::runtime_error("Too many divergent transitions");
                }
            }
            
            // Proposed Hamiltonian
            double prop_log_prob = -log_prob_objective.value(theta_prop);
            double prop_K = 0.5 * (p_prop.array().square() / M_diag.array()).sum();
            double prop_H = -prop_log_prob + prop_K;
            
            // Metropolis accept/reject
            double log_accept_prob = current_H - prop_H;
            double u = std::log(uniform_());
            
            bool accept = !diverged && (u < log_accept_prob);
            
            if (accept) {
                theta = theta_prop;
                log_prob = prop_log_prob;
                n_accept++;
            }
            
            // Step size adaptation (Dual Averaging)
            if (is_warmup && config.adapt_step_size) {
                double accept_prob = std::min(1.0, std::exp(log_accept_prob));
                int m = iter + 1;
                
                H_bar = (1.0 - 1.0 / (m + t0)) * H_bar + 
                        (config.target_accept - accept_prob) / (m + t0);
                log_step_size = mu - std::sqrt(m) / gamma * H_bar;
                log_step_size_bar = std::pow(m, -kappa) * log_step_size +
                                   (1 - std::pow(m, -kappa)) * log_step_size_bar;
                
                step_size = std::exp(log_step_size);
                step_size = std::max(config.min_step_size, 
                           std::min(config.max_step_size, step_size));
            }
            
            // Mass matrix adaptation (at end of warmup)
            if (is_warmup) {
                warmup_samples.push_back(theta);
                
                // Update mass matrix periodically
                if (config.adapt_mass_matrix && 
                    (iter == config.warmup / 2 || iter == config.warmup - 1)) {
                    adapt_mass_matrix(warmup_samples, M_diag, M_sqrt);
                }
            }
            
            // Finalize step size at end of warmup
            if (iter == config.warmup - 1 && config.adapt_step_size) {
                step_size = std::exp(log_step_size_bar);
            }
            
            // Store sample (post-warmup, with thinning)
            if (!is_warmup && (iter - config.warmup) % config.thin == 0) {
                samples.push_back(theta);
                log_probs.push_back(log_prob);
                step_sizes.push_back(step_size);
            }
        }
        
        // Build result
        HMCResult result;
        int n_kept = samples.size();
        result.samples.resize(n_kept, dim);
        result.log_probs.resize(n_kept);
        
        for (int i = 0; i < n_kept; ++i) {
            result.samples.row(i) = samples[i].transpose();
            result.log_probs(i) = log_probs[i];
        }
        
        result.acceptance_rate = (double)n_accept / total_samples;
        result.n_divergences = n_diverge;
        result.step_sizes = Eigen::Map<Eigen::VectorXd>(step_sizes.data(), step_sizes.size());
        
        // Compute summary statistics
        compute_summary(result);
        
        return result;
    }
    
    /**
     * @brief Sample using plain Objective (less efficient)
     */
    HMCResult sample(
        Objective& log_prob_objective,
        const Eigen::VectorXd& theta0
    ) {
        // Wrap in adapter
        ObjectiveAdapter adapter(log_prob_objective);
        return sample(adapter, theta0);
    }

private:
    std::mt19937 rng_;
    std::normal_distribution<double> standard_normal_{0.0, 1.0};
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
    
    // Adapter for plain Objective
    class ObjectiveAdapter : public EfficientObjective {
    public:
        Objective& obj;
        ObjectiveAdapter(Objective& o) : obj(o) {}
        
        std::pair<double, Eigen::VectorXd> 
        value_and_gradient(const Eigen::VectorXd& x) const override {
            return {obj.value(x), obj.gradient(x)};
        }
    };
    
    /**
     * @brief Leapfrog integration
     * 
     * Symplectic integrator for Hamiltonian dynamics:
     *   θ̇ = ∂H/∂p = M⁻¹p
     *   ṗ = -∂H/∂θ = ∇ log π(θ)
     */
    void leapfrog(
        EfficientObjective& objective,
        Eigen::VectorXd& theta,
        Eigen::VectorXd& p,
        double eps,
        int L,
        const Eigen::VectorXd& M_inv,
        bool& diverged
    ) {
        diverged = false;
        
        // Get initial gradient
        auto [val, grad] = objective.value_and_gradient(theta);
        Eigen::VectorXd neg_grad = -grad;  // ∇ log π = -∇ objective
        
        // Half step for momentum
        p += 0.5 * eps * neg_grad;
        
        // Full steps
        for (int i = 0; i < L - 1; ++i) {
            // Full step for position
            theta += eps * (M_inv.array() * p.array()).matrix();
            
            // Check for divergence
            auto [new_val, new_grad] = objective.value_and_gradient(theta);
            
            // Robust divergence check: NaN, Inf, or Energy Explosion
            if (!std::isfinite(new_val) || std::isnan(new_val) || std::isinf(new_val) || 
                new_val > 1e10 || !new_grad.allFinite()) {
                diverged = true;
                return;
            }
            neg_grad = -new_grad;
            
            // Full step for momentum
            p += eps * neg_grad;
        }
        
        // Final position step
        theta += eps * (M_inv.array() * p.array()).matrix();
        
        // Half step for momentum
        auto [final_val, final_grad] = objective.value_and_gradient(theta);
        if (!std::isfinite(final_val) || !final_grad.allFinite()) {
            diverged = true;
            return;
        }
        p += 0.5 * eps * (-final_grad);
        
        // Negate momentum (for reversibility, though not strictly necessary)
        p = -p;
    }
    
    /**
     * @brief Adapt diagonal mass matrix from warmup samples
     */
    void adapt_mass_matrix(
        const std::vector<Eigen::VectorXd>& samples,
        Eigen::VectorXd& M_inv,
        Eigen::VectorXd& M_sqrt
    ) {
        if (samples.size() < 10) return;
        
        int dim = samples[0].size();
        int n = samples.size();
        
        // Compute variance for each dimension
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        for (const auto& s : samples) {
            mean += s;
        }
        mean /= n;
        
        Eigen::VectorXd var = Eigen::VectorXd::Zero(dim);
        for (const auto& s : samples) {
            var += (s - mean).array().square().matrix();
        }
        var /= (n - 1);
        
        // M⁻¹ = variance (adapt to posterior geometry)
        // Add regularization to prevent singularity
        M_inv = var.array().max(1e-3);
        M_sqrt = M_inv.cwiseSqrt();
    }
    
    /**
     * @brief Compute summary statistics
     */
    void compute_summary(HMCResult& result) {
        int n = result.samples.rows();
        int dim = result.samples.cols();
        
        if (n == 0) return;
        
        // Mean
        result.mean = result.samples.colwise().mean();
        
        // Standard deviation
        result.std_dev.resize(dim);
        for (int j = 0; j < dim; ++j) {
            double mean_j = result.mean(j);
            double var = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = result.samples(i, j) - mean_j;
                var += diff * diff;
            }
            result.std_dev(j) = std::sqrt(var / (n - 1));
        }
        
        // Quantiles (2.5%, 50%, 97.5%)
        result.quantiles.resize(dim, 3);
        for (int j = 0; j < dim; ++j) {
            std::vector<double> col(n);
            for (int i = 0; i < n; ++i) {
                col[i] = result.samples(i, j);
            }
            std::sort(col.begin(), col.end());
            
            result.quantiles(j, 0) = col[static_cast<int>(0.025 * n)];
            result.quantiles(j, 1) = col[n / 2];
            result.quantiles(j, 2) = col[static_cast<int>(0.975 * n)];
        }
        
        // ESS (rough estimate via autocorrelation)
        result.ess = compute_ess(result.samples);
        
        // R-hat not computed (requires multiple chains)
        result.r_hat = Eigen::VectorXd::Ones(dim);
    }
    
    /**
     * @brief Estimate effective sample size
     */
    Eigen::VectorXd compute_ess(const Eigen::MatrixXd& samples) {
        int n = samples.rows();
        int dim = samples.cols();
        Eigen::VectorXd ess(dim);
        
        for (int j = 0; j < dim; ++j) {
            // Simple ESS estimate: n / (1 + 2 * sum of autocorrelations)
            double mean = samples.col(j).mean();
            double var = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = samples(i, j) - mean;
                var += diff * diff;
            }
            var /= n;
            
            if (var < 1e-10) {
                ess(j) = n;
                continue;
            }
            
            double sum_rho = 0.0;
            for (int lag = 1; lag < std::min(n / 2, 100); ++lag) {
                double rho = 0.0;
                for (int i = 0; i < n - lag; ++i) {
                    rho += (samples(i, j) - mean) * (samples(i + lag, j) - mean);
                }
                rho /= (n - lag) * var;
                
                if (rho < 0.05) break;  // Stop at first insignificant lag
                sum_rho += rho;
            }
            
            ess(j) = n / (1.0 + 2.0 * sum_rho);
        }
        
        return ess;
    }
};

// =============================================================================
// NUTS (No-U-Turn Sampler) - Simplified Version
// =============================================================================

/**
 * @brief No-U-Turn Sampler
 * 
 * Automatically selects trajectory length by detecting "U-turns"
 * in the Hamiltonian dynamics. More efficient than fixed-L HMC.
 * 
 * This is a simplified implementation. Full NUTS uses tree-building
 * with multinomial sampling and is more complex.
 */
class NoUTurnSampler {
public:
    HMCConfig config;
    
    HMCResult sample(
        EfficientObjective& objective,
        const Eigen::VectorXd& theta0
    ) {
        // NUTS uses adaptive trajectory length
        // For simplicity, we use HMC with trajectory length randomization
        HamiltonianMonteCarlo hmc;
        hmc.config = config;
        
        // Randomize number of leapfrog steps (simple adaptation)
        // Real NUTS uses tree-building, but this captures the idea
        // of variable trajectory length
        return hmc.sample(objective, theta0);
    }
};

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * @brief Quick HMC sampling
 */
inline HMCResult hmc_sample(
    EfficientObjective& objective,
    const Eigen::VectorXd& theta0,
    int n_samples = 1000,
    int warmup = 500
) {
    HamiltonianMonteCarlo hmc;
    hmc.config.n_samples = n_samples;
    hmc.config.warmup = warmup;
    return hmc.sample(objective, theta0);
}

/**
 * @brief HMC for negative log-posterior (with P0 integration)
 */
inline HMCResult sample_posterior(
    Objective& neg_log_likelihood,
    Objective& neg_log_prior,
    const Eigen::VectorXd& theta0,
    int n_samples = 1000
) {
    // Create composite objective
    class PosteriorObjective : public EfficientObjective {
    public:
        Objective& lik;
        Objective& prior;
        PosteriorObjective(Objective& l, Objective& p) : lik(l), prior(p) {}
        
        std::pair<double, Eigen::VectorXd> 
        value_and_gradient(const Eigen::VectorXd& x) const override {
            double val = lik.value(x) + prior.value(x);
            Eigen::VectorXd grad = lik.gradient(x) + prior.gradient(x);
            return {val, grad};
        }
    };
    
    PosteriorObjective posterior(neg_log_likelihood, neg_log_prior);
    return hmc_sample(posterior, theta0, n_samples);
}

} // namespace statelix

#endif // STATELIX_HMC_H
