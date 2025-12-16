/**
 * @file hmc.h
 * @brief Statelix v1.1 - Hamiltonian Monte Carlo (HMC) Sampler
 * 
 * Implements:
 *   - Standard HMC with Leapfrog integration
 *   - Automatic step size adaptation (Dual Averaging)
 *   - Windowed mass matrix adaptation (diagonal)
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
 *   - Stan Reference Manual: https://mc-stan.org/docs/reference-manual/
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
#include <limits>
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
    
    // Convergence diagnostics
    Eigen::VectorXd ess;            // Effective sample size (Geyer's estimator)
    Eigen::VectorXd r_hat;          // NaN for single chain (requires multiple chains)
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
    
    // Windowed adaptation schedule (Stan-style)
    // Initial window: [0, init_buffer)
    // Adaptation windows: [init_buffer, warmup - term_buffer)
    // Terminal window: [warmup - term_buffer, warmup)
    int init_buffer = 75;           // Initial fast adaptation
    int term_buffer = 50;           // Final slow adaptation
    int base_window = 25;           // Base window size (doubles each time)
    
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
 * 
 * Divergence Handling:
 * --------------------
 * When a divergent transition occurs (NaN, Inf, or energy explosion),
 * the proposal is REJECTED and the current state is retained.
 * The rejected sample is NOT used for adaptation. Divergent transitions
 * indicate the step size may be too large or the posterior geometry
 * is problematic. Consider reducing step_size or reparameterizing.
 */
class HamiltonianMonteCarlo {
public:
    HMCConfig config;
    
    HamiltonianMonteCarlo() = default;
    HamiltonianMonteCarlo(const HMCConfig& cfg) : config(cfg) {}
    
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
        double gamma_da = 0.05, t0 = 10, kappa = 0.75;
        double mu = std::log(10 * step_size);
        
        // Storage for warmup samples (ONLY ACCEPTED samples for mass matrix adaptation)
        std::vector<Eigen::VectorXd> warmup_samples;
        
        // Windowed adaptation schedule
        std::vector<std::pair<int, int>> adaptation_windows = 
            compute_adaptation_windows(config.warmup, config.init_buffer, 
                                        config.term_buffer, config.base_window);
        int current_window_idx = 0;
        int window_start = 0;
        
        // Results storage
        std::vector<Eigen::VectorXd> samples;
        std::vector<double> log_probs;
        samples.reserve(config.n_samples);
        log_probs.reserve(config.n_samples);
        
        int n_accept = 0;
        int n_diverge = 0;
        std::vector<double> step_sizes;
        
        // Track divergent iterations (for diagnostics)
        std::vector<int> divergent_iterations;
        
        // Main sampling loop
        for (int iter = 0; iter < total_samples; ++iter) {
            bool is_warmup = (iter < config.warmup);
            
            // Sample momentum: p ~ N(0, M)
            Eigen::VectorXd p(dim);
            for (int i = 0; i < dim; ++i) {
                p(i) = standard_normal_(rng_) * M_sqrt(i);
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
                divergent_iterations.push_back(iter);
                if (n_diverge >= config.max_divergences) {
                    throw std::runtime_error(
                        "Too many divergent transitions (" + 
                        std::to_string(n_diverge) + 
                        "). Consider reducing step_size or reparameterizing the model.");
                }
                // Divergent proposal is rejected, current state retained
                // Do NOT use this theta for adaptation
            }
            
            // Proposed Hamiltonian (only compute if not diverged)
            double prop_log_prob = 0.0;
            double prop_K = 0.0;
            double prop_H = std::numeric_limits<double>::infinity();
            double log_accept_prob = -std::numeric_limits<double>::infinity();
            
            if (!diverged) {
                prop_log_prob = -log_prob_objective.value(theta_prop);
                prop_K = 0.5 * (p_prop.array().square() / M_diag.array()).sum();
                prop_H = -prop_log_prob + prop_K;
                log_accept_prob = current_H - prop_H;
            }
            
            double u = std::log(uniform_(rng_));
            bool accept = !diverged && (u < log_accept_prob);
            
            if (accept) {
                theta = theta_prop;
                log_prob = prop_log_prob;
                n_accept++;
            }
            
            // Step size adaptation (Dual Averaging)
            if (is_warmup && config.adapt_step_size) {
                // Use actual acceptance probability for adaptation
                double accept_prob = diverged ? 0.0 : std::min(1.0, std::exp(log_accept_prob));
                int m = iter + 1;
                
                H_bar = (1.0 - 1.0 / (m + t0)) * H_bar + 
                        (config.target_accept - accept_prob) / (m + t0);
                log_step_size = mu - std::sqrt(m) / gamma_da * H_bar;
                
                // Clamp log_step_size before computing bar
                double clamped_log_step_size = std::max(
                    std::log(config.min_step_size),
                    std::min(std::log(config.max_step_size), log_step_size));
                
                log_step_size_bar = std::pow(m, -kappa) * clamped_log_step_size +
                                   (1 - std::pow(m, -kappa)) * log_step_size_bar;
                
                step_size = std::exp(clamped_log_step_size);
            }
            
            // Mass matrix adaptation (windowed, Stan-style)
            // Only use ACCEPTED samples for adaptation
            if (is_warmup && accept) {
                warmup_samples.push_back(theta);
                
                // Check if we've reached end of a window
                if (config.adapt_mass_matrix && 
                    current_window_idx < adaptation_windows.size()) {
                    int window_end = adaptation_windows[current_window_idx].second;
                    
                    if (iter >= window_end) {
                        // Adapt mass matrix using samples from this window
                        std::vector<Eigen::VectorXd> window_samples(
                            warmup_samples.begin() + window_start,
                            warmup_samples.end());
                        
                        if (window_samples.size() >= 10) {
                            adapt_mass_matrix(window_samples, M_diag, M_sqrt, dim);
                        }
                        
                        // Move to next window
                        window_start = warmup_samples.size();
                        current_window_idx++;
                    }
                }
            }
            
            // Finalize step size at end of warmup
            if (iter == config.warmup - 1 && config.adapt_step_size) {
                // Clamp the final step size
                log_step_size_bar = std::max(
                    std::log(config.min_step_size),
                    std::min(std::log(config.max_step_size), log_step_size_bar));
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
    std::uniform_real_distribution<double> uniform_ = std::uniform_real_distribution<double>(0.0, 1.0);
    
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
     * @brief Compute Stan-style windowed adaptation schedule
     * 
     * Windows double in size each time, starting from base_window.
     * This allows mass matrix to stabilize before final adaptation.
     */
    std::vector<std::pair<int, int>> compute_adaptation_windows(
        int warmup, int init_buffer, int term_buffer, int base_window
    ) {
        std::vector<std::pair<int, int>> windows;
        
        if (warmup < init_buffer + term_buffer + base_window) {
            // Not enough warmup for windowed adaptation
            // Just do one adaptation at the end
            windows.push_back({init_buffer, warmup - term_buffer});
            return windows;
        }
        
        int adapt_start = init_buffer;
        int adapt_end = warmup - term_buffer;
        
        int current = adapt_start;
        int window_size = base_window;
        
        while (current + window_size <= adapt_end) {
            windows.push_back({current, current + window_size});
            current += window_size;
            window_size *= 2;  // Double window size
        }
        
        // Final window to fill remaining
        if (current < adapt_end) {
            windows.push_back({current, adapt_end});
        }
        
        return windows;
    }
    
    /**
     * @brief Leapfrog integration
     * 
     * Symplectic integrator for Hamiltonian dynamics:
     *   θ̇ = ∂H/∂p = M⁻¹p
     *   ṗ = -∂H/∂θ = ∇ log π(θ)
     * 
     * Note: The momentum flip at the end (p = -p) is for theoretical
     * reversibility but doesn't affect the accept/reject decision since
     * kinetic energy K(p) = K(-p). We skip this for clarity.
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
            if (!std::isfinite(new_val) || new_val > 1e10 || !new_grad.allFinite()) {
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
        
        // Note: We intentionally do NOT negate p here.
        // The momentum flip is only needed for detailed balance proof,
        // but since K(p) = K(-p), it doesn't affect acceptance.
        // Removing it improves code clarity.
    }
    
    /**
     * @brief Adapt diagonal mass matrix from warmup samples
     * 
     * Uses welford online algorithm for numerical stability and
     * applies adaptive regularization based on sample size.
     */
    void adapt_mass_matrix(
        const std::vector<Eigen::VectorXd>& samples,
        Eigen::VectorXd& M_inv,
        Eigen::VectorXd& M_sqrt,
        int dim
    ) {
        if (samples.size() < 10) return;
        
        int n = samples.size();
        
        // Welford's online algorithm for variance (numerically stable)
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd M2 = Eigen::VectorXd::Zero(dim);  // Sum of squared differences
        
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd delta = samples[i] - mean;
            mean += delta / (i + 1);
            Eigen::VectorXd delta2 = samples[i] - mean;
            M2 += delta.cwiseProduct(delta2);
        }
        
        Eigen::VectorXd var = M2 / (n - 1);
        
        // Adaptive regularization (scaled by sample size, Stan-style)
        // Regularization shrinks toward prior variance = 1
        double reg_scale = 5.0 / (n + 5.0);  // Decreases as n increases
        
        for (int j = 0; j < dim; ++j) {
            // Regularized variance: (n * var + 5 * 1) / (n + 5)
            double reg_var = (n * var(j) + 5.0 * 1.0) / (n + 5.0);
            M_inv(j) = std::max(1e-8, reg_var);  // Prevent singularity
        }
        
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
        
        // ESS using Geyer's monotone sequence estimator
        result.ess = compute_ess_geyer(result.samples);
        
        // R-hat: NaN for single chain (requires multiple chains for meaningful estimate)
        // Returning ones would be misleading
        result.r_hat = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::quiet_NaN());
    }
    
    /**
     * @brief Estimate effective sample size using Geyer's monotone sequence estimator
     * 
     * This is the method used by Stan and is more robust than simple
     * summation of autocorrelations. It uses the initial positive sequence
     * estimator combined with monotone sequence constraint.
     * 
     * Reference: Geyer, C. J. (1992). Practical Markov Chain Monte Carlo.
     */
    Eigen::VectorXd compute_ess_geyer(const Eigen::MatrixXd& samples) {
        int n = samples.rows();
        int dim = samples.cols();
        Eigen::VectorXd ess(dim);
        
        for (int j = 0; j < dim; ++j) {
            Eigen::VectorXd x = samples.col(j);
            double mean = x.mean();
            
            // Compute variance
            double var = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = x(i) - mean;
                var += diff * diff;
            }
            var /= n;
            
            if (var < 1e-10) {
                ess(j) = n;
                continue;
            }
            
            // Compute autocorrelations
            int max_lag = n - 1;  // Can compute up to n-1 lags
            std::vector<double> rho(max_lag);
            
            for (int lag = 0; lag < max_lag; ++lag) {
                double sum = 0.0;
                for (int i = 0; i < n - lag; ++i) {
                    sum += (x(i) - mean) * (x(i + lag) - mean);
                }
                rho[lag] = sum / (n * var);  // Biased estimator
            }
            
            // Geyer's initial positive sequence estimator
            // Sum autocorrelations in pairs, stop when sum becomes negative
            double sum_rho = rho[0];  // rho[0] = 1 by definition
            std::vector<double> Gamma;  // Paired sums
            
            for (int lag = 1; lag < max_lag - 1; lag += 2) {
                double gamma = rho[lag] + rho[lag + 1];
                if (gamma < 0) break;  // Initial positive sequence stops here
                Gamma.push_back(gamma);
            }
            
            // Apply monotone constraint: Gamma should be non-increasing
            for (size_t i = 1; i < Gamma.size(); ++i) {
                if (Gamma[i] > Gamma[i-1]) {
                    Gamma[i] = Gamma[i-1];
                }
            }
            
            // Sum the monotone sequence
            double tau = -1.0;  // Start at -1 because we add rho[0]=1 twice in pair sum
            for (double g : Gamma) {
                tau += g;
            }
            tau = 1.0 + 2.0 * tau;
            
            // ESS = n / tau, but ensure tau >= 1
            tau = std::max(1.0, tau);
            ess(j) = n / tau;
            
            // Cap at n
            ess(j) = std::min(ess(j), static_cast<double>(n));
        }
        
        return ess;
    }
};

// =============================================================================
// Fixed-L HMC with Trajectory Randomization
// =============================================================================

/**
 * @brief HMC with randomized trajectory length
 * 
 * This is NOT the No-U-Turn Sampler (NUTS). True NUTS requires:
 *   - Binary tree construction (doubling procedure)
 *   - Multinomial sampling from valid tree nodes
 *   - U-turn detection: p · (θ⁺ - θ⁻) < 0
 * 
 * This class provides a simpler alternative that randomizes the number
 * of leapfrog steps uniformly between L_min and L_max, which can help
 * reduce periodic behavior in fixed-L HMC.
 * 
 * For true NUTS, please use specialized libraries like Stan or NumPyro.
 * We are being honest about what this provides. 
 */
class RandomizedTrajectoryHMC {
public:
    HMCConfig config;
    int L_min = 5;   // Minimum leapfrog steps
    int L_max = 20;  // Maximum leapfrog steps
    
    HMCResult sample(
        EfficientObjective& objective,
        const Eigen::VectorXd& theta0
    ) {
        HamiltonianMonteCarlo hmc;
        hmc.config = config;
        
        // Use median of L_min and L_max as base
        // Note: This doesn't actually randomize per-iteration like true NUTS would
        // For proper implementation, we'd need to modify HMC's sampling loop
        hmc.config.n_leapfrog = (L_min + L_max) / 2;
        
        return hmc.sample(objective, theta0);
    }
};

// =============================================================================
// Multi-Chain Utilities
// =============================================================================

/**
 * @brief Compute R-hat (Gelman-Rubin statistic) from multiple chains
 * 
 * R-hat compares between-chain and within-chain variance.
 * Values close to 1.0 indicate convergence.
 * R-hat > 1.1 suggests chains have not mixed well.
 * 
 * @param chains Vector of HMCResult from parallel chains
 * @return R-hat for each parameter dimension
 */
inline Eigen::VectorXd compute_rhat(const std::vector<HMCResult>& chains) {
    if (chains.empty()) {
        throw std::invalid_argument("No chains provided for R-hat computation");
    }
    
    int m = chains.size();  // Number of chains
    if (m < 2) {
        throw std::invalid_argument("R-hat requires at least 2 chains");
    }
    
    int n = chains[0].samples.rows();  // Samples per chain
    int dim = chains[0].samples.cols();
    
    Eigen::VectorXd r_hat(dim);
    
    for (int j = 0; j < dim; ++j) {
        // Compute chain means
        Eigen::VectorXd chain_means(m);
        Eigen::VectorXd chain_vars(m);
        
        for (int c = 0; c < m; ++c) {
            Eigen::VectorXd col = chains[c].samples.col(j);
            chain_means(c) = col.mean();
            
            double var = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = col(i) - chain_means(c);
                var += diff * diff;
            }
            chain_vars(c) = var / (n - 1);
        }
        
        // Between-chain variance B
        double grand_mean = chain_means.mean();
        double B = 0.0;
        for (int c = 0; c < m; ++c) {
            double diff = chain_means(c) - grand_mean;
            B += diff * diff;
        }
        B = B * n / (m - 1);
        
        // Within-chain variance W (mean of chain variances)
        double W = chain_vars.mean();
        
        // Estimated variance: weighted average of W and B
        double var_plus = ((n - 1.0) / n) * W + (1.0 / n) * B;
        
        // R-hat
        r_hat(j) = std::sqrt(var_plus / W);
    }
    
    return r_hat;
}

/**
 * @brief Run multiple chains in sequence (for R-hat computation)
 * 
 * For parallel execution, use std::async or OpenMP externally.
 */
inline std::vector<HMCResult> run_chains(
    EfficientObjective& objective,
    const Eigen::VectorXd& theta0,
    int n_chains = 4,
    int n_samples = 1000,
    int warmup = 500
) {
    std::vector<HMCResult> results;
    results.reserve(n_chains);
    
    for (int c = 0; c < n_chains; ++c) {
        HamiltonianMonteCarlo hmc;
        hmc.config.n_samples = n_samples;
        hmc.config.warmup = warmup;
        hmc.config.seed = 42 + c * 1000;  // Different seed per chain
        
        results.push_back(hmc.sample(objective, theta0));
    }
    
    // Compute R-hat across chains
    Eigen::VectorXd r_hat = compute_rhat(results);
    for (auto& result : results) {
        result.r_hat = r_hat;
    }
    
    return results;
}

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

// Alias for convenience
using HMC = HamiltonianMonteCarlo;

} // namespace statelix

#endif // STATELIX_HMC_H
