#ifndef STATELIX_MCMC_H
#define STATELIX_MCMC_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace statelix {

struct MCMCResult {
    Eigen::MatrixXd samples; // N x dim
    Eigen::VectorXd log_probs;
    double acceptance_rate;
};

/**
 * @brief Adaptive Metropolis-Hastings Sampler
 * 
 * Features:
 * - Configurable RNG seed for reproducibility.
 * - Per-dimension step sizes or single global step size.
 * - Optional Adaptive Metropolis (AM) covariance updates (Haario et al. 2001).
 * - Thinning to reduce autocorrelation.
 * - Basic error handling for non-finite log probabilities.
 * 
 * @tparam LogProbFunction Callable with signature double(const Eigen::VectorXd&)
 */
template <typename LogProbFunction>
class MetropolisHastings {
public:
    int n_samples = 1000;        ///< Number of samples to collect (post burn-in, post-thinning)
    int burn_in = 100;           ///< Number of burn-in iterations (discarded)
    double step_size = 0.5;      ///< Global proposal scale (used if step_sizes is empty)
    Eigen::VectorXd step_sizes;  ///< Per-dimension proposal scales (overrides step_size if non-empty)
    int thinning = 1;            ///< Keep every n-th sample
    bool adaptive = true;        ///< Enable Adaptive Metropolis covariance updates
    int adaptation_start = 100;  ///< Start adapting covariance after this many iterations
    double epsilon = 1e-6;       ///< Regularization term for adaptive covariance

    /**
     * @brief Construct a new Metropolis-Hastings sampler.
     * @param seed RNG seed for reproducibility. Use std::random_device{}() for random.
     */
    explicit MetropolisHastings(unsigned int seed = 42) : rng_(seed) {}

    /**
     * @brief Run the MCMC sampler.
     * 
     * @param log_prob_func Callable returning log probability at x.
     * @param x0 Initial state vector.
     * @return MCMCResult containing samples, log_probs, and acceptance rate.
     * @throws std::runtime_error if initial log_prob is non-finite.
     */
    MCMCResult sample(LogProbFunction& log_prob_func, const Eigen::VectorXd& x0) {
        const int dim = static_cast<int>(x0.size());
        const int total_iters = burn_in + n_samples * thinning;
        
        // Scale factor for adaptive proposals: 2.38^2 / d (optimal for Gaussian)
        const double s_d = 2.38 * 2.38 / static_cast<double>(dim);
        
        Eigen::MatrixXd samples(n_samples, dim);
        Eigen::VectorXd log_probs(n_samples);
        
        Eigen::VectorXd current_x = x0;
        double current_log_prob = log_prob_func(current_x);
        
        if (!std::isfinite(current_log_prob)) {
            throw std::runtime_error("Initial log_prob is non-finite (NaN or Inf).");
        }
        
        // Initialize step sizes
        Eigen::VectorXd scales(dim);
        if (step_sizes.size() == dim) {
            scales = step_sizes;
        } else {
            scales.setConstant(step_size);
        }
        
        // Adaptive Metropolis: running mean and covariance
        Eigen::VectorXd mean = current_x;
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(dim, dim) * (step_size * step_size);
        Eigen::LLT<Eigen::MatrixXd> llt(cov);  // Cholesky for sampling
        
        std::normal_distribution<double> std_normal(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        
        int accepted = 0;
        int sample_idx = 0;
        int iter_count = 0;
        
        for (int iter = 0; iter < total_iters; ++iter) {
            // Generate proposal
            Eigen::VectorXd proposal(dim);
            
            if (adaptive && iter > adaptation_start && iter > burn_in) {
                // Sample from N(current_x, s_d * cov + epsilon * I)
                Eigen::VectorXd z(dim);
                for (int d = 0; d < dim; ++d) z(d) = std_normal(rng_);
                proposal = current_x + std::sqrt(s_d) * llt.matrixL() * z;
            } else {
                // Non-adaptive: independent Gaussian per dimension
                for (int d = 0; d < dim; ++d) {
                    proposal(d) = current_x(d) + scales(d) * std_normal(rng_);
                }
            }
            
            double proposal_log_prob = log_prob_func(proposal);
            
            // Reject non-finite proposals
            bool accept = false;
            if (std::isfinite(proposal_log_prob)) {
                double log_alpha = proposal_log_prob - current_log_prob;
                if (std::log(uniform(rng_)) < log_alpha) {
                    accept = true;
                }
            }
            
            if (accept) {
                current_x = proposal;
                current_log_prob = proposal_log_prob;
                if (iter >= burn_in) accepted++;
            }
            
            iter_count = iter + 1;
            
            // Update adaptive covariance (always, even during burn-in)
            if (adaptive && iter_count > 1) {
                Eigen::VectorXd delta = current_x - mean;
                Eigen::VectorXd new_mean = mean + delta / static_cast<double>(iter_count);
                Eigen::VectorXd delta2 = current_x - new_mean;
                
                // Incremental covariance update: C_n = ((n-1)/n) * C_{n-1} + (1/n) * delta * delta2^T
                cov = ((iter_count - 1.0) / iter_count) * cov 
                    + (1.0 / iter_count) * (delta * delta2.transpose())
                    + (epsilon / iter_count) * Eigen::MatrixXd::Identity(dim, dim);
                
                mean = new_mean;
                
                // Update Cholesky decomposition periodically for efficiency
                if (iter % 100 == 0) {
                    Eigen::MatrixXd reg_cov = cov + epsilon * Eigen::MatrixXd::Identity(dim, dim);
                    llt.compute(reg_cov);
                    if (llt.info() != Eigen::Success) {
                        // Fallback to identity if decomposition fails
                        llt.compute(Eigen::MatrixXd::Identity(dim, dim));
                    }
                }
            }
            
            // Store sample (post burn-in, with thinning)
            if (iter >= burn_in && (iter - burn_in) % thinning == 0) {
                samples.row(sample_idx) = current_x;
                log_probs(sample_idx) = current_log_prob;
                sample_idx++;
            }
        }
        
        double acceptance_rate = static_cast<double>(accepted) / static_cast<double>(n_samples * thinning);
        return {samples, log_probs, acceptance_rate};
    }

private:
    std::mt19937 rng_;
};

} // namespace statelix

#endif // STATELIX_MCMC_H

