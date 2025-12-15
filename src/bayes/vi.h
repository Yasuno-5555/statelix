#ifndef STATELIX_VI_H
#define STATELIX_VI_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <functional>
#include <random>
#include <limits>

namespace statelix {

/**
 * @brief Variational Inference Result
 */
struct VIResult {
    Eigen::VectorXd mean;         ///< Variational mean μ
    Eigen::VectorXd variance;     ///< Variational variance σ² (diagonal)
    double elbo;                  ///< Evidence Lower Bound
    int iterations;
    bool converged;
};

/**
 * @brief Stochastic Variational Inference with Reparameterization Gradient
 * 
 * Approximates posterior p(z|x) with q(z) = N(μ, diag(σ²)).
 * Uses Monte Carlo gradient estimation with the reparameterization trick:
 *   z = μ + σ ⊙ ε,  where ε ~ N(0, I)
 * 
 * ELBO = E_q[log p(x,z)] + H[q]
 *      = E_q[log p(x,z)] + 0.5*d*(1 + log(2π)) + Σ log(σ_i)
 * 
 * @tparam LogJointFunction Callable: double(const Eigen::VectorXd& z)
 * @tparam GradLogJointFunction Callable: Eigen::VectorXd(const Eigen::VectorXd& z)
 */
template <typename LogJointFunction, typename GradLogJointFunction>
class StochasticVI {
public:
    int max_iter = 1000;         ///< Maximum iterations
    double tol = 1e-3;           ///< Convergence tolerance for ELBO change
    double learning_rate = 0.01; ///< Step size for gradient ascent
    int n_mc_samples = 10;       ///< Number of Monte Carlo samples for gradient estimation
    unsigned int seed = 42;      ///< RNG seed for reproducibility

    /**
     * @brief Fit the variational approximation.
     * 
     * @param log_joint Callable returning log p(x, z) for given z.
     * @param grad_log_joint Callable returning ∇_z log p(x, z).
     * @param z0 Initial mean estimate.
     * @return VIResult containing variational parameters and diagnostics.
     */
    VIResult fit(LogJointFunction& log_joint,
                 GradLogJointFunction& grad_log_joint,
                 const Eigen::VectorXd& z0) {
        
        const int dim = static_cast<int>(z0.size());
        
        // Initialize variational parameters
        Eigen::VectorXd mu = z0;
        Eigen::VectorXd log_sigma = Eigen::VectorXd::Constant(dim, -1.0);  // σ ≈ 0.37
        
        double prev_elbo = -std::numeric_limits<double>::infinity();
        
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        
        int iter = 0;
        bool converged = false;
        double elbo = prev_elbo;
        
        for (iter = 0; iter < max_iter; ++iter) {
            Eigen::VectorXd sigma = log_sigma.array().exp().matrix();
            
            // Monte Carlo gradient estimate (average over n_mc_samples)
            Eigen::VectorXd grad_mu = Eigen::VectorXd::Zero(dim);
            Eigen::VectorXd grad_log_sigma_accum = Eigen::VectorXd::Zero(dim);
            double elbo_log_p_sum = 0.0;
            
            for (int s = 0; s < n_mc_samples; ++s) {
                // Sample ε ~ N(0, I)
                Eigen::VectorXd eps(dim);
                for (int d = 0; d < dim; ++d) {
                    eps(d) = normal(rng);
                }
                
                // Reparameterization: z = μ + σ ⊙ ε
                Eigen::VectorXd z = mu + sigma.cwiseProduct(eps);
                
                // Evaluate log joint and its gradient
                double log_p = log_joint(z);
                Eigen::VectorXd grad_z = grad_log_joint(z);
                
                // Accumulate gradients
                // ∇_μ ELBO = E[∇_z log p]
                grad_mu += grad_z;
                
                // ∇_{log σ} ELBO = E[∇_z log p ⊙ ε ⊙ σ] + 1 (entropy term added later)
                grad_log_sigma_accum += grad_z.cwiseProduct(eps).cwiseProduct(sigma);
                
                elbo_log_p_sum += log_p;
            }
            
            // Average over samples
            grad_mu /= static_cast<double>(n_mc_samples);
            Eigen::VectorXd grad_log_sigma = grad_log_sigma_accum / static_cast<double>(n_mc_samples);
            
            // Add entropy gradient: ∂H/∂(log σ_j) = 1
            grad_log_sigma.array() += 1.0;
            
            // Gradient ascent update
            mu += learning_rate * grad_mu;
            log_sigma += learning_rate * grad_log_sigma;
            
            // Compute ELBO estimate
            // ELBO = E_q[log p(x,z)] + H[q]
            // H[q] = 0.5 * d * (1 + log(2π)) + Σ log(σ_j)
            double entropy = 0.5 * dim * (1.0 + std::log(2.0 * M_PI)) + log_sigma.sum();
            double avg_log_p = elbo_log_p_sum / static_cast<double>(n_mc_samples);
            elbo = avg_log_p + entropy;
            
            // Check convergence
            if (std::abs(elbo - prev_elbo) < tol) {
                converged = true;
                break;
            }
            prev_elbo = elbo;
        }
        
        // Compute variance: σ² = exp(log σ)² = exp(2 * log σ)
        Eigen::VectorXd variance = log_sigma.array().exp().square().matrix();
        
        return {mu, variance, elbo, iter + 1, converged};
    }
};

// Backward compatibility alias
template <typename LogJointFunction>
using MeanFieldVI = StochasticVI<LogJointFunction, std::function<Eigen::VectorXd(const Eigen::VectorXd&)>>;

} // namespace statelix

#endif // STATELIX_VI_H

