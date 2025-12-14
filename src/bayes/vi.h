#ifndef STATELIX_VI_H
#define STATELIX_VI_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <functional>

namespace statelix {

// Mean-Field Variational Inference Result
struct VIResult {
    Eigen::VectorXd mean;         // Variational mean
    Eigen::VectorXd variance;     // Variational variance (diagonal)
    double elbo;                  // Evidence Lower Bound
    int iterations;
    bool converged;
};

// Mean-Field Gaussian VI
// Approximates posterior p(z|x) with q(z) = N(mu, diag(sigma^2))
// Uses Coordinate Ascent VI (CAVI) or Stochastic VI
template <typename LogJointFunction>
class MeanFieldVI {
public:
    int max_iter = 100;
    double tol = 1e-4;
    double learning_rate = 0.01; // For stochastic VI
    
    // CAVI for Gaussian mean-field
    // log_joint_func(z) returns log p(x, z) 
    // grad_log_joint_func(z) returns gradient of log p(x, z) w.r.t. z
    VIResult fit(LogJointFunction& log_joint_func,
                 std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad_log_joint,
                 const Eigen::VectorXd& z0) {
        
        int dim = z0.size();
        
        // Initialize variational parameters
        Eigen::VectorXd mu = z0;
        Eigen::VectorXd log_sigma = Eigen::VectorXd::Zero(dim); // log(sigma) for numerical stability
        
        double elbo = -1e10;
        double prev_elbo = -1e10;
        int iter = 0;
        bool converged = false;
        
        // ELBO = E_q[log p(x,z)] - E_q[log q(z)]
        //      = E_q[log p(x,z)] + H[q]
        // For Gaussian q: H[q] = 0.5 * dim * (1 + log(2*pi)) + sum(log_sigma)
        
        for(iter = 0; iter < max_iter; ++iter) {
            // Reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I)
            // For CAVI, we use natural gradient updates
            
            // Compute gradient of ELBO w.r.t. mu and log_sigma
            // Using Monte Carlo estimate with single sample
            
            Eigen::VectorXd sigma = log_sigma.array().exp().matrix();
            Eigen::VectorXd eps = Eigen::VectorXd::Random(dim); // Simple random
            Eigen::VectorXd z = mu + sigma.cwiseProduct(eps);
            
            // Gradient of log p(x, z) w.r.t. z
            Eigen::VectorXd grad_z = grad_log_joint(z);
            
            // Gradient w.r.t. mu = grad_z
            // Gradient w.r.t. log_sigma = grad_z * eps * sigma + 1 (entropy term)
            Eigen::VectorXd grad_mu = grad_z;
            Eigen::VectorXd grad_log_sigma = grad_z.cwiseProduct(eps).cwiseProduct(sigma);
            grad_log_sigma.array() += 1.0; // Entropy gradient
            
            // Update
            mu += learning_rate * grad_mu;
            log_sigma += learning_rate * grad_log_sigma;
            
            // Compute ELBO estimate
            double log_p = log_joint_func(z);
            double entropy = 0.5 * dim * (1.0 + std::log(2.0 * M_PI)) + log_sigma.sum();
            elbo = log_p + entropy;
            
            // Check convergence
            if (std::abs(elbo - prev_elbo) < tol) {
                converged = true;
                break;
            }
            prev_elbo = elbo;
        }
        
        Eigen::VectorXd variance = (2.0 * log_sigma).array().exp().matrix();
        
        return {mu, variance, elbo, iter, converged};
    }
};

} // namespace statelix

#endif // STATELIX_VI_H
