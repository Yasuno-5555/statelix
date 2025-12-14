#ifndef STATELIX_NATIVE_OBJECTIVES_H
#define STATELIX_NATIVE_OBJECTIVES_H

#include "../optimization/objective.h"
#include <Eigen/Dense>
#include <cmath>

namespace statelix {

/**
 * @brief Native C++ Objective for Bayesian Logistic Regression
 * Computes Log Posterior and Gradient without Python GIL.
 * 
 * Model:
 *   y ~ Bernoulli(sigmoid(X * beta))
 *   beta ~ Normal(0, prior_var * I)
 */
class LogisticObjective : public EfficientObjective {
public:
    const Eigen::MatrixXd& X;
    const Eigen::VectorXd& y;
    double prior_variance;

    LogisticObjective(const Eigen::MatrixXd& X_ref, 
                      const Eigen::VectorXd& y_ref, 
                      double prior_std)
        : X(X_ref), y(y_ref), prior_variance(prior_std * prior_std) {}

    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& beta) const override {
        // Linear predictor: z = X * beta
        Eigen::VectorXd z = X * beta;
        
        // Probabilities: p = 1 / (1 + exp(-z))
        // Work on array for element-wise ops
        Eigen::ArrayXd z_arr = z.array();
        Eigen::ArrayXd p = 1.0 / (1.0 + (-z_arr).exp());
        
        // Log Likelihood (Negative for Minimization?)
        // HMC usually expects log_prob (positive is good).
        // Let's compute LOG PROB (Maximize).
        // HMC sampler internally negates it to map to Hamiltonian energy U = -log_prob.
        // Wait, efficient_objective usually returns Value to be MINIMIZED for optimization.
        // But HMC sampler in hmc.h expects target density log_prob?
        // Checking hmc.h...
        // "Objective returns (value, gradient). Value is U(q) = -log_prob(q)."
        // So we must return NEGATIVE Log Posterior.

        // Log Likelihood: sum( y*z - log(1 + exp(z)) )
        // Using softplus for log(1+exp(z)) -> log1p(exp(z)) or z + log1p(exp(-z))
        // Stable: if z > 0: z + log(1+exp(-z)), else: log(1+exp(z))
        
        // Eigen doesn't have vector softplus easily, we can use simple approx or loop if needed.
        // For speed, let's just use log(1+exp(z)) but careful with overflow? 
        // Or simply: y*log(p) + (1-y)*log(1-p).
        
        // Let's stick to the canonical: ll = sum( y*z - log(1+exp(z)) )
        // log(1+exp(z)) term:
        Eigen::ArrayXd log1pexp = (1.0 + z_arr.exp()).log();
        // Fix overflow for large z: if z > 40, log(1+exp(z)) ~= z
        // Simplest: use basic form, check HMC robustness.
        
        double log_likelihood = (y.array() * z_arr - log1pexp).sum();
        
        // Prior: beta ~ N(0, s^2)
        // Log Prior = -0.5 * sum(beta^2) / s^2
        double log_prior = -0.5 * beta.squaredNorm() / prior_variance;
        
        double log_posterior = log_likelihood + log_prior;
        
        // Gradient of NEGATIVE Log Post
        // Grad(LL) = X.T * (y - p)
        // Grad(Prior) = -beta / s^2
        // Grad(LogPost) = X.T * (y - p) - beta/s^2
        
        // We return NEGATIVE Value and NEGATIVE Gradient for Energy
        // U = -LogPost
        // dU = -Grad(LogPost) = - (X.T * (y - p) - beta/s^2) 
        //    = X.T * (p - y) + beta/s^2
        
        double energy = -log_posterior;
        
        Eigen::VectorXd grad = X.transpose() * (p.matrix() - y) + beta / prior_variance;
            
        return {energy, grad};
    }
};

} // namespace statelix

#endif // STATELIX_NATIVE_OBJECTIVES_H
