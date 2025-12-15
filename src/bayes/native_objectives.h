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

    /**
     * @brief Compute energy U(beta) = -log_posterior and its gradient.
     * 
     * Uses numerically stable implementations:
     * - log1pexp(z) = max(z,0) + log1p(exp(-|z|))   (avoids exp overflow)
     * - sigmoid(z)  = exp(-max(-z,0)) / (exp(-max(-z,0)) + exp(-max(z,0)))
     * 
     * @note Caller must ensure X and y outlive this object.
     */
    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& beta) const override {
        // Linear predictor: z = X * beta
        Eigen::VectorXd z = X * beta;
        Eigen::ArrayXd za = z.array();

        // Stable sigmoid: p = 1 / (1 + exp(-z))
        // For z >= 0: p = 1 / (1 + exp(-z))
        // For z <  0: p = exp(z) / (1 + exp(z))
        // Combined:   p = exp(-max(-z,0)) / (exp(-max(-z,0)) + exp(-max(z,0)))
        Eigen::ArrayXd neg_part = (-za).cwiseMax(0.0);  // max(-z, 0)
        Eigen::ArrayXd pos_part = za.cwiseMax(0.0);     // max(z, 0)
        Eigen::ArrayXd exp_neg = (-neg_part).exp();     // exp(-max(-z,0))
        Eigen::ArrayXd exp_pos = (-pos_part).exp();     // exp(-max(z,0))
        Eigen::ArrayXd p = exp_neg / (exp_neg + exp_pos);

        // Stable log(1 + exp(z)) = max(z,0) + log1p(exp(-|z|))
        Eigen::ArrayXd abs_z = za.abs();
        Eigen::ArrayXd log1pexp = za.cwiseMax(0.0) + (-abs_z).exp().log1p();

        // Log Likelihood: sum( y*z - log(1+exp(z)) )
        double log_likelihood = (y.array() * za - log1pexp).sum();
        
        // Log Prior: beta ~ N(0, prior_variance * I)
        double log_prior = -0.5 * beta.squaredNorm() / prior_variance;
        
        double log_posterior = log_likelihood + log_prior;
        
        // Energy U = -log_posterior
        double energy = -log_posterior;
        
        // Gradient: dU/dbeta = X.T * (p - y) + beta / prior_variance
        Eigen::VectorXd grad = X.transpose() * (p.matrix() - y) + beta / prior_variance;
            
        return {energy, grad};
    }
};

} // namespace statelix

#endif // STATELIX_NATIVE_OBJECTIVES_H
