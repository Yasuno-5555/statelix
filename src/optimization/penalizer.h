/**
 * @file penalizer.h
 * @brief Statelix v1.1 - Unified Penalizer/Regularizer Interface
 * 
 * Supports L1, L2, ElasticNet with proximal operators for efficient optimization.
 */
#ifndef STATELIX_PENALIZER_H
#define STATELIX_PENALIZER_H

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <memory>

namespace statelix {

/**
 * @brief Abstract base class for regularization penalties
 * 
 * Key feature: prox() operator enables proximal gradient methods,
 * which are essential for non-smooth penalties like L1.
 */
class Penalizer {
public:
    virtual ~Penalizer() = default;
    
    /**
     * @brief Compute penalty value
     * @param x Parameter vector
     * @return Penalty value (always >= 0)
     */
    virtual double penalty(const Eigen::VectorXd& x) const = 0;
    
    /**
     * @brief Compute gradient of penalty (if smooth)
     * @param x Parameter vector
     * @return Gradient vector (same dimension as x)
     * @note For non-smooth penalties (L1), returns subgradient
     */
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& x) const = 0;
    
    /**
     * @brief Proximal operator: prox_{lambda*g}(x) = argmin_z { 0.5||z-x||^2 + lambda*g(z) }
     * @param x Input vector
     * @param lambda Step size (regularization strength multiplied by learning rate)
     * @return Proximal point
     */
    virtual Eigen::VectorXd prox(const Eigen::VectorXd& x, double lambda) const = 0;
    
    /**
     * @brief Check if this penalty is smooth (differentiable everywhere)
     * @note L1 is NOT smooth, L2 IS smooth
     */
    virtual bool is_smooth() const = 0;
};

/**
 * @brief No regularization (identity proximal operator)
 */
class NoPenalty : public Penalizer {
public:
    double penalty(const Eigen::VectorXd&) const override { 
        return 0.0; 
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override { 
        return Eigen::VectorXd::Zero(x.size()); 
    }
    
    Eigen::VectorXd prox(const Eigen::VectorXd& x, double) const override { 
        return x; 
    }
    
    bool is_smooth() const override { return true; }
};

/**
 * @brief L1 (Lasso) penalty: lambda * ||x||_1
 * 
 * Proximal operator: soft thresholding
 *   prox(x_i) = sign(x_i) * max(|x_i| - lambda, 0)
 */
class L1Penalty : public Penalizer {
public:
    double lambda;
    
    explicit L1Penalty(double lam = 1.0) : lambda(lam) {}
    
    double penalty(const Eigen::VectorXd& x) const override {
        return lambda * x.lpNorm<1>();
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        // Subgradient: sign(x) * lambda, with 0 at origin (arbitrary choice)
        Eigen::VectorXd g(x.size());
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) > 0) g(i) = lambda;
            else if (x(i) < 0) g(i) = -lambda;
            else g(i) = 0.0;
        }
        return g;
    }
    
    Eigen::VectorXd prox(const Eigen::VectorXd& x, double step) const override {
        // Soft thresholding
        double threshold = lambda * step;
        Eigen::VectorXd result(x.size());
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) > threshold) {
                result(i) = x(i) - threshold;
            } else if (x(i) < -threshold) {
                result(i) = x(i) + threshold;
            } else {
                result(i) = 0.0;
            }
        }
        return result;
    }
    
    bool is_smooth() const override { return false; }
};

/**
 * @brief L2 (Ridge) penalty: 0.5 * lambda * ||x||_2^2
 * 
 * Proximal operator: shrinkage
 *   prox(x) = x / (1 + lambda * step)
 */
class L2Penalty : public Penalizer {
public:
    double lambda;
    
    explicit L2Penalty(double lam = 1.0) : lambda(lam) {}
    
    double penalty(const Eigen::VectorXd& x) const override {
        return 0.5 * lambda * x.squaredNorm();
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        return lambda * x;
    }
    
    Eigen::VectorXd prox(const Eigen::VectorXd& x, double step) const override {
        // Shrinkage: x / (1 + lambda * step)
        return x / (1.0 + lambda * step);
    }
    
    bool is_smooth() const override { return true; }
};

/**
 * @brief Elastic Net penalty: λ₁||x||₁ + ½λ₂||x||₂²
 * 
 * Combines L1 and L2. The proximal operator has a closed-form solution.
 * 
 * Mathematical derivation of prox operator:
 * -----------------------------------------
 * Objective: prox_{t·g}(v) = argmin_x { ½||x-v||² + t·λ₁||x||₁ + ½t·λ₂||x||² }
 * 
 * Solution (separable, apply per-coordinate):
 *   1. Define: γ = 1 / (1 + t·λ₂)           [L2 shrinkage factor]
 *   2. Define: τ = t·λ₁·γ                   [scaled threshold]
 *   3. Apply:  x_i = γ · S_{τ}(v_i)         [shrink then soft-threshold]
 * 
 * where S_τ(v) = sign(v)·max(|v| - τ, 0) is soft-thresholding.
 * 
 * Reference: Hastie, Tibshirani, Friedman - "Elements of Statistical Learning"
 *            Section 3.8.5 (Elastic Net)
 */
class ElasticNetPenalty : public Penalizer {
public:
    double lambda1; // L1 coefficient (sparsity)
    double lambda2; // L2 coefficient (grouping)
    
    ElasticNetPenalty(double l1 = 1.0, double l2 = 1.0) 
        : lambda1(l1), lambda2(l2) {}
        
    /**
     * @brief Construct from overall lambda and l1_ratio (glmnet style)
     * @param lambda Overall regularization strength
     * @param l1_ratio Mixing ratio: 1.0 = pure Lasso, 0.0 = pure Ridge
     */
    static ElasticNetPenalty from_ratio(double lambda, double l1_ratio) {
        return ElasticNetPenalty(lambda * l1_ratio, lambda * (1.0 - l1_ratio));
    }
    
    double penalty(const Eigen::VectorXd& x) const override {
        return lambda1 * x.lpNorm<1>() + 0.5 * lambda2 * x.squaredNorm();
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        // L1 subgradient + L2 gradient
        Eigen::VectorXd g = lambda2 * x;
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) > 0) g(i) += lambda1;
            else if (x(i) < 0) g(i) -= lambda1;
        }
        return g;
    }
    
    /**
     * @brief Closed-form proximal operator for Elastic Net
     * 
     * Algorithm:
     *   γ = 1 / (1 + step·λ₂)       // L2 shrinkage
     *   τ = step·λ₁·γ               // scaled threshold
     *   x_i = γ · S_τ(v_i)          // shrink, then soft-threshold
     * 
     * Note: Order matters! We first shrink by L2, then apply L1 thresholding
     *       to the shrunk value. This follows the standard derivation.
     */
    Eigen::VectorXd prox(const Eigen::VectorXd& x, double step) const override {
        // Step 1: L2 shrinkage factor
        const double gamma = 1.0 / (1.0 + lambda2 * step);
        
        // Step 2: Scaled L1 threshold
        const double tau = lambda1 * step * gamma;
        
        // Step 3: Apply γ · S_τ(x)
        Eigen::VectorXd result(x.size());
        for (int i = 0; i < x.size(); ++i) {
            double shrunk = x(i) * gamma;  // First shrink
            // Then soft-threshold
            if (shrunk > tau) {
                result(i) = shrunk - tau;
            } else if (shrunk < -tau) {
                result(i) = shrunk + tau;
            } else {
                result(i) = 0.0;
            }
        }
        return result;
    }
    
    bool is_smooth() const override { return false; } // L1 part is non-smooth
};

/**
 * @brief Factory function to create penalizers
 */
inline std::unique_ptr<Penalizer> make_penalizer(
    const std::string& type, 
    double lambda = 1.0,
    double l1_ratio = 1.0  // only for elastic net
) {
    if (type == "none" || type == "None") {
        return std::make_unique<NoPenalty>();
    } else if (type == "l1" || type == "L1" || type == "lasso") {
        return std::make_unique<L1Penalty>(lambda);
    } else if (type == "l2" || type == "L2" || type == "ridge") {
        return std::make_unique<L2Penalty>(lambda);
    } else if (type == "elasticnet" || type == "elastic_net") {
        return std::make_unique<ElasticNetPenalty>(
            lambda * l1_ratio, lambda * (1.0 - l1_ratio));
    }
    throw std::invalid_argument("Unknown penalizer type: " + type);
}

} // namespace statelix

#endif // STATELIX_PENALIZER_H
