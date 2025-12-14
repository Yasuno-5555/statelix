/**
 * @file objective.h
 * @brief Statelix v1.1 - Unified Optimization Objective Interface
 * 
 * All optimization problems (GLM, MAP, FM, etc.) derive from this base.
 */
#ifndef STATELIX_OBJECTIVE_H
#define STATELIX_OBJECTIVE_H

#include <Eigen/Dense>
#include <utility>
#include <stdexcept>
#include <memory>

namespace statelix {

/**
 * @brief Abstract base class for differentiable objective functions
 * 
 * Usage:
 *   struct MyObjective : Objective {
 *       double value(const Eigen::VectorXd& x) const override { ... }
 *       Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override { ... }
 *   };
 */
class Objective {
public:
    virtual ~Objective() = default;
    
    /**
     * @brief Compute objective function value at x
     * @param x Parameter vector
     * @return Objective value (to be minimized)
     */
    virtual double value(const Eigen::VectorXd& x) const = 0;
    
    /**
     * @brief Compute gradient of objective at x
     * @param x Parameter vector
     * @return Gradient vector (same dimension as x)
     */
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& x) const = 0;
    
    /**
     * @brief Compute Hessian matrix at x (optional)
     * @param x Parameter vector
     * @return Hessian matrix (n x n where n = x.size())
     * @throws std::runtime_error if not implemented
     */
    virtual Eigen::MatrixXd hessian(const Eigen::VectorXd& x) const {
        (void)x; // suppress unused warning
        throw std::runtime_error("Hessian not implemented for this objective");
    }
    
    /**
     * @brief Check if this objective provides Hessian
     */
    virtual bool has_hessian() const { return false; }
    
    /**
     * @brief Get the dimension of the parameter space
     */
    virtual int dimension() const { return -1; } // -1 means undefined/dynamic
};

/**
 * @brief Efficient objective that computes value and gradient together
 * 
 * Many objectives (e.g., log-likelihood) share computation between
 * value and gradient. This interface avoids redundant work.
 * 
 * L-BFGS and HMC prefer this interface for performance.
 */
class EfficientObjective : public Objective {
public:
    /**
     * @brief Compute value and gradient in a single pass
     * @param x Parameter vector
     * @return Pair of (value, gradient)
     */
    virtual std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& x) const = 0;
    
    // Default implementations delegate to value_and_gradient
    double value(const Eigen::VectorXd& x) const override {
        return value_and_gradient(x).first;
    }
    
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        return value_and_gradient(x).second;
    }
};

/**
 * @brief Wrapper to adapt old-style functor to new Objective interface
 * 
 * The old L-BFGS used: double operator()(const VectorXd& x, VectorXd& grad)
 * This adapter enables gradual migration.
 */
template<typename Functor>
class FunctorObjectiveAdapter : public EfficientObjective {
public:
    Functor& functor;
    
    explicit FunctorObjectiveAdapter(Functor& f) : functor(f) {}
    
    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd grad(x.size());
        double val = functor(x, grad);
        return {val, grad};
    }
};

/**
 * @brief Composite objective: f(x) = base(x) + penalty(x)
 * 
 * Used for penalized optimization (Ridge, Lasso, etc.)
 */
class RegularizedObjective : public EfficientObjective {
public:
    std::shared_ptr<Objective> base_objective;
    std::shared_ptr<class Penalizer> penalizer; // forward declaration

    RegularizedObjective(std::shared_ptr<Objective> base, 
                         std::shared_ptr<class Penalizer> pen)
        : base_objective(std::move(base)), penalizer(std::move(pen)) {}
    
    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& x) const override;
    
    // Non-smooth penalties break Hessian - never provide it
    bool has_hessian() const override { return false; }
};

} // namespace statelix

#endif // STATELIX_OBJECTIVE_H
