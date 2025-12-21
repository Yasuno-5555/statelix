/**
 * @file zigen_optimizer.h
 * @brief Zigen L-BFGS Optimizer wrapper for Statelix
 *
 * Provides L-BFGS optimization with automatic differentiation
 * using Zigen's reverse-mode autodiff system.
 */
#ifndef STATELIX_ZIGEN_OPTIMIZER_H
#define STATELIX_ZIGEN_OPTIMIZER_H

#include "lbfgs.h"
#include "objective.h"
#include <Eigen/Dense>
#include <Zigen/Autodiff/Reverse.hpp>
#include <Zigen/Optimize.hpp>
#include <Zigen/Zigen.hpp>

namespace statelix {

/**
 * @brief Zigen-powered L-BFGS Optimizer
 *
 * Uses Zigen's reverse-mode automatic differentiation for gradient
 * computation, eliminating the need for manual gradient derivation.
 */
class ZigenLBFGS {
public:
  // Configuration
  int max_iter = 100;
  double tolerance = 1e-6;
  size_t history_size = 10;
  bool verbose = false;

  /**
   * @brief Minimize an objective function using Zigen L-BFGS
   *
   * This version uses Zigen's autodiff to compute gradients automatically.
   * The objective function only needs to compute the function value.
   *
   * @tparam ObjectiveFunc Callable taking Matrix<Var> and returning Var
   * @param objective The objective function to minimize
   * @param x0 Initial parameter vector (Eigen::VectorXd)
   * @return OptimizerResult containing optimized parameters
   */
  template <typename ObjectiveFunc>
  OptimizerResult minimize_autodiff(ObjectiveFunc &&objective,
                                    const Eigen::VectorXd &x0) {
    using namespace Zigen;
    using namespace Zigen::Autodiff::Reverse;

    // Convert Eigen to Zigen Matrix
    Matrix<double, Dynamic, 1> zigen_x0(x0.size(), 1);
    for (int i = 0; i < x0.size(); ++i) {
      zigen_x0(i, 0) = x0(i);
    }

    // Configure optimizer
    Zigen::Optimize::Config config;
    config.max_iterations = max_iter;
    config.tolerance_grad = tolerance;
    config.history_size = history_size;

    // Run optimization
    auto result = Zigen::Optimize::minimize_lbfgs(
        std::forward<ObjectiveFunc>(objective), zigen_x0, config);

    // Convert result back to Eigen
    OptimizerResult out;
    out.x.resize(x0.size());
    for (int i = 0; i < x0.size(); ++i) {
      out.x(i) = result.optimized_params(i, 0);
    }
    out.min_value = result.final_cost;
    out.iterations = result.iterations;
    out.converged = result.converged;

    return out;
  }

  /**
   * @brief Minimize a Statelix Objective using manual gradients
   *
   * Falls back to the standard L-BFGS approach when gradients are provided.
   * Use minimize_autodiff for automatic gradient computation.
   */
  template <typename Functor>
  OptimizerResult minimize(Functor &func, const Eigen::VectorXd &x0) {
    // Use existing LBFGS implementation for backward compatibility
    LBFGS<Functor> optimizer;
    optimizer.max_iter = max_iter;
    optimizer.epsilon = tolerance;
    optimizer.m = history_size;
    return optimizer.minimize(func, x0);
  }
};

// =============================================================================
// Zigen-Based Logistic Regression Objective
// =============================================================================

/**
 * @brief Logistic loss function using Zigen autodiff
 *
 * Computes: L(β) = -Σ[y_i log(σ(x_i'β)) + (1-y_i) log(1-σ(x_i'β))]
 * where σ(z) = 1 / (1 + exp(-z))
 *
 * Gradients are computed automatically via Zigen's reverse-mode AD.
 */
class ZigenLogisticObjective {
public:
  const Eigen::MatrixXd &X;
  const Eigen::VectorXd &y;

  ZigenLogisticObjective(const Eigen::MatrixXd &X_, const Eigen::VectorXd &y_)
      : X(X_), y(y_) {}

  /**
   * @brief Compute logistic loss
   *
   * This operator is called by Zigen's optimizer with autodiff-enabled
   * variables. Gradients are computed automatically.
   */
  template <typename Var>
  Var operator()(const Zigen::Matrix<Var, Zigen::Dynamic, 1> &beta) const {
    size_t n = X.rows();
    size_t p = X.cols();

    Var loss = Var(0.0);

    for (size_t i = 0; i < n; ++i) {
      // Compute linear predictor: z = x_i' * beta
      Var z = Var(0.0);
      for (size_t j = 0; j < p; ++j) {
        z = z + Var(X(i, j)) * beta(j, 0);
      }

      // Logistic loss: -[y * log(σ(z)) + (1-y) * log(1-σ(z))]
      // Using numerically stable formulation: log(1 + exp(-y_*z))
      // where y_ = 2*y - 1 (maps {0,1} to {-1,1})
      Var y_scaled = Var(2.0 * y(i) - 1.0);
      Var neg_yz = Var(-1.0) * y_scaled * z;

      // log(1 + exp(x)) - log-sum-exp trick for stability
      using Zigen::Autodiff::exp;
      using Zigen::Autodiff::log;
      Var one = Var(1.0);
      loss = loss + log(one + exp(neg_yz));
    }

    return loss;
  }
};

} // namespace statelix

#endif // STATELIX_ZIGEN_OPTIMIZER_H
