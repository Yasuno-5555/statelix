/**
 * @file logistic.cpp
 * @brief Logistic Regression using Zigen Autodiff Optimizer
 */
#include "logistic.h"
#include "../optimization/zigen_optimizer.h"
#include <cmath>
#include <memory>

namespace statelix {

LogisticResult LogisticRegression::fit(const Eigen::MatrixXd &X,
                                       const Eigen::VectorXd &y) {
  // Preprocess intercept
  Eigen::MatrixXd X_train;
  if (fit_intercept) {
    X_train.resize(X.rows(), X.cols() + 1);
    X_train.leftCols(X.cols()) = X;
    X_train.col(X.cols()).setOnes();
  } else {
    X_train = X;
  }

  // Initialize parameters (zero)
  // Note: Zigen optimizer will resize if needed, but good to start correct
  Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(X_train.cols());

  // Setup Zigen objective and optimizer
  ZigenLogisticObjective objective(X_train, y);
  ZigenLBFGS optimizer;
  optimizer.max_iter = max_iter;
  optimizer.tolerance = tol;

  // Run optimization with Autodiff
  OptimizerResult opt_res = optimizer.minimize_autodiff(objective, beta_init);

  // Convert results
  LogisticResult result;
  if (fit_intercept) {
    result.coef = opt_res.x.head(X.cols());
    result.intercept = opt_res.x(X.cols());
  } else {
    result.coef = opt_res.x;
    result.intercept = 0.0;
  }

  result.iterations = opt_res.iterations;
  result.converged = opt_res.converged;

  // Calculate Deviance: -2 * LogLikelihood
  // Zigen minimizes Negative Log Likelihood, so deviance = 2 * min_value
  result.deviance = 2.0 * opt_res.min_value;

  // Calculate AIC: 2*k + Deviance
  double k = (double)opt_res.x.size();
  result.aic = 2.0 * k + result.deviance;

  return result;
}

Eigen::VectorXd LogisticRegression::predict_prob(const Eigen::MatrixXd &X,
                                                 const Eigen::VectorXd &coef,
                                                 double intercept) {
  Eigen::VectorXd eta = X * coef;
  if (fit_intercept) {
    eta.array() += intercept;
  }

  // Apply logistic function
  Eigen::VectorXd probs(eta.size());
  for (int i = 0; i < eta.size(); ++i) {
    if (eta(i) >= 0) {
      double ez = std::exp(-eta(i));
      probs(i) = 1.0 / (1.0 + ez);
    } else {
      double ez = std::exp(eta(i));
      probs(i) = ez / (1.0 + ez);
    }
  }
  return probs;
}

} // namespace statelix
