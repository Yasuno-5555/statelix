/**
 * @file logistic.h
 * @brief Logistic Regression using Zigen Autodiff Optimizer
 */
#ifndef STATELIX_LOGISTIC_H
#define STATELIX_LOGISTIC_H

#include <Eigen/Dense>

namespace statelix {

struct LogisticResult {
  Eigen::VectorXd coef;
  double intercept = 0.0;
  int iterations;
  bool converged;
  double deviance = 0.0;
  double aic = 0.0;
};

class LogisticRegression {
public:
  int max_iter = 100;
  double tol = 1e-6;
  bool fit_intercept = true;

  LogisticResult fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

  Eigen::VectorXd predict_prob(const Eigen::MatrixXd &X,
                               const Eigen::VectorXd &coef,
                               double intercept = 0.0);
};

} // namespace statelix

#endif // STATELIX_LOGISTIC_H
