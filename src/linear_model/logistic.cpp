#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "../glm/glm_models.h"

namespace statelix {

inline double sigmoid(double z) {
    if (z >= 0) {
        double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = std::exp(z);
        return ez / (1.0 + ez);
    }
}

LogisticResult LogisticRegression::fit(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y
) {
    int n = X.rows();
    int p = X.cols();

    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd eta(n), mu(n), z(n), w(n);

    LogisticResult result;
    result.iterations = 0;
    result.converged = false;

    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        eta = X * beta;

        for (int i = 0; i < n; i++) {
            mu(i) = sigmoid(eta(i));
            double m = mu(i);
            w(i) = m * (1.0 - m);

            if (w(i) < 1e-12) w(i) = 1e-12;

            z(i) = eta(i) + (y(i) - m) / w(i);
        }

        Eigen::MatrixXd W = w.asDiagonal();
        Eigen::MatrixXd XtW = X.transpose() * W;
        Eigen::MatrixXd XtWX = XtW * X;
        Eigen::VectorXd XtWz = XtW * z;

        Eigen::VectorXd beta_new = XtWX.ldlt().solve(XtWz);

        if ((beta_new - beta).norm() < tol) {
            beta = beta_new;
            result.converged = true;
            break;
        }

        beta = beta_new;
    }

    result.coef = beta;
    return result;
}

Eigen::VectorXd LogisticRegression::predict_prob(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& coef
) {
    Eigen::VectorXd eta = X * coef;
    Eigen::VectorXd probs(eta.size());
    for (int i = 0; i < eta.size(); ++i) {
        probs(i) = sigmoid(eta(i));
    }
    return probs;
}

} // namespace statelix
