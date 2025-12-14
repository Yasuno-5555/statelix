#ifndef STATELIX_RIDGE_H
#define STATELIX_RIDGE_H

#include <Eigen/Dense>

namespace statelix {

struct RidgeResult {
    Eigen::VectorXd coef;
    // Add more if needed
};

class RidgeRegression {
public:
    double alpha = 1.0; // Lambda

    RidgeResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        int p = X.cols();
        // (X'X + alpha*I) * beta = X'y
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(p, p);
        Eigen::VectorXd coef = (X.transpose() * X + alpha * I).ldlt().solve(X.transpose() * y);
        return {coef};
    }
};

} // namespace statelix

#endif // STATELIX_RIDGE_H
