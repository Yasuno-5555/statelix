#ifndef STATELIX_GLM_MODELS_H
#define STATELIX_GLM_MODELS_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

// 1. Logistic Regression (Binary)
struct LogisticResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

class LogisticRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    LogisticResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict_prob(const Eigen::MatrixXd& X, const Eigen::VectorXd& coef);
};

// 2. Poisson Regression (Count)
struct PoissonResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

class PoissonRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    PoissonResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

// 3. Negative Binomial Regression (Overdispersed Count)
struct NegBinResult {
    Eigen::VectorXd coef;
    double theta;
    int iterations;
    bool converged;
};

class NegBinRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    NegBinResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

// 4. Gamma Regression (Continuous Positive)
struct GammaResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

class GammaRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    GammaResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

// 5. Probit Regression (Binary Probit Link)
struct ProbitResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

class ProbitRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    ProbitResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

} // namespace statelix

#endif // STATELIX_GLM_MODELS_H
