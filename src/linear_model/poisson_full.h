#ifndef STATELIX_POISSON_FULL_H
#define STATELIX_POISSON_FULL_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {
namespace poisson_detail {

struct PoissonResult {
    Eigen::VectorXd coef;
    double intercept;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd z_values;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd conf_int;
    Eigen::VectorXd fitted_values;
    Eigen::VectorXd linear_predictors;
    Eigen::VectorXd deviance_residuals;
    Eigen::VectorXd pearson_residuals;
    double log_likelihood;
    double deviance;
    double null_deviance;
    double aic;
    double bic;
    double pseudo_r_squared;
    Eigen::MatrixXd vcov;
    int iterations;
    bool converged;
    int n_obs;
    int n_params;
};

PoissonResult fit_poisson(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    bool fit_intercept = true,
    const Eigen::VectorXd& offset = Eigen::VectorXd(),
    int max_iter = 50,
    double tol = 1e-8,
    double conf_level = 0.95
);

Eigen::VectorXd predict_poisson(
    const PoissonResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept = true,
    const Eigen::VectorXd& offset = Eigen::VectorXd(),
    bool return_log = false
);

} // namespace poisson_detail
} // namespace statelix

#endif // STATELIX_POISSON_FULL_H
