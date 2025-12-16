#pragma once
#include <Eigen/Dense>

namespace statelix {

// OLS回帰の結果を格納する構造体
struct OLSResult {
    Eigen::VectorXd coef;
    double intercept;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd conf_int;  // shape: (p, 2) - [lower, upper]
    Eigen::VectorXd residuals;
    Eigen::VectorXd fitted_values;
    double r_squared;
    double adj_r_squared;
    double f_statistic;
    double f_pvalue;
    double residual_std_error;
    double aic;
    double bic;
    double log_likelihood;
    Eigen::MatrixXd vcov;
    int n_obs;
    int n_params;
};

// Basic OLS (QR)
Eigen::VectorXd fit_ols_qr(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

// Full OLS
OLSResult fit_ols_full(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    bool fit_intercept = true,
    double conf_level = 0.95
);

// Prediction
Eigen::VectorXd predict_ols(
    const OLSResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept = true
);

// Prediction with Interval
struct PredictionInterval {
    Eigen::VectorXd predictions;
    Eigen::VectorXd lower_bound;
    Eigen::VectorXd upper_bound;
};

PredictionInterval predict_with_interval(
    const OLSResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept = true,
    double conf_level = 0.95
);

} // namespace statelix
