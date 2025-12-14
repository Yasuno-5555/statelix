#pragma once
#include <Eigen/Dense>

namespace statelix {

// OLS回帰の結果を格納する構造体
struct OLSResult {
    // 回帰係数
    Eigen::VectorXd coef;
    
    // 切片（インターセプト）
    double intercept;
    
    // 標準誤差（係数の）
    Eigen::VectorXd std_errors;
    
    // t統計量
    Eigen::VectorXd t_values;
    
    // p値（両側検定）
    Eigen::VectorXd p_values;
    
    // 信頼区間（95%デフォルト）
    Eigen::MatrixXd conf_int;  // shape: (p, 2) - [lower, upper]
    
    // 残差
    Eigen::VectorXd residuals;
    
    // 予測値
    Eigen::VectorXd fitted_values;
    
    // 決定係数
    double r_squared;
    
    // 調整済み決定係数
    double adj_r_squared;
    
    // F統計量
    double f_statistic;
    
    // F統計量のp値
    double f_pvalue;
    
    // 残差標準誤差
    double residual_std_error;
    
    // AIC (Akaike Information Criterion)
    double aic;
    
    // BIC (Bayesian Information Criterion)
    double bic;
    
    // 対数尤度
    double log_likelihood;
    
    // 分散共分散行列
    Eigen::MatrixXd vcov;
    
    // サンプルサイズ
    int n_obs;
    
    // パラメータ数（切片含む）
    int n_params;
};

// 基本的なOLS回帰（QR分解使用）
Eigen::VectorXd fit_ols_qr(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

// 完全なOLS回帰（統計量を含む）
OLSResult fit_ols_full(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    bool fit_intercept = true,
    double conf_level = 0.95
);

// 予測
Eigen::VectorXd predict_ols(
    const OLSResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept = true
);

// 予測区間を含む予測
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
