#ifndef STATELIX_NEGATIVE_BINOMIAL_H
#define STATELIX_NEGATIVE_BINOMIAL_H

#include <Eigen/Dense>

namespace statelix {

// 負の二項回帰の結果構造体
struct NegativeBinomialResult {
    // 回帰係数
    Eigen::VectorXd coef;
    
    // 切片
    double intercept;
    
    // 分散パラメータ θ (過分散パラメータ)
    double theta;
    
    // 標準誤差
    Eigen::VectorXd std_errors;
    
    // z統計量
    Eigen::VectorXd z_values;
    
    // p値
    Eigen::VectorXd p_values;
    
    // 信頼区間
    Eigen::MatrixXd conf_int;
    
    // 予測値（カウント空間）
    Eigen::VectorXd fitted_values;
    
    // 線形予測子（対数空間）
    Eigen::VectorXd linear_predictors;
    
    // デビアンス残差
    Eigen::VectorXd deviance_residuals;
    
    // ピアソン残差
    Eigen::VectorXd pearson_residuals;
    
    // 対数尤度
    double log_likelihood;
    
    // デビアンス
    double deviance;
    
    // ヌルデビアンス
    double null_deviance;
    
    // AIC
    double aic;
    
    // BIC
    double bic;
    
    // 擬似R²
    double pseudo_r_squared;
    
    // 分散共分散行列
    Eigen::MatrixXd vcov;
    
    // 反復回数
    int iterations;
    
    // 収束フラグ
    bool converged;
    
    // サンプルサイズ
    int n_obs;
    
    // パラメータ数
    int n_params;
};

// 負の二項回帰のフィッティング
NegativeBinomialResult fit_negative_binomial(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    bool fit_intercept = true,
    const Eigen::VectorXd& offset = Eigen::VectorXd(),
    double theta_init = 1.0,
    int max_iter = 50,
    double tol = 1e-8,
    double conf_level = 0.95
);

// 予測関数
Eigen::VectorXd predict_negative_binomial(
    const NegativeBinomialResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept = true,
    const Eigen::VectorXd& offset = Eigen::VectorXd(),
    bool return_log = false
);

} // namespace statelix

#endif // STATELIX_NEGATIVE_BINOMIAL_H
