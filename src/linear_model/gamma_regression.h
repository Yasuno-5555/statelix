#ifndef STATELIX_GAMMA_REGRESSION_H
#define STATELIX_GAMMA_REGRESSION_H

#include <Eigen/Dense>
#include <string>

namespace statelix {

// ガンマ回帰のリンク関数
enum class GammaLink {
    LOG,      // log(μ) = η (デフォルト)
    INVERSE,  // 1/μ = η (canonical link)
    IDENTITY  // μ = η
};

// ガンマ回帰の結果構造体
struct GammaResult {
    // 回帰係数
    Eigen::VectorXd coef;
    
    // 切片
    double intercept;
    
    // 形状パラメータ φ (dispersion parameter)
    double phi;
    
    // 標準誤差
    Eigen::VectorXd std_errors;
    
    // z統計量
    Eigen::VectorXd z_values;
    
    // p値
    Eigen::VectorXd p_values;
    
    // 信頼区間
    Eigen::MatrixXd conf_int;
    
    // 予測値
    Eigen::VectorXd fitted_values;
    
    // 線形予測子
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
    
    // リンク関数
    GammaLink link;
};

// ガンマ回帰のフィッティング
GammaResult fit_gamma(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    GammaLink link = GammaLink::LOG,
    bool fit_intercept = true,
    int max_iter = 50,
    double tol = 1e-8,
    double conf_level = 0.95
);

// 予測関数
Eigen::VectorXd predict_gamma(
    const GammaResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept = true
);

} // namespace statelix

#endif // STATELIX_GAMMA_REGRESSION_H
