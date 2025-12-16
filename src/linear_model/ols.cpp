#include "ols.h"
#include <Eigen/Dense>

#include <cmath>
#include <algorithm>
#include "../stats/math_utils.h"

namespace statelix {

// Util functions are now in statelix::stats namespace

// 基本的なOLS（既存の実装を維持しつつ堅牢化）
Eigen::VectorXd fit_ols_qr(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // Check dimensions
    if (X.rows() < X.cols()) {
         throw std::runtime_error("Sample size is smaller than the number of parameters.");
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
    if (qr.rank() < X.cols()) {
        throw std::runtime_error("Singular matrix detected in OLS (AR Model). The time series might be constant or perfectly collinear.");
    }
    return qr.solve(y);
}

// 完全なOLS実装
OLSResult fit_ols_full(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    bool fit_intercept,
    double conf_level
) {
    int n = X.rows();
    int p_original = X.cols();
    
    // Check dimensions
    if (n < p_original + (fit_intercept ? 1 : 0)) {
        throw std::runtime_error("Sample size is smaller than the number of parameters. Cannot solve OLS.");
    }
    
    OLSResult result;
    
    // 切片を追加する場合
    Eigen::MatrixXd X_design;
    if (fit_intercept) {
        X_design.resize(n, p_original + 1);
        X_design.col(0) = Eigen::VectorXd::Ones(n);
        X_design.rightCols(p_original) = X;
    } else {
        X_design = X;
    }
    
    int p = X_design.cols();
    
    // Use ColPivHouseholderQR for rank-revealing decomposition
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_design);
    
    // Check Rank
    if (qr.rank() < p) {
        throw std::runtime_error("Singular matrix detected. The design matrix is rank deficient (perfect multicollinearity). Cannot compute unique coefficients or standard errors.");
    }

    Eigen::VectorXd beta_all = qr.solve(y);
    
    // 切片と係数を分離
    if (fit_intercept) {
        result.intercept = beta_all(0);
        result.coef = beta_all.tail(p_original);
    } else {
        result.intercept = 0.0;
        result.coef = beta_all;
    }
    
    // 予測値と残差
    result.fitted_values = X_design * beta_all;
    result.residuals = y - result.fitted_values;
    
    // 残差平方和 (SSE)
    double sse = result.residuals.squaredNorm();
    
    // 自由度
    int df_resid = n - p;
    int df_model = p - (fit_intercept ? 1 : 0);
    
    if (df_resid <= 0) {
        throw std::runtime_error("Degrees of freedom for residuals is <= 0. Cannot compute statistics.");
    }
    
    // 残差標準誤差
    result.residual_std_error = std::sqrt(sse / df_resid);
    
    // 平均からの偏差平方和 (SST)
    double y_mean = y.mean();
    double sst = (y.array() - y_mean).square().sum();
    
    // R²
    if (std::abs(sst) < 1e-12) {
        // Constant target variable
        result.r_squared = 0.0; // or 1.0? 0 is safer.
    } else {
        result.r_squared = 1.0 - (sse / sst);
    }
    
    // 調整済みR²
    result.adj_r_squared = 1.0 - ((sse / df_resid) / (sst / (n - 1)));
    
    // F統計量
    double mse_model = (sst - sse) / df_model;
    double mse_resid = sse / df_resid;
    if (mse_resid < 1e-12) {
        // Perfect fit or overfitting
        result.f_statistic = std::numeric_limits<double>::infinity();
        result.f_pvalue = 0.0;
    } else {
        result.f_statistic = mse_model / mse_resid;
        result.f_pvalue = stats::f_pvalue_approx(result.f_statistic, df_model, df_resid);
    }
    
    // 分散共分散行列
    // XtX inverse can be problematic if singular (checked above, but still)
    // Use QR to compute (X^T X)^-1 = (R^T Q^T Q R)^-1 = (R^T R)^-1 = R^-1 (R^T)^-1
    // Efficient way: qr.inverse() but ColPivHouseholderQR doesn't have it directly for XtX
    // Just invert XtX explicitly as before, usually safe if rank check passed.
    
    Eigen::MatrixXd XtX = X_design.transpose() * X_design;
    
    // Check Condition Number roughly?
    // Using LDAP or just inverse.
    // If we passed rank check, inverse should exist, though might be unstable.
    result.vcov = (sse / df_resid) * XtX.inverse();
    
    // 標準誤差（対角要素の平方根）
    Eigen::VectorXd se_all(p);
    for (int i = 0; i < p; ++i) {
        double var = result.vcov(i, i);
        if (var < 0 || std::isnan(var)) se_all(i) = 0.0; // Should not happen if positive definite
        else se_all(i) = std::sqrt(var);
    }
    
    if (fit_intercept) {
        result.std_errors = se_all.tail(p_original);
    } else {
        result.std_errors = se_all;
    }
    
    // t統計量とp値
    result.t_values.resize(p_original);
    result.p_values.resize(p_original);
    
    for (int i = 0; i < p_original; ++i) {
        int idx = fit_intercept ? i + 1 : i;
        if (se_all(idx) < 1e-12) {
             result.t_values(i) = 0.0; // Avoid division by zero
             result.p_values(i) = 1.0;
        } else {
            result.t_values(i) = beta_all(idx) / se_all(idx);
            
            // 両側検定のp値
            double t_abs = std::abs(result.t_values(i));
            double p_one_sided = 1.0 - stats::t_cdf_approx(t_abs, df_resid);
            result.p_values(i) = 2.0 * p_one_sided;
        }
    }
    
    // 信頼区間
    double alpha = 1.0 - conf_level;
    // t分布の臨界値（簡易近似：大サンプルでは正規分布と近似）
    double t_crit = 1.96; // 95%信頼区間の場合
    if (conf_level == 0.95) {
        if (df_resid < 30) {
            // 小サンプルの場合の簡易調整
            t_crit = 2.0 + 0.5 / std::sqrt(static_cast<double>(df_resid));
        }
    } else if (conf_level == 0.99) {
        t_crit = 2.576;
    } else if (conf_level == 0.90) {
        t_crit = 1.645;
    }
    
    result.conf_int.resize(p_original, 2);
    for (int i = 0; i < p_original; ++i) {
        double margin = t_crit * result.std_errors(i);
        result.conf_int(i, 0) = result.coef(i) - margin;
        result.conf_int(i, 1) = result.coef(i) + margin;
    }
    
    // AIC, BIC
    if (sse < 1e-12) {
        // Perfect fit
        result.log_likelihood = std::numeric_limits<double>::infinity();
        result.aic = -std::numeric_limits<double>::infinity();
        result.bic = -std::numeric_limits<double>::infinity();
    } else {
        result.log_likelihood = -0.5 * n * (std::log(2.0 * M_PI) + std::log(sse / n) + 1.0);
        result.aic = -2.0 * result.log_likelihood + 2.0 * p;
        result.bic = -2.0 * result.log_likelihood + p * std::log(static_cast<double>(n));
    }
    
    // メタデータ
    result.n_obs = n;
    result.n_params = p;
    
    return result;
}

// 予測
Eigen::VectorXd predict_ols(
    const OLSResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept
) {
    int n_new = X_new.rows();
    Eigen::VectorXd predictions(n_new);
    
    if (fit_intercept) {
        predictions = Eigen::VectorXd::Constant(n_new, result.intercept);
        predictions += X_new * result.coef;
    } else {
        predictions = X_new * result.coef;
    }
    
    return predictions;
}

// 予測区間を含む予測
PredictionInterval predict_with_interval(
    const OLSResult& result,
    const Eigen::MatrixXd& X_new,
    bool fit_intercept,
    double conf_level
) {
    PredictionInterval pred_int;
    pred_int.predictions = predict_ols(result, X_new, fit_intercept);
    
    int n_new = X_new.rows();
    int p_original = result.coef.size();
    
    // t分布の臨界値
    double t_crit = 1.96; // 95%の場合
    if (conf_level == 0.95) {
        int df = result.n_obs - result.n_params;
        if (df < 30) {
            t_crit = 2.0 + 0.5 / std::sqrt(static_cast<double>(df));
        }
    } else if (conf_level == 0.99) {
        t_crit = 2.576;
    } else if (conf_level == 0.90) {
        t_crit = 1.645;
    }
    
    pred_int.lower_bound.resize(n_new);
    pred_int.upper_bound.resize(n_new);
    
    // 予測区間の計算
    for (int i = 0; i < n_new; ++i) {
        Eigen::VectorXd x_i;
        if (fit_intercept) {
            x_i.resize(p_original + 1);
            x_i(0) = 1.0;
            x_i.tail(p_original) = X_new.row(i).transpose();
        } else {
            x_i = X_new.row(i).transpose();
        }
        
        // 予測の分散: σ²(1 + x'(X'X)⁻¹x)
        double var_pred = result.residual_std_error * result.residual_std_error;
        var_pred *= (1.0 + x_i.dot(result.vcov * x_i));
        double se_pred = std::sqrt(std::max(0.0, var_pred));
        
        double margin = t_crit * se_pred;
        pred_int.lower_bound(i) = pred_int.predictions(i) - margin;
        pred_int.upper_bound(i) = pred_int.predictions(i) + margin;
    }
    
    return pred_int;
}

} // namespace statelix

// Python bindings moved to src/bindings/python_bindings.cpp

