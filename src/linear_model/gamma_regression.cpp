#include "linear_model/gamma_regression.h"
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <algorithm>
#include <limits>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace statelix {

// 正規分布の累積分布関数
double gamma_norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// リンク関数: η = g(μ)
double link_function(double mu, GammaLink link) {
    switch (link) {
        case GammaLink::LOG:
            return std::log(mu);
        case GammaLink::INVERSE:
            return 1.0 / mu;
        case GammaLink::IDENTITY:
            return mu;
        default:
            return std::log(mu);
    }
}

// 逆リンク関数: μ = g^(-1)(η)
double inverse_link(double eta, GammaLink link) {
    switch (link) {
        case GammaLink::LOG:
            return std::exp(eta);
        case GammaLink::INVERSE:
            return 1.0 / eta;
        case GammaLink::IDENTITY:
            return eta;
        default:
            return std::exp(eta);
    }
}

// リンク関数の導関数: dη/dμ
double link_derivative(double mu, GammaLink link) {
    switch (link) {
        case GammaLink::LOG:
            return 1.0 / mu;
        case GammaLink::INVERSE:
            return -1.0 / (mu * mu);
        case GammaLink::IDENTITY:
            return 1.0;
        default:
            return 1.0 / mu;
    }
}

// ガンマ分布のデビアンス
double gamma_deviance(const VectorXd& y, const VectorXd& mu) {
    double dev = 0.0;
    int n = y.size();
    
    for (int i = 0; i < n; ++i) {
        dev += -2.0 * (std::log(y(i) / mu(i)) - (y(i) - mu(i)) / mu(i));
    }
    
    return dev;
}

// ガンマ分布の対数尤度（φは既知として）
double gamma_log_likelihood(const VectorXd& y, const VectorXd& mu, double phi) {
    double ll = 0.0;
    int n = y.size();
    
    for (int i = 0; i < n; ++i) {
        // log f(y|mu,phi) = -log(y) - log(Gamma(1/phi)) + (1/phi)*log(y/(phi*mu))
        //                   - y/(phi*mu) - (1/phi)*log(phi)
        // 簡易版（定数項は省略）
        ll += -std::log(y(i)) - y(i) / mu(i) - std::log(mu(i));
    }
    
    return ll / phi;  // 正規化
}

// φパラメータの推定（モーメント法）
double estimate_phi_gamma(const VectorXd& y, const VectorXd& mu) {
    int n = y.size();
    
    // Pearson統計量を使用
    double pearson_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = (y(i) - mu(i)) / mu(i);
        pearson_sum += r * r;
    }
    
    double phi = pearson_sum / n;
    
    if (phi <= 0 || std::isnan(phi) || std::isinf(phi)) {
        phi = 1.0;
    }
    
    return phi;
}

// ガンマ回帰のフィッティング
GammaResult fit_gamma(
    const MatrixXd& X,
    const VectorXd& y,
    GammaLink link,
    bool fit_intercept,
    int max_iter,
    double tol,
    double conf_level
) {
    int n = X.rows();
    int p_original = X.cols();
    
    GammaResult result;
    result.n_obs = n;
    result.link = link;
    
    // yは全て正である必要がある
    for (int i = 0; i < n; ++i) {
        if (y(i) <= 0) {
            throw std::runtime_error("Gamma regression requires all y > 0");
        }
    }
    
    // デザイン行列の作成
    MatrixXd X_design;
    if (fit_intercept) {
        X_design.resize(n, p_original + 1);
        X_design.col(0) = VectorXd::Ones(n);
        X_design.rightCols(p_original) = X;
    } else {
        X_design = X;
    }
    
    int p = X_design.cols();
    result.n_params = p;
    
    // 初期値: μ = y の平均
    double y_mean = y.mean();
    VectorXd mu = VectorXd::Constant(n, y_mean);
    VectorXd eta(n);
    for (int i = 0; i < n; ++i) {
        eta(i) = link_function(mu(i), link);
    }
    
    VectorXd beta = VectorXd::Zero(p);
    
    // ヌルモデル
    VectorXd mu_null = VectorXd::Constant(n, y_mean);
    result.null_deviance = gamma_deviance(y, mu_null);
    
    // IRLS
    result.converged = false;
    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        
        // 重み: W_i = 1 / (g'(μ)^2 * V(μ))
        // ガンマ分布で V(μ) = μ^2
        VectorXd W(n);
        VectorXd z(n);
        
        for (int i = 0; i < n; ++i) {
            double g_prime = link_derivative(mu(i), link);
            double variance = mu(i) * mu(i);  // V(μ) = μ^2
            W(i) = 1.0 / (g_prime * g_prime * variance);
            
            if (W(i) < 1e-12) W(i) = 1e-12;
            
            // 作業応答変数
            z(i) = eta(i) + (y(i) - mu(i)) * g_prime;
        }
        
        // 重み付き最小二乗法
        MatrixXd XtW = X_design.transpose() * W.asDiagonal();
        MatrixXd XtWX = XtW * X_design;
        VectorXd XtWz = XtW * z;
        
        VectorXd beta_new = XtWX.ldlt().solve(XtWz);
        
        // 収束判定
        if ((beta_new - beta).norm() < tol) {
            beta = beta_new;
            result.converged = true;
            break;
        }
        
        beta = beta_new;
        eta = X_design * beta;
        
        // μの更新
        for (int i = 0; i < n; ++i) {
            mu(i) = inverse_link(eta(i), link);
            // 数値安定性のためμは正の値を保持
            if (mu(i) <= 1e-12) mu(i) = 1e-12;
        }
    }
    
    // 最終結果
    result.linear_predictors = eta;
    result.fitted_values = mu;
    
    // 係数の抽出
    if (fit_intercept) {
        result.intercept = beta(0);
        result.coef = beta.tail(p_original);
    } else {
        result.intercept = 0.0;
        result.coef = beta;
    }
    
    // φパラメータの推定
    result.phi = estimate_phi_gamma(y, mu);
    
    // デビアンス残差
    result.deviance_residuals.resize(n);
    for (int i = 0; i < n; ++i) {
        double sign = (y(i) > mu(i)) ? 1.0 : -1.0;
        double di = -2.0 * (std::log(y(i) / mu(i)) - (y(i) - mu(i)) / mu(i));
        result.deviance_residuals(i) = sign * std::sqrt(std::abs(di));
    }
    
    // ピアソン残差
    result.pearson_residuals.resize(n);
    for (int i = 0; i < n; ++i) {
        result.pearson_residuals(i) = (y(i) - mu(i)) / mu(i);  // V(μ) = μ^2 なので
    }
    
    // デビアンスと尤度
    result.deviance = gamma_deviance(y, mu);
    result.log_likelihood = gamma_log_likelihood(y, mu, result.phi);
    
    // 擬似R²
    result.pseudo_r_squared = 1.0 - (result.deviance / result.null_deviance);
    
    // AIC, BIC
    result.aic = -2.0 * result.log_likelihood + 2.0 * p;
    result.bic = -2.0 * result.log_likelihood + p * std::log(static_cast<double>(n));
    
    // 分散共分散行列
    VectorXd W_final(n);
    for (int i = 0; i < n; ++i) {
        double g_prime = link_derivative(mu(i), link);
        double variance = mu(i) * mu(i);
        W_final(i) = 1.0 / (g_prime * g_prime * variance);
        if (W_final(i) < 1e-12) W_final(i) = 1e-12;
    }
    
    MatrixXd Fisher = X_design.transpose() * W_final.asDiagonal() * X_design;
    result.vcov = result.phi * Fisher.ldlt().solve(MatrixXd::Identity(p, p));
    
    // 標準誤差
    VectorXd se_all(p);
    for (int i = 0; i < p; ++i) {
        se_all(i) = std::sqrt(std::max(0.0, result.vcov(i, i)));
    }
    
    if (fit_intercept) {
        result.std_errors = se_all.tail(p_original);
    } else {
        result.std_errors = se_all;
    }
    
    // z統計量とp値
    result.z_values.resize(p_original);
    result.p_values.resize(p_original);
    
    for (int i = 0; i < p_original; ++i) {
        int idx = fit_intercept ? i + 1 : i;
        result.z_values(i) = beta(idx) / se_all(idx);
        
        double z_abs = std::abs(result.z_values(i));
        double p_one_sided = 1.0 - gamma_norm_cdf(z_abs);
        result.p_values(i) = 2.0 * p_one_sided;
    }
    
    // 信頼区間
    double z_crit = 1.96;
    if (conf_level == 0.99) {
        z_crit = 2.576;
    } else if (conf_level == 0.90) {
        z_crit = 1.645;
    }
    
    result.conf_int.resize(p_original, 2);
    for (int i = 0; i < p_original; ++i) {
        double margin = z_crit * result.std_errors(i);
        result.conf_int(i, 0) = result.coef(i) - margin;
        result.conf_int(i, 1) = result.coef(i) + margin;
    }
    
    return result;
}

// 予測関数
VectorXd predict_gamma(
    const GammaResult& result,
    const MatrixXd& X_new,
    bool fit_intercept
) {
    int n_new = X_new.rows();
    
    VectorXd eta;
    if (fit_intercept) {
        eta = VectorXd::Constant(n_new, result.intercept);
        eta += X_new * result.coef;
    } else {
        eta = X_new * result.coef;
    }
    
    VectorXd predictions(n_new);
    for (int i = 0; i < n_new; ++i) {
        predictions(i) = inverse_link(eta(i), result.link);
    }
    
    return predictions;
}

} // namespace statelix
