#include "negative_binomial.h"
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
double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// 対数ガンマ関数の近似
double lgamma_approx(double x) {
    if (x <= 0) return std::numeric_limits<double>::infinity();
    return std::lgamma(x);
}

// 負の二項分布の対数尤度
double negbin_log_likelihood(const VectorXd& y, const VectorXd& mu, double theta) {
    double ll = 0.0;
    int n = y.size();
    
    for (int i = 0; i < n; ++i) {
        double yi = y(i);
        double mui = mu(i);
        
        // log P(Y=y|mu,theta) = lgamma(y+theta) - lgamma(theta) - lgamma(y+1)
        //                      + theta*log(theta) - theta*log(theta+mu)
        //                      + y*log(mu) - y*log(theta+mu)
        
        ll += lgamma_approx(yi + theta) - lgamma_approx(theta) - lgamma_approx(yi + 1.0);
        ll += theta * std::log(theta / (theta + mui));
        ll += yi * std::log(mui / (theta + mui));
    }
    
    return ll;
}

// 負の二項デビアンスの計算
double negbin_deviance(const VectorXd& y, const VectorXd& mu, double theta) {
    double dev = 0.0;
    int n = y.size();
    
    for (int i = 0; i < n; ++i) {
        double yi = y(i);
        double mui = mu(i);
        
        if (yi > 0) {
            dev += 2.0 * (yi * std::log(yi / mui) - (yi + theta) * std::log((yi + theta) / (mui + theta)));
        } else {
            dev += 2.0 * theta * std::log(theta / (mui + theta));
        }
    }
    
    return dev;
}

// θパラメータの推定（モーメント法）
double estimate_theta_moment(const VectorXd& y, const VectorXd& mu) {
    int n = y.size();
    
    // Var(Y) = mu + mu^2/theta
    // theta = mu^2 / (Var(Y) - mu)
    
    double mean_y = y.mean();
    double var_y = (y.array() - mean_y).square().sum() / (n - 1.0);
    
    double mean_mu = mu.mean();
    
    double theta_est = mean_mu * mean_mu / std::max(1e-6, var_y - mean_mu);
    
    // θは正の値である必要がある
    if (theta_est <= 0 || std::isnan(theta_est) || std::isinf(theta_est)) {
        theta_est = 1.0;
    }
    
    return theta_est;
}

// 負の二項回帰のフィッティング
NegativeBinomialResult fit_negative_binomial(
    const MatrixXd& X,
    const VectorXd& y,
    bool fit_intercept,
    const VectorXd& offset,
    double theta_init,
    int max_iter,
    double tol,
    double conf_level
) {
    int n = X.rows();
    int p_original = X.cols();
    
    NegativeBinomialResult result;
    result.n_obs = n;
    
    // オフセットの処理
    VectorXd offset_vec;
    if (offset.size() == 0) {
        offset_vec = VectorXd::Zero(n);
    } else {
        offset_vec = offset;
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
    result.n_params = p + 1; // β + θ
    
    // 初期値
    VectorXd beta = VectorXd::Zero(p);
    VectorXd eta = X_design * beta + offset_vec;
    VectorXd mu = eta.array().exp();
    
    double theta = theta_init;
    
    // ヌルモデル
    double y_mean = y.mean();
    VectorXd mu_null = VectorXd::Constant(n, y_mean);
    result.null_deviance = negbin_deviance(y, mu_null, theta);
    
    // IRLS with theta estimation
    result.converged = false;
    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        
        // θの更新（モーメント法）
        theta = estimate_theta_moment(y, mu);
        theta = std::max(0.01, std::min(theta, 1000.0)); // 安定性のため範囲制限
        
        // 重み: W = mu / (1 + mu/theta)
        VectorXd W(n);
        for (int i = 0; i < n; ++i) {
            W(i) = mu(i) / (1.0 + mu(i) / theta);
            if (W(i) < 1e-12) W(i) = 1e-12;
        }
        
        // 作業応答変数: z = eta + (y - mu) / mu * (1 + mu/theta)
        VectorXd z(n);
        for (int i = 0; i < n; ++i) {
            z(i) = eta(i) + (y(i) - mu(i)) / mu(i) * (1.0 + mu(i) / theta);
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
        eta = X_design * beta + offset_vec;
        mu = eta.array().exp();
    }
    
    // 最終結果
    result.theta = theta;
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
    
    // デビアンス残差
    result.deviance_residuals.resize(n);
    for (int i = 0; i < n; ++i) {
        double sign = (y(i) > mu(i)) ? 1.0 : -1.0;
        double di = 0.0;
        if (y(i) > 0) {
            di = 2.0 * (y(i) * std::log(y(i) / mu(i)) 
                - (y(i) + theta) * std::log((y(i) + theta) / (mu(i) + theta)));
        } else {
            di = 2.0 * theta * std::log(theta / (mu(i) + theta));
        }
        result.deviance_residuals(i) = sign * std::sqrt(std::abs(di));
    }
    
    // ピアソン残差
    result.pearson_residuals.resize(n);
    for (int i = 0; i < n; ++i) {
        double var_i = mu(i) + mu(i) * mu(i) / theta;
        result.pearson_residuals(i) = (y(i) - mu(i)) / std::sqrt(var_i);
    }
    
    // デビアンスと尤度
    result.deviance = negbin_deviance(y, mu, theta);
    result.log_likelihood = negbin_log_likelihood(y, mu, theta);
    
    // 擬似R²
    result.pseudo_r_squared = 1.0 - (result.deviance / result.null_deviance);
    
    // AIC, BIC
    result.aic = -2.0 * result.log_likelihood + 2.0 * result.n_params;
    result.bic = -2.0 * result.log_likelihood + result.n_params * std::log(static_cast<double>(n));
    
    // 分散共分散行列（Fisher情報行列の逆行列）
    VectorXd W_final(n);
    for (int i = 0; i < n; ++i) {
        W_final(i) = mu(i) / (1.0 + mu(i) / theta);
        if (W_final(i) < 1e-12) W_final(i) = 1e-12;
    }
    MatrixXd Fisher = X_design.transpose() * W_final.asDiagonal() * X_design;
    result.vcov = Fisher.ldlt().solve(MatrixXd::Identity(p, p));
    
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
        double p_one_sided = 1.0 - norm_cdf(z_abs);
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
VectorXd predict_negative_binomial(
    const NegativeBinomialResult& result,
    const MatrixXd& X_new,
    bool fit_intercept,
    const VectorXd& offset,
    bool return_log
) {
    int n_new = X_new.rows();
    
    VectorXd offset_vec;
    if (offset.size() == 0) {
        offset_vec = VectorXd::Zero(n_new);
    } else {
        offset_vec = offset;
    }
    
    VectorXd eta;
    if (fit_intercept) {
        eta = VectorXd::Constant(n_new, result.intercept);
        eta += X_new * result.coef + offset_vec;
    } else {
        eta = X_new * result.coef + offset_vec;
    }
    
    if (return_log) {
        return eta;
    } else {
        return eta.array().exp();
    }
}

} // namespace statelix

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(statelix_negative_binomial, m) {
    m.doc() = "Negative Binomial regression (overdispersed count data) module";
    
    py::class_<statelix::NegativeBinomialResult>(m, "NegativeBinomialResult")
        .def_readonly("coef", &statelix::NegativeBinomialResult::coef, "Regression coefficients")
        .def_readonly("intercept", &statelix::NegativeBinomialResult::intercept, "Intercept term")
        .def_readonly("theta", &statelix::NegativeBinomialResult::theta, "Dispersion parameter theta")
        .def_readonly("std_errors", &statelix::NegativeBinomialResult::std_errors, "Standard errors")
        .def_readonly("z_values", &statelix::NegativeBinomialResult::z_values, "z-statistics")
        .def_readonly("p_values", &statelix::NegativeBinomialResult::p_values, "p-values")
        .def_readonly("conf_int", &statelix::NegativeBinomialResult::conf_int, "Confidence intervals")
        .def_readonly("fitted_values", &statelix::NegativeBinomialResult::fitted_values, "Fitted values")
        .def_readonly("linear_predictors", &statelix::NegativeBinomialResult::linear_predictors, "Linear predictors")
        .def_readonly("deviance_residuals", &statelix::NegativeBinomialResult::deviance_residuals, "Deviance residuals")
        .def_readonly("pearson_residuals", &statelix::NegativeBinomialResult::pearson_residuals, "Pearson residuals")
        .def_readonly("log_likelihood", &statelix::NegativeBinomialResult::log_likelihood, "Log-likelihood")
        .def_readonly("deviance", &statelix::NegativeBinomialResult::deviance, "Deviance")
        .def_readonly("null_deviance", &statelix::NegativeBinomialResult::null_deviance, "Null deviance")
        .def_readonly("aic", &statelix::NegativeBinomialResult::aic, "AIC")
        .def_readonly("bic", &statelix::NegativeBinomialResult::bic, "BIC")
        .def_readonly("pseudo_r_squared", &statelix::NegativeBinomialResult::pseudo_r_squared, "Pseudo R-squared")
        .def_readonly("vcov", &statelix::NegativeBinomialResult::vcov, "Variance-covariance matrix")
        .def_readonly("iterations", &statelix::NegativeBinomialResult::iterations, "Number of iterations")
        .def_readonly("converged", &statelix::NegativeBinomialResult::converged, "Convergence flag")
        .def_readonly("n_obs", &statelix::NegativeBinomialResult::n_obs, "Number of observations")
        .def_readonly("n_params", &statelix::NegativeBinomialResult::n_params, "Number of parameters");
    
    m.def("fit_negative_binomial", &statelix::fit_negative_binomial,
          "Fit Negative Binomial regression using IRLS",
          py::arg("X"), py::arg("y"),
          py::arg("fit_intercept") = true,
          py::arg("offset") = VectorXd(),
          py::arg("theta_init") = 1.0,
          py::arg("max_iter") = 50,
          py::arg("tol") = 1e-8,
          py::arg("conf_level") = 0.95);
    
    m.def("predict_negative_binomial", &statelix::predict_negative_binomial,
          "Make predictions using fitted Negative Binomial model",
          py::arg("result"), py::arg("X_new"),
          py::arg("fit_intercept") = true,
          py::arg("offset") = VectorXd(),
          py::arg("return_log") = false);
}
