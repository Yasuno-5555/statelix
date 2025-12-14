#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <algorithm>
#include <limits>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace statelix {

// Poisson回帰の結果構造体
struct PoissonResult {
    // 回帰係数
    VectorXd coef;
    
    // 切片
    double intercept;
    
    // 標準誤差
    VectorXd std_errors;
    
    // z統計量（GLMではtではなくz）
    VectorXd z_values;
    
    // p値
    VectorXd p_values;
    
    // 信頼区間
    MatrixXd conf_int;
    
    // 予測値（カウント空間）
    VectorXd fitted_values;
    
    // 線形予測子（対数空間）
    VectorXd linear_predictors;
    
    // デビアンス残差
    VectorXd deviance_residuals;
    
    // ピアソン残差
    VectorXd pearson_residuals;
    
    // 対数尤度
    double log_likelihood;
    
    // デビアンス（逸脱度）
    double deviance;
    
    // ヌルデビアンス
    double null_deviance;
    
    // AIC
    double aic;
    
    // BIC
    double bic;
    
    // 擬似R² (McFadden's R²)
    double pseudo_r_squared;
    
    // 分散共分散行列
    MatrixXd vcov;
    
    // 反復回数
    int iterations;
    
    // 収束フラグ
    bool converged;
    
    // サンプルサイズ
    int n_obs;
    
    // パラメータ数
    int n_params;
};

// 正規分布の累積分布関数（標準正規分布）
double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Poisson対数尤度の計算
double poisson_log_likelihood(const VectorXd& y, const VectorXd& mu) {
    double ll = 0.0;
    for (int i = 0; i < y.size(); ++i) {
        if (mu(i) > 0) {
            ll += y(i) * std::log(mu(i)) - mu(i);
            // 対数階乗項は定数なので省略（デビアンス計算では相殺される）
        }
    }
    return ll;
}

// Poissonデビアンスの計算
double poisson_deviance(const VectorXd& y, const VectorXd& mu) {
    double dev = 0.0;
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) > 0) {
            dev += 2.0 * (y(i) * std::log(y(i) / mu(i)) - (y(i) - mu(i)));
        } else {
            dev += 2.0 * mu(i);
        }
    }
    return dev;
}

// Poisson回帰のフィッティング
PoissonResult fit_poisson(
    const MatrixXd& X,
    const VectorXd& y,
    bool fit_intercept = true,
    const VectorXd& offset = VectorXd(),
    int max_iter = 50,
    double tol = 1e-8,
    double conf_level = 0.95
) {
    int n = X.rows();
    int p_original = X.cols();
    
    PoissonResult result;
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
    result.n_params = p;
    
    // 初期値: β = 0
    VectorXd beta = VectorXd::Zero(p);
    VectorXd eta = X_design * beta + offset_vec;
    VectorXd mu = eta.array().exp();
    
    // ヌルモデルの計算（切片のみ）
    double y_mean = y.mean();
    VectorXd mu_null = VectorXd::Constant(n, y_mean);
    result.null_deviance = poisson_deviance(y, mu_null);
    
    // IRLS (Iteratively Reweighted Least Squares)
    result.converged = false;
    for (int iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        
        // 重み: W = mu（Poissonの分散 = 平均）
        VectorXd W = mu;
        
        // 数値安定性のため、極小の重みを防ぐ
        for (int i = 0; i < n; ++i) {
            if (W(i) < 1e-12) W(i) = 1e-12;
        }
        
        // 作業応答変数: z = eta + (y - mu) / mu
        VectorXd z = eta + (y - mu).array() / mu.array();
        
        // 重み付き最小二乗法
        MatrixXd XtW = X_design.transpose() * W.asDiagonal();
        MatrixXd XtWX = XtW * X_design;
        VectorXd XtWz = XtW * z;
        
        // 新しい係数の計算
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
    
    // 最終的な予測値と残差
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
        double di = 0.0;
        if (y(i) > 0) {
            di = 2.0 * (y(i) * std::log(y(i) / mu(i)) - (y(i) - mu(i)));
        } else {
            di = 2.0 * mu(i);
        }
        result.deviance_residuals(i) = std::sqrt(di) * (y(i) > mu(i) ? 1.0 : -1.0);
    }
    
    // ピアソン残差
    result.pearson_residuals = (y - mu).array() / mu.array().sqrt();
    
    // デビアンスと尤度
    result.deviance = poisson_deviance(y, mu);
    result.log_likelihood = poisson_log_likelihood(y, mu);
    
    // 擬似R²
    result.pseudo_r_squared = 1.0 - (result.deviance / result.null_deviance);
    
    // AIC, BIC
    result.aic = -2.0 * result.log_likelihood + 2.0 * p;
    result.bic = -2.0 * result.log_likelihood + p * std::log(static_cast<double>(n));
    
    // 分散共分散行列（Fisher情報行列の逆行列）
    VectorXd W_final = mu;
    for (int i = 0; i < n; ++i) {
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
        
        // 両側検定のp値（標準正規分布）
        double z_abs = std::abs(result.z_values(i));
        double p_one_sided = 1.0 - norm_cdf(z_abs);
        result.p_values(i) = 2.0 * p_one_sided;
    }
    
    // 信頼区間
    double z_crit = 1.96; // 95%
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
VectorXd predict_poisson(
    const PoissonResult& result,
    const MatrixXd& X_new,
    bool fit_intercept = true,
    const VectorXd& offset = VectorXd(),
    bool return_log = false
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
        return eta; // 対数空間での予測
    } else {
        return eta.array().exp(); // カウント空間での予測
    }
}

} // namespace statelix

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(statelix_poisson, m) {
    m.doc() = "Poisson regression (GLM with log link) module";
    
    // PoissonResult構造体のバインディング
    py::class_<statelix::PoissonResult>(m, "PoissonResult")
        .def_readonly("coef", &statelix::PoissonResult::coef, "Regression coefficients")
        .def_readonly("intercept", &statelix::PoissonResult::intercept, "Intercept term")
        .def_readonly("std_errors", &statelix::PoissonResult::std_errors, "Standard errors")
        .def_readonly("z_values", &statelix::PoissonResult::z_values, "z-statistics")
        .def_readonly("p_values", &statelix::PoissonResult::p_values, "p-values")
        .def_readonly("conf_int", &statelix::PoissonResult::conf_int, "Confidence intervals")
        .def_readonly("fitted_values", &statelix::PoissonResult::fitted_values, "Fitted values (counts)")
        .def_readonly("linear_predictors", &statelix::PoissonResult::linear_predictors, "Linear predictors (log scale)")
        .def_readonly("deviance_residuals", &statelix::PoissonResult::deviance_residuals, "Deviance residuals")
        .def_readonly("pearson_residuals", &statelix::PoissonResult::pearson_residuals, "Pearson residuals")
        .def_readonly("log_likelihood", &statelix::PoissonResult::log_likelihood, "Log-likelihood")
        .def_readonly("deviance", &statelix::PoissonResult::deviance, "Deviance")
        .def_readonly("null_deviance", &statelix::PoissonResult::null_deviance, "Null deviance")
        .def_readonly("aic", &statelix::PoissonResult::aic, "AIC")
        .def_readonly("bic", &statelix::PoissonResult::bic, "BIC")
        .def_readonly("pseudo_r_squared", &statelix::PoissonResult::pseudo_r_squared, "Pseudo R-squared (McFadden)")
        .def_readonly("vcov", &statelix::PoissonResult::vcov, "Variance-covariance matrix")
        .def_readonly("iterations", &statelix::PoissonResult::iterations, "Number of iterations")
        .def_readonly("converged", &statelix::PoissonResult::converged, "Convergence flag")
        .def_readonly("n_obs", &statelix::PoissonResult::n_obs, "Number of observations")
        .def_readonly("n_params", &statelix::PoissonResult::n_params, "Number of parameters");
    
    // 関数のバインディング
    m.def("fit_poisson", &statelix::fit_poisson,
          "Fit Poisson regression using IRLS",
          py::arg("X"), py::arg("y"),
          py::arg("fit_intercept") = true,
          py::arg("offset") = VectorXd(),
          py::arg("max_iter") = 50,
          py::arg("tol") = 1e-8,
          py::arg("conf_level") = 0.95);
    
    m.def("predict_poisson", &statelix::predict_poisson,
          "Make predictions using fitted Poisson model",
          py::arg("result"), py::arg("X_new"),
          py::arg("fit_intercept") = true,
          py::arg("offset") = VectorXd(),
          py::arg("return_log") = false);
}
