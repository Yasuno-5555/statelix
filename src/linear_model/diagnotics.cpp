#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace statelix {

// 診断統計量の結果構造体
struct DiagnosticsResult {
    VectorXd vif;                    // 分散膨張因子
    VectorXd cooks_distance;         // Cook's距離
    VectorXd leverage;               // レバレッジ（hatvalues）
    VectorXd studentized_residuals;  // スチューデント化残差
    MatrixXd dfbetas;                // DFBETAS（各係数への影響）
    VectorXd dffits;                 // DFFITS（予測値への影響）
    VectorXd covratio;               // COVRATIO（共分散行列への影響）
};

// VIF (Variance Inflation Factor) の計算
VectorXd vif(const MatrixXd& X) {
    int p = X.cols();
    VectorXd out(p);
    for (int j = 0; j < p; ++j) {
        // X_jを他の変数で回帰
        MatrixXd Xr(X.rows(), p - 1);
        int idx = 0;
        for (int k = 0; k < p; ++k) {
            if (k != j) Xr.col(idx++) = X.col(k);
        }
        VectorXd y = X.col(j);
        VectorXd coef = (Xr.transpose() * Xr).ldlt().solve(Xr.transpose() * y);
        VectorXd pred = Xr * coef;
        double ssr = (pred - y).squaredNorm();
        double sst = (y.array() - y.mean()).matrix().squaredNorm();
        double r2 = 1.0 - ssr / sst;
        
        // VIF = 1 / (1 - R²)
        out(j) = 1.0 / (1.0 - r2);
    }
    return out;
}

// Hat行列の対角要素（レバレッジ）を計算
VectorXd compute_leverage(const MatrixXd& X) {
    int n = X.rows();
    int p = X.cols();
    
    // Hat行列: H = X(X'X)^{-1}X'
    // レバレッジ = diag(H)
    MatrixXd XtX = X.transpose() * X;
    MatrixXd XtX_inv = XtX.inverse();
    
    VectorXd leverage(n);
    for (int i = 0; i < n; ++i) {
        VectorXd xi = X.row(i).transpose();
        leverage(i) = xi.dot(XtX_inv * xi);
    }
    
    return leverage;
}

// Cook's距離の計算
VectorXd cooks_distance(
    const MatrixXd& X,
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    int n = X.rows();
    int p = X.cols();
    
    VectorXd cooks_d(n);
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // Cook's D_i = (e_i² / (p * MSE)) * (h_ii / (1 - h_ii)²)
        double numerator = e_i * e_i * h_ii;
        double denominator = p * mse * std::pow(1.0 - h_ii, 2.0);
        
        cooks_d(i) = numerator / denominator;
    }
    
    return cooks_d;
}

// スチューデント化残差の計算
VectorXd studentized_residuals(
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    int n = residuals.size();
    VectorXd stud_resid(n);
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // スチューデント化残差 = e_i / (sqrt(MSE * (1 - h_ii)))
        double se = std::sqrt(mse * (1.0 - h_ii));
        stud_resid(i) = e_i / se;
    }
    
    return stud_resid;
}

// DFBETAS の計算（各観測値が各係数に与える影響）
MatrixXd compute_dfbetas(
    const MatrixXd& X,
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    int n = X.rows();
    int p = X.cols();
    
    MatrixXd XtX_inv = (X.transpose() * X).inverse();
    MatrixXd dfbetas(n, p);
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        VectorXd xi = X.row(i).transpose();
        
        // DFBETAS_i = (e_i / (1 - h_ii)) * (X'X)^{-1} * x_i / sqrt(MSE * diag((X'X)^{-1}))
        double scale = e_i / (1.0 - h_ii);
        VectorXd influence = scale * (XtX_inv * xi);
        
        for (int j = 0; j < p; ++j) {
            double se_j = std::sqrt(mse * XtX_inv(j, j));
            dfbetas(i, j) = influence(j) / se_j;
        }
    }
    
    return dfbetas;
}

// DFFITS の計算（各観測値が予測値に与える影響）
VectorXd compute_dffits(
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    int n = residuals.size();
    VectorXd dffits(n);
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // DFFITS_i = e_i / sqrt(MSE * (1 - h_ii)) * sqrt(h_ii / (1 - h_ii))
        double stud_res = e_i / std::sqrt(mse * (1.0 - h_ii));
        dffits(i) = stud_res * std::sqrt(h_ii / (1.0 - h_ii));
    }
    
    return dffits;
}

// COVRATIO の計算（各観測値が共分散行列に与える影響）
VectorXd compute_covratio(
    int p,
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    int n = residuals.size();
    VectorXd covratio(n);
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // MSE(i) = ((n - p) * MSE - e_i² / (1 - h_ii)) / (n - p - 1)
        double mse_i = ((n - p) * mse - (e_i * e_i) / (1.0 - h_ii)) / (n - p - 1.0);
        
        // COVRATIO_i = (MSE(i) / MSE)^p * (1 / (1 - h_ii))
        covratio(i) = std::pow(mse_i / mse, static_cast<double>(p)) / (1.0 - h_ii);
    }
    
    return covratio;
}

// 完全な診断統計量の計算
DiagnosticsResult compute_diagnostics(
    const MatrixXd& X,
    const VectorXd& y,
    const VectorXd& fitted_values,
    const VectorXd& residuals
) {
    int n = X.rows();
    int p = X.cols();
    
    DiagnosticsResult result;
    
    // MSE（平均二乗誤差）
    double sse = residuals.squaredNorm();
    double mse = sse / (n - p);
    
    // レバレッジ
    result.leverage = compute_leverage(X);
    
    // VIF
    result.vif = vif(X);
    
    // Cook's距離
    result.cooks_distance = cooks_distance(X, residuals, mse, result.leverage);
    
    // スチューデント化残差
    result.studentized_residuals = studentized_residuals(residuals, mse, result.leverage);
    
    // DFBETAS
    result.dfbetas = compute_dfbetas(X, residuals, mse, result.leverage);
    
    // DFFITS
    result.dffits = compute_dffits(residuals, mse, result.leverage);
    
    // COVRATIO
    result.covratio = compute_covratio(p, residuals, mse, result.leverage);
    
    return result;
}

// 外れ値検出（スチューデント化残差の絶対値が閾値を超える点）
std::vector<int> detect_outliers(
    const VectorXd& studentized_residuals,
    double threshold = 3.0
) {
    std::vector<int> outliers;
    for (int i = 0; i < studentized_residuals.size(); ++i) {
        if (std::abs(studentized_residuals(i)) > threshold) {
            outliers.push_back(i);
        }
    }
    return outliers;
}

// 高レバレッジポイントの検出
std::vector<int> detect_high_leverage(
    const VectorXd& leverage,
    int p,
    int n
) {
    std::vector<int> high_lev;
    double threshold = 2.0 * p / n;  // 一般的な閾値: 2p/n
    
    for (int i = 0; i < leverage.size(); ++i) {
        if (leverage(i) > threshold) {
            high_lev.push_back(i);
        }
    }
    return high_lev;
}

// 影響力のある観測値の検出（Cook's距離）
std::vector<int> detect_influential(
    const VectorXd& cooks_distance,
    double threshold = 1.0
) {
    std::vector<int> influential;
    for (int i = 0; i < cooks_distance.size(); ++i) {
        if (cooks_distance(i) > threshold) {
            influential.push_back(i);
        }
    }
    return influential;
}

} // namespace statelix

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(statelix_diagnostics, m) {
    m.doc() = "Regression diagnostics module (VIF, Cook's D, leverage, influence measures)";
    
    // DiagnosticsResult構造体
    py::class_<statelix::DiagnosticsResult>(m, "DiagnosticsResult")
        .def_readonly("vif", &statelix::DiagnosticsResult::vif, "Variance Inflation Factors")
        .def_readonly("cooks_distance", &statelix::DiagnosticsResult::cooks_distance, "Cook's distances")
        .def_readonly("leverage", &statelix::DiagnosticsResult::leverage, "Leverage values (hat values)")
        .def_readonly("studentized_residuals", &statelix::DiagnosticsResult::studentized_residuals, "Studentized residuals")
        .def_readonly("dfbetas", &statelix::DiagnosticsResult::dfbetas, "DFBETAS (influence on coefficients)")
        .def_readonly("dffits", &statelix::DiagnosticsResult::dffits, "DFFITS (influence on fitted values)")
        .def_readonly("covratio", &statelix::DiagnosticsResult::covratio, "COVRATIO (influence on covariance matrix)");
    
    // 関数
    m.def("vif", &statelix::vif,
          "Compute Variance Inflation Factors",
          py::arg("X"));
    
    m.def("compute_leverage", &statelix::compute_leverage,
          "Compute leverage (hat values)",
          py::arg("X"));
    
    m.def("cooks_distance", &statelix::cooks_distance,
          "Compute Cook's distances",
          py::arg("X"), py::arg("residuals"), py::arg("mse"), py::arg("leverage"));
    
    m.def("studentized_residuals", &statelix::studentized_residuals,
          "Compute studentized residuals",
          py::arg("residuals"), py::arg("mse"), py::arg("leverage"));
    
    m.def("compute_dfbetas", &statelix::compute_dfbetas,
          "Compute DFBETAS (influence on coefficients)",
          py::arg("X"), py::arg("residuals"), py::arg("mse"), py::arg("leverage"));
    
    m.def("compute_dffits", &statelix::compute_dffits,
          "Compute DFFITS (influence on fitted values)",
          py::arg("residuals"), py::arg("mse"), py::arg("leverage"));
    
    m.def("compute_covratio", &statelix::compute_covratio,
          "Compute COVRATIO (influence on covariance matrix)",
          py::arg("p"), py::arg("residuals"), py::arg("mse"), py::arg("leverage"));
    
    m.def("compute_diagnostics", &statelix::compute_diagnostics,
          "Compute all regression diagnostics",
          py::arg("X"), py::arg("y"), py::arg("fitted_values"), py::arg("residuals"));
    
    m.def("detect_outliers", &statelix::detect_outliers,
          "Detect outliers based on studentized residuals",
          py::arg("studentized_residuals"), py::arg("threshold") = 3.0);
    
    m.def("detect_high_leverage", &statelix::detect_high_leverage,
          "Detect high leverage points",
          py::arg("leverage"), py::arg("p"), py::arg("n"));
    
    m.def("detect_influential", &statelix::detect_influential,
          "Detect influential observations based on Cook's distance",
          py::arg("cooks_distance"), py::arg("threshold") = 1.0);
}
