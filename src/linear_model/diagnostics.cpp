#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <vector>

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

/**
 * @brief VIF (Variance Inflation Factor) の計算
 * 
 * @note X should NOT include an intercept column for meaningful VIF.
 *       If X contains an intercept (column of 1s), VIF for that column is meaningless.
 *       Consider passing X without the intercept term.
 */
VectorXd vif(const MatrixXd& X) {
    const int p = static_cast<int>(X.cols());
    VectorXd out(p);
    
    for (int j = 0; j < p; ++j) {
        // X_jを他の変数で回帰
        MatrixXd Xr(X.rows(), p - 1);
        int idx = 0;
        for (int k = 0; k < p; ++k) {
            if (k != j) Xr.col(idx++) = X.col(k);
        }
        VectorXd y = X.col(j);
        
        // Use LDLT for numerical stability
        VectorXd coef = (Xr.transpose() * Xr).ldlt().solve(Xr.transpose() * y);
        VectorXd pred = Xr * coef;
        double ssr = (pred - y).squaredNorm();
        double sst = (y.array() - y.mean()).matrix().squaredNorm();
        
        // Handle edge case: constant column
        if (sst < 1e-12) {
            out(j) = std::numeric_limits<double>::infinity();
        } else {
            double r2 = 1.0 - ssr / sst;
            // VIF = 1 / (1 - R²)
            out(j) = 1.0 / std::max(1.0 - r2, 1e-12);
        }
    }
    return out;
}

/**
 * @brief Hat行列の対角要素（レバレッジ）を計算（ベクトル化版）
 * 
 * h_ii = x_i^T (X^T X)^{-1} x_i = [X @ (X^T X)^{-1}]_{i,:} · x_i
 * Vectorized: h = rowwise_sum( X .* (X @ XtX_inv) )
 */
VectorXd compute_leverage(const MatrixXd& X) {
    // Compute (X^T X)^{-1} using LDLT for numerical stability
    MatrixXd XtX_inv = (X.transpose() * X).ldlt().solve(
        MatrixXd::Identity(X.cols(), X.cols())
    );
    
    // Vectorized leverage computation: h = rowwise_sum(X .* (X * XtX_inv))
    MatrixXd H_diag_contrib = X.array() * (X * XtX_inv).array();
    return H_diag_contrib.rowwise().sum();
}

/**
 * @brief Compute leverage with pre-computed XtX_inv (for efficiency in compute_diagnostics)
 */
VectorXd compute_leverage_with_inv(const MatrixXd& X, const MatrixXd& XtX_inv) {
    MatrixXd H_diag_contrib = X.array() * (X * XtX_inv).array();
    return H_diag_contrib.rowwise().sum();
}

// Cook's距離の計算
VectorXd cooks_distance(
    const MatrixXd& X,
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    
    VectorXd cooks_d(n);
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // Cook's D_i = (e_i² / (p * MSE)) * (h_ii / (1 - h_ii)²)
        double one_minus_h = std::max(1.0 - h_ii, 1e-12);
        double numerator = e_i * e_i * h_ii;
        double denominator = p * mse * one_minus_h * one_minus_h;
        
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
    const int n = static_cast<int>(residuals.size());
    VectorXd stud_resid(n);
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // スチューデント化残差 = e_i / (sqrt(MSE * (1 - h_ii)))
        double one_minus_h = std::max(1.0 - h_ii, 1e-12);
        double se = std::sqrt(mse * one_minus_h);
        stud_resid(i) = e_i / se;
    }
    
    return stud_resid;
}

/**
 * @brief DFBETAS の計算（各観測値が各係数に与える影響）
 * 
 * Uses pre-computed XtX_inv for efficiency.
 */
MatrixXd compute_dfbetas_with_inv(
    const MatrixXd& X,
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage,
    const MatrixXd& XtX_inv
) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    
    MatrixXd dfbetas(n, p);
    
    // Precompute standard errors for each coefficient
    VectorXd se_coef(p);
    for (int j = 0; j < p; ++j) {
        se_coef(j) = std::sqrt(mse * XtX_inv(j, j));
    }
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        VectorXd xi = X.row(i).transpose();
        
        // DFBETAS_i = (e_i / (1 - h_ii)) * (X'X)^{-1} * x_i / se_j
        double one_minus_h = std::max(1.0 - h_ii, 1e-12);
        double scale = e_i / one_minus_h;
        VectorXd influence = scale * (XtX_inv * xi);
        
        for (int j = 0; j < p; ++j) {
            dfbetas(i, j) = influence(j) / se_coef(j);
        }
    }
    
    return dfbetas;
}

// DFBETAS の計算（スタンドアロン版）
MatrixXd compute_dfbetas(
    const MatrixXd& X,
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    MatrixXd XtX_inv = (X.transpose() * X).ldlt().solve(
        MatrixXd::Identity(X.cols(), X.cols())
    );
    return compute_dfbetas_with_inv(X, residuals, mse, leverage, XtX_inv);
}

// DFFITS の計算（各観測値が予測値に与える影響）
VectorXd compute_dffits(
    const VectorXd& residuals,
    double mse,
    const VectorXd& leverage
) {
    const int n = static_cast<int>(residuals.size());
    VectorXd dffits(n);
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // DFFITS_i = e_i / sqrt(MSE * (1 - h_ii)) * sqrt(h_ii / (1 - h_ii))
        double one_minus_h = std::max(1.0 - h_ii, 1e-12);
        double stud_res = e_i / std::sqrt(mse * one_minus_h);
        dffits(i) = stud_res * std::sqrt(h_ii / one_minus_h);
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
    const int n = static_cast<int>(residuals.size());
    VectorXd covratio(n);
    
    for (int i = 0; i < n; ++i) {
        double h_ii = leverage(i);
        double e_i = residuals(i);
        
        // MSE(i) = ((n - p) * MSE - e_i² / (1 - h_ii)) / (n - p - 1)
        double one_minus_h = std::max(1.0 - h_ii, 1e-12);
        double mse_i = ((n - p) * mse - (e_i * e_i) / one_minus_h) / (n - p - 1.0);
        
        // Ensure mse_i is positive
        mse_i = std::max(mse_i, 1e-12);
        
        // COVRATIO_i = (MSE(i) / MSE)^p * (1 / (1 - h_ii))
        covratio(i) = std::pow(mse_i / mse, static_cast<double>(p)) / one_minus_h;
    }
    
    return covratio;
}

/**
 * @brief 完全な診断統計量の計算（最適化版）
 * 
 * Computes XtX_inv once and shares it across all functions.
 */
DiagnosticsResult compute_diagnostics(
    const MatrixXd& X,
    const VectorXd& y,
    const VectorXd& fitted_values,
    const VectorXd& residuals
) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    
    DiagnosticsResult result;
    
    // MSE（平均二乗誤差）
    double sse = residuals.squaredNorm();
    double mse = sse / (n - p);
    
    // Compute (X^T X)^{-1} once using LDLT for numerical stability
    MatrixXd XtX_inv = (X.transpose() * X).ldlt().solve(
        MatrixXd::Identity(p, p)
    );
    
    // レバレッジ (using pre-computed XtX_inv)
    result.leverage = compute_leverage_with_inv(X, XtX_inv);
    
    // VIF
    result.vif = vif(X);
    
    // Cook's距離
    result.cooks_distance = cooks_distance(X, residuals, mse, result.leverage);
    
    // スチューデント化残差
    result.studentized_residuals = studentized_residuals(residuals, mse, result.leverage);
    
    // DFBETAS (using pre-computed XtX_inv)
    result.dfbetas = compute_dfbetas_with_inv(X, residuals, mse, result.leverage, XtX_inv);
    
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

/**
 * @brief 影響力のある観測値の検出（Cook's距離）
 * 
 * @param cooks_distance Cook's distance vector
 * @param threshold Threshold for influential points. 
 *                  Common choices: 4/n (default), 4/(n-p-1), or 1.0 for extreme cases.
 */
std::vector<int> detect_influential(
    const VectorXd& cooks_distance,
    int n,  // Added n parameter for default threshold
    double threshold = -1.0  // -1 means use 4/n
) {
    if (threshold < 0) {
        threshold = 4.0 / n;  // Rule of thumb: 4/n
    }
    
    std::vector<int> influential;
    for (int i = 0; i < cooks_distance.size(); ++i) {
        if (cooks_distance(i) > threshold) {
            influential.push_back(i);
        }
    }
    return influential;
}

// Backward-compatible version
std::vector<int> detect_influential_simple(
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

