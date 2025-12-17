#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include "solver.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace statelix {

// Ridge回帰のクロスバリデーション結果
struct RidgeCVResult {
    double best_lambda;                  // 最適λ
    VectorXd coef;                       // 最適λでの係数
    double intercept;                    // 最適λでの切片
    std::vector<double> lambda_path;     // 試したλの値
    std::vector<double> cv_scores;       // 各λでのCVスコア（MSE）
    std::vector<double> cv_std;          // CVスコアの標準偏差
    std::vector<double> gcv_scores;      // 一般化クロスバリデーションスコア
};

// Ridge回帰の基本ソルバー (WeightedSolver使用)
VectorXd ridge_solver_internal(
    const MatrixXd& X,
    const VectorXd& y,
    double lambda,
    bool fit_intercept = true
) {
    int n = X.rows();
    int p_original = X.cols();
    
    MatrixXd X_design;
    if (fit_intercept) {
        X_design.resize(n, p_original + 1);
        X_design.col(0) = VectorXd::Ones(n);
        X_design.rightCols(p_original) = X;
    } else {
        X_design = X;
    }
    
    int p = X_design.cols();
    
    // Use WeightedSolver with unit weights
    VectorXd weights = VectorXd::Ones(n);
    WeightedDesignMatrix wdm(X_design, weights);
    
    // Compute X^T X + λI manually, then use LDLT
    MatrixXd XtX = wdm.compute_gram();
    
    // Add ridge penalty (skip intercept)
    if (fit_intercept) {
        for (int j = 1; j < p; ++j) {
            XtX(j, j) += lambda;
        }
    } else {
        for (int j = 0; j < p; ++j) {
            XtX(j, j) += lambda;
        }
    }
    
    VectorXd Xty = wdm.compute_XTWy(y);
    
    // Solve via LDLT (numerically stable)
    return XtX.ldlt().solve(Xty);
}

// 一般化クロスバリデーション (GCV) スコアの計算
double compute_gcv(
    const MatrixXd& X,
    const VectorXd& y,
    double lambda,
    bool fit_intercept = true
) {
    int n = X.rows();
    int p_original = X.cols();
    
    MatrixXd X_design;
    if (fit_intercept) {
        X_design.resize(n, p_original + 1);
        X_design.col(0) = VectorXd::Ones(n);
        X_design.rightCols(p_original) = X;
    } else {
        X_design = X;
    }
    
    int p = X_design.cols();
    
    // Ridge解の計算
    VectorXd beta = ridge_solver_internal(X, y, lambda, fit_intercept);
    
    // 残差平方和
    VectorXd y_pred = X_design * beta;
    double rss = (y - y_pred).squaredNorm();
    
    // 有効自由度の計算（トレース(hat matrix)）
    MatrixXd XtX = X_design.transpose() * X_design;
    MatrixXd penalty = MatrixXd::Zero(p, p);
    if (fit_intercept) {
        penalty.bottomRightCorner(p_original, p_original) = 
            lambda * MatrixXd::Identity(p_original, p_original);
    } else {
        penalty = lambda * MatrixXd::Identity(p, p);
    }
    
    MatrixXd A_inv = (XtX + penalty).ldlt().solve(MatrixXd::Identity(p, p));
    double df = (A_inv * XtX).trace();
    
    // GCVスコア: RSS / (n * (1 - df/n)^2)
    double denom = 1.0 - df / n;
    if (std::abs(denom) < 1e-10) return std::numeric_limits<double>::infinity();
    double gcv = rss / (n * denom * denom);
    
    return gcv;
}

// K-fold クロスバリデーション
std::pair<double, double> cross_validate_ridge(
    const MatrixXd& X,
    const VectorXd& y,
    double lambda,
    int n_folds = 5,
    bool fit_intercept = true
) {
    int n = X.rows();
    int fold_size = n / n_folds;
    
    std::vector<double> fold_errors;
    
    for (int fold = 0; fold < n_folds; ++fold) {
        int test_start = fold * fold_size;
        int test_end = (fold == n_folds - 1) ? n : (fold + 1) * fold_size;
        int test_size = test_end - test_start;
        int train_size = n - test_size;
        
        MatrixXd X_train(train_size, X.cols());
        VectorXd y_train(train_size);
        MatrixXd X_test(test_size, X.cols());
        VectorXd y_test(test_size);
        
        int train_idx = 0;
        int test_idx = 0;
        
        for (int i = 0; i < n; ++i) {
            if (i >= test_start && i < test_end) {
                X_test.row(test_idx) = X.row(i);
                y_test(test_idx) = y(i);
                test_idx++;
            } else {
                X_train.row(train_idx) = X.row(i);
                y_train(train_idx) = y(i);
                train_idx++;
            }
        }
        
        VectorXd beta = ridge_solver_internal(X_train, y_train, lambda, fit_intercept);
        
        MatrixXd X_test_design;
        int p_original = X.cols();
        if (fit_intercept) {
            X_test_design.resize(test_size, p_original + 1);
            X_test_design.col(0) = VectorXd::Ones(test_size);
            X_test_design.rightCols(p_original) = X_test;
        } else {
            X_test_design = X_test;
        }
        
        VectorXd y_pred = X_test_design * beta;
        
        double mse = (y_test - y_pred).squaredNorm() / test_size;
        fold_errors.push_back(mse);
    }
    
    double mean_error = 0.0;
    for (double err : fold_errors) mean_error += err;
    mean_error /= fold_errors.size();
    
    double std_error = 0.0;
    for (double err : fold_errors) std_error += std::pow(err - mean_error, 2.0);
    std_error = std::sqrt(std_error / fold_errors.size());
    
    return std::make_pair(mean_error, std_error);
}

// Ridge回帰のクロスバリデーション
RidgeCVResult fit_ridge_cv(
    const MatrixXd& X,
    const VectorXd& y,
    const std::vector<double>& lambda_values = {},
    int n_folds = 5,
    bool fit_intercept = true,
    bool use_gcv = false
) {
    RidgeCVResult result;
    
    std::vector<double> lambdas;
    if (lambda_values.empty()) {
        for (int i = 0; i < 10; ++i) {
            lambdas.push_back(std::pow(10.0, -2.0 + i * 0.5));
        }
    } else {
        lambdas = lambda_values;
    }
    
    result.lambda_path = lambdas;
    
    for (double lambda : lambdas) {
        if (use_gcv) {
            double gcv = compute_gcv(X, y, lambda, fit_intercept);
            result.gcv_scores.push_back(gcv);
            result.cv_scores.push_back(gcv);
            result.cv_std.push_back(0.0);
        } else {
            auto [cv_score, cv_std] = cross_validate_ridge(X, y, lambda, n_folds, fit_intercept);
            result.cv_scores.push_back(cv_score);
            result.cv_std.push_back(cv_std);
            double gcv = compute_gcv(X, y, lambda, fit_intercept);
            result.gcv_scores.push_back(gcv);
        }
    }
    
    auto min_it = std::min_element(result.cv_scores.begin(), result.cv_scores.end());
    int best_idx = std::distance(result.cv_scores.begin(), min_it);
    result.best_lambda = lambdas[best_idx];
    
    VectorXd beta_all = ridge_solver_internal(X, y, result.best_lambda, fit_intercept);
    
    int p_original = X.cols();
    if (fit_intercept) {
        result.intercept = beta_all(0);
        result.coef = beta_all.tail(p_original);
    } else {
        result.intercept = 0.0;
        result.coef = beta_all;
    }
    
    return result;
}

// 予測
VectorXd predict_ridge(
    const RidgeCVResult& result,
    const MatrixXd& X_new
) {
    int n_new = X_new.rows();
    VectorXd predictions = VectorXd::Constant(n_new, result.intercept);
    predictions += X_new * result.coef;
    return predictions;
}

} // namespace statelix
