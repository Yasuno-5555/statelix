#include "gbdt.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace statelix {

// ---------------- DecisionTreeRegressor ----------------

double calculate_variance(const Eigen::VectorXd& y) {
    if (y.size() < 2) return 0.0;
    double mean = y.mean();
    return (y.array() - mean).square().sum() / y.size();
}

void DecisionTreeRegressor::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    root = build_tree(X, y, 0);
}

double DecisionTreeRegressor::predict_one(const Eigen::VectorXd& x) const {
    TreeNode* node = root.get();
    while (!node->is_leaf) {
        if (x(node->feature_index) <= node->threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node->value;
}

std::unique_ptr<TreeNode> DecisionTreeRegressor::build_tree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int depth) {
    auto node = std::make_unique<TreeNode>();
    
    int n_samples = X.rows();
    int n_features = X.cols();

    // Stopping criteria
    if (depth >= max_depth || n_samples < min_samples_split) {
        node->is_leaf = true;
        node->value = (n_samples > 0) ? y.mean() : 0.0;
        return node;
    }

    double best_mse = std::numeric_limits<double>::infinity();
    int best_feature = -1;
    double best_threshold = 0.0;
    
    // Greedy split search (very simple loop, O(N*F))
    // Optimization: Use quantiles or histograms for large data
    for (int f = 0; f < n_features; ++f) {
        // Simple strategy: Try every distinct value as threshold? Too slow.
        // Try quantiles or just min/max steps?
        // Let's implement extremely simple: 10 candidate splits per feature.
        
        double min_val = X.col(f).minCoeff();
        double max_val = X.col(f).maxCoeff();
        
        if (min_val == max_val) continue;

        for (int k = 1; k < 10; ++k) {
            double thresh = min_val + k * (max_val - min_val) / 10.0;
            
            // Vectorized filtering is clumsy in generic Eigen without masking
            // Use logical indexing mask simulation
            
            // Mask
            // Eigen::Array<bool, ...> is not directly usable for indexing rows easily without custom loop or plugin
            // Loop is simpler for clarity here
            
            std::vector<int> left_idx, right_idx;
            left_idx.reserve(n_samples); 
            right_idx.reserve(n_samples);

            for(int i=0; i<n_samples; ++i) {
                if (X(i, f) <= thresh) left_idx.push_back(i);
                else right_idx.push_back(i);
            }
            
            if (left_idx.empty() || right_idx.empty()) continue;
            
            // Calc MSE improvement
            // MSE = ( n_l * var_l + n_r * var_r ) / n
            // Actually just need to minimize Sum Squared Error since n is constant
            // SSE = sum((y_l - mean_l)^2) + sum((y_r - mean_r)^2)
            
            double sum_l = 0.0, sq_sum_l = 0.0;
            for(int idx : left_idx) { sum_l += y(idx); sq_sum_l += y(idx)*y(idx); }
            double sse_l = sq_sum_l - (sum_l*sum_l)/left_idx.size();
            
            double sum_r = 0.0, sq_sum_r = 0.0;
            for(int idx : right_idx) { sum_r += y(idx); sq_sum_r += y(idx)*y(idx); }
            double sse_r = sq_sum_r - (sum_r*sum_r)/right_idx.size();
            
            double total_sse = sse_l + sse_r;
            
            if (total_sse < best_mse) {
                best_mse = total_sse;
                best_feature = f;
                best_threshold = thresh;
            }
        }
    }

    if (best_feature == -1) { // No valid split found
        node->is_leaf = true;
        node->value = y.mean();
        return node;
    }

    // Perform best split
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->is_leaf = false;
    
    std::vector<int> left_idx, right_idx;
    for(int i=0; i<n_samples; ++i) {
        if (X(i, best_feature) <= best_threshold) left_idx.push_back(i);
        else right_idx.push_back(i);
    }
    
    // Construct sub-matrices (Copying data is slow but safe for now)
    Eigen::MatrixXd X_left(left_idx.size(), n_features);
    Eigen::VectorXd y_left(left_idx.size());
    for(size_t i=0; i<left_idx.size(); ++i) {
        X_left.row(i) = X.row(left_idx[i]);
        y_left(i) = y(left_idx[i]);
    }
    
    Eigen::MatrixXd X_right(right_idx.size(), n_features);
    Eigen::VectorXd y_right(right_idx.size());
    for(size_t i=0; i<right_idx.size(); ++i) {
        X_right.row(i) = X.row(right_idx[i]);
        y_right(i) = y(right_idx[i]);
    }

    node->left = build_tree(X_left, y_left, depth + 1);
    node->right = build_tree(X_right, y_right, depth + 1);
    
    return node;
}


// ---------------- GradientBoostingRegressor ----------------

void GradientBoostingRegressor::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int n_samples = X.rows();
    trees.clear();
    
    // F_0(x) = mean(y)
    initial_prediction = y.mean();
    Eigen::VectorXd F = Eigen::VectorXd::Constant(n_samples, initial_prediction);
    
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    // Random engine
    std::mt19937 rng(42); // Seed for reproducibility

    for (int i = 0; i < n_estimators; ++i) {
        // Residuals r_i = y_i - F(x_i)
        Eigen::VectorXd residuals = y - F;
        
        // Subsampling
        Eigen::MatrixXd X_train;
        Eigen::VectorXd res_train;
        
        if (subsample < 1.0) {
            std::shuffle(indices.begin(), indices.end(), rng);
            int n_sub = std::max(1, (int)(n_samples * subsample));
            
            X_train.resize(n_sub, X.cols());
            res_train.resize(n_sub);
            
            for(int k=0; k<n_sub; ++k) {
                X_train.row(k) = X.row(indices[k]);
                res_train(k) = residuals(indices[k]);
            }
        } else {
            X_train = X;
            res_train = residuals;
        }

        // Fit tree to residuals (using subsample)
        auto tree = std::make_unique<DecisionTreeRegressor>(max_depth);
        tree->fit(X_train, res_train);
        
        // Update predictions for ALL samples (requires predicting on X)
        // Note: For Stochastic Gradient Boosting, we update F for all x using the tree trained on subset.
        for (int j = 0; j < n_samples; ++j) {
            F(j) += learning_rate * tree->predict_one(X.row(j));
        }
        
        trees.push_back(std::move(tree));
    }
}

Eigen::VectorXd GradientBoostingRegressor::predict(const Eigen::MatrixXd& X) {
    int n = X.rows();
    Eigen::VectorXd preds = Eigen::VectorXd::Constant(n, initial_prediction);
    
    for (const auto& tree : trees) {
        for (int i = 0; i < n; ++i) {
            preds(i) += learning_rate * tree->predict_one(X.row(i));
        }
    }
    return preds;
}

} // namespace statelix
