#include "gbdt.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>

namespace statelix {

// ---------------- DecisionTreeRegressor ----------------

double calculate_variance(const Eigen::VectorXd& y) {
    if (y.size() < 2) return 0.0;
    double mean = y.mean();
    return (y.array() - mean).square().sum() / y.size();
}

void DecisionTreeRegressor::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    root = build_tree(X, y, indices, 0);
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

std::unique_ptr<TreeNode> DecisionTreeRegressor::build_tree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, 
                                                            const std::vector<int>& indices, int depth) {
    auto node = std::make_unique<TreeNode>();
    
    int n_samples = indices.size();
    int n_features = X.cols();

    // Calculate mean of current node
    double mean = 0.0;
    if (n_samples > 0) {
        for(int idx : indices) mean += y(idx);
        mean /= n_samples;
    }

    // Stopping criteria
    if (depth >= max_depth || n_samples < min_samples_split) {
        node->is_leaf = true;
        node->value = mean;
        return node;
    }

    double best_mse = std::numeric_limits<double>::infinity();
    int best_feature = -1;
    double best_threshold = 0.0;
    
    // Greedy split search
    // Optimization: using indices to avoid copy
    // Improvement: increased steps to 20 for better resolution
    
    // Pre-calculate sum and sum_sq for the node to quickly compute SSE
    double total_sum = 0.0;
    double total_sq_sum = 0.0;
    for(int idx : indices) {
        double val = y(idx);
        total_sum += val;
        total_sq_sum += val * val;
    }
    double total_sse = total_sq_sum - (total_sum * total_sum) / n_samples;

    for (int f = 0; f < n_features; ++f) {
        // Find min/max for this feature within current indices
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        
        for(int idx : indices) {
            double val = X(idx, f);
            if(val < min_val) min_val = val;
            if(val > max_val) max_val = val;
        }
        
        if (std::abs(max_val - min_val) < 1e-9) continue;

        // Try 20 split points
        for (int k = 1; k < 20; ++k) {
            double thresh = min_val + k * (max_val - min_val) / 20.0;
            
            // Collect stats for left/right
            double sum_l = 0.0, sq_sum_l = 0.0;
            int n_l = 0;
            
            // Single pass to gather left stats (right is total - left)
            for(int idx : indices) {
                if (X(idx, f) <= thresh) {
                    double val = y(idx);
                    sum_l += val;
                    sq_sum_l += val * val;
                    n_l++;
                }
            }
            
            int n_r = n_samples - n_l;
            if (n_l == 0 || n_r == 0) continue;

            double sse_l = sq_sum_l - (sum_l * sum_l) / n_l;
            
            double sum_r = total_sum - sum_l;
            double sq_sum_r = total_sq_sum - sq_sum_l;
            double sse_r = sq_sum_r - (sum_r * sum_r) / n_r;
            
            double current_total_sse = sse_l + sse_r;
            
            if (current_total_sse < best_mse) {
                best_mse = current_total_sse;
                best_feature = f;
                best_threshold = thresh;
            }
        }
    }

    if (best_feature == -1) { // No valid split found
        node->is_leaf = true;
        node->value = mean;
        return node;
    }

    // Perform best split
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->is_leaf = false;
    
    std::vector<int> left_idx, right_idx;
    left_idx.reserve(n_samples);
    right_idx.reserve(n_samples);

    for(int idx : indices) {
        if (X(idx, best_feature) <= best_threshold) left_idx.push_back(idx);
        else right_idx.push_back(idx);
    }
    
    // Recursion without data copy
    node->left = build_tree(X, y, left_idx, depth + 1);
    node->right = build_tree(X, y, right_idx, depth + 1);
    
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
