/**
 * @file gbdt.cpp
 * @brief Gradient Boosted Decision Trees with Histogram-based Split Finding
 *
 * Optimizations:
 *   - Histogram-based split finding (O(bins) instead of O(n))
 *   - Pre-sorted feature indices for efficient binning
 *   - Vectorized gradient/hessian accumulation
 */
#include "gbdt.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace statelix {

// Number of histogram bins for split finding
constexpr int N_BINS = 256;

// ---------------- Histogram-based Split Finder ----------------

struct Histogram {
  double sum_grad[N_BINS];
  double sum_hess[N_BINS]; // For L2-squared loss, hessian = 1.0 per sample
  int count[N_BINS];

  void reset() {
    std::fill_n(sum_grad, N_BINS, 0.0);
    std::fill_n(sum_hess, N_BINS, 0.0);
    std::fill_n(count, N_BINS, 0);
  }
};

// Bin boundaries for one feature
struct FeatureBins {
  std::vector<double> thresholds; // N_BINS - 1 thresholds

  void build(const Eigen::MatrixXd &X, int feature,
             const std::vector<int> &indices) {
    if (indices.empty())
      return;

    // Collect values
    std::vector<double> vals;
    vals.reserve(indices.size());
    for (int idx : indices) {
      vals.push_back(X(idx, feature));
    }
    std::sort(vals.begin(), vals.end());

    // Quantile-based binning
    thresholds.clear();
    thresholds.reserve(N_BINS - 1);

    for (int b = 1; b < N_BINS; ++b) {
      size_t pos = static_cast<size_t>(b * vals.size() / N_BINS);
      pos = std::min(pos, vals.size() - 1);
      double thresh = vals[pos];

      // Avoid duplicates
      if (thresholds.empty() || thresh > thresholds.back() + 1e-9) {
        thresholds.push_back(thresh);
      }
    }
  }

  int get_bin(double value) const {
    auto it = std::lower_bound(thresholds.begin(), thresholds.end(), value);
    return static_cast<int>(it - thresholds.begin());
  }
};

// ---------------- DecisionTreeRegressor ----------------

void DecisionTreeRegressor::fit(const Eigen::MatrixXd &X,
                                const Eigen::VectorXd &y) {
  std::vector<int> indices(X.rows());
  std::iota(indices.begin(), indices.end(), 0);
  root = build_tree(X, y, indices, 0);
}

double DecisionTreeRegressor::predict_one(const Eigen::VectorXd &x) const {
  TreeNode *node = root.get();
  while (!node->is_leaf) {
    if (x(node->feature_index) <= node->threshold) {
      node = node->left.get();
    } else {
      node = node->right.get();
    }
  }
  return node->value;
}

std::unique_ptr<TreeNode>
DecisionTreeRegressor::build_tree(const Eigen::MatrixXd &X,
                                  const Eigen::VectorXd &y,
                                  const std::vector<int> &indices, int depth) {
  auto node = std::make_unique<TreeNode>();

  int n_samples = indices.size();
  int n_features = X.cols();

  // Calculate mean
  double sum = 0.0;
  for (int idx : indices)
    sum += y(idx);
  double mean = (n_samples > 0) ? sum / n_samples : 0.0;

  // Stopping criteria
  if (depth >= max_depth || n_samples < min_samples_split) {
    node->is_leaf = true;
    node->value = mean;
    return node;
  }

  // Pre-compute statistics for histogram-based splitting
  double total_sum = sum;
  double total_sum_sq = 0.0;
  for (int idx : indices) {
    double val = y(idx);
    total_sum_sq += val * val;
  }

  // Build feature bins (quantile-based)
  static thread_local std::vector<FeatureBins> feature_bins;
  if (depth == 0 || feature_bins.size() != static_cast<size_t>(n_features)) {
    feature_bins.resize(n_features);
    for (int f = 0; f < n_features; ++f) {
      feature_bins[f].build(X, f, indices);
    }
  }

  double best_gain = 0.0;
  int best_feature = -1;
  double best_threshold = 0.0;

  // Histogram for each feature
  static thread_local Histogram hist;

  for (int f = 0; f < n_features; ++f) {
    const FeatureBins &bins = feature_bins[f];
    if (bins.thresholds.empty())
      continue;

    // Build histogram
    hist.reset();

    for (int idx : indices) {
      int bin = bins.get_bin(X(idx, f));
      hist.sum_grad[bin] += y(idx);
      hist.sum_hess[bin] += 1.0;
      hist.count[bin]++;
    }

    // Scan for best split
    double left_sum = 0.0;
    int left_count = 0;

    int n_thresholds = bins.thresholds.size();
    for (int b = 0; b < n_thresholds; ++b) {
      left_sum += hist.sum_grad[b];
      left_count += hist.count[b];

      int right_count = n_samples - left_count;
      if (left_count == 0 || right_count == 0)
        continue;

      double right_sum = total_sum - left_sum;

      // Gain = reduction in SSE
      // SSE_parent = Σ(y - mean)^2 = Σy^2 - (Σy)^2/n
      // SSE_left = Σy_L^2 - (Σy_L)^2/n_L
      // SSE_right = Σy_R^2 - (Σy_R)^2/n_R
      // Gain = (Σy_L)^2/n_L + (Σy_R)^2/n_R - (Σy)^2/n

      double gain = (left_sum * left_sum / left_count) +
                    (right_sum * right_sum / right_count) -
                    (total_sum * total_sum / n_samples);

      if (gain > best_gain) {
        best_gain = gain;
        best_feature = f;
        best_threshold = bins.thresholds[b];
      }
    }
  }

  if (best_feature == -1) {
    node->is_leaf = true;
    node->value = mean;
    return node;
  }

  // Perform split
  node->feature_index = best_feature;
  node->threshold = best_threshold;
  node->is_leaf = false;

  std::vector<int> left_idx, right_idx;
  left_idx.reserve(n_samples);
  right_idx.reserve(n_samples);

  for (int idx : indices) {
    if (X(idx, best_feature) <= best_threshold) {
      left_idx.push_back(idx);
    } else {
      right_idx.push_back(idx);
    }
  }

  node->left = build_tree(X, y, left_idx, depth + 1);
  node->right = build_tree(X, y, right_idx, depth + 1);

  return node;
}

// ---------------- GradientBoostingRegressor ----------------

void GradientBoostingRegressor::fit(const Eigen::MatrixXd &X,
                                    const Eigen::VectorXd &y) {
  int n_samples = X.rows();
  trees.clear();

  initial_prediction = y.mean();
  Eigen::VectorXd F = Eigen::VectorXd::Constant(n_samples, initial_prediction);

  std::vector<int> indices(n_samples);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(42);

  for (int i = 0; i < n_estimators; ++i) {
    Eigen::VectorXd residuals = y - F;

    // Subsampling
    Eigen::MatrixXd X_train;
    Eigen::VectorXd res_train;

    if (subsample < 1.0) {
      std::shuffle(indices.begin(), indices.end(), rng);
      int n_sub = std::max(1, static_cast<int>(n_samples * subsample));

      X_train.resize(n_sub, X.cols());
      res_train.resize(n_sub);

      for (int k = 0; k < n_sub; ++k) {
        X_train.row(k) = X.row(indices[k]);
        res_train(k) = residuals(indices[k]);
      }
    } else {
      X_train = X;
      res_train = residuals;
    }

    auto tree = std::make_unique<DecisionTreeRegressor>(max_depth);
    tree->fit(X_train, res_train);

    // Update predictions for all samples
    for (int j = 0; j < n_samples; ++j) {
      F(j) += learning_rate * tree->predict_one(X.row(j));
    }

    trees.push_back(std::move(tree));
  }
}

Eigen::VectorXd GradientBoostingRegressor::predict(const Eigen::MatrixXd &X) {
  int n = X.rows();
  Eigen::VectorXd preds = Eigen::VectorXd::Constant(n, initial_prediction);

  for (const auto &tree : trees) {
    for (int i = 0; i < n; ++i) {
      preds(i) += learning_rate * tree->predict_one(X.row(i));
    }
  }
  return preds;
}

} // namespace statelix
