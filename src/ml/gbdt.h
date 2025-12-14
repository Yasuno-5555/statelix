#ifndef STATELIX_GBDT_H
#define STATELIX_GBDT_H

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace statelix {

// Simple Binary Decision Tree
struct TreeNode {
    bool is_leaf;
    int feature_index;
    double threshold;
    double value; // Prediction for leaf
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    TreeNode() : is_leaf(false), feature_index(-1), threshold(0.0), value(0.0) {}
};

class DecisionTreeRegressor {
public:
    int max_depth;
    int min_samples_split;
    std::unique_ptr<TreeNode> root;

    DecisionTreeRegressor(int depth = 3, int min_split = 2) 
        : max_depth(depth), min_samples_split(min_split) {}

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    double predict_one(const Eigen::VectorXd& x) const;
    
private:
    std::unique_ptr<TreeNode> build_tree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int depth);
};

struct GBDTResult {
    Eigen::VectorXd predictions; // Final predictions on training set (redundant?)
    // Could store feature importances here
};

class GradientBoostingRegressor {
public:
    int n_estimators = 100;
    double learning_rate = 0.1;
    int max_depth = 3;
    double subsample = 1.0; // Fraction of samples to use for fitting each tree
    
    // Store trees
    std::vector<std::unique_ptr<DecisionTreeRegressor>> trees;
    double initial_prediction = 0.0; // F_0(x)

    GradientBoostingRegressor() = default;

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X);
};

} // namespace statelix

#endif // STATELIX_GBDT_H
