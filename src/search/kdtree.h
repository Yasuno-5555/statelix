#ifndef STATELIX_KDTREE_H
#define STATELIX_KDTREE_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <algorithm>

namespace statelix {

struct KDNode {
    int point_index; // Index in the original data matrix
    int axis;        // Splitting axis
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(int idx, int ax) : point_index(idx), axis(ax), left(nullptr), right(nullptr) {}
};

struct KNNSearchResult {
    std::vector<int> indices;
    std::vector<double> distances;
};

class KDTree {
public:
    KDTree() = default;
    
    // Build tree from data matrix X (n_samples x n_features)
    void fit(const Eigen::MatrixXd& X);
    
    // Find k nearest neighbors for a single query point
    KNNSearchResult query(const Eigen::VectorXd& point, int k);

private:
    std::unique_ptr<KDNode> root;
    Eigen::MatrixXd data; // Store copy or reference? Copy is safer for now.
    int n_samples;
    int n_features;

    // Recursive builder
    std::unique_ptr<KDNode> build_recursive(std::vector<int>& indices, int depth);

    // Recursive search
    void search_recursive(const KDNode* node, const Eigen::VectorXd& point, int k, 
                          std::vector<std::pair<double, int>>& heap);
};

} // namespace statelix

#endif // STATELIX_KDTREE_H
