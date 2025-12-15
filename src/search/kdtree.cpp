#include "kdtree.h"
#include <cmath>
#include <queue>
#include <iostream>

namespace statelix {

void KDTree::fit(const Eigen::MatrixXd& X) {
    data = X;
    n_samples = X.rows();
    n_features = X.cols();
    
    std::vector<int> indices(n_samples);
    for(int i = 0; i < n_samples; ++i) indices[i] = i;

    // Use in-place construction
    root = build_recursive(indices, 0, n_samples, 0);
}

// Optimized in-place build: O(N log N) time, O(1) extra space per node (stack only)
std::unique_ptr<KDNode> KDTree::build_recursive(std::vector<int>& indices, int start, int end, int depth) {
    if (start >= end) return nullptr;

    // Cycle through axes
    int axis = depth % n_features;

    // Find median using nth_element on the range [start, end)
    int len = end - start;
    int mid = start + len / 2;
    
    std::nth_element(indices.begin() + start, 
                     indices.begin() + mid, 
                     indices.begin() + end,
        [&](int a, int b) {
            return data(a, axis) < data(b, axis);
        }
    );

    int median_idx = indices[mid];
    auto node = std::make_unique<KDNode>(median_idx, axis);

    // Recursively build subtrees
    // Left: [start, mid)
    // Right: [mid + 1, end)
    node->left = build_recursive(indices, start, mid, depth + 1);
    node->right = build_recursive(indices, mid + 1, end, depth + 1);

    return node;
}

KNNSearchResult KDTree::query(const Eigen::VectorXd& point, int k) {
    if (!root) return {{}, {}};
    
    // Max heap to keep nearest k neighbors (we want smallest distances, 
    // but a priority_queue pops the largest. So if size > k, pop largest).
    std::vector<std::pair<double, int>> heap; 
    
    search_recursive(root.get(), point, k, heap);

    std::sort(heap.begin(), heap.end()); // sort by distance asc

    std::vector<int> indices;
    std::vector<double> distances;
    for(auto& p : heap) {
        distances.push_back(std::sqrt(p.first)); // Convert squared dist to euclidean
        indices.push_back(p.second);
    }

    return {indices, distances};
}

void KDTree::search_recursive(const KDNode* node, const Eigen::VectorXd& point, int k, 
                              std::vector<std::pair<double, int>>& heap) {
    if (!node) return;

    // Calculate distance to current node point
    double dist_sq = (data.row(node->point_index) - point.transpose()).squaredNorm();

    // Maintain heap of size k
    // Format: {distance_sq, index}
    // We want to KEEP smallest items. 
    // If heap size < k, push.
    // If heap size == k, compare with max element. If smaller, replace max.
    
    // Using vector + sort/make_heap for simplicity in this logic block, 
    // or just linear scan if k is small (k usually < 20). 
    // For standard "heap", std::priority_queue is best but iterating is hard.
    // Let's just use vector and push_back + sort approach for simplicity unless slow.
    // Optimization: Keep sorted vector.

    bool added = false;
    if (heap.size() < k) {
        heap.push_back({dist_sq, node->point_index});
        std::push_heap(heap.begin(), heap.end()); // Max heap
        added = true;
    } else {
        // heap.front() is largest distance
        if (dist_sq < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.pop_back();
            heap.push_back({dist_sq, node->point_index});
            std::push_heap(heap.begin(), heap.end());
            added = true;
        }
    }

    // Determine which side to search first
    double axis_diff = point(node->axis) - data(node->point_index, node->axis);
    
    const KDNode* near = (axis_diff < 0) ? node->left.get() : node->right.get();
    const KDNode* far  = (axis_diff < 0) ? node->right.get() : node->left.get();

    // Search near side
    search_recursive(near, point, k, heap);

    // Pruning: Check if we need to search far side
    // If distance from query point to splitting plane is less than the 
    // "worst" distance in our current k-nearest set, we must check far side.
    // Dist to plane = axis_diff^2
    
    double dist_to_plane_sq = axis_diff * axis_diff;
    // If heap not full, MUST check. 
    // If heap full, check if plane < max_dist
    if (heap.size() < k || dist_to_plane_sq < heap.front().first) {
        search_recursive(far, point, k, heap);
    }
}

} // namespace statelix
