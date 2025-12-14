/**
 * @file louvain.h
 * @brief Statelix v1.1 - Louvain Community Detection
 * 
 * Implements the Louvain method for community detection:
 *   - Fast modularity optimization
 *   - Hierarchical community structure
 *   - Sparse matrix integration
 * 
 * Algorithm:
 * ----------
 * Phase 1 (Local): Move nodes to maximize modularity gain
 * Phase 2 (Aggregate): Collapse communities into super-nodes
 * Repeat until no improvement
 * 
 * Modularity:
 *   Q = (1/2m) Σ_ij [A_ij - k_i k_j / 2m] δ(c_i, c_j)
 * 
 * where A = adjacency, k_i = degree(i), m = total edges
 * 
 * Reference: Blondel, V. et al. (2008). Fast unfolding of communities in
 *            large networks. Journal of Statistical Mechanics.
 */
#ifndef STATELIX_LOUVAIN_H
#define STATELIX_LOUVAIN_H

#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace statelix {
namespace graph {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Louvain clustering result
 */
struct LouvainResult {
    std::vector<int> labels;            // Community assignment for each node
    int n_communities;                  // Number of detected communities
    double modularity;                  // Final modularity score
    
    // Hierarchical structure
    std::vector<std::vector<int>> hierarchy;  // Labels at each level
    std::vector<double> modularity_history;   // Modularity at each level
    
    // Statistics
    std::vector<int> community_sizes;   // Size of each community
    int n_iterations;
    int n_levels;
};

// =============================================================================
// Louvain Algorithm
// =============================================================================

/**
 * @brief Louvain community detection algorithm
 * 
 * Usage:
 *   Louvain louvain;
 *   auto result = louvain.fit(adjacency_matrix);
 */
class Louvain {
public:
    // Parameters
    double resolution = 1.0;        // Resolution parameter (higher = smaller communities)
    int max_iterations = 100;       // Max iterations per level
    double min_modularity_gain = 1e-7;
    bool randomize_order = true;    // Randomize node processing order
    unsigned int seed = 42;
    
    /**
     * @brief Detect communities in a graph
     * 
     * @param adjacency Sparse adjacency matrix (n × n), can be weighted
     * @return LouvainResult with community assignments
     */
    LouvainResult fit(const Eigen::SparseMatrix<double>& adjacency) {
        int n = adjacency.rows();
        if (n != adjacency.cols()) {
            throw std::invalid_argument("Adjacency matrix must be square");
        }
        
        LouvainResult result;
        rng_.seed(seed);
        
        // Make symmetric (undirected)
        Eigen::SparseMatrix<double> A = (adjacency + Eigen::SparseMatrix<double>(adjacency.transpose())) / 2.0;
        
        // Initialize: each node is its own community
        std::vector<int> labels(n);
        std::iota(labels.begin(), labels.end(), 0);
        
        // Precompute graph properties
        double m = A.sum() / 2.0;  // Total edge weight
        if (m < 1e-10) {
            // Empty graph
            result.labels = labels;
            result.n_communities = n;
            result.modularity = 0.0;
            return result;
        }
        
        Eigen::VectorXd k = Eigen::VectorXd::Zero(n);  // Degree (weighted)
        for (int i = 0; i < A.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                k(it.row()) += it.value();
            }
        }
        
        result.hierarchy.push_back(labels);
        result.modularity_history.push_back(compute_modularity(A, labels, k, m));
        
        bool improved = true;
        int level = 0;
        
        while (improved) {
            // Phase 1: Local optimization
            improved = local_optimization(A, labels, k, m);
            
            // Renumber communities
            renumber_communities(labels);
            
            result.hierarchy.push_back(labels);
            double Q = compute_modularity(A, labels, k, m);
            result.modularity_history.push_back(Q);
            
            // Phase 2: Aggregate
            if (improved) {
                int n_comm = *std::max_element(labels.begin(), labels.end()) + 1;
                if (n_comm >= A.rows()) break;  // No aggregation possible
                
                auto [A_new, k_new] = aggregate(A, labels, k);
                A = A_new;
                k = k_new;
                
                labels.resize(A.rows());
                std::iota(labels.begin(), labels.end(), 0);
            }
            
            level++;
            if (level > 100) break;  // Safety limit
        }
        
        // Expand labels back to original nodes
        result.labels = expand_labels(result.hierarchy);
        renumber_communities(result.labels);
        result.n_communities = *std::max_element(result.labels.begin(), result.labels.end()) + 1;
        result.modularity = result.modularity_history.back();
        result.n_levels = result.hierarchy.size();
        result.n_iterations = level;
        
        // Compute community sizes
        result.community_sizes.resize(result.n_communities, 0);
        for (int label : result.labels) {
            result.community_sizes[label]++;
        }
        
        return result;
    }
    
    /**
     * @brief Fit from edge list
     */
    LouvainResult fit_edges(
        int n_nodes,
        const std::vector<std::pair<int, int>>& edges,
        const std::vector<double>& weights = {}
    ) {
        Eigen::SparseMatrix<double> A(n_nodes, n_nodes);
        std::vector<Eigen::Triplet<double>> triplets;
        
        bool weighted = !weights.empty();
        for (size_t i = 0; i < edges.size(); ++i) {
            int u = edges[i].first;
            int v = edges[i].second;
            double w = weighted ? weights[i] : 1.0;
            triplets.emplace_back(u, v, w);
            triplets.emplace_back(v, u, w);
        }
        
        A.setFromTriplets(triplets.begin(), triplets.end());
        return fit(A);
    }

private:
    std::mt19937 rng_;
    
    /**
     * @brief Compute modularity
     * Q = (1/2m) Σ_ij [A_ij - γ k_i k_j / 2m] δ(c_i, c_j)
     */
    double compute_modularity(
        const Eigen::SparseMatrix<double>& A,
        const std::vector<int>& labels,
        const Eigen::VectorXd& k,
        double m
    ) {
        double Q = 0.0;
        
        for (int i = 0; i < A.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                int j = it.row();
                if (labels[i] == labels[j]) {
                    Q += it.value() - resolution * k(i) * k(j) / (2.0 * m);
                }
            }
        }
        
        return Q / (2.0 * m);
    }
    
    /**
     * @brief Phase 1: Local node movement optimization
     */
    bool local_optimization(
        const Eigen::SparseMatrix<double>& A,
        std::vector<int>& labels,
        const Eigen::VectorXd& k,
        double m
    ) {
        int n = A.rows();
        bool any_improvement = false;
        
        // Create node ordering (optionally randomized)
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        if (randomize_order) {
            std::shuffle(order.begin(), order.end(), rng_);
        }
        
        // Precompute community properties
        int n_comm = *std::max_element(labels.begin(), labels.end()) + 1;
        std::vector<double> sum_in(n_comm, 0.0);    // Sum of weights inside community
        std::vector<double> sum_tot(n_comm, 0.0);   // Sum of total weights to community
        
        for (int i = 0; i < n; ++i) {
            sum_tot[labels[i]] += k(i);
        }
        
        for (int i = 0; i < A.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if (labels[i] == labels[it.row()]) {
                    sum_in[labels[i]] += it.value();
                }
            }
        }
        for (int c = 0; c < n_comm; ++c) {
            sum_in[c] /= 2.0;  // Each edge counted twice
        }
        
        // Iterate until convergence
        for (int iter = 0; iter < max_iterations; ++iter) {
            bool improved = false;
            
            for (int idx : order) {
                int i = idx;
                int c_i = labels[i];
                
                // Compute edges to each neighbor community
                std::unordered_map<int, double> k_i_in;  // edges from i to each community
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                    int j = it.row();
                    k_i_in[labels[j]] += it.value();
                }
                
                // Remove i from its community
                sum_tot[c_i] -= k(i);
                sum_in[c_i] -= k_i_in[c_i];
                labels[i] = -1;  // Temporarily unassigned
                
                // Find best community
                double best_delta = 0.0;
                int best_c = c_i;
                
                for (const auto& [c, k_ic] : k_i_in) {
                    // Modularity gain: ΔQ = k_ic/m - γ * k_i * Σ_tot_c / 2m²
                    double delta = k_ic / m - resolution * k(i) * sum_tot[c] / (2.0 * m * m);
                    
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_c = c;
                    }
                }
                
                // Also consider original community
                if (k_i_in.find(c_i) != k_i_in.end()) {
                    double delta = k_i_in[c_i] / m - resolution * k(i) * sum_tot[c_i] / (2.0 * m * m);
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_c = c_i;
                    }
                }
                
                // Assign to best community  
                labels[i] = best_c;
                sum_tot[best_c] += k(i);
                sum_in[best_c] += k_i_in[best_c];
                
                if (best_c != c_i && best_delta > min_modularity_gain) {
                    improved = true;
                    any_improvement = true;
                }
            }
            
            if (!improved) break;
        }
        
        return any_improvement;
    }
    
    /**
     * @brief Renumber communities to be contiguous 0..n_comm-1
     */
    void renumber_communities(std::vector<int>& labels) {
        std::unordered_map<int, int> mapping;
        int next_id = 0;
        
        for (int& label : labels) {
            if (mapping.find(label) == mapping.end()) {
                mapping[label] = next_id++;
            }
            label = mapping[label];
        }
    }
    
    /**
     * @brief Phase 2: Aggregate communities into super-nodes
     */
    std::pair<Eigen::SparseMatrix<double>, Eigen::VectorXd> aggregate(
        const Eigen::SparseMatrix<double>& A,
        const std::vector<int>& labels,
        const Eigen::VectorXd& k
    ) {
        int n_comm = *std::max_element(labels.begin(), labels.end()) + 1;
        
        // Build aggregated adjacency
        std::vector<Eigen::Triplet<double>> triplets;
        std::unordered_map<long long, double> edge_weights;
        
        for (int i = 0; i < A.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                int c1 = labels[i];
                int c2 = labels[it.row()];
                long long key = static_cast<long long>(c1) * n_comm + c2;
                edge_weights[key] += it.value();
            }
        }
        
        for (const auto& [key, w] : edge_weights) {
            int c1 = key / n_comm;
            int c2 = key % n_comm;
            triplets.emplace_back(c1, c2, w);
        }
        
        Eigen::SparseMatrix<double> A_new(n_comm, n_comm);
        A_new.setFromTriplets(triplets.begin(), triplets.end());
        
        // Aggregate degrees
        Eigen::VectorXd k_new = Eigen::VectorXd::Zero(n_comm);
        for (int i = 0; i < (int)labels.size(); ++i) {
            k_new(labels[i]) += k(i);
        }
        
        return {A_new, k_new};
    }
    
    /**
     * @brief Expand hierarchical labels to original node labels
     */
    std::vector<int> expand_labels(const std::vector<std::vector<int>>& hierarchy) {
        if (hierarchy.empty()) return {};
        
        std::vector<int> labels = hierarchy[0];
        
        for (size_t level = 1; level < hierarchy.size(); ++level) {
            const auto& level_labels = hierarchy[level];
            for (int& label : labels) {
                label = level_labels[label];
            }
        }
        
        return labels;
    }
};

} // namespace graph
} // namespace statelix

#endif // STATELIX_LOUVAIN_H
