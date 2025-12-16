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
 *   Q = (1/2m) Σ_ij [A_ij - γ k_i k_j / 2m] δ(c_i, c_j)
 * 
 * where A = adjacency, k_i = degree(i), m = total edges
 * 
 * Note on undirected graphs:
 * --------------------------
 * This implementation assumes undirected graphs. Input matrices are
 * symmetrized. Self-loops are NOT expected (A_ii = 0). Each edge (i,j)
 * appears twice in the adjacency matrix (A_ij and A_ji).
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
        
        // Make symmetric (undirected) - note: this doubles edge weights if already symmetric
        // We use (A + A^T) / 2 to handle both symmetric and asymmetric inputs
        Eigen::SparseMatrix<double> A = (adjacency + Eigen::SparseMatrix<double>(adjacency.transpose())) / 2.0;
        
        // Initialize: each node is its own community
        std::vector<int> labels(n);
        std::iota(labels.begin(), labels.end(), 0);
        
        // Precompute graph properties
        // m = total edge weight = sum(A) / 2 (since A is symmetric)
        double m = A.sum() / 2.0;
        if (m < 1e-10) {
            // Empty graph
            result.labels = labels;
            result.n_communities = n;
            result.modularity = 0.0;
            return result;
        }
        
        // Weighted degree: k(i) = sum_j A_ij
        Eigen::VectorXd k = Eigen::VectorXd::Zero(n);
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
     * @brief Compute modularity (corrected version)
     * 
     * Q = (1/2m) Σ_ij [A_ij - γ k_i k_j / 2m] δ(c_i, c_j)
     * 
     * Since A is symmetric, we only iterate over upper triangle (i < j)
     * and multiply by 2. Self-loops (i == i) are handled separately.
     * 
     * Note: For efficiency, we rewrite as:
     *   Q = Σ_c [Σ_in(c) / 2m - γ * (Σ_tot(c) / 2m)^2]
     * where Σ_in(c) = sum of edge weights within community c
     *       Σ_tot(c) = sum of degrees of nodes in community c
     */
    double compute_modularity(
        const Eigen::SparseMatrix<double>& A,
        const std::vector<int>& labels,
        const Eigen::VectorXd& k,
        double m
    ) {
        int n_comm = *std::max_element(labels.begin(), labels.end()) + 1;
        
        // sum_in[c] = sum of edge weights inside community c (each edge counted once)
        // sum_tot[c] = sum of degrees of nodes in community c
        std::vector<double> sum_in(n_comm, 0.0);
        std::vector<double> sum_tot(n_comm, 0.0);
        
        for (int i = 0; i < (int)labels.size(); ++i) {
            sum_tot[labels[i]] += k(i);
        }
        
        // Count internal edges - iterate over upper triangle only
        for (int j = 0; j < A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                int i = it.row();
                if (i >= j) continue;  // Only upper triangle (i < j)
                if (labels[i] == labels[j]) {
                    sum_in[labels[i]] += it.value();  // Each edge once
                }
            }
        }
        
        // Q = Σ_c [Σ_in(c) / m - γ * (Σ_tot(c) / 2m)^2]
        // Note: Σ_in / m because m = sum(A)/2 and we counted each edge once
        double Q = 0.0;
        for (int c = 0; c < n_comm; ++c) {
            Q += sum_in[c] / m - resolution * std::pow(sum_tot[c] / (2.0 * m), 2);
        }
        
        return Q;
    }
    
    /**
     * @brief Phase 1: Local node movement optimization
     * 
     * For each node, compute the modularity gain of moving to each
     * neighbor community, and move to the one with maximum gain.
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
        
        // sum_in[c] = sum of internal edge weights (each edge counted once)
        // sum_tot[c] = sum of degrees in community c
        std::vector<double> sum_in(n_comm, 0.0);
        std::vector<double> sum_tot(n_comm, 0.0);
        
        for (int i = 0; i < n; ++i) {
            sum_tot[labels[i]] += k(i);
        }
        
        // Count internal edges (upper triangle only)
        for (int j = 0; j < A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                int i = it.row();
                if (i >= j) continue;
                if (labels[i] == labels[j]) {
                    sum_in[labels[i]] += it.value();
                }
            }
        }
        
        // Pre-allocate vector for k_i_in (faster than unordered_map for small communities)
        std::vector<double> k_i_in(n_comm, 0.0);
        std::vector<int> neighbor_comms;
        neighbor_comms.reserve(n_comm);
        
        // Iterate until convergence
        for (int iter = 0; iter < max_iterations; ++iter) {
            bool improved = false;
            
            for (int idx : order) {
                int i = idx;
                int c_i = labels[i];
                
                // Reset k_i_in for this node
                for (int c : neighbor_comms) {
                    k_i_in[c] = 0.0;
                }
                neighbor_comms.clear();
                
                // Compute edges from node i to each neighbor community
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                    int j = it.row();
                    int c_j = labels[j];
                    if (k_i_in[c_j] == 0.0 && c_j != c_i) {
                        neighbor_comms.push_back(c_j);
                    }
                    k_i_in[c_j] += it.value();
                }
                
                // Also include current community if not already
                if (k_i_in[c_i] == 0.0) {
                    // k_i_in[c_i] is already 0, node has no intra-community edges
                }
                neighbor_comms.push_back(c_i);  // Always consider staying
                
                double k_i = k(i);
                
                // Remove node i from its current community for gain calculation
                // ΔQ for removing i from c_i = -k_i_in[c_i]/m + γ * k_i * (sum_tot[c_i] - k_i) / (2m^2)
                double remove_cost = k_i_in[c_i] / m - 
                                     resolution * k_i * (sum_tot[c_i] - k_i) / (2.0 * m * m);
                
                // Find best community to move to
                double best_gain = 0.0;
                int best_c = c_i;
                
                for (int c : neighbor_comms) {
                    if (c == c_i) {
                        // Staying in current community: gain = 0
                        if (0.0 > best_gain) {
                            best_gain = 0.0;
                            best_c = c_i;
                        }
                    } else {
                        // Moving to community c:
                        // ΔQ = k_i_in[c]/m - γ * k_i * sum_tot[c] / (2m^2) - remove_cost
                        double add_gain = k_i_in[c] / m - 
                                          resolution * k_i * sum_tot[c] / (2.0 * m * m);
                        double total_gain = add_gain - remove_cost;
                        
                        if (total_gain > best_gain) {
                            best_gain = total_gain;
                            best_c = c;
                        }
                    }
                }
                
                // Move node if beneficial
                if (best_c != c_i && best_gain > min_modularity_gain) {
                    // Update sum_tot
                    sum_tot[c_i] -= k_i;
                    sum_tot[best_c] += k_i;
                    
                    // Update sum_in (approximate - will be recomputed next iteration)
                    // Edges from i to nodes in c_i become external (remove from sum_in[c_i])
                    // Edges from i to nodes in best_c become internal (add to sum_in[best_c])
                    sum_in[c_i] -= k_i_in[c_i];
                    sum_in[best_c] += k_i_in[best_c];
                    
                    labels[i] = best_c;
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
     * 
     * Each community becomes a node. Edges between communities become
     * weighted edges between super-nodes. Internal edges become self-loops.
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
        
        for (int j = 0; j < A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                int c1 = labels[j];  // Column community
                int c2 = labels[it.row()];  // Row community
                long long key = static_cast<long long>(c1) * n_comm + c2;
                edge_weights[key] += it.value();
            }
        }
        
        for (const auto& [key, w] : edge_weights) {
            int c1 = key / n_comm;
            int c2 = key % n_comm;
            triplets.emplace_back(c2, c1, w);  // Note: (row, col)
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
