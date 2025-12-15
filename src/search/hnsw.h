/**
 * @file hnsw.h
 * @brief Statelix v1.1 - Hierarchical Navigable Small World (HNSW)
 * 
 * Approximate Nearest Neighbor search with:
 *   - O(log N) search time
 *   - High recall (typically > 95%)
 *   - Incremental index building
 * 
 * Algorithm:
 * ----------
 * HNSW builds a multi-layer graph where:
 *   - Layer 0: Contains all points with dense local connections
 *   - Layer L: Contains points with sparser long-range connections
 *   - Entry point: Highest layer node, used to begin traversal
 * 
 * Search proceeds top-down: start at entry point in highest layer,
 * greedily descend to layer 0, then perform local search.
 * 
 * Reference: Malkov, Y. & Yashunin, D. (2018). Efficient and Robust
 *            Approximate Nearest Neighbor using Hierarchical Navigable
 *            Small World Graphs. IEEE PAMI.
 */
#ifndef STATELIX_HNSW_H
#define STATELIX_HNSW_H

#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <stdexcept>

namespace statelix {
namespace search {

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief HNSW construction and search parameters
 */
struct HNSWConfig {
    // Construction parameters
    int M = 16;                     // Max edges per node per layer
    int M_max_0 = 32;               // Max edges at layer 0 (typically 2*M)
    int ef_construction = 200;      // Search width during construction
    
    // Search parameters
    int ef_search = 50;             // Search width (higher = better recall, slower)
    
    // Distance metric
    enum class Distance { L2, COSINE, INNER_PRODUCT };
    Distance distance = Distance::L2;
    
    // Random level generation
    double level_mult = 1.0 / std::log(M);  // m_L in paper
    
    // Memory/performance tradeoffs
    bool precompute_norms = true;   // For cosine/IP distances
    
    // Random seed
    unsigned int seed = 42;
};

// =============================================================================
// Search Result
// =============================================================================

/**
 * @brief Single nearest neighbor result
 */
struct Neighbor {
    int id;
    double distance;
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
    
    bool operator>(const Neighbor& other) const {
        return distance > other.distance;
    }
};

/**
 * @brief k-NN search result
 */
struct HNSWSearchResult {
    std::vector<int> indices;           // Nearest neighbor indices
    std::vector<double> distances;      // Corresponding distances
    int n_comparisons = 0;              // Number of distance computations
};

// =============================================================================
// HNSW Index
// =============================================================================

/**
 * @brief HNSW Approximate Nearest Neighbor Index
 * 
 * Usage:
 *   HNSW index;
 *   index.build(data);  // data: (n_points, dim)
 *   auto result = index.query(query_point, k);
 */
class HNSW {
public:
    HNSWConfig config;
    
    HNSW() = default;
    explicit HNSW(const HNSWConfig& cfg) : config(cfg) {}
    
    /**
     * @brief Build index from data matrix
     * @param data Points as rows (n_points, dim)
     */
    void build(const Eigen::MatrixXd& data) {
        int n = data.rows();
        dim_ = data.cols();
        
        if (n == 0) return;
        
        // Store data
        data_ = data;
        
        // Precompute norms for cosine/IP
        if (config.precompute_norms && 
            config.distance != HNSWConfig::Distance::L2) {
            norms_.resize(n);
            for (int i = 0; i < n; ++i) {
                norms_(i) = data.row(i).norm();
            }
        }
        
        // Initialize graph structure
        rng_.seed(config.seed);
        max_level_ = 0;
        entry_point_ = 0;
        
        // Allocate adjacency lists
        layers_.clear();
        layers_.resize(1);  // At least layer 0
        layers_[0].resize(n);
        
        // Add points incrementally
        for (int i = 0; i < n; ++i) {
            insert(i);
        }
    }
    
    /**
     * @brief Find k nearest neighbors
     * @param query Query point (dim,)
     * @param k Number of neighbors
     */
    HNSWSearchResult query(const Eigen::VectorXd& query, int k) const {
        if (data_.rows() == 0) {
            return HNSWSearchResult();
        }
        
        HNSWSearchResult result;
        result.n_comparisons = 0;
        
        // Search from top layer down
        int current = entry_point_;
        
        for (int level = max_level_; level > 0; --level) {
            current = search_layer_greedy(query, current, level, result.n_comparisons);
        }
        
        // Search layer 0 with beam search
        auto candidates = search_layer(query, current, 0, 
                                       std::max(k, config.ef_search),
                                       result.n_comparisons);
        
        // Return top k
        while (candidates.size() > static_cast<size_t>(k)) {
            candidates.pop();
        }
        
        result.indices.resize(candidates.size());
        result.distances.resize(candidates.size());
        
        int idx = candidates.size() - 1;
        while (!candidates.empty()) {
            result.indices[idx] = candidates.top().id;
            result.distances[idx] = candidates.top().distance;
            candidates.pop();
            idx--;
        }
        
        return result;
    }
    
    /**
     * @brief Batch query
     */
    std::vector<HNSWSearchResult> query_batch(
        const Eigen::MatrixXd& queries, int k
    ) const {
        int n_queries = queries.rows();
        std::vector<HNSWSearchResult> results(n_queries);
        
        // Parallel execution
        #pragma omp parallel for
        for (int i = 0; i < n_queries; ++i) {
            results[i] = query(queries.row(i), k);
        }
        
        return results;
    }
    
    /**
     * @brief Compute recall against brute-force search
     */
    double compute_recall(
        const Eigen::MatrixXd& queries,
        int k
    ) const {
        int n_correct = 0;
        int n_total = 0;
        
        for (int i = 0; i < queries.rows(); ++i) {
            auto hnsw_result = query(queries.row(i), k);
            auto exact_result = brute_force_knn(queries.row(i), k);
            
            std::unordered_set<int> exact_set(
                exact_result.indices.begin(), exact_result.indices.end());
            
            for (int idx : hnsw_result.indices) {
                if (exact_set.count(idx)) {
                    n_correct++;
                }
            }
            n_total += k;
        }
        
        return static_cast<double>(n_correct) / n_total;
    }
    
    /**
     * @brief Brute-force k-NN (for comparison)
     */
    HNSWSearchResult brute_force_knn(const Eigen::VectorXd& query, int k) const {
        std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> pq;
        
        for (int i = 0; i < data_.rows(); ++i) {
            double d = distance(query, i);
            pq.push({i, d});
            if (static_cast<int>(pq.size()) > k) {
                pq.pop();
            }
        }
        
        HNSWSearchResult result;
        result.indices.resize(pq.size());
        result.distances.resize(pq.size());
        
        int idx = pq.size() - 1;
        while (!pq.empty()) {
            result.indices[idx] = pq.top().id;
            result.distances[idx] = pq.top().distance;
            pq.pop();
            idx--;
        }
        
        return result;
    }
    
    /**
     * @brief Save index to file
     */
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for writing: " + path);
        
        // Header
        int n = data_.rows();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
        out.write(reinterpret_cast<const char*>(&entry_point_), sizeof(entry_point_));
        
        // Data
        out.write(reinterpret_cast<const char*>(data_.data()), 
                  n * dim_ * sizeof(double));
        
        // Graph structure
        int n_layers = layers_.size();
        out.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));
        
        for (const auto& layer : layers_) {
            int layer_size = layer.size();
            out.write(reinterpret_cast<const char*>(&layer_size), sizeof(layer_size));
            
            for (const auto& neighbors : layer) {
                int n_neighbors = neighbors.size();
                out.write(reinterpret_cast<const char*>(&n_neighbors), sizeof(n_neighbors));
                out.write(reinterpret_cast<const char*>(neighbors.data()), 
                          n_neighbors * sizeof(int));
            }
        }
    }
    
    /**
     * @brief Load index from file
     */
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for reading: " + path);
        
        // Header
        int n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
        in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
        in.read(reinterpret_cast<char*>(&entry_point_), sizeof(entry_point_));
        
        // Data
        data_.resize(n, dim_);
        in.read(reinterpret_cast<char*>(data_.data()), n * dim_ * sizeof(double));
        
        // Graph structure
        int n_layers;
        in.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
        
        layers_.resize(n_layers);
        for (auto& layer : layers_) {
            int layer_size;
            in.read(reinterpret_cast<char*>(&layer_size), sizeof(layer_size));
            layer.resize(layer_size);
            
            for (auto& neighbors : layer) {
                int n_neighbors;
                in.read(reinterpret_cast<char*>(&n_neighbors), sizeof(n_neighbors));
                neighbors.resize(n_neighbors);
                in.read(reinterpret_cast<char*>(neighbors.data()), 
                        n_neighbors * sizeof(int));
            }
        }
        
        // Recompute norms if needed
        if (config.precompute_norms && 
            config.distance != HNSWConfig::Distance::L2) {
            norms_.resize(n);
            for (int i = 0; i < n; ++i) {
                norms_(i) = data_.row(i).norm();
            }
        }
    }
    
    // Accessors
    int size() const { return data_.rows(); }
    int dim() const { return dim_; }
    int n_layers() const { return layers_.size(); }

private:
    Eigen::MatrixXd data_;
    Eigen::VectorXd norms_;             // Precomputed norms
    int dim_ = 0;
    int max_level_ = 0;
    int entry_point_ = 0;
    
    // Adjacency lists: layers_[level][node] = list of neighbor indices
    std::vector<std::vector<std::vector<int>>> layers_;
    
    std::mt19937 rng_;
    
    /**
     * @brief Compute distance between query and stored point
     */
    double distance(const Eigen::VectorXd& query, int idx) const {
        switch (config.distance) {
            case HNSWConfig::Distance::L2:
                return (query - data_.row(idx).transpose()).squaredNorm();
            
            case HNSWConfig::Distance::COSINE: {
                double dot = query.dot(data_.row(idx).transpose());
                double norm_q = query.norm();
                double norm_d = (norms_.size() > 0) ? norms_(idx) : 
                               data_.row(idx).norm();
                return 1.0 - dot / (norm_q * norm_d + 1e-10);
            }
            
            case HNSWConfig::Distance::INNER_PRODUCT:
                return -query.dot(data_.row(idx).transpose());  // Negative for min-heap
            
            default:
                return (query - data_.row(idx).transpose()).squaredNorm();
        }
    }
    
    /**
     * @brief Compute distance between two stored points
     */
    double distance(int i, int j) const {
        return distance(data_.row(i).transpose(), j);
    }
    
    /**
     * @brief Random level generation (geometric distribution)
     */
    int random_level() {
        std::uniform_real_distribution<double> dist(1e-10, 1.0);
        double r = dist(rng_);
        return static_cast<int>(-std::log(r) * config.level_mult);
    }
    
    /**
     * @brief Insert a point into the index
     */
    void insert(int id) {
        int level = random_level();
        level = std::min(level, max_level_ + 1);  // Don't jump too many levels
        
        // Ensure we have enough layers
        while (static_cast<int>(layers_.size()) <= level) {
            layers_.emplace_back();
        }
        
        // Initialize empty neighbor lists for this point
        for (int l = 0; l <= level; ++l) {
            if (static_cast<int>(layers_[l].size()) <= id) {
                layers_[l].resize(id + 1);
            }
        }
        
        // First point is special
        if (id == 0) {
            entry_point_ = 0;
            max_level_ = level;
            return;
        }
        
        Eigen::VectorXd point = data_.row(id).transpose();
        int current = entry_point_;
        int dummy_comparisons = 0;
        
        // Descend from top to insertion level + 1
        for (int l = max_level_; l > level; --l) {
            current = search_layer_greedy(point, current, l, dummy_comparisons);
        }
        
        // Insert at each level from level down to 0
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            // Find ef_construction nearest neighbors
            auto candidates = search_layer(point, current, l, 
                                          config.ef_construction, 
                                          dummy_comparisons);
            
            // Select M neighbors
            int M = (l == 0) ? config.M_max_0 : config.M;
            auto neighbors = select_neighbors(candidates, M);
            
            // Add bidirectional edges
            layers_[l][id] = neighbors;
            
            for (int neighbor : neighbors) {
                layers_[l][neighbor].push_back(id);
                
                // Prune if too many edges
                int max_edges = (l == 0) ? config.M_max_0 : config.M;
                if (static_cast<int>(layers_[l][neighbor].size()) > max_edges) {
                    prune_connections(neighbor, l, max_edges);
                }
            }
            
            if (!neighbors.empty()) {
                current = neighbors[0];  // Use closest as entry for next level
            }
        }
        
        // Update entry point if this point is higher
        if (level > max_level_) {
            entry_point_ = id;
            max_level_ = level;
        }
    }
    
    /**
     * @brief Greedy search in a layer (returns single closest point)
     */
    int search_layer_greedy(
        const Eigen::VectorXd& query,
        int entry,
        int level,
        int& n_comparisons
    ) const {
        int current = entry;
        double current_dist = distance(query, current);
        n_comparisons++;
        
        bool improved = true;
        while (improved) {
            improved = false;
            
            for (int neighbor : layers_[level][current]) {
                double d = distance(query, neighbor);
                n_comparisons++;
                
                if (d < current_dist) {
                    current = neighbor;
                    current_dist = d;
                    improved = true;
                }
            }
        }
        
        return current;
    }
    
    /**
     * @brief Beam search in a layer (returns priority queue of candidates)
     */
    std::priority_queue<Neighbor> search_layer(
        const Eigen::VectorXd& query,
        int entry,
        int level,
        int ef,
        int& n_comparisons
    ) const {
        std::unordered_set<int> visited;
        
        // Min-heap for candidates to explore
        std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> 
            candidates;
        
        // Max-heap for results (worst at top)
        std::priority_queue<Neighbor> results;
        
        double entry_dist = distance(query, entry);
        n_comparisons++;
        
        candidates.push({entry, entry_dist});
        results.push({entry, entry_dist});
        visited.insert(entry);
        
        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();
            
            // Stop if current candidate is worse than worst result
            if (static_cast<int>(results.size()) >= ef && 
                current.distance > results.top().distance) {
                break;
            }
            
            // Explore neighbors
            for (int neighbor : layers_[level][current.id]) {
                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);
                
                double d = distance(query, neighbor);
                n_comparisons++;
                
                if (static_cast<int>(results.size()) < ef || d < results.top().distance) {
                    candidates.push({neighbor, d});
                    results.push({neighbor, d});
                    
                    if (static_cast<int>(results.size()) > ef) {
                        results.pop();
                    }
                }
            }
        }
        
        return results;
    }
    
    /**
     * @brief Select M best neighbors from candidates
     */
    std::vector<int> select_neighbors(
        std::priority_queue<Neighbor>& candidates,
        int M
    ) {
        std::vector<int> result;
        result.reserve(M);
        
        // Simple selection: take M closest
        std::vector<Neighbor> sorted;
        while (!candidates.empty()) {
            sorted.push_back(candidates.top());
            candidates.pop();
        }

        
        for (int i = 0; i < std::min(M, static_cast<int>(sorted.size())); ++i) {
            result.push_back(sorted[i].id);
        }
        
        return result;
    }
    
    /**
     * @brief Prune connections to keep at most max_edges
     */
    void prune_connections(int node, int level, int max_edges) {
        auto& neighbors = layers_[level][node];
        
        if (static_cast<int>(neighbors.size()) <= max_edges) return;
        
        // Sort by distance and keep closest
        Eigen::VectorXd node_vec = data_.row(node).transpose();
        std::vector<Neighbor> scored;
        for (int n : neighbors) {
            scored.push_back({n, distance(node_vec, n)});
        }
        std::sort(scored.begin(), scored.end());
        
        neighbors.clear();
        for (int i = 0; i < max_edges; ++i) {
            neighbors.push_back(scored[i].id);
        }
    }
};

} // namespace search
} // namespace statelix

#endif // STATELIX_HNSW_H
