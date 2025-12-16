/**
 * @file pagerank.h
 * @brief Statelix v1.1 - PageRank Algorithm
 * 
 * Implements:
 *   - Power iteration PageRank
 *   - Personalized PageRank (PPR)
 *   - Sparse matrix × vector operations
 * 
 * Algorithm:
 * ----------
 * Standard PageRank (Google variant):
 *   r = d * M * r + d * (dangling_sum / n) * 1 + (1 - d) * v
 * where:
 *   M = column-normalized adjacency (transition matrix)
 *   d = damping factor (typically 0.85)
 *   v = personalization vector (uniform by default)
 *   dangling_sum = sum of rank mass on dangling nodes
 * 
 * Dangling Nodes Handling:
 * ------------------------
 * For standard PageRank, dangling nodes (no outgoing edges) redistribute
 * their rank uniformly to all nodes (Google's approach).
 * 
 * For Personalized PageRank (PPR), there are two common variants:
 *   - Dangling → uniform (this implementation)
 *   - Dangling → personalization vector (alternative)
 * This implementation uses dangling → uniform for standard and
 * dangling → personalization for PPR (Page et al. original).
 * 
 * Graph Type:
 * -----------
 * This implementation assumes a DIRECTED graph. For undirected graphs,
 * the input adjacency matrix should be symmetric (A_ij = A_ji).
 * The transition matrix M is column-normalized (column-stochastic).
 * 
 * Reference: Page, L. et al. (1999). The PageRank Citation Ranking:
 *            Bringing Order to the Web. Stanford InfoLab.
 */
#ifndef STATELIX_PAGERANK_H
#define STATELIX_PAGERANK_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace statelix {
namespace graph {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief PageRank result
 */
struct PageRankResult {
    Eigen::VectorXd scores;         // PageRank scores (sum to 1)
    std::vector<int> ranking;       // Node indices sorted by score (descending)
    int iterations;
    bool converged;
    double residual;                // Final residual norm
};

// =============================================================================
// PageRank Algorithm
// =============================================================================

/**
 * @brief PageRank algorithm with power iteration
 */
class PageRank {
public:
    double damping = 0.85;          // Damping factor
    int max_iter = 100;
    double tol = 1e-6;
    bool normalize_output = true;   // Scores sum to 1
    
    // Dangling node handling strategy
    enum class DanglingStrategy {
        UNIFORM,         // Distribute to all nodes uniformly (Google standard)
        PERSONALIZATION  // Distribute according to personalization vector
    };
    DanglingStrategy dangling_strategy = DanglingStrategy::UNIFORM;
    
    /**
     * @brief Compute PageRank scores
     * 
     * @param adjacency Sparse adjacency matrix (directed graph, A_ij = edge i→j)
     * @return PageRankResult with scores and ranking
     */
    PageRankResult compute(const Eigen::SparseMatrix<double>& adjacency) {
        int n = adjacency.rows();
        if (n != adjacency.cols()) {
            throw std::invalid_argument("Adjacency matrix must be square");
        }
        
        // Uniform personalization for standard PageRank
        Eigen::VectorXd personalization = Eigen::VectorXd::Constant(n, 1.0 / n);
        
        return compute_internal(adjacency, personalization, DanglingStrategy::UNIFORM);
    }
    
    /**
     * @brief Personalized PageRank (PPR)
     * 
     * @param adjacency Adjacency matrix
     * @param seeds Seed nodes (restart from these)
     * @param restart_prob Probability of restart (1 - damping)
     */
    PageRankResult personalized(
        const Eigen::SparseMatrix<double>& adjacency,
        const std::vector<int>& seeds,
        double restart_prob = 0.15
    ) {
        int n = adjacency.rows();
        
        // Build personalization vector concentrated on seeds
        Eigen::VectorXd personalization = Eigen::VectorXd::Zero(n);
        for (int seed : seeds) {
            if (seed >= 0 && seed < n) {
                personalization(seed) = 1.0;
            }
        }
        if (personalization.sum() > 0) {
            personalization /= personalization.sum();
        } else {
            personalization.setConstant(1.0 / n);
        }
        
        // Save and modify damping
        double saved_damping = damping;
        damping = 1.0 - restart_prob;
        
        // For PPR, dangling nodes go to personalization vector (Page et al.)
        auto result = compute_internal(adjacency, personalization, 
                                        DanglingStrategy::PERSONALIZATION);
        
        damping = saved_damping;
        return result;
    }
    
    /**
     * @brief Personalized PageRank with custom personalization vector
     * 
     * @param adjacency Adjacency matrix
     * @param personalization Custom personalization vector (will be normalized)
     * @param use_personalization_for_dangling If true, dangling nodes use 
     *        personalization vector; otherwise uniform
     */
    PageRankResult compute_personalized(
        const Eigen::SparseMatrix<double>& adjacency,
        Eigen::VectorXd personalization,
        bool use_personalization_for_dangling = true
    ) {
        // Normalize personalization
        double sum = personalization.sum();
        if (sum > 0) {
            personalization /= sum;
        } else {
            int n = adjacency.rows();
            personalization = Eigen::VectorXd::Constant(n, 1.0 / n);
        }
        
        DanglingStrategy strategy = use_personalization_for_dangling 
            ? DanglingStrategy::PERSONALIZATION 
            : DanglingStrategy::UNIFORM;
        
        return compute_internal(adjacency, personalization, strategy);
    }

private:
    std::vector<double> out_degree_;
    std::vector<int> dangling_nodes_;  // Cache dangling node indices
    
    /**
     * @brief Build column-normalized transition matrix
     * 
     * M_ij = A_ij / out_degree(j)
     * Columns with zero out-degree are left as zero (handled separately).
     */
    Eigen::SparseMatrix<double> build_transition_matrix(
        const Eigen::SparseMatrix<double>& A
    ) {
        int n = A.rows();
        
        // Compute out-degrees (column sums for column-stochastic)
        out_degree_.assign(n, 0.0);
        dangling_nodes_.clear();
        
        for (int j = 0; j < A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                out_degree_[j] += it.value();
            }
        }
        
        // Identify dangling nodes
        for (int j = 0; j < n; ++j) {
            if (out_degree_[j] == 0.0) {
                dangling_nodes_.push_back(j);
            }
        }
        
        // Build normalized matrix (column-stochastic)
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(A.nonZeros());
        
        for (int j = 0; j < A.outerSize(); ++j) {
            if (out_degree_[j] > 0) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                    triplets.emplace_back(it.row(), j, it.value() / out_degree_[j]);
                }
            }
            // Dangling nodes: column remains zero (handled in power iteration)
        }
        
        Eigen::SparseMatrix<double> M(n, n);
        M.setFromTriplets(triplets.begin(), triplets.end());
        
        return M;
    }
    
    /**
     * @brief Internal power iteration implementation
     */
    PageRankResult compute_internal(
        const Eigen::SparseMatrix<double>& adjacency,
        const Eigen::VectorXd& personalization,
        DanglingStrategy dangling_strat
    ) {
        int n = adjacency.rows();
        
        PageRankResult result;
        result.scores.resize(n);
        
        if (n == 0) {
            result.converged = true;
            result.iterations = 0;
            result.residual = 0;
            return result;
        }
        
        // Build column-normalized transition matrix
        Eigen::SparseMatrix<double> M = build_transition_matrix(adjacency);
        
        // Dangling distribution vector
        Eigen::VectorXd dangling_dist(n);
        if (dangling_strat == DanglingStrategy::UNIFORM) {
            dangling_dist.setConstant(1.0 / n);
        } else {
            dangling_dist = personalization;
        }
        
        // Power iteration
        Eigen::VectorXd r = personalization;  // Initial: personalization
        Eigen::VectorXd r_new(n);
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // Compute dangling mass
            double dangling_sum = 0.0;
            for (int j : dangling_nodes_) {
                dangling_sum += r(j);
            }
            
            // r_new = d * M * r + d * dangling_sum * dangling_dist + (1 - d) * personalization
            r_new = damping * (M * r);
            r_new += damping * dangling_sum * dangling_dist;
            r_new += (1.0 - damping) * personalization;
            
            // Check convergence (L1 norm is more common for PageRank)
            double residual = (r_new - r).lpNorm<1>();
            
            if (residual < tol) {
                result.scores = r_new;
                result.iterations = iter + 1;
                result.converged = true;
                result.residual = residual;
                break;
            }
            
            r = r_new;
            result.iterations = iter + 1;
            result.converged = false;
            result.residual = residual;
        }
        
        if (!result.converged) {
            result.scores = r_new;
        }
        
        // Normalize output
        if (normalize_output) {
            double sum = result.scores.sum();
            if (sum > 0) {
                result.scores /= sum;
            }
        }
        
        // Compute ranking
        result.ranking = compute_ranking(result.scores);
        
        return result;
    }
    
    /**
     * @brief Get ranking (indices sorted by descending score)
     */
    std::vector<int> compute_ranking(const Eigen::VectorXd& scores) {
        std::vector<int> ranking(scores.size());
        std::iota(ranking.begin(), ranking.end(), 0);
        
        std::sort(ranking.begin(), ranking.end(),
                  [&scores](int a, int b) { return scores(a) > scores(b); });
        
        return ranking;
    }
};

} // namespace graph
} // namespace statelix

#endif // STATELIX_PAGERANK_H
