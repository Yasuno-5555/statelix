/**
 * @file pagerank.h
 * @brief Statelix v1.1 - PageRank Algorithm
 * 
 * Implements:
 *   - Power iteration PageRank
 *   - Personalized PageRank
 *   - Sparse matrix × vector benchmark
 * 
 * Algorithm:
 * ----------
 * Standard PageRank:
 *   r = d * M * r + (1 - d) * v
 * where:
 *   M = column-normalized adjacency (transition matrix)
 *   d = damping factor (typically 0.85)
 *   v = personalization vector (uniform by default)
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
#include <algorithm>
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
    
    /**
     * @brief Compute PageRank scores
     * 
     * @param adjacency Sparse adjacency matrix (directed or undirected)
     * @return PageRankResult with scores and ranking
     */
    PageRankResult compute(const Eigen::SparseMatrix<double>& adjacency) {
        int n = adjacency.rows();
        if (n != adjacency.cols()) {
            throw std::invalid_argument("Adjacency matrix must be square");
        }
        
        PageRankResult result;
        result.scores.resize(n);
        
        if (n == 0) {
            result.converged = true;
            result.iterations = 0;
            result.residual = 0;
            return result;
        }
        
        // Build column-normalized transition matrix
        // M_ij = A_ij / out_degree(j)
        Eigen::SparseMatrix<double> M = build_transition_matrix(adjacency);
        
        // Uniform personalization
        Eigen::VectorXd personalization = Eigen::VectorXd::Constant(n, 1.0 / n);
        
        // Power iteration
        Eigen::VectorXd r = personalization;  // Initial: uniform
        Eigen::VectorXd r_new(n);
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // r_new = d * M * r + (1 - d) * personalization
            r_new = damping * (M * r) + (1.0 - damping) * personalization;
            
            // Handle dangling nodes (no outgoing edges)
            double dangling_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                if (out_degree_[i] == 0) {
                    dangling_sum += r(i);
                }
            }
            r_new.array() += damping * dangling_sum / n;
            
            // Check convergence
            double residual = (r_new - r).norm();
            
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
        
        // Normalize
        if (normalize_output) {
            result.scores /= result.scores.sum();
        }
        
        // Compute ranking
        result.ranking = compute_ranking(result.scores);
        
        return result;
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
        
        // Compute with modified damping
        double saved_damping = damping;
        damping = 1.0 - restart_prob;
        
        auto result = compute_with_personalization(adjacency, personalization);
        
        damping = saved_damping;
        return result;
    }

private:
    std::vector<double> out_degree_;
    
    /**
     * @brief Build column-normalized transition matrix
     */
    Eigen::SparseMatrix<double> build_transition_matrix(
        const Eigen::SparseMatrix<double>& A
    ) {
        int n = A.rows();
        
        // Compute out-degrees
        out_degree_.resize(n, 0.0);
        for (int j = 0; j < A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                out_degree_[j] += it.value();
            }
        }
        
        // Build normalized matrix (column-stochastic)
        std::vector<Eigen::Triplet<double>> triplets;
        
        for (int j = 0; j < A.outerSize(); ++j) {
            if (out_degree_[j] > 0) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                    triplets.emplace_back(it.row(), j, it.value() / out_degree_[j]);
                }
            }
        }
        
        Eigen::SparseMatrix<double> M(n, n);
        M.setFromTriplets(triplets.begin(), triplets.end());
        
        return M;
    }
    
    /**
     * @brief Compute with custom personalization vector
     */
    PageRankResult compute_with_personalization(
        const Eigen::SparseMatrix<double>& adjacency,
        const Eigen::VectorXd& personalization
    ) {
        int n = adjacency.rows();
        
        PageRankResult result;
        result.scores.resize(n);
        
        Eigen::SparseMatrix<double> M = build_transition_matrix(adjacency);
        
        Eigen::VectorXd r = personalization;
        Eigen::VectorXd r_new(n);
        
        for (int iter = 0; iter < max_iter; ++iter) {
            r_new = damping * (M * r) + (1.0 - damping) * personalization;
            
            // Handle dangling nodes
            double dangling_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                if (out_degree_[i] == 0) {
                    dangling_sum += r(i);
                }
            }
            r_new += damping * dangling_sum * personalization;
            
            double residual = (r_new - r).norm();
            
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
        
        if (normalize_output) {
            result.scores /= result.scores.sum();
        }
        
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
