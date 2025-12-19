/**
 * @file psm.h
 * @brief Statelix v2.3 - Propensity Score Matching
 * 
 * Implements:
 *   - Propensity score estimation (logistic regression via IRLS)
 *   - Nearest neighbor matching:
 *       - 1D PS matching: O(n log n) via sorted binary search
 *       - Multivariate matching: O(log n) via HNSW (when using covariates)
 *   - Caliper matching
 *   - Radius matching
 *   - Kernel matching
 *   - Inverse probability weighting (IPW)
 *   - Doubly robust estimation (AIPW)
 *   - Balance diagnostics (standardized differences)
 * 
 * Theory:
 * -------
 * Propensity Score: e(X) = P(D=1|X)
 * 
 * Under unconfoundedness (Y(0), Y(1) ⊥ D | X):
 *   ATT = E[Y(1) - Y(0) | D=1]
 *   ATE = E[Y(1) - Y(0)]
 *   ATC = E[Y(1) - Y(0) | D=0]
 * 
 * Matching on PS:
 *   For 1D PS, we use sorted binary search for O(log n) per query.
 *   For multivariate covariate matching, HNSW provides O(log n) ANN.
 * 
 * Reference:
 *   - Rosenbaum, P.R. & Rubin, D.B. (1983). The Central Role of the Propensity Score
 *   - Abadie, A. & Imbens, G.W. (2006). Large Sample Properties of Matching Estimators
 */
#ifndef STATELIX_PSM_H
#define STATELIX_PSM_H

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <stdexcept>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Optional HNSW for multivariate matching
#ifdef STATELIX_USE_HNSW
#include "../search/hnsw.h"
#endif

#include "../linear_model/solver.h"

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

enum class MatchingMethod {
    NEAREST_NEIGHBOR,   // k-NN on propensity score (1D binary search)
    CALIPER,            // k-NN with caliper constraint
    RADIUS,             // All matches within radius
    KERNEL,             // Kernel-weighted matching
    COVARIATE           // Multivariate k-NN on covariates (uses HNSW if available)
};

struct PropensityScoreResult {
    Eigen::VectorXd scores;         // Predicted propensity scores
    Eigen::VectorXd coef;           // Logistic regression coefficients
    double intercept;
    int n_treated;
    int n_control;
    double overlap_min;             // Min common support
    double overlap_max;             // Max common support
};

struct MatchingResult {
    // Treatment effect estimates
    double att;                     // Average Treatment effect on Treated
    double att_se;
    double att_t;
    double att_pvalue;
    double att_ci_lower;
    double att_ci_upper;
    
    double ate;                     // Average Treatment Effect (proper estimate)
    double ate_se;
    
    double atc;                     // Average Treatment effect on Control (proper estimate)
    double atc_se;
    
    // Matching info
    std::vector<std::vector<int>> matches;  // For each treated, matched control indices
    std::vector<std::vector<int>> matches_atc;  // For ATC: each control's matched treated
    int n_matched_treated;
    int n_matched_control;
    double caliper_used;
    
    // Balance diagnostics
    Eigen::VectorXd std_diff_before;    // Standardized differences before matching
    Eigen::VectorXd std_diff_after;     // After matching
    double mean_std_diff_before;
    double mean_std_diff_after;
    
    MatchingMethod method;
};

struct IPWResult {
    double att;
    double att_se;
    double ate;
    double ate_se;
    double atc;
    double atc_se;
    Eigen::VectorXd weights;        // IPW weights
    int n_trimmed;                  // Observations trimmed due to extreme weights
};

struct AIPWResult {
    double att;
    double att_se;
    double ate;
    double ate_se;
    double atc;
    double atc_se;
    double efficiency_gain;         // % reduction in variance vs IPW
};

// =============================================================================
// Sorted Index for 1D Matching (O(log n) per query)
// =============================================================================

/**
 * @brief Sorted index for efficient 1D nearest neighbor search
 * 
 * For propensity score matching, this is MORE efficient than HNSW
 * because PS is 1-dimensional. Build: O(n log n), Query: O(k log n).
 */
class SortedIndex1D {
public:
    void build(const Eigen::VectorXd& values, const std::vector<int>& indices) {
        sorted_.clear();
        sorted_.reserve(indices.size());
        for (int idx : indices) {
            sorted_.push_back({values(idx), idx});
        }
        std::sort(sorted_.begin(), sorted_.end());
    }
    
    /**
     * @brief Find k nearest neighbors to query value
     * 
     * @param query Query value
     * @param k Number of neighbors
     * @param max_dist Maximum distance (caliper)
     * @param exclude Set of indices to exclude (for without-replacement)
     * @return Vector of (distance, index) pairs, sorted by distance
     */
    std::vector<std::pair<double, int>> query(
        double query,
        int k,
        double max_dist = std::numeric_limits<double>::infinity(),
        const std::unordered_set<int>* exclude = nullptr
    ) const {
        if (sorted_.empty()) return {};
        
        // Binary search for insertion point
        auto it = std::lower_bound(sorted_.begin(), sorted_.end(), 
                                   std::make_pair(query, 0));
        
        std::vector<std::pair<double, int>> result;
        result.reserve(k);
        
        // Expand left and right from insertion point
        auto left = (it == sorted_.begin()) ? sorted_.begin() : std::prev(it);
        auto right = it;
        
        while (result.size() < static_cast<size_t>(k)) {
            bool has_left = (left >= sorted_.begin() && left < sorted_.end());
            bool has_right = (right < sorted_.end());
            
            if (!has_left && !has_right) break;
            
            double dist_left = has_left ? std::abs(left->first - query) : 
                               std::numeric_limits<double>::infinity();
            double dist_right = has_right ? std::abs(right->first - query) : 
                                std::numeric_limits<double>::infinity();
            
            // Take closer one
            if (dist_left <= dist_right) {
                if (dist_left <= max_dist) {
                    if (!exclude || exclude->find(left->second) == exclude->end()) {
                        result.push_back({dist_left, left->second});
                    }
                }
                if (left == sorted_.begin()) {
                    has_left = false;
                    left = sorted_.end();  // Mark as exhausted
                } else {
                    --left;
                }
            } else {
                if (dist_right <= max_dist) {
                    if (!exclude || exclude->find(right->second) == exclude->end()) {
                        result.push_back({dist_right, right->second});
                    }
                }
                ++right;
            }
            
            // Early termination if remaining elements are too far
            double min_possible = std::min(
                has_left ? std::abs(left->first - query) : std::numeric_limits<double>::infinity(),
                has_right ? std::abs(right->first - query) : std::numeric_limits<double>::infinity()
            );
            if (min_possible > max_dist && result.size() >= static_cast<size_t>(k)) break;
        }
        
        return result;
    }
    
    /**
     * @brief Find all neighbors within radius
     */
    std::vector<std::pair<double, int>> radius_query(
        double query,
        double radius,
        const std::unordered_set<int>* exclude = nullptr
    ) const {
        if (sorted_.empty()) return {};
        
        std::vector<std::pair<double, int>> result;
        
        // Binary search for lower bound
        auto lower = std::lower_bound(sorted_.begin(), sorted_.end(),
                                       std::make_pair(query - radius, INT_MIN));
        auto upper = std::upper_bound(sorted_.begin(), sorted_.end(),
                                       std::make_pair(query + radius, INT_MAX));
        
        for (auto it = lower; it != upper; ++it) {
            double dist = std::abs(it->first - query);
            if (dist <= radius) {
                if (!exclude || exclude->find(it->second) == exclude->end()) {
                    result.push_back({dist, it->second});
                }
            }
        }
        
        std::sort(result.begin(), result.end());
        return result;
    }
    
    int size() const { return sorted_.size(); }

private:
    std::vector<std::pair<double, int>> sorted_;  // (value, original_index)
};

// =============================================================================
// Propensity Score Matching
// =============================================================================

/**
 * @brief Propensity Score Matching estimator
 */
class PropensityScoreMatching {
public:
    MatchingMethod method = MatchingMethod::NEAREST_NEIGHBOR;
    int n_neighbors = 1;            // Number of matches per treated
    bool with_replacement = true;
    double caliper = -1;            // -1 for no caliper, else fraction of SD
    double radius = -1;             // For RADIUS method
    double kernel_bandwidth = 0.06;
    double trim_threshold = 0.01;   // Trim extreme propensity scores
    double conf_level = 0.95;
    unsigned int seed = 42;
    
    /**
     * @brief Estimate propensity scores
     */
    PropensityScoreResult estimate_propensity(
        const Eigen::VectorXd& D,
        const Eigen::MatrixXd& X
    ) {
        int n = D.size();
        int k = X.cols();
        
        PropensityScoreResult result;
        
        // Add intercept
        Eigen::MatrixXd X_aug(n, k + 1);
        X_aug.col(0).setOnes();
        X_aug.rightCols(k) = X;
        
        // Logistic regression via IRLS
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(k + 1);
        
        // Weighted Solver for robust IRLS
        // Using AUTO strategy to fallback to QR if XT W X is singular (e.g. collinearity)
        WeightedSolver solver(SolverStrategy::AUTO);
        
        for (int iter = 0; iter < 50; ++iter) {
            Eigen::VectorXd eta = X_aug * beta;
            Eigen::VectorXd p(n);
            for (int i = 0; i < n; ++i) {
                p(i) = 1.0 / (1.0 + std::exp(-eta(i)));
                p(i) = std::max(1e-10, std::min(1.0 - 1e-10, p(i)));
            }
            
            Eigen::VectorXd w = p.array() * (1.0 - p.array());
            Eigen::VectorXd z = (eta.array() + (D - p).array() / w.array()).matrix();
            
            // Use WeightedSolver for efficiency and robustness
            // Re-create WeightedDesignMatrix as weights change every iteration
            WeightedDesignMatrix wdm(X_aug, w);
            
            // Reset solver to force re-decomposition
            solver.reset();
            
            Eigen::VectorXd beta_new;
            try {
                beta_new = solver.solve(wdm, z);
            } catch (const std::exception& e) {
                // If even QR fails, break or throw? 
                // For now, stop iteration.
                break;
            }
            
            if ((beta_new - beta).norm() < 1e-8) {
                beta = beta_new;
                break;
            }
            beta = beta_new;
        }
        
        result.intercept = beta(0);
        result.coef = beta.tail(k);
        
        // Compute propensity scores
        result.scores.resize(n);
        Eigen::VectorXd eta = X_aug * beta;
        for (int i = 0; i < n; ++i) {
            result.scores(i) = 1.0 / (1.0 + std::exp(-eta(i)));
        }
        
        // Count treated/control
        result.n_treated = 0;
        result.n_control = 0;
        for (int i = 0; i < n; ++i) {
            if (D(i) > 0.5) result.n_treated++;
            else result.n_control++;
        }
        
        // Common support
        double min_t = 1, max_t = 0, min_c = 1, max_c = 0;
        for (int i = 0; i < n; ++i) {
            if (D(i) > 0.5) {
                min_t = std::min(min_t, result.scores(i));
                max_t = std::max(max_t, result.scores(i));
            } else {
                min_c = std::min(min_c, result.scores(i));
                max_c = std::max(max_c, result.scores(i));
            }
        }
        result.overlap_min = std::max(min_t, min_c);
        result.overlap_max = std::min(max_t, max_c);
        
        return result;
    }
    
    /**
     * @brief Perform matching and estimate treatment effect
     */
    MatchingResult match(
        const Eigen::VectorXd& Y,
        const Eigen::VectorXd& D,
        const Eigen::MatrixXd& X,
        const PropensityScoreResult& ps
    ) {
        int n = Y.size();
        MatchingResult result;
        result.method = method;
        
        // Separate treated and control indices
        std::vector<int> treated_idx, control_idx;
        for (int i = 0; i < n; ++i) {
            if (D(i) > 0.5) treated_idx.push_back(i);
            else control_idx.push_back(i);
        }
        
        int n_t = treated_idx.size();
        int n_c = control_idx.size();
        
        // Balance before matching
        compute_balance(X, D, result.std_diff_before);
        result.mean_std_diff_before = result.std_diff_before.array().abs().mean();
        
        // Determine caliper
        if (caliper < 0) {
            result.caliper_used = std::numeric_limits<double>::infinity();
        } else {
            double sd_ps = std::sqrt((ps.scores.array() - ps.scores.mean()).square().mean());
            result.caliper_used = caliper * sd_ps;
        }
        
        // Build sorted index for controls (for ATT)
        SortedIndex1D control_index;
        control_index.build(ps.scores, control_idx);
        
        // Build sorted index for treated (for ATC)
        SortedIndex1D treated_index;
        treated_index.build(ps.scores, treated_idx);
        
        // === ATT Matching ===
        result.matches.resize(n_t);
        std::unordered_set<int> used_controls;
        
        match_units(ps.scores, treated_idx, control_idx, control_index,
                    result.matches, used_controls, result.caliper_used);
        
        // === ATC Matching (controls matched to treated) ===
        result.matches_atc.resize(n_c);
        std::unordered_set<int> used_treated;
        
        match_units(ps.scores, control_idx, treated_idx, treated_index,
                    result.matches_atc, used_treated, result.caliper_used);
        
        // Count matched
        result.n_matched_treated = 0;
        for (const auto& m : result.matches) {
            if (!m.empty()) result.n_matched_treated++;
        }
        result.n_matched_control = used_controls.size();
        
        // === Estimate ATT ===
        double att = 0, att_var = 0;
        estimate_effect(Y, treated_idx, result.matches, ps.scores, att, att_var);
        result.att = att;
        result.att_se = std::sqrt(att_var);
        
        // === Estimate ATC ===
        double atc = 0, atc_var = 0;
        estimate_effect_atc(Y, control_idx, result.matches_atc, ps.scores, atc, atc_var);
        result.atc = atc;
        result.atc_se = std::sqrt(atc_var);
        
        // === Estimate ATE ===
        // ATE = p * ATT + (1-p) * ATC where p = P(D=1)
        double p_treated = static_cast<double>(n_t) / n;
        result.ate = p_treated * result.att + (1 - p_treated) * result.atc;
        result.ate_se = std::sqrt(p_treated * p_treated * att_var + 
                                  (1 - p_treated) * (1 - p_treated) * atc_var);
        
        // Inference for ATT
        result.att_t = result.att / result.att_se;
        result.att_pvalue = 2.0 * (1.0 - normal_cdf(std::abs(result.att_t)));
        
        double z_crit = normal_quantile(0.5 + conf_level / 2);
        result.att_ci_lower = result.att - z_crit * result.att_se;
        result.att_ci_upper = result.att + z_crit * result.att_se;
        
        // Balance after matching
        compute_balance_after(X, D, result.matches, treated_idx, control_idx, 
                              result.std_diff_after);
        result.mean_std_diff_after = result.std_diff_after.array().abs().mean();
        
        return result;
    }
    
    /**
     * @brief Inverse Probability Weighting estimator
     */
    IPWResult ipw(
        const Eigen::VectorXd& Y,
        const Eigen::VectorXd& D,
        const PropensityScoreResult& ps
    ) {
        int n = Y.size();
        IPWResult result;
        
        // Compute IPW weights
        result.weights.resize(n);
        result.n_trimmed = 0;
        
        double sum_d = 0;
        for (int i = 0; i < n; ++i) {
            sum_d += D(i);
        }
        double p_treated = sum_d / n;
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            
            // Trim extreme propensity scores
            if (e < trim_threshold || e > 1 - trim_threshold) {
                result.weights(i) = 0;
                result.n_trimmed++;
            } else if (D(i) > 0.5) {
                result.weights(i) = 1.0;  // Treated weight for ATT
            } else {
                result.weights(i) = e / (1.0 - e);  // Control weight for ATT
            }
        }
        
        // ATT via IPW
        double sum_y1 = 0, sum_w1 = 0;
        double sum_y0w = 0, sum_w0 = 0;
        
        for (int i = 0; i < n; ++i) {
            if (result.weights(i) == 0) continue;
            
            if (D(i) > 0.5) {
                sum_y1 += Y(i);
                sum_w1 += 1.0;
            } else {
                sum_y0w += Y(i) * result.weights(i);
                sum_w0 += result.weights(i);
            }
        }
        
        double y1_mean = (sum_w1 > 0) ? sum_y1 / sum_w1 : 0;
        double y0w_mean = (sum_w0 > 0) ? sum_y0w / sum_w0 : 0;
        
        result.att = y1_mean - y0w_mean;
        
        // === ATE via IPW ===
        double sum_ate_1 = 0, sum_ate_0 = 0;
        int n_valid = 0;
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            if (e < trim_threshold || e > 1 - trim_threshold) continue;
            n_valid++;
            
            if (D(i) > 0.5) {
                sum_ate_1 += Y(i) / e;
            } else {
                sum_ate_0 += Y(i) / (1.0 - e);
            }
        }
        result.ate = (n_valid > 0) ? (sum_ate_1 - sum_ate_0) / n_valid : 0;
        
        // === ATC via IPW ===
        double sum_atc_1w = 0, sum_atc_1_w = 0;
        double sum_atc_0 = 0, sum_atc_0_n = 0;
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            if (e < trim_threshold || e > 1 - trim_threshold) continue;
            
            if (D(i) > 0.5) {
                double w = (1.0 - e) / e;
                sum_atc_1w += Y(i) * w;
                sum_atc_1_w += w;
            } else {
                sum_atc_0 += Y(i);
                sum_atc_0_n += 1.0;
            }
        }
        double y1w_mean_atc = (sum_atc_1_w > 0) ? sum_atc_1w / sum_atc_1_w : 0;
        double y0_mean_atc = (sum_atc_0_n > 0) ? sum_atc_0 / sum_atc_0_n : 0;
        result.atc = y1w_mean_atc - y0_mean_atc;
        
        // Variance via influence function
        std::vector<double> psi_att, psi_ate, psi_atc;
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            if (e < trim_threshold || e > 1 - trim_threshold) continue;
            
            // ATT influence
            if (D(i) > 0.5) {
                psi_att.push_back((Y(i) - y1_mean) - result.att);
            } else {
                double w = e / (1.0 - e);
                psi_att.push_back(-w * (Y(i) - y0w_mean));
            }
            
            // ATE influence
            if (D(i) > 0.5) {
                psi_ate.push_back(Y(i) / e - result.ate);
            } else {
                psi_ate.push_back(-Y(i) / (1.0 - e));
            }
            
            // ATC influence
            if (D(i) > 0.5) {
                double w = (1.0 - e) / e;
                psi_atc.push_back(w * (Y(i) - y1w_mean_atc));
            } else {
                psi_atc.push_back((y0_mean_atc - Y(i)) - result.atc);
            }
        }
        
        // Compute variances
        auto compute_var = [](const std::vector<double>& psi) {
            double sum = 0, sum2 = 0;
            for (double p : psi) {
                sum += p;
                sum2 += p * p;
            }
            int n = psi.size();
            return (n > 1) ? sum2 / (n * n) : 0.0;
        };
        
        result.att_se = std::sqrt(compute_var(psi_att));
        result.ate_se = std::sqrt(compute_var(psi_ate));
        result.atc_se = std::sqrt(compute_var(psi_atc));
        
        return result;
    }
    
    /**
     * @brief Augmented IPW (Doubly Robust) estimator
     * 
     * AIPW ATT = E[D(Y - μ₀(X)) / P(D=1)] + E[(1-D)e(X)/(1-e(X)) * (Y - μ₀(X)) / P(D=1)]
     */
    AIPWResult aipw(
        const Eigen::VectorXd& Y,
        const Eigen::VectorXd& D,
        const Eigen::MatrixXd& X,
        const PropensityScoreResult& ps
    ) {
        int n = Y.size();
        int k = X.cols();
        AIPWResult result;
        
        // Estimate outcome models μ₁(X) = E[Y|D=1,X] and μ₀(X) = E[Y|D=0,X]
        std::vector<int> t_idx, c_idx;
        for (int i = 0; i < n; ++i) {
            if (D(i) > 0.5) t_idx.push_back(i);
            else c_idx.push_back(i);
        }
        
        // OLS for treated: Y = Xβ₁
        Eigen::MatrixXd X1(t_idx.size(), k + 1);
        Eigen::VectorXd Y1(t_idx.size());
        for (size_t j = 0; j < t_idx.size(); ++j) {
            X1(j, 0) = 1.0;
            X1.row(j).tail(k) = X.row(t_idx[j]);
            Y1(j) = Y(t_idx[j]);
        }
        Eigen::VectorXd beta1 = (X1.transpose() * X1).ldlt().solve(X1.transpose() * Y1);
        
        // OLS for control: Y = Xβ₀
        Eigen::MatrixXd X0(c_idx.size(), k + 1);
        Eigen::VectorXd Y0(c_idx.size());
        for (size_t j = 0; j < c_idx.size(); ++j) {
            X0(j, 0) = 1.0;
            X0.row(j).tail(k) = X.row(c_idx[j]);
            Y0(j) = Y(c_idx[j]);
        }
        Eigen::VectorXd beta0 = (X0.transpose() * X0).ldlt().solve(X0.transpose() * Y0);
        
        // Predict μ₁(X) and μ₀(X) for all observations
        Eigen::VectorXd mu1(n), mu0(n);
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd x(k + 1);
            x(0) = 1.0;
            x.tail(k) = X.row(i).transpose();
            mu1(i) = x.dot(beta1);
            mu0(i) = x.dot(beta0);
        }
        
        // AIPW estimator for ATT
        // ATT = (1/n₁) Σ_i [D_i(Y_i - μ₀(X_i)) - (1-D_i)e(X_i)/(1-e(X_i))(Y_i - μ₀(X_i))]
        double sum_att = 0;
        int n_t = 0;
        std::vector<double> psi_att;
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            if (e < trim_threshold || e > 1 - trim_threshold) continue;
            
            double psi;
            if (D(i) > 0.5) {
                psi = Y(i) - mu0(i);
                n_t++;
            } else {
                psi = -e / (1.0 - e) * (Y(i) - mu0(i));
            }
            sum_att += psi;
            psi_att.push_back(psi);
        }
        result.att = (n_t > 0) ? sum_att / n_t : 0;
        
        // AIPW estimator for ATE
        // ATE = (1/n) Σ_i [μ₁(X_i) - μ₀(X_i) + D_i(Y_i-μ₁)/e - (1-D_i)(Y_i-μ₀)/(1-e)]
        double sum_ate = 0;
        int n_valid = 0;
        std::vector<double> psi_ate;
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            if (e < trim_threshold || e > 1 - trim_threshold) continue;
            
            double psi = mu1(i) - mu0(i);
            if (D(i) > 0.5) {
                psi += (Y(i) - mu1(i)) / e;
            } else {
                psi -= (Y(i) - mu0(i)) / (1.0 - e);
            }
            sum_ate += psi;
            n_valid++;
            psi_ate.push_back(psi);
        }
        result.ate = (n_valid > 0) ? sum_ate / n_valid : 0;
        
        // AIPW estimator for ATC
        double sum_atc = 0;
        int n_c = 0;
        std::vector<double> psi_atc;
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            if (e < trim_threshold || e > 1 - trim_threshold) continue;
            
            double psi;
            if (D(i) > 0.5) {
                psi = (1.0 - e) / e * (Y(i) - mu1(i));
            } else {
                psi = mu1(i) - Y(i);
                n_c++;
            }
            sum_atc += psi;
            psi_atc.push_back(psi);
        }
        result.atc = (n_c > 0) ? sum_atc / n_c : 0;
        
        // Variances
        auto compute_var = [](const std::vector<double>& psi, double mean, int denom) {
            double sum2 = 0;
            for (double p : psi) {
                double centered = p - mean * (psi.size() > 0);
                sum2 += centered * centered;
            }
            return sum2 / (denom * denom);
        };
        
        result.att_se = std::sqrt(compute_var(psi_att, result.att, n_t));
        result.ate_se = std::sqrt(compute_var(psi_ate, result.ate, n_valid));
        result.atc_se = std::sqrt(compute_var(psi_atc, result.atc, n_c));
        
        // Efficiency gain vs IPW
        IPWResult ipw_result = ipw(Y, D, ps);
        result.efficiency_gain = (ipw_result.att_se > 0) ?
            (ipw_result.att_se - result.att_se) / ipw_result.att_se * 100 : 0;
        
        return result;
    }
    
private:
    /**
     * @brief Match units using sorted index
     */
    void match_units(
        const Eigen::VectorXd& scores,
        const std::vector<int>& query_idx,      // Units to match (treated for ATT)
        const std::vector<int>& target_idx,     // Pool to match from (control for ATT)
        const SortedIndex1D& target_index,
        std::vector<std::vector<int>>& matches,
        std::unordered_set<int>& used_targets,
        double caliper_dist
    ) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (size_t t = 0; t < query_idx.size(); ++t) {
            int i = query_idx[t];
            double ps_i = scores(i);
            
            std::vector<std::pair<double, int>> neighbors;
            
            if (method == MatchingMethod::RADIUS) {
                double r = (radius > 0) ? radius : caliper_dist;
                neighbors = target_index.radius_query(
                    ps_i, r, 
                    with_replacement ? nullptr : &used_targets);
            } else if (method == MatchingMethod::KERNEL) {
                // Kernel: get all targets within bandwidth
                neighbors = target_index.radius_query(
                    ps_i, kernel_bandwidth * 3,  // 3 sigma
                    nullptr);  // Kernel always uses all
            } else {
                // NEAREST_NEIGHBOR or CALIPER
                neighbors = target_index.query(
                    ps_i, n_neighbors, caliper_dist,
                    with_replacement ? nullptr : &used_targets);
            }
            
            for (const auto& [dist, j] : neighbors) {
                matches[t].push_back(j);
                if (!with_replacement) {
                    used_targets.insert(j);
                }
            }
        }
    }
    
    /**
     * @brief Estimate ATT from matches
     */
    void estimate_effect(
        const Eigen::VectorXd& Y,
        const std::vector<int>& treated_idx,
        const std::vector<std::vector<int>>& matches,
        const Eigen::VectorXd& scores,
        double& effect,
        double& variance
    ) {
        std::vector<double> diffs;
        
        for (size_t t = 0; t < treated_idx.size(); ++t) {
            if (matches[t].empty()) continue;
            
            int i = treated_idx[t];
            double y_t = Y(i);
            
            double y_c = 0, w_sum = 0;
            
            if (method == MatchingMethod::KERNEL) {
                for (int j : matches[t]) {
                    double dist = std::abs(scores(i) - scores(j));
                    double k = epanechnikov_kernel(dist / kernel_bandwidth);
                    y_c += k * Y(j);
                    w_sum += k;
                }
            } else {
                for (int j : matches[t]) {
                    y_c += Y(j);
                    w_sum += 1.0;
                }
            }
            
            if (w_sum > 0) {
                y_c /= w_sum;
                diffs.push_back(y_t - y_c);
            }
        }
        
        if (diffs.empty()) {
            effect = 0;
            variance = 0;
            return;
        }
        
        // Mean
        effect = 0;
        for (double d : diffs) effect += d;
        effect /= diffs.size();
        
        // Variance (Abadie-Imbens style)
        double var = 0;
        for (double d : diffs) {
            var += (d - effect) * (d - effect);
        }
        var /= (diffs.size() - 1);
        variance = var / diffs.size();
    }
    
    /**
     * @brief Estimate ATC from matches (control matched to treated)
     */
    void estimate_effect_atc(
        const Eigen::VectorXd& Y,
        const std::vector<int>& control_idx,
        const std::vector<std::vector<int>>& matches_atc,
        const Eigen::VectorXd& scores,
        double& effect,
        double& variance
    ) {
        std::vector<double> diffs;
        
        for (size_t c = 0; c < control_idx.size(); ++c) {
            if (matches_atc[c].empty()) continue;
            
            int i = control_idx[c];
            double y_c = Y(i);
            
            double y_t = 0, w_sum = 0;
            
            if (method == MatchingMethod::KERNEL) {
                for (int j : matches_atc[c]) {
                    double dist = std::abs(scores(i) - scores(j));
                    double k = epanechnikov_kernel(dist / kernel_bandwidth);
                    y_t += k * Y(j);
                    w_sum += k;
                }
            } else {
                for (int j : matches_atc[c]) {
                    y_t += Y(j);
                    w_sum += 1.0;
                }
            }
            
            if (w_sum > 0) {
                y_t /= w_sum;
                diffs.push_back(y_t - y_c);  // Treatment effect for this control
            }
        }
        
        if (diffs.empty()) {
            effect = 0;
            variance = 0;
            return;
        }
        
        effect = 0;
        for (double d : diffs) effect += d;
        effect /= diffs.size();
        
        double var = 0;
        for (double d : diffs) {
            var += (d - effect) * (d - effect);
        }
        var /= (diffs.size() - 1);
        variance = var / diffs.size();
    }
    
    void compute_balance(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& D,
        Eigen::VectorXd& std_diff
    ) {
        int k = X.cols();
        std_diff.resize(k);
        
        for (int j = 0; j < k; ++j) {
            double sum_t = 0, sum_c = 0;
            double sum_t2 = 0, sum_c2 = 0;
            int n_t = 0, n_c = 0;
            
            for (int i = 0; i < X.rows(); ++i) {
                if (D(i) > 0.5) {
                    sum_t += X(i, j);
                    sum_t2 += X(i, j) * X(i, j);
                    n_t++;
                } else {
                    sum_c += X(i, j);
                    sum_c2 += X(i, j) * X(i, j);
                    n_c++;
                }
            }
            
            double mean_t = sum_t / n_t;
            double mean_c = sum_c / n_c;
            double var_t = sum_t2 / n_t - mean_t * mean_t;
            double var_c = sum_c2 / n_c - mean_c * mean_c;
            
            double pooled_sd = std::sqrt((var_t + var_c) / 2);
            std_diff(j) = (pooled_sd > 1e-10) ? (mean_t - mean_c) / pooled_sd : 0;
        }
    }
    
    void compute_balance_after(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& D,
        const std::vector<std::vector<int>>& matches,
        const std::vector<int>& treated_idx,
        const std::vector<int>& control_idx,
        Eigen::VectorXd& std_diff
    ) {
        int k = X.cols();
        std_diff.resize(k);
        
        // Get matched treated and their controls
        std::vector<int> matched_t, matched_c;
        for (size_t t = 0; t < matches.size(); ++t) {
            if (!matches[t].empty()) {
                matched_t.push_back(treated_idx[t]);
                for (int j : matches[t]) {
                    matched_c.push_back(j);
                }
            }
        }
        
        for (int j = 0; j < k; ++j) {
            double sum_t = 0, sum_c = 0;
            double sum_t2 = 0, sum_c2 = 0;
            
            for (int i : matched_t) {
                sum_t += X(i, j);
                sum_t2 += X(i, j) * X(i, j);
            }
            for (int i : matched_c) {
                sum_c += X(i, j);
                sum_c2 += X(i, j) * X(i, j);
            }
            
            int n_t = matched_t.size();
            int n_c = matched_c.size();
            
            double mean_t = (n_t > 0) ? sum_t / n_t : 0;
            double mean_c = (n_c > 0) ? sum_c / n_c : 0;
            double var_t = (n_t > 0) ? sum_t2 / n_t - mean_t * mean_t : 0;
            double var_c = (n_c > 0) ? sum_c2 / n_c - mean_c * mean_c : 0;
            
            double pooled_sd = std::sqrt((var_t + var_c) / 2);
            std_diff(j) = (pooled_sd > 1e-10) ? (mean_t - mean_c) / pooled_sd : 0;
        }
    }
    
    double epanechnikov_kernel(double u) {
        if (std::abs(u) > 1) return 0;
        return 0.75 * (1.0 - u * u);
    }
    
    double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    
    double normal_quantile(double p) {
        // Approximation using inverse error function
        double a = 0.147;
        double x = 2 * p - 1;
        double ln = std::log(1 - x * x);
        double s = (x > 0 ? 1 : -1);
        return s * std::sqrt(std::sqrt((2/(M_PI*a) + ln/2) * (2/(M_PI*a) + ln/2) - ln/a) 
                            - (2/(M_PI*a) + ln/2)) * std::sqrt(2);
    }
};

// =============================================================================
// Multivariate Matching with HNSW (for covariate matching)
// =============================================================================

#ifdef STATELIX_USE_HNSW

/**
 * @brief Covariate Matching using HNSW
 * 
 * For high-dimensional covariate matching (Mahalanobis distance, etc.),
 * HNSW provides O(log n) approximate nearest neighbor search.
 * 
 * This is useful when:
 *   - Matching directly on covariates (not PS)
 *   - Matching on PS + covariates combined
 *   - Very high-dimensional feature spaces
 */
class CovariateMatching {
public:
    int n_neighbors = 1;
    bool with_replacement = true;
    search::HNSWConfig hnsw_config;
    
    /**
     * @brief Match treated to controls using covariate distance
     * 
     * @param X Covariates (n × d)
     * @param D Treatment indicator
     * @param mahalanobis If true, use Mahalanobis distance
     */
    std::vector<std::vector<int>> match(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& D,
        bool mahalanobis = false
    ) {
        int n = X.rows();
        
        // Transform to Mahalanobis if requested
        Eigen::MatrixXd X_use = X;
        if (mahalanobis) {
            Eigen::MatrixXd cov = (X.transpose() * X) / n;
            Eigen::LLT<Eigen::MatrixXd> llt(cov);
            X_use = llt.solve(X.transpose()).transpose();
        }
        
        // Separate indices
        std::vector<int> treated_idx, control_idx;
        for (int i = 0; i < n; ++i) {
            if (D(i) > 0.5) treated_idx.push_back(i);
            else control_idx.push_back(i);
        }
        
        // Build HNSW index on controls
        Eigen::MatrixXd X_control(control_idx.size(), X_use.cols());
        for (size_t i = 0; i < control_idx.size(); ++i) {
            X_control.row(i) = X_use.row(control_idx[i]);
        }
        
        search::HNSW index(hnsw_config);
        index.build(X_control);
        
        // Query for each treated
        std::vector<std::vector<int>> matches(treated_idx.size());
        std::unordered_set<int> used;
        
        for (size_t t = 0; t < treated_idx.size(); ++t) {
            Eigen::VectorXd query = X_use.row(treated_idx[t]).transpose();
            
            // Query HNSW
            int k_query = with_replacement ? n_neighbors : 
                          std::min(n_neighbors * 2, (int)control_idx.size());
            auto result = index.query(query, k_query);
            
            int matched = 0;
            for (size_t j = 0; j < result.indices.size() && matched < n_neighbors; ++j) {
                int ctrl_local = result.indices[j];
                int ctrl_global = control_idx[ctrl_local];
                
                if (!with_replacement && used.count(ctrl_global)) continue;
                
                matches[t].push_back(ctrl_global);
                used.insert(ctrl_global);
                matched++;
            }
        }
        
        return matches;
    }
};

#endif // STATELIX_USE_HNSW

} // namespace statelix

#endif // STATELIX_PSM_H
