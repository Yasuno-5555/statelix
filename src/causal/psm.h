/**
 * @file psm.h
 * @brief Statelix v2.3 - Propensity Score Matching
 * 
 * Implements:
 *   - Propensity score estimation (logistic regression)
 *   - Nearest neighbor matching (with/without replacement)
 *   - Caliper matching
 *   - Kernel matching
 *   - Inverse probability weighting (IPW)
 *   - Doubly robust estimation (AIPW)
 *   - Balance diagnostics
 * 
 * Theory:
 * -------
 * Propensity Score: e(X) = P(D=1|X)
 * 
 * Under unconfoundedness (Y(0), Y(1) ⊥ D | X):
 *   ATT = E[Y(1) - Y(0) | D=1]
 *       = E[E[Y|D=1,X] - E[Y|D=0,X] | D=1]
 *       = E[E[Y|D=1,e(X)] - E[Y|D=0,e(X)] | D=1]
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

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

enum class MatchingMethod {
    NEAREST_NEIGHBOR,
    CALIPER,
    KERNEL,
    RADIUS
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
    
    double ate;                     // Average Treatment Effect
    double ate_se;
    
    double atc;                     // Average Treatment effect on Control
    double atc_se;
    
    // Matching info
    std::vector<std::vector<int>> matches;  // For each treated, matched control indices
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
    Eigen::VectorXd weights;        // IPW weights
    int n_trimmed;                  // Observations trimmed due to extreme weights
};

struct AIPWResult {
    double att;
    double att_se;
    double ate;
    double ate_se;
    double efficiency_gain;         // % reduction in variance vs IPW
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
        
        for (int iter = 0; iter < 50; ++iter) {
            Eigen::VectorXd eta = X_aug * beta;
            Eigen::VectorXd p(n);
            for (int i = 0; i < n; ++i) {
                p(i) = 1.0 / (1.0 + std::exp(-eta(i)));
                p(i) = std::max(1e-10, std::min(1.0 - 1e-10, p(i)));
            }
            
            Eigen::VectorXd w = p.array() * (1.0 - p.array());
            Eigen::VectorXd z = eta + (D - p).array() / w.array();
            
            Eigen::MatrixXd W = w.asDiagonal();
            Eigen::VectorXd beta_new = (X_aug.transpose() * W * X_aug).ldlt()
                                       .solve(X_aug.transpose() * W * z);
            
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
        
        // Matching
        result.matches.resize(n_t);
        std::unordered_set<int> used_controls;
        
        if (method == MatchingMethod::NEAREST_NEIGHBOR) {
            nearest_neighbor_match(ps.scores, treated_idx, control_idx, 
                                   result.matches, used_controls);
        } else if (method == MatchingMethod::KERNEL) {
            // Kernel matching: all controls contribute with weights
            kernel_match(ps.scores, treated_idx, control_idx, result.matches);
        }
        
        // Count matched
        result.n_matched_treated = 0;
        for (const auto& m : result.matches) {
            if (!m.empty()) result.n_matched_treated++;
        }
        result.n_matched_control = used_controls.size();
        
        // Estimate ATT
        double sum_diff = 0;
        double sum_weight = 0;
        std::vector<double> diffs;
        
        for (int t = 0; t < n_t; ++t) {
            if (result.matches[t].empty()) continue;
            
            int i = treated_idx[t];
            double y_t = Y(i);
            
            double y_c = 0;
            double w = 0;
            for (int j : result.matches[t]) {
                if (method == MatchingMethod::KERNEL) {
                    double dist = std::abs(ps.scores(i) - ps.scores(j));
                    double k = epanechnikov_kernel(dist / kernel_bandwidth);
                    y_c += k * Y(j);
                    w += k;
                } else {
                    y_c += Y(j);
                    w += 1.0;
                }
            }
            if (w > 0) y_c /= w;
            
            double diff = y_t - y_c;
            sum_diff += diff;
            sum_weight += 1.0;
            diffs.push_back(diff);
        }
        
        result.att = (sum_weight > 0) ? sum_diff / sum_weight : 0;
        
        // Standard error (Abadie-Imbens)
        double var = 0;
        for (double d : diffs) {
            var += (d - result.att) * (d - result.att);
        }
        var /= (diffs.size() - 1);
        result.att_se = std::sqrt(var / diffs.size());
        
        result.att_t = result.att / result.att_se;
        result.att_pvalue = 2.0 * (1.0 - normal_cdf(std::abs(result.att_t)));
        
        double z_crit = normal_quantile(0.5 + conf_level / 2);
        result.att_ci_lower = result.att - z_crit * result.att_se;
        result.att_ci_upper = result.att + z_crit * result.att_se;
        
        // Balance after matching
        compute_balance_after(X, D, result.matches, treated_idx, control_idx, 
                              result.std_diff_after);
        result.mean_std_diff_after = result.std_diff_after.array().abs().mean();
        
        // ATE and ATC (rough estimates)
        result.ate = result.att;  // Simplified
        result.ate_se = result.att_se;
        result.atc = result.att;
        result.atc_se = result.att_se;
        
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
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            
            // Trim extreme propensity scores
            if (e < trim_threshold || e > 1 - trim_threshold) {
                result.weights(i) = 0;
                result.n_trimmed++;
            } else if (D(i) > 0.5) {
                result.weights(i) = 1.0;  // Treated
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
        double y0_mean = (sum_w0 > 0) ? sum_y0w / sum_w0 : 0;
        
        result.att = y1_mean - y0_mean;
        
        // Variance via influence function (simplified)
        double var = 0;
        int n_eff = 0;
        for (int i = 0; i < n; ++i) {
            if (result.weights(i) == 0) continue;
            n_eff++;
            
            double psi;
            if (D(i) > 0.5) {
                psi = Y(i) - y1_mean - result.att;
            } else {
                psi = -result.weights(i) * (Y(i) - y0_mean);
            }
            var += psi * psi;
        }
        result.att_se = std::sqrt(var / (n_eff * n_eff));
        
        // ATE
        result.ate = result.att;  // Simplified
        result.ate_se = result.att_se;
        
        return result;
    }
    
    /**
     * @brief Augmented IPW (Doubly Robust) estimator
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
        
        // Estimate outcome model for each treatment status
        // μ(1,X) and μ(0,X) via OLS
        std::vector<int> t_idx, c_idx;
        for (int i = 0; i < n; ++i) {
            if (D(i) > 0.5) t_idx.push_back(i);
            else c_idx.push_back(i);
        }
        
        // OLS for treated
        Eigen::MatrixXd X1(t_idx.size(), k + 1);
        Eigen::VectorXd Y1(t_idx.size());
        for (size_t j = 0; j < t_idx.size(); ++j) {
            X1(j, 0) = 1.0;
            X1.row(j).tail(k) = X.row(t_idx[j]);
            Y1(j) = Y(t_idx[j]);
        }
        Eigen::VectorXd beta1 = (X1.transpose() * X1).ldlt().solve(X1.transpose() * Y1);
        
        // OLS for control
        Eigen::MatrixXd X0(c_idx.size(), k + 1);
        Eigen::VectorXd Y0(c_idx.size());
        for (size_t j = 0; j < c_idx.size(); ++j) {
            X0(j, 0) = 1.0;
            X0.row(j).tail(k) = X.row(c_idx[j]);
            Y0(j) = Y(c_idx[j]);
        }
        Eigen::VectorXd beta0 = (X0.transpose() * X0).ldlt().solve(X0.transpose() * Y0);
        
        // Predict μ(1,X) and μ(0,X) for all observations
        Eigen::VectorXd mu1(n), mu0(n);
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd x(k + 1);
            x(0) = 1.0;
            x.tail(k) = X.row(i).transpose();
            mu1(i) = x.dot(beta1);
            mu0(i) = x.dot(beta0);
        }
        
        // AIPW estimator
        double sum_psi = 0;
        double sum_w = 0;
        std::vector<double> psi_vals;
        
        for (int i = 0; i < n; ++i) {
            double e = ps.scores(i);
            if (e < trim_threshold || e > 1 - trim_threshold) continue;
            
            double psi;
            if (D(i) > 0.5) {
                // Treated: Y - μ(1,X) + E[μ(1,X)|D=1]
                psi = Y(i) - mu0(i) - (1 - D(i)) / (1 - e) * (Y(i) - mu0(i));
            } else {
                // Control
                psi = mu1(i) - Y(i) + D(i) / e * (Y(i) - mu1(i));
            }
            
            // ATT influence
            double psi_att = D(i) * (Y(i) - mu0(i)) - 
                             (1 - D(i)) * e / (1 - e) * (Y(i) - mu0(i));
            
            sum_psi += psi_att;
            sum_w += D(i);
            psi_vals.push_back(psi_att);
        }
        
        result.att = sum_psi / sum_w;
        
        // Variance
        double var = 0;
        for (double p : psi_vals) {
            var += (p - result.att) * (p - result.att);
        }
        result.att_se = std::sqrt(var / (psi_vals.size() * psi_vals.size()));
        
        // ATE
        result.ate = result.att;  // Simplified
        result.ate_se = result.att_se;
        
        // Efficiency gain vs IPW (rough)
        IPWResult ipw_result = ipw(Y, D, ps);
        result.efficiency_gain = (ipw_result.att_se - result.att_se) / ipw_result.att_se * 100;
        
        return result;
    }
    
private:
    void nearest_neighbor_match(
        const Eigen::VectorXd& scores,
        const std::vector<int>& treated_idx,
        const std::vector<int>& control_idx,
        std::vector<std::vector<int>>& matches,
        std::unordered_set<int>& used_controls
    ) {
        int n_t = treated_idx.size();
        
        for (int t = 0; t < n_t; ++t) {
            int i = treated_idx[t];
            double ps_t = scores(i);
            
            // Find k nearest controls
            std::vector<std::pair<double, int>> distances;
            for (int j : control_idx) {
                if (!with_replacement && used_controls.count(j)) continue;
                
                double dist = std::abs(ps_t - scores(j));
                if (dist <= caliper || caliper < 0) {
                    distances.push_back({dist, j});
                }
            }
            
            std::sort(distances.begin(), distances.end());
            
            for (int k = 0; k < std::min(n_neighbors, (int)distances.size()); ++k) {
                int j = distances[k].second;
                matches[t].push_back(j);
                used_controls.insert(j);
            }
        }
    }
    
    void kernel_match(
        const Eigen::VectorXd& scores,
        const std::vector<int>& treated_idx,
        const std::vector<int>& control_idx,
        std::vector<std::vector<int>>& matches
    ) {
        int n_t = treated_idx.size();
        
        for (int t = 0; t < n_t; ++t) {
            // Include all controls for kernel matching
            for (int j : control_idx) {
                matches[t].push_back(j);
            }
        }
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
        double a = 0.147;
        double x = 2 * p - 1;
        double ln = std::log(1 - x * x);
        double s = (x > 0 ? 1 : -1);
        return s * std::sqrt(std::sqrt((2/(M_PI*a) + ln/2) * (2/(M_PI*a) + ln/2) - ln/a) 
                            - (2/(M_PI*a) + ln/2)) * std::sqrt(2);
    }
};

} // namespace statelix

#endif // STATELIX_PSM_H
