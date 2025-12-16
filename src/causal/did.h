#pragma once

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "../linear_model/solver.h"
#include "../stats/math_utils.h"

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Basic DID estimation result
 */
struct DIDResult {
    // ATT estimate
    double att;                     // Treatment effect (Post × Treat coefficient)
    double att_std_error;
    double t_stat;
    double pvalue;
    double conf_lower;
    double conf_upper;
    
    // Full model coefficients: [intercept, post, treat, post×treat]
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    
    // Parallel trends test (pre-treatment)
    double pre_trend_diff;          // Pre-treatment difference-in-slopes
    double pre_trend_pvalue;        // H0: parallel trends
    bool parallel_trends_valid;     // true if p > 0.05
    
    // Model diagnostics
    double r_squared;
    int n_obs;
    int n_treated;
    int n_control;
    int n_pre_periods;
    int n_post_periods;
    
    // Fitted values
    Eigen::VectorXd fitted_values;
    Eigen::VectorXd residuals;
};

/**
 * @brief TWFE (Two-Way Fixed Effects) estimation result
 */
struct TWFEResult {
    // Main treatment effect
    double delta;                   // Treatment effect estimate
    double delta_std_error;
    double t_stat;
    double pvalue;
    double conf_lower;
    double conf_upper;
    
    // Fixed effects (demeaned coefficients)
    Eigen::VectorXd unit_fe;        // Unit fixed effects
    Eigen::VectorXd time_fe;        // Time fixed effects
    
    // Control variable coefficients (if any)
    Eigen::VectorXd controls_coef;
    Eigen::VectorXd controls_se;
    
    // Diagnostics for staggered adoption
    bool has_staggered_adoption;    // Multiple treatment timing
    int n_treatment_cohorts;        // Number of distinct adoption times
    double bacon_weight_negative;   // Fraction of negative weights (Goodman-Bacon)
    bool twfe_potentially_biased;   // Warning flag
    
    // Model fit
    double r_squared_within;
    double r_squared_overall;
    int n_obs;
    int n_units;
    int n_periods;
    
    // Cluster-robust standard errors
    bool clustered_se;
    int n_clusters;
};

/**
 * @brief Event study result (dynamic treatment effects)
 */
struct EventStudyResult {
    // Relative time coefficients (τ = -K, ..., -1, 0, 1, ..., L)
    Eigen::VectorXd coefficients;   // Treatment effects by period
    Eigen::VectorXd std_errors;
    Eigen::VectorXd conf_lower;
    Eigen::VectorXd conf_upper;
    std::vector<int> rel_time;      // Relative time labels
    int reference_period;           // Normalized to 0 (typically -1)
    
    // Pre-trend test
    double pre_trend_f_stat;
    double pre_trend_pvalue;
    bool parallel_trends_valid;
    
    // Summary statistics
    int n_obs;
    int n_units;
    int n_periods;
};

// =============================================================================
// Basic Difference-in-Differences
// =============================================================================

/**
 * @brief Basic 2x2 Difference-in-Differences estimator
 * 
 * For simple two-group, two-period designs.
 */
class DifferenceInDifferences {
public:
    double conf_level = 0.95;
    bool robust_se = false;
    
    /**
     * @brief Fit basic DID model
     * 
     * @param Y Outcome variable (n,)
     * @param treated Treatment group indicator (0/1)
     * @param post Post-treatment period indicator (0/1)
     * @return DIDResult with ATT estimate and diagnostics
     */
    DIDResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::VectorXi& treated,
        const Eigen::VectorXi& post
    ) {
        int n = Y.size();
        if (treated.size() != n || post.size() != n) {
            throw std::invalid_argument("Y, treated, and post must have same length");
        }
        
        DIDResult result;
        result.n_obs = n;
        
        // Count groups
        result.n_treated = treated.sum();
        result.n_control = n - result.n_treated;
        result.n_post_periods = post.sum();
        result.n_pre_periods = n - result.n_post_periods;
        
        // Build design matrix: [1, post, treat, post×treat]
        Eigen::MatrixXd X(n, 4);
        X.col(0).setOnes();                              // Intercept
        X.col(1) = post.cast<double>();                  // Post
        X.col(2) = treated.cast<double>();               // Treat
        X.col(3) = (post.array() * treated.array()).cast<double>();  // Interaction
        
        // OLS estimation with WeightedSolver
        Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
        WeightedDesignMatrix wdm(X, w);
        WeightedSolver solver(SolverStrategy::AUTO);
        
        try {
            result.coef = solver.solve(wdm, Y);
        } catch (const std::exception& e) {
            throw std::runtime_error("DID estimation failed: " + std::string(e.what()));
        }
        
        // ATT is the interaction coefficient
        result.att = result.coef(3);
        
        // Residuals and standard errors
        result.fitted_values = X * result.coef;
        result.residuals = Y - result.fitted_values;
        
        double sse = result.residuals.squaredNorm();
        int df = n - 4;
        
        // Solver variance_covariance() returns (X'X)^-1 (unscaled)
        // So vcov_solver IS XtX_inv
        Eigen::MatrixXd vcov_solver = solver.variance_covariance();
        Eigen::MatrixXd XtX_inv = vcov_solver;
        
        if (robust_se) {
            result.std_errors.resize(4);
            // Robust Sandwich
            Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(4, 4);
            for (int i=0; i<n; ++i) {
                Eigen::VectorXd xi = X.row(i).transpose();
                meat += result.residuals(i) * result.residuals(i) * xi * xi.transpose();
            }
            Eigen::MatrixXd vcov_robust = XtX_inv * meat * XtX_inv;
            for (int j=0; j<4; ++j) result.std_errors(j) = std::sqrt(vcov_robust(j,j));
        } else {
            result.std_errors.resize(4);
            for(int j=0; j<4; ++j) {
                result.std_errors(j) = std::sqrt(vcov_solver(j,j)); 
            }
        }
        
        result.att_std_error = result.std_errors(3);
        result.t_stat = result.att / result.att_std_error;
        result.pvalue = 2.0 * (1.0 - statelix::stats::t_cdf(std::abs(result.t_stat), df));
        
        double t_crit = statelix::stats::t_quantile(1.0 - (1.0 - conf_level) / 2.0, df);
        result.conf_lower = result.att - t_crit * result.att_std_error;
        result.conf_upper = result.att + t_crit * result.att_std_error;
        
        // R-squared
        double sst = (Y.array() - Y.mean()).square().sum();
        result.r_squared = 1.0 - sse / sst;
        
        result.parallel_trends_valid = true; 
        result.pre_trend_diff = 0.0;
        result.pre_trend_pvalue = 1.0;
        
        return result;
    }
    
    /**
     * @brief Fit DID with pre-period parallel trends test
     * 
     * @param Y Outcome variable
     * @param treated Treatment indicator
     * @param time Time period (integers)
     * @param treatment_time When treatment begins for treated units
     */
    DIDResult fit_with_pretest(
        const Eigen::VectorXd& Y,
        const Eigen::VectorXi& treated,
        const Eigen::VectorXi& time,
        int treatment_time
    ) {
        int n = Y.size();
        
        // Create post indicator
        Eigen::VectorXi post(n);
        for (int i = 0; i < n; ++i) {
            post(i) = (time(i) >= treatment_time) ? 1 : 0;
        }
        
        // Basic DID
        DIDResult result = fit(Y, treated, post);
        
        // Pre-trend test: regress Y on time × treated in pre-period
        std::vector<int> pre_idx;
        for (int i = 0; i < n; ++i) {
            if (post(i) == 0) pre_idx.push_back(i);
        }
        
        if (pre_idx.size() > 3) {
            int n_pre = pre_idx.size();
            Eigen::MatrixXd X_pre(n_pre, 3);
            Eigen::VectorXd Y_pre(n_pre);
            
            for (int i = 0; i < n_pre; ++i) {
                int idx = pre_idx[i];
                Y_pre(i) = Y(idx);
                X_pre(i, 0) = 1.0;
                X_pre(i, 1) = time(idx);
                X_pre(i, 2) = time(idx) * treated(idx);  // time × treat
            }
            
            // WeightedSolver for pre-trend test
            Eigen::VectorXd w_pre = Eigen::VectorXd::Ones(n_pre);
            WeightedDesignMatrix wdm_pre(X_pre, w_pre);
            WeightedSolver solver_pre(SolverStrategy::AUTO);
            Eigen::VectorXd beta = solver_pre.solve(wdm_pre, Y_pre);
            
            Eigen::VectorXd resid = Y_pre - X_pre * beta;
            double sigma2_pre = resid.squaredNorm() / (n_pre - 3);
            
            // Solver returns Unscaled (X'X)^-1
            Eigen::MatrixXd XtX_inv_pre = solver_pre.variance_covariance();
            
            result.pre_trend_diff = beta(2);  // time × treat coefficient
            double se = std::sqrt(sigma2_pre * XtX_inv_pre(2, 2));
            double t = result.pre_trend_diff / se;
            result.pre_trend_pvalue = 2.0 * (1.0 - statelix::stats::t_cdf(std::abs(t), n_pre - 3));
            result.parallel_trends_valid = (result.pre_trend_pvalue > 0.05);
        }
        
        return result;
    }

private:
    Eigen::MatrixXd compute_robust_vcov(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& residuals,
        const Eigen::MatrixXd& XtX_inv
    ) {
        int n = residuals.size();
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(X.cols(), X.cols());
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd xi = X.row(i).transpose();
            meat += residuals(i) * residuals(i) * xi * xi.transpose();
        }
        return XtX_inv * meat * XtX_inv;
    }
    // removed local static math functions
};

// =============================================================================
// Two-Way Fixed Effects
// =============================================================================

/**
 * @brief Two-Way Fixed Effects estimator
 * 
 * For panel data with unit and time fixed effects.
 * Uses within-transformation (demeaning) for computational efficiency.
 * 
 * WARNING: With staggered treatment timing, TWFE can produce biased
 * estimates. Check diagnostics and consider alternative estimators.
 */
class TwoWayFixedEffects {
public:
    double conf_level = 0.95;
    bool cluster_se = true;         // Cluster at unit level by default
    
    /**
     * @brief Fit TWFE model
     * 
     * @param Y Outcome (n,)
     * @param D Treatment indicator (n,) - 0/1
     * @param unit_id Unit identifiers (n,)
     * @param time_id Time period identifiers (n,)
     * @param X_controls Optional control variables (n, k)
     */
    TWFEResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::VectorXi& D,
        const Eigen::VectorXi& unit_id,
        const Eigen::VectorXi& time_id,
        const Eigen::MatrixXd& X_controls = Eigen::MatrixXd()
    ) {
        int n = Y.size();
        TWFEResult result;
        result.n_obs = n;
        
        // Get unique units and times
        std::unordered_set<int> units_set(unit_id.data(), unit_id.data() + n);
        std::unordered_set<int> times_set(time_id.data(), time_id.data() + n);
        std::vector<int> units(units_set.begin(), units_set.end());
        std::vector<int> times(times_set.begin(), times_set.end());
        std::sort(units.begin(), units.end());
        std::sort(times.begin(), times.end());
        
        result.n_units = units.size();
        result.n_periods = times.size();
        
        // Create unit and time index maps
        std::unordered_map<int, int> unit_map, time_map;
        for (int i = 0; i < (int)units.size(); ++i) unit_map[units[i]] = i;
        for (int i = 0; i < (int)times.size(); ++i) time_map[times[i]] = i;
        
        // Check for staggered adoption
        check_staggered_adoption(D, unit_id, time_id, unit_map, result);
        
        // Within transformation (demean by unit and time)
        Eigen::VectorXd Y_demean = within_transform(Y, unit_id, time_id, unit_map, time_map);
        Eigen::VectorXd D_demean = within_transform(D.cast<double>(), unit_id, time_id, unit_map, time_map);
        
        // Demean controls if present
        int k = X_controls.cols();
        Eigen::MatrixXd X_demean(n, k);
        for (int j = 0; j < k; ++j) {
            X_demean.col(j) = within_transform(X_controls.col(j), unit_id, time_id, unit_map, time_map);
        }
        
        // Stack regressors: [D, controls]
        int p = 1 + k;
        Eigen::MatrixXd X(n, p);
        X.col(0) = D_demean;
        if (k > 0) X.rightCols(k) = X_demean;
        
        // OLS on demeaned data using WeightedSolver
        Eigen::VectorXd w_vec = Eigen::VectorXd::Ones(n);
        WeightedDesignMatrix wdm(X, w_vec);
        WeightedSolver solver(SolverStrategy::AUTO);
        
        Eigen::VectorXd beta;
        try {
            beta = solver.solve(wdm, Y_demean);
        } catch (const std::exception& e) {
             throw std::runtime_error("TWFE estimation failed: " + std::string(e.what()));
        }

        // Solver variance_covariance() returns (X'X)^-1
        Eigen::MatrixXd vcov_solver = solver.variance_covariance();
        Eigen::MatrixXd XtX_inv = vcov_solver;
        
        result.delta = beta(0);
        if (k > 0) result.controls_coef = beta.segment(1, k);
        
        // Residuals
        Eigen::VectorXd resid = Y_demean - X * beta;
        
        // Standard errors
        int df = n - result.n_units - result.n_periods + 1 - k;
        Eigen::MatrixXd vcov;
        
        if (cluster_se) {
            vcov = compute_cluster_vcov(X, resid, unit_id, unit_map, XtX_inv);
            result.clustered_se = true;
            result.n_clusters = result.n_units;
        } else {
            double sigma2 = resid.squaredNorm() / df;
            vcov = sigma2 * XtX_inv;
            result.clustered_se = false;
            result.n_clusters = 0;
        }
        
        result.delta_std_error = std::sqrt(vcov(0, 0));
        result.t_stat = result.delta / result.delta_std_error;
        
        int df_t = cluster_se ? result.n_units - 1 : df;
        result.pvalue = 2.0 * (1.0 - statelix::stats::t_cdf(std::abs(result.t_stat), df_t));
        
        double t_crit = statelix::stats::t_quantile(1.0 - (1.0 - conf_level) / 2.0, df_t);
        result.conf_lower = result.delta - t_crit * result.delta_std_error;
        result.conf_upper = result.delta + t_crit * result.delta_std_error;
        
        // Control variable SEs
        if (k > 0) {
            result.controls_se.resize(k);
            for (int j = 0; j < k; ++j) {
                result.controls_se(j) = std::sqrt(vcov(1 + j, 1 + j));
            }
        }
        
        // R-squared
        double sst = Y_demean.squaredNorm();
        double sse = resid.squaredNorm();
        result.r_squared_within = 1.0 - sse / sst;
        
        double sst_overall = (Y.array() - Y.mean()).square().sum();
        result.r_squared_overall = 1.0 - sse / sst_overall;
        
        // Compute fixed effects
        compute_fixed_effects(Y, D, X_controls, beta, unit_id, time_id, 
                             unit_map, time_map, result);
        
        return result;
    }

private:
    // Within transformation: x_it - x̄_i - x̄_t + x̄
    Eigen::VectorXd within_transform(
        const Eigen::VectorXd& x,
        const Eigen::VectorXi& unit_id,
        const Eigen::VectorXi& time_id,
        const std::unordered_map<int, int>& unit_map,
        const std::unordered_map<int, int>& time_map
    ) {
        int n = x.size();
        int n_units = unit_map.size();
        int n_times = time_map.size();
        
        // Compute means
        Eigen::VectorXd unit_sum = Eigen::VectorXd::Zero(n_units);
        Eigen::VectorXi unit_count = Eigen::VectorXi::Zero(n_units);
        Eigen::VectorXd time_sum = Eigen::VectorXd::Zero(n_times);
        Eigen::VectorXi time_count = Eigen::VectorXi::Zero(n_times);
        double total = 0.0;
        
        for (int i = 0; i < n; ++i) {
            int u = unit_map.at(unit_id(i));
            int t = time_map.at(time_id(i));
            unit_sum(u) += x(i);
            unit_count(u)++;
            time_sum(t) += x(i);
            time_count(t)++;
            total += x(i);
        }
        
        Eigen::VectorXd unit_mean = unit_sum.array() / unit_count.cast<double>().array();
        Eigen::VectorXd time_mean = time_sum.array() / time_count.cast<double>().array();
        double grand_mean = total / n;
        
        // Demean
        Eigen::VectorXd result(n);
        for (int i = 0; i < n; ++i) {
            int u = unit_map.at(unit_id(i));
            int t = time_map.at(time_id(i));
            result(i) = x(i) - unit_mean(u) - time_mean(t) + grand_mean;
        }
        
        return result;
    }
    
    // Check for staggered treatment adoption
    void check_staggered_adoption(
        const Eigen::VectorXi& D,
        const Eigen::VectorXi& unit_id,
        const Eigen::VectorXi& time_id,
        const std::unordered_map<int, int>& unit_map,
        TWFEResult& result
    ) {
        std::unordered_map<int, int> treatment_start;  // unit -> first treatment time
        
        for (int i = 0; i < D.size(); ++i) {
            if (D(i) == 1) {
                int u = unit_id(i);
                int t = time_id(i);
                if (treatment_start.find(u) == treatment_start.end() || 
                    t < treatment_start[u]) {
                    treatment_start[u] = t;
                }
            }
        }
        
        // Count distinct treatment cohorts
        std::unordered_set<int> cohorts;
        for (const auto& [unit, start_time] : treatment_start) {
            cohorts.insert(start_time);
        }
        
        result.n_treatment_cohorts = cohorts.size();
        result.has_staggered_adoption = (cohorts.size() > 1);
        result.twfe_potentially_biased = result.has_staggered_adoption;
        // Requires full Goodman-Bacon decomposition to calculate weights accurately.
        // This is a placeholder for future implementation.
        result.bacon_weight_negative = 0.0; 
    }
    
    // Cluster-robust variance (Liang-Zeger)
    Eigen::MatrixXd compute_cluster_vcov(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& resid,
        const Eigen::VectorXi& cluster_id,
        const std::unordered_map<int, int>& cluster_map,
        const Eigen::MatrixXd& XtX_inv
    ) {
        int n = resid.size();
        int p = X.cols();
        int G = cluster_map.size();
        
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(p, p);
        
        for (const auto& [cid, cidx] : cluster_map) {
            Eigen::VectorXd score = Eigen::VectorXd::Zero(p);
            for (int i = 0; i < n; ++i) {
                if (cluster_id(i) == cid) {
                    score += resid(i) * X.row(i).transpose();
                }
            }
            meat += score * score.transpose();
        }
        
        // Small sample correction: G/(G-1) * (n-1)/(n-p)
        double correction = (double)G / (G - 1) * (n - 1) / (n - p);
        
        return correction * XtX_inv * meat * XtX_inv;
    }
    
    // Compute unit and time fixed effects
    void compute_fixed_effects(
        const Eigen::VectorXd& Y,
        const Eigen::VectorXi& D,
        const Eigen::MatrixXd& X_controls,
        const Eigen::VectorXd& beta,
        const Eigen::VectorXi& unit_id,
        const Eigen::VectorXi& time_id,
        const std::unordered_map<int, int>& unit_map,
        const std::unordered_map<int, int>& time_map,
        TWFEResult& result
    ) {
        int n = Y.size();
        int n_units = unit_map.size();
        int n_times = time_map.size();
        
        // Residuals after removing treatment and controls
        Eigen::VectorXd Y_adj = Y;
        Y_adj -= beta(0) * D.cast<double>();
        if (X_controls.cols() > 0) {
            Y_adj -= X_controls * beta.segment(1, X_controls.cols());
        }
        
        // Compute means as fixed effects
        result.unit_fe.resize(n_units);
        result.time_fe.resize(n_times);
        result.unit_fe.setZero();
        result.time_fe.setZero();
        
        Eigen::VectorXi unit_count = Eigen::VectorXi::Zero(n_units);
        Eigen::VectorXi time_count = Eigen::VectorXi::Zero(n_times);
        
        for (int i = 0; i < n; ++i) {
            int u = unit_map.at(unit_id(i));
            int t = time_map.at(time_id(i));
            result.unit_fe(u) += Y_adj(i);
            unit_count(u)++;
            result.time_fe(t) += Y_adj(i);
            time_count(t)++;
        }
        
        result.unit_fe = result.unit_fe.array() / unit_count.cast<double>().array();
        result.time_fe = result.time_fe.array() / time_count.cast<double>().array();
    }
};

} // namespace statelix
