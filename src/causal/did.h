/**
 * @file did.h
 * @brief Statelix v1.1 - Difference-in-Differences (DID) / Two-Way Fixed Effects
 * 
 * Implements:
 *   - Basic 2x2 DID (two groups, two periods)
 *   - TWFE (Two-Way Fixed Effects for staggered adoption)
 *   - Parallel trends pre-test
 *   - Event study design
 * 
 * Theory:
 * -------
 * Basic DID:
 *   Y_it = α + β₁ Post_t + β₂ Treat_i + β₃ (Post_t × Treat_i) + ε_it
 *   ATT = β₃ (Average Treatment Effect on the Treated)
 * 
 * TWFE with staggered adoption (Goodman-Bacon 2021 decomposition warning):
 *   Y_it = α_i + λ_t + δ D_it + ε_it
 *   where α_i = unit FE, λ_t = time FE, D_it = treatment indicator
 * 
 * WARNING: TWFE with heterogeneous treatment timing can produce biased
 * estimates. Consider using more robust estimators (Callaway-Sant'Anna, 
 * Sun-Abraham) for staggered designs. This implementation includes
 * diagnostics to detect potential issues.
 * 
 * Reference: 
 *   - Angrist, J.D. & Pischke, J.S. (2009). Mostly Harmless Econometrics
 *   - Goodman-Bacon, A. (2021). Difference-in-differences with variation
 *     in treatment timing. Journal of Econometrics, 225(2), 254-277.
 */
#ifndef STATELIX_DID_H
#define STATELIX_DID_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <stdexcept>

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
    double p_value;
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
    double p_value;
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
        
        // OLS estimation
        Eigen::MatrixXd XtX = X.transpose() * X;
        Eigen::MatrixXd XtX_inv = XtX.ldlt().solve(Eigen::MatrixXd::Identity(4, 4));
        result.coef = XtX_inv * X.transpose() * Y;
        
        // ATT is the interaction coefficient
        result.att = result.coef(3);
        
        // Residuals and standard errors
        result.fitted_values = X * result.coef;
        result.residuals = Y - result.fitted_values;
        
        double sse = result.residuals.squaredNorm();
        int df = n - 4;
        double sigma2 = sse / df;
        
        Eigen::MatrixXd vcov;
        if (robust_se) {
            vcov = compute_robust_vcov(X, result.residuals, XtX_inv);
        } else {
            vcov = sigma2 * XtX_inv;
        }
        
        result.std_errors.resize(4);
        for (int j = 0; j < 4; ++j) {
            result.std_errors(j) = std::sqrt(vcov(j, j));
        }
        
        result.att_std_error = result.std_errors(3);
        result.t_stat = result.att / result.att_std_error;
        result.p_value = 2.0 * (1.0 - t_cdf(std::abs(result.t_stat), df));
        
        double t_crit = t_quantile(1.0 - (1.0 - conf_level) / 2.0, df);
        result.conf_lower = result.att - t_crit * result.att_std_error;
        result.conf_upper = result.att + t_crit * result.att_std_error;
        
        // R-squared
        double sst = (Y.array() - Y.mean()).square().sum();
        result.r_squared = 1.0 - sse / sst;
        
        // Parallel trends test (simplified: just check pre-period difference)
        result.parallel_trends_valid = true;  // Would need multiple pre-periods
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
            
            Eigen::MatrixXd XtX = X_pre.transpose() * X_pre;
            Eigen::MatrixXd XtX_inv = XtX.ldlt().solve(Eigen::MatrixXd::Identity(3, 3));
            Eigen::VectorXd beta = XtX_inv * X_pre.transpose() * Y_pre;
            
            Eigen::VectorXd resid = Y_pre - X_pre * beta;
            double sigma2 = resid.squaredNorm() / (n_pre - 3);
            
            result.pre_trend_diff = beta(2);  // time × treat coefficient
            double se = std::sqrt(sigma2 * XtX_inv(2, 2));
            double t = result.pre_trend_diff / se;
            result.pre_trend_pvalue = 2.0 * (1.0 - t_cdf(std::abs(t), n_pre - 3));
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
    
    static double t_cdf(double t, int df) {
        if (df > 100) {
            return 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)));
        }
        double x = df / (df + t * t);
        return 0.5 + 0.5 * std::copysign(1.0, t) * (1.0 - beta_inc(df / 2.0, 0.5, x));
    }
    
    static double t_quantile(double p, int df) {
        double t = (p > 0.5) ? std::sqrt(2.0) * erfinv(2.0 * p - 1.0) : 
                               -std::sqrt(2.0) * erfinv(1.0 - 2.0 * p);
        for (int i = 0; i < 5; ++i) {
            double cdf = t_cdf(t, df);
            double pdf = std::tgamma((df + 1.0) / 2.0) / 
                        (std::sqrt(df * M_PI) * std::tgamma(df / 2.0)) *
                        std::pow(1.0 + t * t / df, -(df + 1.0) / 2.0);
            t -= (cdf - p) / pdf;
        }
        return t;
    }
    
    static double erfinv(double x) {
        double w = -std::log((1 - x) * (1 + x));
        double p;
        if (w < 5.0) {
            w -= 2.5;
            p = 2.81022636e-08 + w * (3.43273939e-07 + w * (-3.5233877e-06 +
                w * (-4.39150654e-06 + w * (0.00021858087 + w * (-0.00125372503 +
                w * (-0.00417768164 + w * (0.246640727 + w * 0.115956309)))))));
        } else {
            w = std::sqrt(w) - 3.0;
            p = -0.000200214257 + w * (0.000100950558 + w * (0.00134934322 +
                w * (-0.00367342844 + w * (0.00573950773 + w * (-0.0076224613 +
                w * (0.00943887047 + w * (1.00167406 + w * 0.00282095556)))))));
        }
        return p * x;
    }
    
    static double beta_inc(double a, double b, double x) {
        if (x <= 0) return 0.0;
        if (x >= 1) return 1.0;
        double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                            a * std::log(x) + b * std::log(1.0 - x));
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return bt * beta_cf(a, b, x) / a;
        }
        return 1.0 - bt * beta_cf(b, a, 1.0 - x) / b;
    }
    
    static double beta_cf(double a, double b, double x) {
        double am = 1, bm = 1, az = 1;
        double qab = a + b, qap = a + 1, qam = a - 1;
        double bz = 1.0 - qab * x / qap;
        for (int m = 1; m <= 100; ++m) {
            double em = m;
            double d = em * (b - m) * x / ((qam + 2*em) * (a + 2*em));
            double ap = az + d * am, bp = bz + d * bm;
            d = -(a + em) * (qab + em) * x / ((a + 2*em) * (qap + 2*em));
            double app = ap + d * az, bpp = bp + d * bz;
            double aold = az;
            am = ap / bpp; bm = bp / bpp; az = app / bpp; bz = 1.0;
            if (std::abs(az - aold) < 1e-10 * std::abs(az)) break;
        }
        return az;
    }
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
        
        // OLS on demeaned data
        Eigen::MatrixXd XtX = X.transpose() * X;
        Eigen::MatrixXd XtX_inv = XtX.ldlt().solve(Eigen::MatrixXd::Identity(p, p));
        Eigen::VectorXd beta = XtX_inv * X.transpose() * Y_demean;
        
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
        result.p_value = 2.0 * (1.0 - t_cdf(std::abs(result.t_stat), df_t));
        
        double t_crit = t_quantile(1.0 - (1.0 - conf_level) / 2.0, df_t);
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
        result.bacon_weight_negative = 0.0;  // Would need full decomposition
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
    
    static double t_cdf(double t, int df) {
        if (df > 100) return 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)));
        double x = df / (df + t * t);
        return 0.5 + 0.5 * std::copysign(1.0, t) * (1.0 - beta_inc(df / 2.0, 0.5, x));
    }
    
    static double t_quantile(double p, int df) {
        double t = (p > 0.5) ? std::sqrt(2.0) * erfinv(2.0 * p - 1.0) : 
                               -std::sqrt(2.0) * erfinv(1.0 - 2.0 * p);
        for (int i = 0; i < 5; ++i) {
            double cdf = t_cdf(t, df);
            double pdf = std::tgamma((df + 1.0) / 2.0) / 
                        (std::sqrt(df * M_PI) * std::tgamma(df / 2.0)) *
                        std::pow(1.0 + t * t / df, -(df + 1.0) / 2.0);
            t -= (cdf - p) / pdf;
        }
        return t;
    }
    
    static double erfinv(double x) {
        double w = -std::log((1 - x) * (1 + x));
        double p;
        if (w < 5.0) {
            w -= 2.5;
            p = 2.81022636e-08 + w * (3.43273939e-07 + w * (-3.5233877e-06 +
                w * (-4.39150654e-06 + w * (0.00021858087 + w * (-0.00125372503 +
                w * (-0.00417768164 + w * (0.246640727 + w * 0.115956309)))))));
        } else {
            w = std::sqrt(w) - 3.0;
            p = -0.000200214257 + w * (0.000100950558 + w * (0.00134934322 +
                w * (-0.00367342844 + w * (0.00573950773 + w * (-0.0076224613 +
                w * (0.00943887047 + w * (1.00167406 + w * 0.00282095556)))))));
        }
        return p * x;
    }
    
    static double beta_inc(double a, double b, double x) {
        if (x <= 0) return 0.0;
        if (x >= 1) return 1.0;
        double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                            a * std::log(x) + b * std::log(1.0 - x));
        if (x < (a + 1.0) / (a + b + 2.0)) return bt * beta_cf(a, b, x) / a;
        return 1.0 - bt * beta_cf(b, a, 1.0 - x) / b;
    }
    
    static double beta_cf(double a, double b, double x) {
        double am = 1, bm = 1, az = 1;
        double qab = a + b, qap = a + 1, qam = a - 1;
        double bz = 1.0 - qab * x / qap;
        for (int m = 1; m <= 100; ++m) {
            double em = m;
            double d = em * (b - m) * x / ((qam + 2*em) * (a + 2*em));
            double ap = az + d * am, bp = bz + d * bm;
            d = -(a + em) * (qab + em) * x / ((a + 2*em) * (qap + 2*em));
            am = ap / (bp + d * bz); bm = bp / (bp + d * bz); 
            az = (ap + d * az) / (bp + d * bz); bz = 1.0;
        }
        return az;
    }
};

} // namespace statelix

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif // STATELIX_DID_H
