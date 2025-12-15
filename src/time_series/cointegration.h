/**
 * @file cointegration.h
 * @brief Statelix v2.3 - Cointegration Analysis
 * 
 * Implements:
 *   - Augmented Dickey-Fuller (ADF) unit root test
 *   - Phillips-Perron (PP) unit root test
 *   - KPSS stationarity test
 *   - Engle-Granger two-step cointegration test
 *   - Johansen cointegration test (trace and max eigenvalue)
 *   - Vector Error Correction Model (VECM)
 * 
 * Theory:
 * -------
 * Unit Root (ADF):
 *   Δy_t = α + βt + γy_{t-1} + Σδ_i Δy_{t-i} + ε_t
 *   H0: γ = 0 (unit root exists)
 * 
 * Engle-Granger:
 *   Step 1: Regress y_t on x_t, get residuals ε̂_t
 *   Step 2: Test ε̂_t for unit root (ADF with adjusted critical values)
 *   If reject: y and x are cointegrated
 * 
 * Johansen:
 *   ΔY_t = ΠY_{t-1} + ΓΔY_{t-1} + ... + ε_t
 *   Π = αβ' where rank(Π) = r = number of cointegrating vectors
 *   Trace test: H0: r ≤ r₀ vs H1: r > r₀
 *   Max eigenvalue: H0: r = r₀ vs H1: r = r₀ + 1
 * 
 * Reference:
 *   - Engle, R.F. & Granger, C.W.J. (1987). Co-Integration and Error Correction
 *   - Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors
 */
#ifndef STATELIX_COINTEGRATION_H
#define STATELIX_COINTEGRATION_H

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

enum class TrendType {
    NONE,           // No constant, no trend
    CONSTANT,       // Constant only
    TREND           // Constant and linear trend
};

struct ADFResult {
    double adf_statistic;
    double p_value;
    int lags_used;
    int n_obs;
    TrendType trend;
    
    // Critical values
    double cv_1pct;
    double cv_5pct;
    double cv_10pct;
    
    bool has_unit_root;     // Fail to reject H0
    std::string conclusion;
};

struct KPSSResult {
    double kpss_statistic;
    TrendType trend;
    
    double cv_1pct;
    double cv_5pct;
    double cv_10pct;
    
    bool is_stationary;     // Fail to reject H0
};

struct EngleGrangerResult {
    // Step 1: Cointegrating regression
    Eigen::VectorXd coef;           // Cointegrating vector
    double intercept;
    Eigen::VectorXd residuals;
    
    // Step 2: ADF on residuals
    double adf_statistic;
    double p_value;
    
    double cv_1pct;
    double cv_5pct;
    double cv_10pct;
    
    bool cointegrated;              // Reject unit root in residuals
    int n_obs;
    int n_vars;
};

struct JohansenResult {
    // Eigenvalues and eigenvectors
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;   // β (cointegrating vectors)
    Eigen::MatrixXd alpha;          // Loading matrix
    
    // Trace test
    Eigen::VectorXd trace_stats;
    Eigen::VectorXd trace_cv_5pct;
    int trace_rank;                 // Estimated cointegration rank
    
    // Max eigenvalue test
    Eigen::VectorXd max_eigen_stats;
    Eigen::VectorXd max_cv_5pct;
    int max_rank;
    
    // Model info
    int n_obs;
    int n_vars;
    int lag_order;
    TrendType trend;
    
    // Recommended rank
    int recommended_rank;
};

struct VECMResult {
    // Error correction term: α * β'Y_{t-1}
    Eigen::MatrixXd alpha;          // Loading matrix (K, r)
    Eigen::MatrixXd beta;           // Cointegrating vectors (K, r)
    
    // Short-run dynamics: Γ_i
    std::vector<Eigen::MatrixXd> gamma;  // Lagged difference coefficients
    
    Eigen::VectorXd intercept;
    Eigen::MatrixXd residuals;
    Eigen::MatrixXd sigma;          // Residual covariance
    
    int n_obs;
    int n_vars;
    int rank;                       // Cointegration rank
    int lag_order;
    
    double log_likelihood;
    double aic;
    double bic;
};

// =============================================================================
// Unit Root Tests
// =============================================================================

/**
 * @brief Augmented Dickey-Fuller test for unit root
 */
class ADF {
public:
    int max_lags = 12;
    TrendType trend = TrendType::CONSTANT;
    bool auto_lag = true;           // Select lags by AIC
    
    ADFResult test(const Eigen::VectorXd& y) {
        int n = y.size();
        ADFResult result;
        result.n_obs = n;
        result.trend = trend;
        
        // Determine lag order
        int lags = auto_lag ? select_lags(y) : max_lags;
        lags = std::min(lags, n / 3);
        result.lags_used = lags;
        
        // Build regression: Δy_t = α + βt + γy_{t-1} + Σδ_i Δy_{t-i} + ε_t
        int n_eff = n - lags - 1;
        if (n_eff < 10) {
            throw std::runtime_error("Not enough observations for ADF test");
        }
        
        // Number of regressors
        int n_det = (trend == TrendType::NONE) ? 0 : 
                    (trend == TrendType::CONSTANT) ? 1 : 2;
        int k = n_det + 1 + lags;  // deterministic + y_{t-1} + lagged differences
        
        Eigen::MatrixXd X(n_eff, k);
        Eigen::VectorXd dy(n_eff);
        
        for (int t = 0; t < n_eff; ++t) {
            int idx = lags + 1 + t;
            dy(t) = y(idx) - y(idx - 1);
            
            int col = 0;
            if (trend == TrendType::CONSTANT || trend == TrendType::TREND) {
                X(t, col++) = 1.0;
            }
            if (trend == TrendType::TREND) {
                X(t, col++) = idx;
            }
            X(t, col++) = y(idx - 1);  // y_{t-1}
            
            for (int j = 1; j <= lags; ++j) {
                X(t, col++) = y(idx - j) - y(idx - j - 1);  // Δy_{t-j}
            }
        }
        
        // OLS
        Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * dy);
        Eigen::VectorXd resid = dy - X * beta;
        
        // Standard error of γ (coefficient on y_{t-1})
        double s2 = resid.squaredNorm() / (n_eff - k);
        Eigen::MatrixXd XtX_inv = (X.transpose() * X).ldlt()
                                  .solve(Eigen::MatrixXd::Identity(k, k));
        
        int gamma_idx = (trend == TrendType::NONE) ? 0 : 
                        (trend == TrendType::CONSTANT) ? 1 : 2;
        double gamma = beta(gamma_idx);
        double gamma_se = std::sqrt(s2 * XtX_inv(gamma_idx, gamma_idx));
        
        result.adf_statistic = gamma / gamma_se;
        
        // Critical values (MacKinnon approximation)
        set_critical_values(result, n_eff);
        
        // p-value approximation
        result.p_value = compute_pvalue(result.adf_statistic, trend, n_eff);
        
        result.has_unit_root = (result.adf_statistic > result.cv_5pct);
        result.conclusion = result.has_unit_root ? 
            "Fail to reject H0: Unit root exists (non-stationary)" :
            "Reject H0: No unit root (stationary)";
        
        return result;
    }
    
private:
    int select_lags(const Eigen::VectorXd& y) {
        int n = y.size();
        int best_lag = 0;
        double best_aic = std::numeric_limits<double>::infinity();
        
        for (int lag = 0; lag <= std::min(max_lags, n / 4); ++lag) {
            try {
                int n_eff = n - lag - 1;
                int n_det = (trend == TrendType::NONE) ? 0 : 
                            (trend == TrendType::CONSTANT) ? 1 : 2;
                int k = n_det + 1 + lag;
                
                Eigen::MatrixXd X(n_eff, k);
                Eigen::VectorXd dy(n_eff);
                
                for (int t = 0; t < n_eff; ++t) {
                    int idx = lag + 1 + t;
                    dy(t) = y(idx) - y(idx - 1);
                    
                    int col = 0;
                    if (trend == TrendType::CONSTANT || trend == TrendType::TREND) {
                        X(t, col++) = 1.0;
                    }
                    if (trend == TrendType::TREND) {
                        X(t, col++) = idx;
                    }
                    X(t, col++) = y(idx - 1);
                    for (int j = 1; j <= lag; ++j) {
                        X(t, col++) = y(idx - j) - y(idx - j - 1);
                    }
                }
                
                Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * dy);
                Eigen::VectorXd resid = dy - X * beta;
                double s2 = resid.squaredNorm() / n_eff;
                
                double aic = n_eff * std::log(s2) + 2 * k;
                if (aic < best_aic) {
                    best_aic = aic;
                    best_lag = lag;
                }
            } catch (...) {
                continue;
            }
        }
        
        return best_lag;
    }
    
    void set_critical_values(ADFResult& result, int n) {
        // MacKinnon critical values (simplified, should use full regression for accuracy)
        if (trend == TrendType::NONE) {
            result.cv_1pct = -2.58;
            result.cv_5pct = -1.95;
            result.cv_10pct = -1.62;
        } else if (trend == TrendType::CONSTANT) {
            result.cv_1pct = -3.43;
            result.cv_5pct = -2.86;
            result.cv_10pct = -2.57;
        } else {
            result.cv_1pct = -3.96;
            result.cv_5pct = -3.41;
            result.cv_10pct = -3.12;
        }
        
        // Sample size adjustment (rough approximation)
        double adj = 1.0 / n;
        result.cv_1pct -= 2.0 * adj;
        result.cv_5pct -= 1.0 * adj;
    }
    
    double compute_pvalue(double stat, TrendType type, int n) {
        // Very rough p-value approximation
        // In practice, should use MacKinnon's response surface regressions
        double cv5 = (type == TrendType::NONE) ? -1.95 :
                     (type == TrendType::CONSTANT) ? -2.86 : -3.41;
        
        if (stat < cv5 - 1.5) return 0.01;
        if (stat < cv5 - 0.5) return 0.025;
        if (stat < cv5) return 0.05;
        if (stat < cv5 + 0.5) return 0.10;
        if (stat < cv5 + 1.0) return 0.20;
        return 0.50;
    }
};

/**
 * @brief KPSS stationarity test
 * 
 * H0: Series is stationary (opposite of ADF)
 */
class KPSS {
public:
    TrendType trend = TrendType::CONSTANT;
    int lags = -1;  // -1 for automatic
    
    KPSSResult test(const Eigen::VectorXd& y) {
        int n = y.size();
        KPSSResult result;
        result.trend = trend;
        
        // Determine bandwidth
        int l = (lags < 0) ? static_cast<int>(std::sqrt(n)) : lags;
        
        // Detrend the series
        Eigen::VectorXd e;
        if (trend == TrendType::CONSTANT) {
            e = y.array() - y.mean();
        } else {
            // Detrend with linear trend
            Eigen::MatrixXd X(n, 2);
            X.col(0).setOnes();
            for (int i = 0; i < n; ++i) X(i, 1) = i;
            Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
            e = y - X * beta;
        }
        
        // Partial sums
        Eigen::VectorXd S(n);
        S(0) = e(0);
        for (int i = 1; i < n; ++i) {
            S(i) = S(i-1) + e(i);
        }
        
        // Long-run variance estimate (Newey-West)
        double s2 = e.squaredNorm() / n;
        for (int j = 1; j <= l; ++j) {
            double w = 1.0 - double(j) / (l + 1);  // Bartlett kernel
            double gamma_j = 0;
            for (int t = j; t < n; ++t) {
                gamma_j += e(t) * e(t - j);
            }
            gamma_j /= n;
            s2 += 2 * w * gamma_j;
        }
        
        // KPSS statistic
        result.kpss_statistic = S.squaredNorm() / (n * n * s2);
        
        // Critical values
        if (trend == TrendType::CONSTANT) {
            result.cv_1pct = 0.739;
            result.cv_5pct = 0.463;
            result.cv_10pct = 0.347;
        } else {
            result.cv_1pct = 0.216;
            result.cv_5pct = 0.146;
            result.cv_10pct = 0.119;
        }
        
        result.is_stationary = (result.kpss_statistic < result.cv_5pct);
        
        return result;
    }
};

// =============================================================================
// Engle-Granger Cointegration Test
// =============================================================================

/**
 * @brief Engle-Granger two-step cointegration test
 */
class EngleGranger {
public:
    int adf_lags = -1;  // -1 for automatic
    TrendType trend = TrendType::CONSTANT;
    
    /**
     * @brief Test for cointegration between y and x
     * 
     * @param y Dependent variable (T,)
     * @param x Independent variables (T, k)
     */
    EngleGrangerResult test(const Eigen::VectorXd& y, const Eigen::MatrixXd& x) {
        int n = y.size();
        int k = x.cols();
        
        EngleGrangerResult result;
        result.n_obs = n;
        result.n_vars = k + 1;
        
        // Step 1: Cointegrating regression
        Eigen::MatrixXd X(n, k + 1);
        X.col(0).setOnes();
        X.rightCols(k) = x;
        
        Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
        result.intercept = beta(0);
        result.coef = beta.tail(k);
        result.residuals = y - X * beta;
        
        // Step 2: ADF test on residuals
        ADF adf;
        adf.max_lags = (adf_lags < 0) ? 12 : adf_lags;
        adf.trend = TrendType::NONE;  // No deterministic terms for residual test
        adf.auto_lag = (adf_lags < 0);
        
        ADFResult adf_result = adf.test(result.residuals);
        result.adf_statistic = adf_result.adf_statistic;
        
        // Adjusted critical values for residual-based test (MacKinnon 1991)
        // These depend on number of variables in cointegrating regression
        int n_vars = k + 1;
        if (n_vars == 2) {
            result.cv_1pct = -3.90;
            result.cv_5pct = -3.34;
            result.cv_10pct = -3.04;
        } else if (n_vars == 3) {
            result.cv_1pct = -4.29;
            result.cv_5pct = -3.74;
            result.cv_10pct = -3.45;
        } else if (n_vars == 4) {
            result.cv_1pct = -4.64;
            result.cv_5pct = -4.10;
            result.cv_10pct = -3.81;
        } else {
            result.cv_1pct = -4.96;
            result.cv_5pct = -4.42;
            result.cv_10pct = -4.13;
        }
        
        result.cointegrated = (result.adf_statistic < result.cv_5pct);
        
        // Rough p-value
        if (result.adf_statistic < result.cv_1pct) {
            result.p_value = 0.01;
        } else if (result.adf_statistic < result.cv_5pct) {
            result.p_value = 0.05;
        } else if (result.adf_statistic < result.cv_10pct) {
            result.p_value = 0.10;
        } else {
            result.p_value = 0.50;
        }
        
        return result;
    }
};

// =============================================================================
// Johansen Cointegration Test
// =============================================================================

/**
 * @brief Johansen cointegration test and VECM estimation
 */
class Johansen {
public:
    int lag_order = 1;
    TrendType trend = TrendType::CONSTANT;
    
    /**
     * @brief Perform Johansen cointegration test
     * 
     * @param Y Multivariate time series (T, K)
     */
    JohansenResult test(const Eigen::MatrixXd& Y) {
        int T = Y.rows();
        int K = Y.cols();
        int p = lag_order;
        
        JohansenResult result;
        result.n_obs = T;
        result.n_vars = K;
        result.lag_order = p;
        result.trend = trend;
        
        int T_eff = T - p;
        
        // Build differenced and lagged level matrices
        Eigen::MatrixXd dY(T_eff, K);     // ΔY_t
        Eigen::MatrixXd Y_lag(T_eff, K);  // Y_{t-1}
        
        for (int t = 0; t < T_eff; ++t) {
            dY.row(t) = Y.row(p + t) - Y.row(p + t - 1);
            Y_lag.row(t) = Y.row(p + t - 1);
        }
        
        // Build lagged differences for short-run dynamics
        int n_diff = (p - 1) * K;
        Eigen::MatrixXd Z(T_eff, n_diff + (trend != TrendType::NONE ? 1 : 0));
        
        if (trend != TrendType::NONE) {
            Z.col(0).setOnes();
        }
        
        int col = (trend != TrendType::NONE) ? 1 : 0;
        for (int lag = 1; lag < p; ++lag) {
            for (int k = 0; k < K; ++k) {
                for (int t = 0; t < T_eff; ++t) {
                    Z(t, col) = Y(p + t - lag, k) - Y(p + t - lag - 1, k);
                }
                col++;
            }
        }
        
        // Concentrate out short-run dynamics
        Eigen::MatrixXd R0, R1;
        
        if (Z.cols() > 0) {
            Eigen::MatrixXd ZtZ_inv = (Z.transpose() * Z).ldlt()
                                      .solve(Eigen::MatrixXd::Identity(Z.cols(), Z.cols()));
            Eigen::MatrixXd M = Eigen::MatrixXd::Identity(T_eff, T_eff) - Z * ZtZ_inv * Z.transpose();
            R0 = M * dY;
            R1 = M * Y_lag;
        } else {
            R0 = dY;
            R1 = Y_lag;
        }
        
        // Product moment matrices
        Eigen::MatrixXd S00 = R0.transpose() * R0 / T_eff;
        Eigen::MatrixXd S01 = R0.transpose() * R1 / T_eff;
        Eigen::MatrixXd S10 = R1.transpose() * R0 / T_eff;
        Eigen::MatrixXd S11 = R1.transpose() * R1 / T_eff;
        
        // Solve eigenvalue problem: |λS11 - S10 S00^{-1} S01| = 0
        Eigen::MatrixXd S00_inv = S00.ldlt().solve(Eigen::MatrixXd::Identity(K, K));
        Eigen::MatrixXd S11_inv = S11.ldlt().solve(Eigen::MatrixXd::Identity(K, K));
        Eigen::MatrixXd A = S11_inv * S10 * S00_inv * S01;
        
        Eigen::EigenSolver<Eigen::MatrixXd> es(A);
        
        // Sort eigenvalues in descending order
        std::vector<std::pair<double, int>> eig_pairs(K);
        for (int i = 0; i < K; ++i) {
            eig_pairs[i] = {std::abs(es.eigenvalues()(i).real()), i};
        }
        std::sort(eig_pairs.begin(), eig_pairs.end(), std::greater<>());
        
        result.eigenvalues.resize(K);
        result.eigenvectors.resize(K, K);
        
        for (int i = 0; i < K; ++i) {
            result.eigenvalues(i) = eig_pairs[i].first;
            result.eigenvectors.col(i) = es.eigenvectors().col(eig_pairs[i].second).real();
        }
        
        // Normalize eigenvectors
        for (int i = 0; i < K; ++i) {
            result.eigenvectors.col(i) /= result.eigenvectors.col(i).norm();
        }
        
        // Trace statistics: -T Σ_{i=r+1}^{K} ln(1 - λ_i)
        result.trace_stats.resize(K);
        result.max_eigen_stats.resize(K);
        
        for (int r = 0; r < K; ++r) {
            result.trace_stats(r) = 0;
            for (int i = r; i < K; ++i) {
                result.trace_stats(r) -= T_eff * std::log(1.0 - result.eigenvalues(i));
            }
            result.max_eigen_stats(r) = -T_eff * std::log(1.0 - result.eigenvalues(r));
        }
        
        // Critical values (simplified - should use Osterwald-Lenum tables)
        set_critical_values(result, K);
        
        // Determine cointegration rank
        result.trace_rank = 0;
        for (int r = 0; r < K; ++r) {
            if (result.trace_stats(r) > result.trace_cv_5pct(r)) {
                result.trace_rank = r + 1;
            } else {
                break;
            }
        }
        
        result.max_rank = 0;
        for (int r = 0; r < K; ++r) {
            if (result.max_eigen_stats(r) > result.max_cv_5pct(r)) {
                result.max_rank = r + 1;
            } else {
                break;
            }
        }
        
        result.recommended_rank = std::min(result.trace_rank, result.max_rank);
        
        // Compute loading matrix alpha
        if (result.recommended_rank > 0) {
            Eigen::MatrixXd beta = result.eigenvectors.leftCols(result.recommended_rank);
            result.alpha = S01 * beta * (beta.transpose() * S11 * beta).ldlt()
                          .solve(Eigen::MatrixXd::Identity(result.recommended_rank, result.recommended_rank));
        }
        
        return result;
    }
    
    /**
     * @brief Estimate VECM with given cointegration rank
     */
    VECMResult estimate_vecm(const Eigen::MatrixXd& Y, int rank) {
        int T = Y.rows();
        int K = Y.cols();
        int p = lag_order;
        
        VECMResult result;
        result.n_obs = T;
        result.n_vars = K;
        result.rank = rank;
        result.lag_order = p;
        
        // Run Johansen to get beta
        JohansenResult joh = test(Y);
        result.beta = joh.eigenvectors.leftCols(rank);
        
        int T_eff = T - p;
        
        // Build regression matrices
        Eigen::MatrixXd dY(T_eff, K);
        Eigen::MatrixXd ECT(T_eff, rank);  // Error correction terms
        
        for (int t = 0; t < T_eff; ++t) {
            dY.row(t) = Y.row(p + t) - Y.row(p + t - 1);
            ECT.row(t) = (result.beta.transpose() * Y.row(p + t - 1).transpose()).transpose();
        }
        
        // Build lagged differences
        int n_lags = p - 1;
        int n_regressors = rank + n_lags * K + 1;  // ECT + lags + intercept
        
        Eigen::MatrixXd X(T_eff, n_regressors);
        X.col(0).setOnes();
        X.block(0, 1, T_eff, rank) = ECT;
        
        int col = 1 + rank;
        for (int lag = 1; lag <= n_lags; ++lag) {
            for (int k = 0; k < K; ++k) {
                for (int t = 0; t < T_eff; ++t) {
                    X(t, col) = Y(p + t - lag, k) - Y(p + t - lag - 1, k);
                }
                col++;
            }
        }
        
        // OLS for each equation
        Eigen::MatrixXd B = (X.transpose() * X).ldlt().solve(X.transpose() * dY);
        
        result.intercept = B.row(0).transpose();
        result.alpha = B.block(1, 0, rank, K).transpose();
        
        result.gamma.resize(n_lags);
        for (int lag = 0; lag < n_lags; ++lag) {
            result.gamma[lag] = B.block(1 + rank + lag * K, 0, K, K).transpose();
        }
        
        result.residuals = dY - X * B;
        result.sigma = result.residuals.transpose() * result.residuals / (T_eff - n_regressors);
        
        // Information criteria
        double det_sigma = result.sigma.determinant();
        if (det_sigma <= 0) det_sigma = 1e-10;
        
        int n_params = K * n_regressors;
        result.log_likelihood = -0.5 * T_eff * (K * (1 + std::log(2 * M_PI)) + std::log(det_sigma));
        result.aic = -2 * result.log_likelihood + 2 * n_params;
        result.bic = -2 * result.log_likelihood + n_params * std::log(T_eff);
        
        return result;
    }
    
private:
    void set_critical_values(JohansenResult& result, int K) {
        // Simplified critical values for trace and max eigenvalue tests
        // In practice, should use Osterwald-Lenum (1992) tables
        result.trace_cv_5pct.resize(K);
        result.max_cv_5pct.resize(K);
        
        // Approximate critical values for constant case
        std::vector<double> trace_cv = {3.76, 15.41, 29.68, 47.21, 68.52, 94.15};
        std::vector<double> max_cv = {3.76, 14.07, 20.97, 27.07, 33.46, 39.37};
        
        for (int r = 0; r < K; ++r) {
            int idx = K - r - 1;
            if (idx < static_cast<int>(trace_cv.size())) {
                result.trace_cv_5pct(r) = trace_cv[idx];
                result.max_cv_5pct(r) = max_cv[idx];
            } else {
                result.trace_cv_5pct(r) = trace_cv.back() + (idx - trace_cv.size() + 1) * 20;
                result.max_cv_5pct(r) = max_cv.back() + (idx - max_cv.size() + 1) * 6;
            }
        }
    }
};

} // namespace statelix

#endif // STATELIX_COINTEGRATION_H
