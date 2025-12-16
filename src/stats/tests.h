// Statelix v2.3 - Econometric Specification Tests
// Implements: Durbin-Watson, White, Breusch-Pagan, Goldfeld-Quandt, Ramsey RESET, Chow, Jarque-Bera, Condition Number, Breusch-Godfrey.
// Reference: Wooldridge (2013), Greene (2018).
#ifndef STATELIX_TESTS_H
#define STATELIX_TESTS_H

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <limits>

namespace statelix {
namespace tests {

// =============================================================================
// Result Structures
// =============================================================================

struct TestResult {
    double statistic;
    double p_value;
    int df;                         // Degrees of freedom (if applicable)
    bool reject_null;               // At 5% level
    std::string test_name;
    std::string null_hypothesis;
    std::string conclusion;
};

struct DurbinWatsonResult {
    double dw_statistic;            // DW ∈ [0, 4]
    double d_lower;                 // Critical value (lower)
    double d_upper;                 // Critical value (upper)
    std::string conclusion;         // "positive", "negative", "no autocorr", "inconclusive"
};

struct WhiteTestResult {
    double chi2_statistic;
    double f_statistic;
    int df;
    double chi2_pvalue;
    double f_pvalue;
    bool heteroskedastic;           // At 5% level
};

struct BreuschPaganResult {
    double lm_statistic;            // Lagrange multiplier
    double f_statistic;
    int df;
    double lm_pvalue;
    double f_pvalue;
    bool heteroskedastic;
};

struct BreuschGodfreyResult {
    double lm_statistic;
    double f_statistic;
    int order;                      // Order of autocorrelation tested
    int df;
    double lm_pvalue;
    double f_pvalue;
    bool serial_correlation;
};

struct ChowTestResult {
    double f_statistic;
    int df1;
    int df2;
    double p_value;
    bool structural_break;
    int break_point;
};

struct RamseyResetResult {
    double f_statistic;
    int df1;
    int df2;
    double p_value;
    bool misspecified;              // Reject H0: correct specification
};

struct JarqueBeraResult {
    double jb_statistic;
    double skewness;
    double kurtosis;
    double p_value;
    bool normal;                    // Fail to reject H0: normality
};

struct ConditionNumberResult {
    double condition_number;
    Eigen::VectorXd singular_values;
    bool multicollinearity;         // CN > 30 typically
    std::string severity;           // "none", "moderate", "severe"
};

// =============================================================================
// Helper Functions (Implemented first to avoid forward declaration issues)
// =============================================================================

// Beta continued fraction
inline double beta_cf(double a, double b, double x) {
    double qab = a + b, qap = a + 1, qam = a - 1;
    double c = 1, d = 1 - qab * x / qap;
    if (std::abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    double h = d;
    
    for (int m = 1; m <= 100; ++m) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = 1 + aa / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1 / d;
        h *= d * c;
        
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = 1 + aa / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1 / d;
        double del = d * c;
        h *= del;
        
        if (std::abs(del - 1) < 1e-10) break;
    }
    return h;
}

inline double beta_inc(double a, double b, double x) {
    if (x < 0 || x > 1) return 0;
    if (x == 0) return 0;
    if (x == 1) return 1;
    
    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                 a * std::log(x) + b * std::log(1 - x));
    
    if (x < (a + 1) / (a + b + 2)) {
        return bt * beta_cf(a, b, x) / a;
    } else {
        return 1 - bt * beta_cf(b, a, 1 - x) / b;
    }
}

inline double gamma_inc(double a, double x) {
    if (x < 0) return 0;
    if (a <= 0) return 1;
    
    if (x < a + 1) {
        double sum = 1.0 / a;
        double term = sum;
        for (int n = 1; n < 200; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < 1e-12 * std::abs(sum)) break;
        }
        return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
    } else {
        double b = x + 1 - a;
        double c = 1e30;
        double d = 1 / b;
        double h = d;
        for (int n = 1; n < 200; ++n) {
            double an = -n * (n - a);
            b += 2;
            d = an * d + b;
            if (std::abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (std::abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (std::abs(del - 1) < 1e-12) break;
        }
        return 1.0 - std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
    }
}

inline double chi2_cdf(double x, int df) {
    if (x <= 0) return 0;
    return gamma_inc(df / 2.0, x / 2.0);
}

inline double f_cdf(double f, int df1, int df2) {
    if (f <= 0) return 0;
    double x = df1 * f / (df1 * f + df2);
    return beta_inc(df1 / 2.0, df2 / 2.0, x);
}

// =============================================================================
// Serial Correlation Tests
// =============================================================================

/**
 * @brief Durbin-Watson test for first-order autocorrelation
 */
inline DurbinWatsonResult durbin_watson(
    const Eigen::VectorXd& residuals,
    int n_obs,
    int k
) {
    DurbinWatsonResult result;
    int n = residuals.size();
    
    // DW = Σ(e_t - e_{t-1})² / Σe_t²
    double num = 0, denom = 0;
    for (int t = 1; t < n; ++t) {
        double diff = residuals(t) - residuals(t - 1);
        num += diff * diff;
    }
    denom = residuals.squaredNorm();
    
    result.dw_statistic = (denom > 1e-12) ? num / denom : 2.0;
    
    // Critical values approximation (very rough, usually looked up)
    // For large n, dL approx 1.6, dU approx 1.7 at 5%
    // We'll return placeholders or basic rule of thumb
    result.d_lower = 1.5;
    result.d_upper = 1.7; // Just for structure
    
    if (result.dw_statistic < result.d_lower) {
        result.conclusion = "Positive Autocorrelation";
    } else if (result.dw_statistic > 4 - result.d_lower) {
        result.conclusion = "Negative Autocorrelation";
    } else if (result.dw_statistic > result.d_upper && result.dw_statistic < 4 - result.d_upper) {
        result.conclusion = "No Autocorrelation";
    } else {
        result.conclusion = "Inconclusive";
    }
    
    return result;
}

inline BreuschGodfreyResult breusch_godfrey(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& residuals,
    int order
) {
    BreuschGodfreyResult result;
    result.order = order;
    
    int n = X.rows();
    int k = X.cols();
    
    // Auxiliary regression: e_t on X_t and e_{t-1}, ..., e_{t-p}
    // Effective samples: n - order (since we need lags)
    // Or fill lags with 0 (standard implementation usually drops first p or fills 0)
    // We will fill 0 for lags to keep sample size n (asymtotically equivalent)
    
    int q = k + order;
    Eigen::MatrixXd Z(n, q);
    Z.leftCols(k) = X;
    
    for (int p = 1; p <= order; ++p) {
        // Lag p column of residuals
        Eigen::VectorXd lag_res = Eigen::VectorXd::Zero(n);
        lag_res.tail(n - p) = residuals.head(n - p);
        Z.col(k + p - 1) = lag_res;
    }
    
    // Regress residuals on Z
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Z.transpose() * Z);
    Eigen::VectorXd gamma = ldlt.solve(Z.transpose() * residuals);
    Eigen::VectorXd fitted = Z * gamma;
    
    // R^2 of auxiliary regression
    double tss = residuals.squaredNorm(); // Mean of residuals is 0 by OLS construction (usually)
    double ess = fitted.squaredNorm();
    double r_squared = (tss > 1e-12) ? ess / tss : 0.0;
    
    // LM statistic = n * R^2
    result.lm_statistic = n * r_squared;
    result.df = order;
    result.lm_pvalue = 1.0 - chi2_cdf(result.lm_statistic, result.df);
    
    // F statistic
    // ((R^2)/p) / ((1-R^2)/(n-k-p))
    double num = r_squared / order;
    double denom = (1.0 - r_squared) / (n - q);
    if (denom < 1e-12) result.f_statistic = 0; // or inf
    else result.f_statistic = num / denom;
    
    result.f_pvalue = 1.0 - f_cdf(result.f_statistic, order, n - q);
    result.serial_correlation = (result.lm_pvalue < 0.05);
    
    return result;
}

// =============================================================================
// Heteroskedasticity Tests
// =============================================================================

inline WhiteTestResult white_test(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& residuals
) {
    WhiteTestResult result;
    
    int n = X.rows();
    int k = X.cols();
    
    // Squared residuals
    Eigen::VectorXd e2 = residuals.array().square();
    
    // Build auxiliary regressors: [1, X, X²] 
    // Simplified version (X and X² only)
    int q = 1 + 2 * k;
    Eigen::MatrixXd Z(n, q);
    Z.col(0).setOnes();
    Z.block(0, 1, n, k) = X;
    Z.block(0, 1 + k, n, k) = X.array().square().matrix();
    
    // Regress e² on Z
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Z.transpose() * Z);
    Eigen::VectorXd gamma = ldlt.solve(Z.transpose() * e2);
    Eigen::VectorXd e2_hat = Z * gamma;
    
    // R²
    double tss = (e2.array() - e2.mean()).square().sum();
    double ess = (e2_hat.array() - e2.mean()).square().sum();
    double r_squared = (tss > 1e-12) ? ess / tss : 0;
    
    // Test statistics
    result.df = q - 1;  // Exclude intercept
    result.chi2_statistic = n * r_squared;
    result.chi2_pvalue = 1.0 - chi2_cdf(result.chi2_statistic, result.df);
    
    // F-version
    result.f_statistic = (r_squared / result.df) / ((1 - r_squared) / (n - q));
    result.f_pvalue = 1.0 - f_cdf(result.f_statistic, result.df, n - q);
    
    result.heteroskedastic = (result.chi2_pvalue < 0.05);
    
    return result;
}

inline BreuschPaganResult breusch_pagan(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& residuals
) {
    BreuschPaganResult result;
    
    int n = X.rows();
    int k = X.cols();
    
    // σ̂² = e'e / n
    double sigma2 = residuals.squaredNorm() / n;
    
    // Scaled squared residuals: g = e² / σ̂²
    Eigen::VectorXd g = residuals.array().square() / sigma2;
    
    // Regress g on X (including constant)
    Eigen::MatrixXd Z(n, k);
    Z.col(0).setOnes();
    if (k > 1) {
        Z.rightCols(k - 1) = X.rightCols(k - 1);
    }
    
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Z.transpose() * Z);
    Eigen::VectorXd gamma = ldlt.solve(Z.transpose() * g);
    Eigen::VectorXd g_hat = Z * gamma;
    
    // ESS
    double ess = (g_hat.array() - g.mean()).square().sum();
    
    // LM = ESS / 2
    result.lm_statistic = ess / 2.0;
    result.df = k - 1;
    result.lm_pvalue = 1.0 - chi2_cdf(result.lm_statistic, result.df);
    
    // F-version (Koenker's studentized version)
    double rss = (g - g_hat).squaredNorm();
    result.f_statistic = (ess / result.df) / (rss / (n - k));
    result.f_pvalue = 1.0 - f_cdf(result.f_statistic, result.df, n - k);
    
    result.heteroskedastic = (result.lm_pvalue < 0.05);
    
    return result;
}

inline TestResult goldfeld_quandt(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    double drop_middle = 0.2
) {
    TestResult result;
    result.test_name = "Goldfeld-Quandt";
    result.null_hypothesis = "Homoskedasticity";
    
    int n = X.rows();
    int k = X.cols();
    
    int n_drop = static_cast<int>(n * drop_middle);
    int n_side = (n - n_drop) / 2;
    
    if (n_side <= k + 1) {
        // Just return default result if not enough data
        result.statistic = 0;
        result.p_value = 1.0;
        result.reject_null = false;
        result.conclusion = "Insufficient observations";
        return result;
    }
    
    // First subsample (low values)
    Eigen::MatrixXd X1 = X.topRows(n_side);
    Eigen::VectorXd y1 = y.head(n_side);
    
    // Second subsample (high values)
    Eigen::MatrixXd X2 = X.bottomRows(n_side);
    Eigen::VectorXd y2 = y.tail(n_side);
    
    // OLS on each subsample
    auto ols_resid = [](const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
        return y - X * beta;
    };
    
    Eigen::VectorXd e1 = ols_resid(X1, y1);
    Eigen::VectorXd e2 = ols_resid(X2, y2);
    
    double s1 = e1.squaredNorm() / (n_side - k);
    double s2 = e2.squaredNorm() / (n_side - k);
    
    // F-statistic: larger variance / smaller variance
    if (s2 > s1) {
        result.statistic = s2 / s1;
    } else {
        result.statistic = s1 / s2;
    }
    
    result.df = n_side - k;
    result.p_value = 2.0 * (1.0 - f_cdf(result.statistic, result.df, result.df));
    result.reject_null = (result.p_value < 0.05);
    result.conclusion = result.reject_null ? "Heteroskedasticity detected" : "No evidence of heteroskedasticity";
    
    return result;
}

// =============================================================================
// Functional Form / Specification Tests
// =============================================================================

inline RamseyResetResult ramsey_reset(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& fitted,
    int power = 3
) {
    RamseyResetResult result;
    
    int n = X.rows();
    int k = X.cols();
    
    // Restricted model RSS
    Eigen::VectorXd e_r = y - fitted;
    double rss_r = e_r.squaredNorm();
    
    // Unrestricted model: add ŷ², ŷ³, ...
    int n_powers = power - 1;  // e.g., power=3 means add ŷ² and ŷ³
    Eigen::MatrixXd Z(n, k + n_powers);
    Z.leftCols(k) = X;
    
    for (int p = 2; p <= power; ++p) {
        Z.col(k + p - 2) = fitted.array().pow(p);
    }
    
    // OLS on Z
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Z.transpose() * Z);
    Eigen::VectorXd gamma = ldlt.solve(Z.transpose() * y);
    Eigen::VectorXd e_u = y - Z * gamma;
    double rss_u = e_u.squaredNorm();
    
    // F-test
    result.df1 = n_powers;
    result.df2 = n - k - n_powers;
    
    if (result.df2 <= 0) {
       result.p_value = 1.0;
       result.misspecified = false;
       return result;
    }
    
    result.f_statistic = ((rss_r - rss_u) / result.df1) / (rss_u / result.df2);
    result.p_value = 1.0 - f_cdf(result.f_statistic, result.df1, result.df2);
    result.misspecified = (result.p_value < 0.05);
    
    return result;
}

// =============================================================================
// Structural Break Tests
// =============================================================================

inline ChowTestResult chow_test(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    int break_point
) {
    ChowTestResult result;
    result.break_point = break_point;
    
    int n = X.rows();
    int k = X.cols();
    
    if (break_point <= k || break_point >= n - k) {
        result.structural_break = false;
        return result;
    }
    
    // Pooled regression
    Eigen::VectorXd beta_pooled = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    Eigen::VectorXd e_pooled = y - X * beta_pooled;
    double rss_pooled = e_pooled.squaredNorm();
    
    // Subsample 1
    Eigen::MatrixXd X1 = X.topRows(break_point);
    Eigen::VectorXd y1 = y.head(break_point);
    Eigen::VectorXd beta1 = (X1.transpose() * X1).ldlt().solve(X1.transpose() * y1);
    double rss1 = (y1 - X1 * beta1).squaredNorm();
    
    // Subsample 2
    Eigen::MatrixXd X2 = X.bottomRows(n - break_point);
    Eigen::VectorXd y2 = y.tail(n - break_point);
    Eigen::VectorXd beta2 = (X2.transpose() * X2).ldlt().solve(X2.transpose() * y2);
    double rss2 = (y2 - X2 * beta2).squaredNorm();
    
    // F-statistic
    double rss_unrestricted = rss1 + rss2;
    result.df1 = k;
    result.df2 = n - 2 * k;
    
    if (result.df2 <= 0) {
        result.p_value = 1.0;
        result.structural_break = false;
        return result;
    }
    
    result.f_statistic = ((rss_pooled - rss_unrestricted) / result.df1) / 
                         (rss_unrestricted / result.df2);
    result.p_value = 1.0 - f_cdf(result.f_statistic, result.df1, result.df2);
    result.structural_break = (result.p_value < 0.05);
    
    return result;
}

// =============================================================================
// Normality Tests
// =============================================================================

inline JarqueBeraResult jarque_bera(const Eigen::VectorXd& residuals) {
    JarqueBeraResult result;
    
    int n = residuals.size();
    double mean = residuals.mean();
    
    // Central moments
    double m2 = 0, m3 = 0, m4 = 0;
    for (int i = 0; i < n; ++i) {
        double dev = residuals(i) - mean;
        double dev2 = dev * dev;
        m2 += dev2;
        m3 += dev2 * dev;
        m4 += dev2 * dev2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    
    // Skewness and kurtosis
    double sigma3 = std::pow(m2, 1.5);
    double sigma4 = m2 * m2;
    
    result.skewness = (sigma3 > 1e-12) ? m3 / sigma3 : 0;
    result.kurtosis = (sigma4 > 1e-12) ? m4 / sigma4 - 3.0 : 0;  // Excess kurtosis
    
    // JB statistic
    result.jb_statistic = (n / 6.0) * (result.skewness * result.skewness + 
                                        result.kurtosis * result.kurtosis / 4.0);
    result.p_value = 1.0 - chi2_cdf(result.jb_statistic, 2);
    result.normal = (result.p_value >= 0.05);
    
    return result;
}

// =============================================================================
// Multicollinearity Diagnostics
// =============================================================================

/**
 * @brief Compute condition number for multicollinearity diagnosis
 * @note Uses Thin SVD options to avoid template instantiation errors in sparse context
 */
inline ConditionNumberResult condition_number(const Eigen::MatrixXd& X) {
    ConditionNumberResult result;
    
    // Standardize X
    Eigen::MatrixXd X_std = X;
    for (int j = 0; j < X.cols(); ++j) {
        double mean = X.col(j).mean();
        double std = std::sqrt((X.col(j).array() - mean).square().mean());
        if (std > 1e-12) {
            X_std.col(j) = (X.col(j).array() - mean) / std;
        }
    }
    
    // SVD with Explicit Options (Fixes Docker/GCC build issue)
    // Using Thin SVD for efficiency and stability
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X_std, Eigen::ComputeThinU | Eigen::ComputeThinV);
    result.singular_values = svd.singularValues();
    
    double max_sv = result.singular_values.maxCoeff();
    double min_sv = result.singular_values.minCoeff();
    
    result.condition_number = (min_sv > 1e-12) ? max_sv / min_sv : 
                               std::numeric_limits<double>::infinity();
    
    if (result.condition_number > 30) {
        result.multicollinearity = true;
        result.severity = "severe";
    } else if (result.condition_number > 10) {
        result.multicollinearity = true;
        result.severity = "moderate";
    } else {
        result.multicollinearity = false;
        result.severity = "none";
    }
    
    return result;
}

// =============================================================================
// Endogeneity Tests
// =============================================================================

inline TestResult durbin_wu_hausman(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X_exog,
    const Eigen::MatrixXd& X_endog,
    const Eigen::MatrixXd& Z
) {
    TestResult result;
    result.test_name = "Durbin-Wu-Hausman";
    result.null_hypothesis = "OLS is consistent (no endogeneity)";
    
    int n = y.size();
    int k_endog = X_endog.cols();
    
    // Build full instrument set: [Z, X_exog]
    Eigen::MatrixXd Z_full(n, Z.cols() + X_exog.cols());
    Z_full.leftCols(Z.cols()) = Z;
    Z_full.rightCols(X_exog.cols()) = X_exog;
    
    // First stage
    Eigen::MatrixXd residuals_first(n, k_endog);
    Eigen::MatrixXd ZtZ_inv = (Z_full.transpose() * Z_full).ldlt()
                               .solve(Eigen::MatrixXd::Identity(Z_full.cols(), Z_full.cols()));
    
    for (int j = 0; j < k_endog; ++j) {
        Eigen::VectorXd pi = ZtZ_inv * Z_full.transpose() * X_endog.col(j);
        Eigen::VectorXd X_endog_hat = Z_full * pi;
        residuals_first.col(j) = X_endog.col(j) - X_endog_hat;
    }
    
    // Second stage (augmented)
    Eigen::MatrixXd X_aug(n, X_exog.cols() + X_endog.cols() + k_endog);
    X_aug << X_exog, X_endog, residuals_first;
    
    // OLS augmented
    Eigen::VectorXd beta_aug = (X_aug.transpose() * X_aug).ldlt()
                               .solve(X_aug.transpose() * y);
    
    Eigen::VectorXd e = y - X_aug * beta_aug;
    double s2 = e.squaredNorm() / (n - X_aug.cols());
    
    Eigen::MatrixXd XtX_inv = (X_aug.transpose() * X_aug).ldlt()
                               .solve(Eigen::MatrixXd::Identity(X_aug.cols(), X_aug.cols()));
    
    // F-test on added residual coefficients
    Eigen::VectorXd gamma = beta_aug.tail(k_endog);
    Eigen::MatrixXd V_gamma = s2 * XtX_inv.bottomRightCorner(k_endog, k_endog);
    
    Eigen::LDLT<Eigen::MatrixXd> ldlt_gamma(V_gamma);
    double wald = gamma.transpose() * ldlt_gamma.solve(gamma);
    
    result.statistic = wald;
    result.df = k_endog;
    result.p_value = 1.0 - chi2_cdf(wald, k_endog);
    result.reject_null = (result.p_value < 0.05);
    result.conclusion = result.reject_null ? 
        "Endogeneity detected: use IV" : "No evidence of endogeneity";
    
    return result;
}

// =============================================================================
// Robust Standard Errors
// =============================================================================

inline Eigen::MatrixXd robust_vcov(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& residuals,
    const std::string& type = "HC1"
) {
    int n = X.rows();
    int k = X.cols();
    
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::MatrixXd XtX_inv = XtX.ldlt().solve(Eigen::MatrixXd::Identity(k, k));
    
    Eigen::VectorXd h(n);
    if (type == "HC2" || type == "HC3") {
        Eigen::MatrixXd H = X * XtX_inv;
        for (int i = 0; i < n; ++i) {
            h(i) = H.row(i).dot(X.row(i));
        }
    }
    
    Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < n; ++i) {
        double e2;
        if (type == "HC0" || type == "HC1") {
            e2 = residuals(i) * residuals(i);
        } else if (type == "HC2") {
            e2 = residuals(i) * residuals(i) / (1.0 - h(i));
        } else {  // HC3
            double denom = 1.0 - h(i);
            e2 = residuals(i) * residuals(i) / (denom * denom);
        }
        meat += e2 * X.row(i).transpose() * X.row(i);
    }
    
    Eigen::MatrixXd vcov = XtX_inv * meat * XtX_inv;
    if (type == "HC1") {
        vcov *= double(n) / (n - k);
    }
    return vcov;
}

inline Eigen::MatrixXd newey_west_vcov(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& residuals,
    int max_lag = -1
) {
    int n = X.rows();
    int k = X.cols();
    
    if (max_lag < 0) {
        max_lag = static_cast<int>(std::floor(4.0 * std::pow(n / 100.0, 2.0 / 9.0)));
    }
    max_lag = std::max(0, std::min(max_lag, n - 1));
    
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::MatrixXd XtX_inv = XtX.ldlt().solve(Eigen::MatrixXd::Identity(k, k));
    
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(k, k);
    for (int t = 0; t < n; ++t) {
        S += residuals(t) * residuals(t) * X.row(t).transpose() * X.row(t);
    }
    
    for (int j = 1; j <= max_lag; ++j) {
        double w = 1.0 - double(j) / (max_lag + 1.0);
        Eigen::MatrixXd Gamma_j = Eigen::MatrixXd::Zero(k, k);
        for (int t = j; t < n; ++t) {
            Gamma_j += residuals(t) * residuals(t - j) * X.row(t).transpose() * X.row(t - j);
        }
        S += w * (Gamma_j + Gamma_j.transpose());
    }
    
    return XtX_inv * S * XtX_inv;
}

inline Eigen::MatrixXd cluster_robust_vcov(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& residuals,
    const Eigen::VectorXi& cluster_id
) {
    int n = X.rows();
    int k = X.cols();
    
    std::unordered_map<int, int> cluster_map;
    for (int i = 0; i < n; ++i) {
        if (cluster_map.find(cluster_id(i)) == cluster_map.end()) {
            cluster_map[cluster_id(i)] = cluster_map.size();
        }
    }
    int G = cluster_map.size();
    
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::MatrixXd XtX_inv = XtX.ldlt().solve(Eigen::MatrixXd::Identity(k, k));
    
    std::vector<Eigen::VectorXd> cluster_sums(G, Eigen::VectorXd::Zero(k));
    for (int i = 0; i < n; ++i) {
        int g = cluster_map[cluster_id(i)];
        cluster_sums[g] += X.row(i).transpose() * residuals(i);
    }
    
    Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(k, k);
    for (int g = 0; g < G; ++g) {
        meat += cluster_sums[g] * cluster_sums[g].transpose();
    }
    
    double correction = double(G) / (G - 1) * double(n - 1) / (n - k);
    return correction * XtX_inv * meat * XtX_inv;
}

} // namespace tests
} // namespace statelix

#endif // STATELIX_TESTS_H
