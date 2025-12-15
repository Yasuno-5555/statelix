/**
 * @file iv.h
 * @brief Statelix v1.1 - Instrumental Variables / Two-Stage Least Squares
 * 
 * Implements:
 *   - 2SLS (Two-Stage Least Squares)
 *   - Weak instrument diagnostics (First-stage F-statistic)
 *   - Overidentification test (Sargan-Hansen J-test)
 * 
 * Theory:
 * -------
 * Model: Y = Xβ + ε  where X is endogenous (correlated with ε)
 * Instruments Z satisfy:
 *   1. Relevance: Cov(Z, X) ≠ 0
 *   2. Exogeneity: Cov(Z, ε) = 0
 * 
 * 2SLS Procedure:
 *   Stage 1: X̂ = Z(Z'Z)⁻¹Z'X = P_Z X
 *   Stage 2: β̂ = (X̂'X̂)⁻¹X̂'Y
 * 
 * Reference: Wooldridge, J.M. (2010). Econometric Analysis of Cross Section
 *            and Panel Data. MIT Press, Chapter 5.
 */
#ifndef STATELIX_IV_H
#define STATELIX_IV_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Complete 2SLS estimation result
 */
struct IVResult {
    // Second-stage coefficients
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd conf_int;       // (p, 2) for [lower, upper]
    
    // First-stage diagnostics
    Eigen::MatrixXd first_stage_coef;   // (m_aug x k1) Matrix of coefs for each endogenous variable
    double first_stage_f;               // F-statistic
    double first_stage_f_pvalue;
    bool weak_instruments;              // true if F < 10 (Stock-Yogo)
    
    // Overidentification test (if #instruments > #endogenous)
    double sargan_stat;                 // Sargan-Hansen J statistic
    double sargan_pvalue;
    int overid_df;                      // degrees of freedom
    bool overid_test_valid;             // true if model is overidentified
    
    // Model fit
    double r_squared;
    double adj_r_squared;
    double residual_std_error;
    int n_obs;
    int n_endog;                        // Number of endogenous regressors
    int n_instruments;                  // Number of instruments
    int n_exog;                         // Number of exogenous controls
    
    // Covariance matrix
    Eigen::MatrixXd vcov;
    
    // Fitted values and residuals
    Eigen::VectorXd fitted_values;
    Eigen::VectorXd residuals;
};

// =============================================================================
// Two-Stage Least Squares
// =============================================================================

/**
 * @brief Two-Stage Least Squares estimator for endogeneity correction
 * 
 * Usage:
 *   TwoStageLeastSquares iv;
 *   // Y = outcome, X_endog = endogenous regressors, 
 *   // X_exog = exogenous controls, Z = instruments
 *   auto result = iv.fit(Y, X_endog, X_exog, Z);
 */
class TwoStageLeastSquares {
public:
    bool fit_intercept = true;
    bool robust_se = false;     // Heteroskedasticity-robust SEs
    double conf_level = 0.95;
    
    /**
     * @brief Fit 2SLS model
     * 
     * @param Y Outcome variable (n,)
     * @param X_endog Endogenous regressors (n, k1) - receives instrument correction
     * @param X_exog Exogenous controls (n, k2) - optional, can be empty matrix
     * @param Z Instruments (n, m) where m >= k1 for identification
     * @return IVResult containing estimates and diagnostics
     * 
     * @throws std::invalid_argument if model is underidentified (m < k1)
     */
    IVResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& X_exog,
        const Eigen::MatrixXd& Z
    ) {
        int n = Y.size();
        int k1 = X_endog.cols();  // Endogenous count
        int k2 = X_exog.cols();   // Exogenous controls (may be 0)
        int m = Z.cols();         // Instrument count
        
        // Check identification
        if (m < k1) {
            throw std::invalid_argument(
                "Underidentified: need at least as many instruments as endogenous variables");
        }
        
        IVResult result;
        result.n_obs = n;
        result.n_endog = k1;
        result.n_instruments = m;
        result.n_exog = k2;
        result.overid_test_valid = (m > k1);
        result.overid_df = m - k1;
        
        // Build augmented instrument matrix [Z, X_exog, 1] 
        // (exogenous variables are their own instruments)
        Eigen::MatrixXd Z_aug = build_augmented_instruments(Z, X_exog, n);
        
        // =====================================================================
        // STAGE 1: Project endogenous onto instruments
        // X̂ = Z_aug (Z_aug' Z_aug)^{-1} Z_aug' X_endog
        // =====================================================================
        Eigen::MatrixXd ZtZ = Z_aug.transpose() * Z_aug;
        Eigen::MatrixXd ZtZ_inv = ZtZ.ldlt().solve(
            Eigen::MatrixXd::Identity(Z_aug.cols(), Z_aug.cols()));
        Eigen::MatrixXd P_Z = Z_aug * ZtZ_inv * Z_aug.transpose();
        
        Eigen::MatrixXd X_endog_hat = P_Z * X_endog;
        
        // First-stage F-statistic (for weak instrument test)
        result.first_stage_f = compute_first_stage_f(X_endog, X_endog_hat, Z_aug, n);
        result.weak_instruments = (result.first_stage_f < 10.0);
        result.first_stage_f_pvalue = 1.0 - f_cdf(result.first_stage_f, m, n - m - 1);
        
        // First-stage coefficients (for reference)
        result.first_stage_coef = ZtZ_inv * Z_aug.transpose() * X_endog;
        
        // =====================================================================
        // STAGE 2: Regress Y on [X̂_endog, X_exog]
        // β̂ = (X̂'X̂)^{-1} X̂'Y (using fitted values from stage 1)
        // =====================================================================
        Eigen::MatrixXd X_hat = build_second_stage_design(X_endog_hat, X_exog, n);
        
        Eigen::MatrixXd XtX_hat = X_hat.transpose() * X_hat;
        Eigen::MatrixXd XtX_hat_inv = XtX_hat.ldlt().solve(
            Eigen::MatrixXd::Identity(X_hat.cols(), X_hat.cols()));
        
        result.coef = XtX_hat_inv * X_hat.transpose() * Y;
        
        // =====================================================================
        // Residuals and standard errors (using ORIGINAL X, not X̂)
        // =====================================================================
        Eigen::MatrixXd X_orig = build_second_stage_design(X_endog, X_exog, n);
        result.fitted_values = X_orig * result.coef;
        result.residuals = Y - result.fitted_values;
        
        double sse = result.residuals.squaredNorm();
        int df = n - result.coef.size();
        double sigma2 = sse / df;
        result.residual_std_error = std::sqrt(sigma2);
        
        // Variance-covariance matrix
        // For 2SLS: Var(β̂) = σ² (X̂'X̂)^{-1}
        if (robust_se) {
            // Heteroskedasticity-robust (sandwich estimator)
            result.vcov = compute_robust_vcov(X_hat, result.residuals, XtX_hat_inv);
        } else {
            result.vcov = sigma2 * XtX_hat_inv;
        }
        
        // Standard errors, t-values, p-values
        compute_inference(result, n);
        
        // R-squared
        double sst = (Y.array() - Y.mean()).square().sum();
        result.r_squared = 1.0 - sse / sst;
        result.adj_r_squared = 1.0 - (1.0 - result.r_squared) * (n - 1) / df;
        
        // =====================================================================
        // Sargan-Hansen overidentification test (if overidentified)
        // J = n * R² from regression of 2SLS residuals on all instruments
        // =====================================================================
        if (result.overid_test_valid) {
            compute_sargan_test(result, Z_aug, n);
        }
        
        return result;
    }
    
    /**
     * @brief Simplified interface: no exogenous controls
     */
    IVResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& Z
    ) {
        return fit(Y, X_endog, Eigen::MatrixXd(Y.size(), 0), Z);
    }

private:
    // Build [Z, X_exog, 1] matrix
    Eigen::MatrixXd build_augmented_instruments(
        const Eigen::MatrixXd& Z,
        const Eigen::MatrixXd& X_exog,
        int n
    ) {
        int m_total = Z.cols() + X_exog.cols() + (fit_intercept ? 1 : 0);
        Eigen::MatrixXd Z_aug(n, m_total);
        
        int col = 0;
        if (fit_intercept) {
            Z_aug.col(col++).setOnes();
        }
        Z_aug.middleCols(col, Z.cols()) = Z;
        col += Z.cols();
        if (X_exog.cols() > 0) {
            Z_aug.middleCols(col, X_exog.cols()) = X_exog;
        }
        return Z_aug;
    }
    
    // Build [X_endog, X_exog, 1] matrix for stage 2
    Eigen::MatrixXd build_second_stage_design(
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& X_exog,
        int n
    ) {
        int p = X_endog.cols() + X_exog.cols() + (fit_intercept ? 1 : 0);
        Eigen::MatrixXd X(n, p);
        
        int col = 0;
        if (fit_intercept) {
            X.col(col++).setOnes();
        }
        X.middleCols(col, X_endog.cols()) = X_endog;
        col += X_endog.cols();
        if (X_exog.cols() > 0) {
            X.middleCols(col, X_exog.cols()) = X_exog;
        }
        return X;
    }
    
    // First-stage F-statistic for weak instrument test
    double compute_first_stage_f(
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& X_endog_hat,
        const Eigen::MatrixXd& Z_aug,
        int n
    ) {
        // F = [(ESS / k) / (RSS / (n-k-1))]
        // For simplicity, use average across endogenous variables
        double total_f = 0.0;
        int k1 = X_endog.cols();
        int m = Z_aug.cols() - (fit_intercept ? 1 : 0);
        
        for (int i = 0; i < k1; ++i) {
            Eigen::VectorXd x_i = X_endog.col(i);
            Eigen::VectorXd x_hat_i = X_endog_hat.col(i);
            double x_mean = x_i.mean();
            
            double ess = (x_hat_i.array() - x_mean).square().sum();
            double rss = (x_i - x_hat_i).squaredNorm();
            
            // F = (ESS/m) / (RSS/(n-m-1))
            double f = (ess / m) / (rss / (n - m - 1));
            total_f += f;
        }
        
        return total_f / k1;  // Average F across endogenous vars
    }
    
    // Heteroskedasticity-robust variance (HC0)
    Eigen::MatrixXd compute_robust_vcov(
        const Eigen::MatrixXd& X_hat,
        const Eigen::VectorXd& residuals,
        const Eigen::MatrixXd& XtX_inv
    ) {
        int n = residuals.size();
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(X_hat.cols(), X_hat.cols());
        
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd xi = X_hat.row(i).transpose();
            meat += residuals(i) * residuals(i) * xi * xi.transpose();
        }
        
        return XtX_inv * meat * XtX_inv;
    }
    
    // Compute t-values, p-values, confidence intervals
    void compute_inference(IVResult& result, int n) {
        int p = result.coef.size();
        int df = n - p;
        
        result.std_errors.resize(p);
        result.t_values.resize(p);
        result.p_values.resize(p);
        result.conf_int.resize(p, 2);
        
        // t critical value for confidence interval
        double t_crit = t_quantile(1.0 - (1.0 - conf_level) / 2.0, df);
        
        for (int j = 0; j < p; ++j) {
            result.std_errors(j) = std::sqrt(result.vcov(j, j));
            result.t_values(j) = result.coef(j) / result.std_errors(j);
            result.p_values(j) = 2.0 * (1.0 - t_cdf(std::abs(result.t_values(j)), df));
            result.conf_int(j, 0) = result.coef(j) - t_crit * result.std_errors(j);
            result.conf_int(j, 1) = result.coef(j) + t_crit * result.std_errors(j);
        }
    }
    
    // Sargan-Hansen J test for overidentification
    void compute_sargan_test(IVResult& result, const Eigen::MatrixXd& Z_aug, int n) {
        // Regress 2SLS residuals on instruments
        Eigen::MatrixXd ZtZ = Z_aug.transpose() * Z_aug;
        Eigen::VectorXd Zte = Z_aug.transpose() * result.residuals;
        Eigen::VectorXd gamma = ZtZ.ldlt().solve(Zte);
        
        Eigen::VectorXd e_hat = Z_aug * gamma;
        double r2 = e_hat.squaredNorm() / result.residuals.squaredNorm();
        
        // J = n * R² ~ χ²(m - k) under H0: instruments are valid
        result.sargan_stat = n * r2;
        result.sargan_pvalue = 1.0 - chi2_cdf(result.sargan_stat, result.overid_df);
    }
    
    // Statistical distribution helpers
    static double t_cdf(double t, int df) {
        // Approximation using normal for large df
        if (df > 100) return normal_cdf(t);
        // Beta function approximation
        double x = df / (df + t * t);
        return 0.5 + 0.5 * std::copysign(1.0, t) * (1.0 - beta_inc(df / 2.0, 0.5, x));
    }
    
    static double t_quantile(double p, int df) {
        // Newton-Raphson approximation
        double t = normal_quantile(p);  // Initial guess
        for (int i = 0; i < 10; ++i) {
            double cdf = t_cdf(t, df);
            double pdf = t_pdf(t, df);
            t -= (cdf - p) / pdf;
        }
        return t;
    }
    
    static double t_pdf(double t, int df) {
        return std::tgamma((df + 1.0) / 2.0) / 
               (std::sqrt(df * M_PI) * std::tgamma(df / 2.0)) *
               std::pow(1.0 + t * t / df, -(df + 1.0) / 2.0);
    }
    
    static double f_cdf(double f, int df1, int df2) {
        double x = df1 * f / (df1 * f + df2);
        return beta_inc(df1 / 2.0, df2 / 2.0, x);
    }
    
    static double chi2_cdf(double x, int df) {
        return gamma_inc(df / 2.0, x / 2.0);
    }
    
    static double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    
    static double normal_quantile(double p) {
        // Rational approximation (Abramowitz & Stegun 26.2.23)
        if (p <= 0) return -8.0;
        if (p >= 1) return 8.0;
        double t = std::sqrt(-2.0 * std::log(p < 0.5 ? p : 1.0 - p));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        double z = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t);
        return p < 0.5 ? -z : z;
    }
    
    // Incomplete beta function (regularized)
    static double beta_inc(double a, double b, double x) {
        // Continued fraction approximation (simplified)
        if (x < 0 || x > 1) return 0.0;
        if (x == 0) return 0.0;
        if (x == 1) return 1.0;
        
        double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                            a * std::log(x) + b * std::log(1.0 - x));
        
        // For x < (a+1)/(a+b+2), use direct series
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return bt * beta_cf(a, b, x) / a;
        } else {
            return 1.0 - bt * beta_cf(b, a, 1.0 - x) / b;
        }
    }
    
    // Continued fraction for beta function
    static double beta_cf(double a, double b, double x) {
        const int max_iter = 100;
        const double eps = 1e-10;
        
        double am = 1.0, bm = 1.0, az = 1.0;
        double qab = a + b, qap = a + 1.0, qam = a - 1.0;
        double bz = 1.0 - qab * x / qap;
        
        for (int m = 1; m <= max_iter; ++m) {
            double em = m;
            double d = em * (b - m) * x / ((qam + 2*em) * (a + 2*em));
            double ap = az + d * am;
            double bp = bz + d * bm;
            d = -(a + em) * (qab + em) * x / ((a + 2*em) * (qap + 2*em));
            double app = ap + d * az;
            double bpp = bp + d * bz;
            
            double aold = az;
            am = ap / bpp;
            bm = bp / bpp;
            az = app / bpp;
            bz = 1.0;
            
            if (std::abs(az - aold) < eps * std::abs(az)) break;
        }
        return az;
    }
    
    // Incomplete gamma function (regularized)
    static double gamma_inc(double a, double x) {
        if (x < 0 || a <= 0) return 0.0;
        if (x == 0) return 0.0;
        
        // Series expansion for small x
        if (x < a + 1.0) {
            double sum = 1.0 / a;
            double term = sum;
            for (int n = 1; n < 100; ++n) {
                term *= x / (a + n);
                sum += term;
                if (std::abs(term) < std::abs(sum) * 1e-10) break;
            }
            return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
        } else {
            // Continued fraction for large x
            double b = x + 1 - a;
            double c = 1e30;
            double d = 1.0 / b;
            double h = d;
            for (int n = 1; n <= 100; ++n) {
                double an = -n * (n - a);
                b += 2.0;
                d = an * d + b;
                if (std::abs(d) < 1e-30) d = 1e-30;
                c = b + an / c;
                if (std::abs(c) < 1e-30) c = 1e-30;
                d = 1.0 / d;
                double del = d * c;
                h *= del;
                if (std::abs(del - 1.0) < 1e-10) break;
            }
            return 1.0 - std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
        }
    }
};

} // namespace statelix

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif // STATELIX_IV_H
