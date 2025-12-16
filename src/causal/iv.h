#ifndef STATELIX_IV_H
#define STATELIX_IV_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "../linear_model/solver.h"

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

struct IVResult {
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd conf_int;
    
    Eigen::MatrixXd first_stage_coef;
    double first_stage_f;
    double first_stage_f_pvalue;
    bool weak_instruments;
    
    double sargan_stat;
    double sargan_pvalue;
    int overid_df;
    bool overid_test_valid;
    
    double r_squared;
    double adj_r_squared;
    double residual_std_error;
    int n_obs;
    int n_endog;
    int n_instruments;
    int n_exog;
    
    Eigen::MatrixXd vcov;
    Eigen::VectorXd fitted_values;
    Eigen::VectorXd residuals;
};

// =============================================================================
// Two-Stage Least Squares
// =============================================================================

class TwoStageLeastSquares {
public:
    bool fit_intercept = true;
    bool robust_se = false;
    double conf_level = 0.95;
    
    IVResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& X_exog,
        const Eigen::MatrixXd& Z
    ) {
        int n = Y.size();
        int k1 = X_endog.cols();
        int k2 = X_exog.cols();
        int m = Z.cols();
        
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
        
        // Z_aug = [Z, X_exog, 1]
        Eigen::MatrixXd Z_aug = build_augmented_instruments(Z, X_exog, n);
            
        // =====================================================================
        // STAGE 1: Regress X_endog on Z_aug
        // =====================================================================
        Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
        WeightedDesignMatrix wdm1(Z_aug, weights);
        WeightedSolver solver1(SolverStrategy::AUTO);
        
        Eigen::MatrixXd X_endog_hat(n, k1);
        // Store first stage coefs [m_aug x k1]
        Eigen::MatrixXd pi_first(Z_aug.cols(), k1);
        
        for(int i=0; i<k1; ++i) {
             // Solve for i-th endogenous variable
             try {
                pi_first.col(i) = solver1.solve(wdm1, X_endog.col(i));
             } catch (const std::exception& e) {
                 throw std::runtime_error("First stage estimation failed for variable " + std::to_string(i) + ": " + e.what());
             }
             X_endog_hat.col(i) = Z_aug * pi_first.col(i);
        }
        result.first_stage_coef = pi_first;
        
        // First-stage diagnostics (using original X for ESS/RSS logic)
        // Note: X_endog_hat is projection P_Z * X
        result.first_stage_f = compute_first_stage_f(X_endog, X_endog_hat, Z_aug, n);
        result.weak_instruments = (result.first_stage_f < 10.0);
        result.first_stage_f_pvalue = 1.0 - f_cdf(result.first_stage_f, m, n - m - 1);
        
        // =====================================================================
        // STAGE 2: Regress Y on [X_endog_hat, X_exog, 1]
        // =====================================================================
        Eigen::MatrixXd X_hat = build_second_stage_design(X_endog_hat, X_exog, n);
        WeightedDesignMatrix wdm2(X_hat, weights);
        WeightedSolver solver2(SolverStrategy::AUTO);
        
        try {
            result.coef = solver2.solve(wdm2, Y);
        } catch (const std::exception& e) {
            throw std::runtime_error("Second stage estimation failed: " + std::string(e.what()));
        }
        
        // =====================================================================
        // Residuals & Variance: MUST USE ORIGINAL X
        // =====================================================================
        Eigen::MatrixXd X_orig = build_second_stage_design(X_endog, X_exog, n);
        result.fitted_values = X_orig * result.coef;
        result.residuals = Y - result.fitted_values;
        
        double sse = result.residuals.squaredNorm();
        int df = n - result.coef.size();
        double sigma2 = sse / df;
        result.residual_std_error = std::sqrt(sigma2);
        
        // Covariance
        // V = sigma2 * (X_hat' X_hat)^-1  (standard 2SLS)
        // With WeightedSolver, we get unscaled inverse
        Eigen::MatrixXd XtX_inv;
        try {
            XtX_inv = solver2.variance_covariance();
        } catch(const std::exception& e) {
            // Rank deficient second stage?
            // Fallback: warn and try regularized? Or just throw.
             throw std::runtime_error("Second stage covariance failed (rank deficiency): " + std::string(e.what()));
        }

        if (robust_se) {
            result.vcov = compute_robust_vcov(X_hat, result.residuals, XtX_inv);
        } else {
            result.vcov = sigma2 * XtX_inv;
        }
        
        compute_inference(result, n);
        
        double sst = (Y.array() - Y.mean()).square().sum();
        result.r_squared = 1.0 - sse / sst;
        result.adj_r_squared = 1.0 - (1.0 - result.r_squared) * (n - 1) / df;
        
        // Overidentification Test
        if (result.overid_test_valid) {
             // Regress 2SLS residuals on instruments Z_aug
             // reusing solver1 logic is fine if we re-solve (WeightedSolver might cache decomposition for same WDM?)
             // solver.h doesn't cache across calls to solve() implicitly unless specifically designed.
             // But WDM is same.
             // We can just create new solver or call solve again.
             // Let's create new for clarity/safety.
             WeightedDesignMatrix wdm_sargan(Z_aug, weights);
             WeightedSolver solver_sargan(SolverStrategy::AUTO);
             Eigen::VectorXd gamma = solver_sargan.solve(wdm_sargan, result.residuals);
             
             Eigen::VectorXd e_aux = Z_aug * gamma;
             double r2_aux = e_aux.squaredNorm() / result.residuals.squaredNorm();
             result.sargan_stat = n * r2_aux;
             result.sargan_pvalue = 1.0 - chi2_cdf(result.sargan_stat, result.overid_df);
        }
        
        return result;
    }
    
    IVResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& Z
    ) {
        return fit(Y, X_endog, Eigen::MatrixXd(Y.size(), 0), Z);
    }

private:
    Eigen::MatrixXd build_augmented_instruments(
        const Eigen::MatrixXd& Z,
        const Eigen::MatrixXd& X_exog,
        int n
    ) {
        int m_total = Z.cols() + X_exog.cols() + (fit_intercept ? 1 : 0);
        Eigen::MatrixXd Z_aug(n, m_total);
        int col = 0;
        if (fit_intercept) Z_aug.col(col++).setOnes();
        Z_aug.middleCols(col, Z.cols()) = Z;
        col += Z.cols();
        if (X_exog.cols() > 0) Z_aug.middleCols(col, X_exog.cols()) = X_exog;
        return Z_aug;
    }
    
    Eigen::MatrixXd build_second_stage_design(
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& X_exog,
        int n
    ) {
        int p = X_endog.cols() + X_exog.cols() + (fit_intercept ? 1 : 0);
        Eigen::MatrixXd X(n, p);
        int col = 0;
        if (fit_intercept) X.col(col++).setOnes();
        X.middleCols(col, X_endog.cols()) = X_endog;
        col += X_endog.cols();
        if (X_exog.cols() > 0) X.middleCols(col, X_exog.cols()) = X_exog;
        return X;
    }
    
    double compute_first_stage_f(
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& X_endog_hat,
        const Eigen::MatrixXd& Z_aug,
        int n
    ) {
        double total_f = 0.0;
        int k1 = X_endog.cols();
        int m = Z_aug.cols() - (fit_intercept ? 1 : 0); // Instruments DoF
        
        for (int i = 0; i < k1; ++i) {
            Eigen::VectorXd x_i = X_endog.col(i);
            Eigen::VectorXd x_hat_i = X_endog_hat.col(i);
            double x_mean = x_i.mean();
            double ess = (x_hat_i.array() - x_mean).square().sum();
            double rss = (x_i - x_hat_i).squaredNorm();
            double f = (ess / m) / (rss / (n - m - 1));
            total_f += f;
        }
        return total_f / k1;
    }
    
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
    
    void compute_inference(IVResult& result, int n) {
        int p = result.coef.size();
        int df = n - p;
        result.std_errors.resize(p);
        result.t_values.resize(p);
        result.p_values.resize(p);
        result.conf_int.resize(p, 2);
        double t_crit = t_quantile(1.0 - (1.0 - conf_level) / 2.0, df);
        for (int j = 0; j < p; ++j) {
            result.std_errors(j) = std::sqrt(result.vcov(j, j));
            if(result.std_errors(j) > 1e-12) {
                 result.t_values(j) = result.coef(j) / result.std_errors(j);
                 result.p_values(j) = 2.0 * (1.0 - t_cdf(std::abs(result.t_values(j)), df));
            } else {
                 result.t_values(j) = 0; result.p_values(j) = 1;
            }
            result.conf_int(j, 0) = result.coef(j) - t_crit * result.std_errors(j);
            result.conf_int(j, 1) = result.coef(j) + t_crit * result.std_errors(j);
        }
    }

    public: // Statistical functions (Internal Copy for now, needed by GMM)
    static double t_cdf(double t, int df) {
        if (df > 100) return normal_cdf(t);
        double x = df / (df + t * t);
        return 0.5 + 0.5 * std::copysign(1.0, t) * (1.0 - beta_inc(df / 2.0, 0.5, x));
    }
    
    static double t_quantile(double p, int df) {
        double t = normal_quantile(p);
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
        if (p <= 0) return -8.0;
        if (p >= 1) return 8.0;
        double t = std::sqrt(-2.0 * std::log(p < 0.5 ? p : 1.0 - p));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        double z = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t);
        return p < 0.5 ? -z : z;
    }
    
    static double beta_inc(double a, double b, double x) {
        if (x < 0 || x > 1) return 0.0;
        if (x == 0) return 0.0;
        if (x == 1) return 1.0;
        double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                            a * std::log(x) + b * std::log(1.0 - x));
        if (x < (a + 1.0) / (a + b + 2.0)) return bt * beta_cf(a, b, x) / a;
        else return 1.0 - bt * beta_cf(b, a, 1.0 - x) / b;
    }
    
    static double beta_cf(double a, double b, double x) {
        const int max_iter = 100;
        const double eps = 1e-10;
        double am = 1, bm = 1, az = 1;
        double qab = a + b, qap = a + 1, qam = a - 1;
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
            am = ap / bpp; bm = bp / bpp; az = app / bpp; bz = 1.0;
            if (std::abs(az - aold) < eps * std::abs(az)) break;
        }
        return az;
    }
    
    static double gamma_inc(double a, double x) {
        if (x < 0 || a <= 0) return 0.0;
        if (x == 0) return 0.0;
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
private:
};

} // namespace statelix

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif // STATELIX_IV_H
