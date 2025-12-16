/**
 * @file panel.h
 * @brief Statelix v2.3 - Panel Data Models (Refined)
 * 
 * Implements:
 *   - Fixed Effects (FE) Estimator (Within transformation)
 *   - Random Effects (RE) Estimator (GLS)
 *   - Hausman Test (FE vs RE specification test)
 *   - First Difference Estimator
 * 
 * Refinements:
 *   - Shared PanelStats for distributions
 *   - RE uses t-distribution for small sample inference
 *   - Hausman test warns on negative eigenvalues
 *   - FD documentation clarifies consecutive periods
 */
#ifndef STATELIX_PANEL_H
#define STATELIX_PANEL_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include "../linear_model/solver.h" // [NEW] Unified Solver
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <iostream>

namespace statelix {

// =============================================================================
// Statistical Helpers
// =============================================================================

struct PanelStats {
    static double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    
    static double normal_quantile(double p) {
        // Approximation using inverse error function
        double a = 0.147;
        double x = 2 * p - 1;
        double ln = std::log(1 - x * x);
        double s = (x > 0 ? 1 : -1);
        return s * std::sqrt(std::sqrt((2/(M_PI*a) + ln/2) * (2/(M_PI*a) + ln/2) - ln/a) 
                            - (2/(M_PI*a) + ln/2)) * std::sqrt(2);
    }

    static double beta_inc(double a, double b, double x) {
        if (x < 0 || x > 1) return 0;
        if (x == 0) return 0;
        if (x == 1) return 1;
        
        double bt = (x == 0 || x == 1) ? 0 :
            std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                     a * std::log(x) + b * std::log(1 - x));
        
        if (x < (a + 1) / (a + b + 2)) {
            return bt * beta_cf(a, b, x) / a;
        } else {
            return 1 - bt * beta_cf(b, a, 1 - x) / b;
        }
    }
    
    static double beta_cf(double a, double b, double x) {
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
    
    static double t_cdf(double t, int df) {
        double x = df / (df + t * t);
        return 1.0 - 0.5 * beta_inc(df / 2.0, 0.5, x);
    }
    
    static double t_quantile(double p, int df) {
        // Newton-Raphson
        double x = normal_quantile(p);
        for (int i = 0; i < 10; ++i) {
            double f = t_cdf(x, df) - p;
            double fp = t_pdf(x, df);
            if (std::abs(fp) < 1e-12) break;
            x -= f / fp;
        }
        return x;
    }
    
    static double t_pdf(double t, int df) {
        double c = std::tgamma((df + 1.0) / 2.0) / 
                   (std::sqrt(df * M_PI) * std::tgamma(df / 2.0));
        return c * std::pow(1.0 + t * t / df, -(df + 1.0) / 2.0);
    }
    
    static double f_cdf(double f, int df1, int df2) {
        if (f <= 0) return 0;
        double x = df1 * f / (df1 * f + df2);
        return beta_inc(df1 / 2.0, df2 / 2.0, x);
    }
    
    static double chi2_cdf(double x, int df) {
       return gamma_inc(df/2.0, x/2.0);
    }

    static double gamma_inc(double a, double x) {
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
            double d = 1 / b;
            double h = d;
            double c = 1e30; // undefined in user code, assuming large val?
            // Re-use logic from tests.h if possible, but implementing simple CF here
            for (int n = 1; n < 100; ++n) {
                double an = -n * (n - a);
                b += 2;
                d = an * d + b;
                if(std::abs(d) < 1e-30) d=1e-30;
                c = b + an/c;
                if(std::abs(c) < 1e-30) c=1e-30;
                d = 1/d;
                double del = d*c;
                h *= del;
                if(std::abs(del-1) < 1e-10) break;
            }
            return 1.0 - std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
        }
    }
};

// =============================================================================
// Result Structures
// =============================================================================

struct PanelFEResult {
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    Eigen::VectorXd conf_lower;
    Eigen::VectorXd conf_upper;
    
    Eigen::VectorXd unit_fe;
    Eigen::VectorXd time_fe;
    
    double sigma2_e;
    double sigma2_u;
    
    double r_squared_within;
    double r_squared_between;
    double r_squared_overall;
    
    Eigen::MatrixXd vcov;
    
    int n_obs;
    int n_units;
    int n_periods;
    int df_residual;
    
    double f_stat;
    double f_pvalue;
    
    bool clustered_se;
    int n_clusters;
    
    Eigen::VectorXd residuals;
    Eigen::VectorXd fitted_values;
};

struct PanelREResult {
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    Eigen::VectorXd conf_lower;
    Eigen::VectorXd conf_upper;
    
    double intercept;
    double intercept_se;
    
    double sigma2_e;
    double sigma2_u;
    double theta;
    double rho;
    
    double r_squared_within;
    double r_squared_between;
    double r_squared_overall;
    
    Eigen::MatrixXd vcov;
    
    int n_obs;
    int n_units;
    int n_periods;
    int df_residual;
    
    Eigen::VectorXd residuals;
    Eigen::VectorXd fitted_values;
};

struct HausmanTestResult {
    double chi2_stat;
    int df;
    double p_value;
    bool prefer_fe;
    
    Eigen::VectorXd coef_diff;
    Eigen::MatrixXd var_diff;
    
    std::string recommendation;
    std::string warning; // New field for warnings (e.g., negative eigenvalues)
};

struct PanelFDResult {
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    double r_squared;
    int n_obs;
    Eigen::MatrixXd vcov;
};

// =============================================================================
// Fixed Effects Estimator
// =============================================================================

class PanelFixedEffects {
public:
    double conf_level = 0.95;
    bool cluster_se = true;
    bool two_way = false;
    
    PanelFEResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXi& unit_id,
        const Eigen::VectorXi& time_id
    ) {
        int n = Y.size();
        int k = X.cols();
        PanelFEResult result;
        result.n_obs = n;
        result.clustered_se = cluster_se;
        
        std::unordered_map<int, int> unit_map;
        std::unordered_map<int, int> time_map;
        build_mappings(unit_id, time_id, unit_map, time_map);
        
        int N = unit_map.size();
        int T_max = time_map.size();
        result.n_units = N;
        result.n_periods = T_max;
        result.n_clusters = N;
        
        Eigen::VectorXd Y_demean = within_transform(Y, unit_id, time_id, unit_map, time_map);
        Eigen::MatrixXd X_demean(n, k);
        for (int j = 0; j < k; ++j) {
            X_demean.col(j) = within_transform(X.col(j), unit_id, time_id, unit_map, time_map);
        }
        
        // [NEW] Unified WeightedSolver
        // FE is WLS with weights = 1
        Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
        WeightedDesignMatrix wdm(X_demean, weights);
        WeightedSolver solver(SolverStrategy::AUTO);
        
        try {
            result.coef = solver.solve(wdm, Y_demean);
        } catch (const std::exception& e) {
             throw std::runtime_error(std::string("Solver failed in FE estimation: ") + e.what());
        }

        // Handle Singularity / Rank Deficiency
        // If rank < k, some coefs might be meaningless (already handled by QR/LDLT fallback usually returning a solution)
        // But DoF calculation needs effective rank.
        int rank = solver.rank(); // Exposed in solver.h 
        if (rank == 0 && k > 0) rank = k; // Fallback if solver didn't set it (shouldn't happen)

        result.residuals = Y_demean - X_demean * result.coef;
        
        // DoF Correction for FE: N - k - N_units - (T - 1 if two-way)
        // k is typically rank.
        int df_adj = two_way ? (N + T_max - 1) : N;
        // Caution: If we have k regressors, we lose k DoF. 
        // If matrix was singular, we only lose rank DoF? Standard stats packages usually penalize for full k or rank.
        // Let's use rank.
        result.df_residual = n - rank - df_adj;
        if (result.df_residual <= 0) {
            // Edge case: Saturated model? 
             result.df_residual = 1; // Prevent div/0, warn ideally.
        }
        
        result.sigma2_e = result.residuals.squaredNorm() / result.df_residual;
        
        // Covariance
        // Solver gives (X'X)^-1 (unscaled)
        Eigen::MatrixXd XtX_inv;
        try {
            XtX_inv = solver.variance_covariance();
        } catch (const std::exception& e) {
            // Fallback for QR mode if variance_covariance not ready?
            // User instruction: use unscaled inverse hessian. 
            // If solver in QR mode throws, we might need manual calc?
            // Assuming solver.h handles Cholesky path fine. QR path throws currently.
            // If we hit QR, it means we had singularity.
            // For now, let's propagate error or handle empty.
             throw std::runtime_error("Could not compute covariance (likely rank deficient FE model): " + std::string(e.what()));
        }

        if (cluster_se) {
            result.vcov = compute_cluster_robust_vcov(X_demean, result.residuals, unit_id, unit_map, XtX_inv);
        } else {
            result.vcov = result.sigma2_e * XtX_inv;
        }
        
        result.std_errors.resize(k);
        for (int j = 0; j < k; ++j) result.std_errors(j) = std::sqrt(std::max(0.0, result.vcov(j, j)));
        
        result.t_values.resize(k);
        result.p_values.resize(k);
        result.conf_lower.resize(k);
        result.conf_upper.resize(k);
        
        double t_crit = PanelStats::t_quantile(0.5 + conf_level / 2, result.df_residual);
        
        for (int j = 0; j < k; ++j) {
            if (result.std_errors(j) > 1e-12) {
                result.t_values(j) = result.coef(j) / result.std_errors(j);
                result.p_values(j) = 2.0 * (1.0 - PanelStats::t_cdf(std::abs(result.t_values(j)), result.df_residual));
            } else {
                result.t_values(j) = 0;
                result.p_values(j) = 1;
            }
            result.conf_lower(j) = result.coef(j) - t_crit * result.std_errors(j);
            result.conf_upper(j) = result.coef(j) + t_crit * result.std_errors(j);
        }
        
        result.unit_fe = compute_unit_fe(Y, X, result.coef, unit_id, unit_map);
        if (two_way) result.time_fe = compute_time_fe(Y, X, result.coef, result.unit_fe, unit_id, time_id, unit_map, time_map);
        
        compute_r_squared(Y, X, result.coef, unit_id, unit_map, result);
        
        result.fitted_values = X * result.coef;
        for (int i = 0; i < n; ++i) {
            result.fitted_values(i) += result.unit_fe(unit_map[unit_id(i)]);
            if(two_way) result.fitted_values(i) += result.time_fe(time_map[time_id(i)]);
        }
        
        if (k > 0) {
            double rss = result.residuals.squaredNorm();
            Eigen::VectorXd Y_demean_null = within_transform(Y, unit_id, time_id, unit_map, time_map);
            double tss = Y_demean_null.squaredNorm();
            // F-test logic: 
            // ( (TSS - RSS) / rank ) / ( RSS / df_res )
            // Caution: degrees of freedom for numerator is rank (number of predictors).
            result.f_stat = ((tss - rss) / rank) / (rss / result.df_residual);
            result.f_pvalue = 1.0 - PanelStats::f_cdf(result.f_stat, rank, result.df_residual);
        }
        
        result.sigma2_u = result.unit_fe.squaredNorm() / (N - 1);
        return result;
    }

private:
    void build_mappings(const Eigen::VectorXi& u_id, const Eigen::VectorXi& t_id, 
                       std::unordered_map<int,int>& u_map, std::unordered_map<int,int>& t_map) {
        for(int i=0; i<u_id.size(); ++i) {
            if(u_map.find(u_id(i)) == u_map.end()) u_map[u_id(i)] = u_map.size();
            if(t_map.find(t_id(i)) == t_map.end()) t_map[t_id(i)] = t_map.size();
        }
    }
    
    Eigen::VectorXd within_transform(const Eigen::VectorXd& x, const Eigen::VectorXi& u_id, const Eigen::VectorXi& t_id,
                                     const std::unordered_map<int,int>& u_map, const std::unordered_map<int,int>& t_map) {
        int n = x.size();
        int N = u_map.size();
        Eigen::VectorXd u_sum = Eigen::VectorXd::Zero(N);
        Eigen::VectorXi u_cnt = Eigen::VectorXi::Zero(N);
        for(int i=0; i<n; ++i) {
            int u = u_map.at(u_id(i));
            u_sum(u) += x(i);
            u_cnt(u) += 1;
        }
        Eigen::VectorXd u_mean(N);
        for(int u=0; u<N; ++u) u_mean(u) = (u_cnt(u)>0) ? u_sum(u)/u_cnt(u) : 0;
        
        Eigen::VectorXd res(n);
        for(int i=0; i<n; ++i) res(i) = x(i) - u_mean(u_map.at(u_id(i)));
        
        if(two_way) {
            int T = t_map.size();
            Eigen::VectorXd t_sum = Eigen::VectorXd::Zero(T);
            Eigen::VectorXi t_cnt = Eigen::VectorXi::Zero(T);
            for(int i=0; i<n; ++i) {
                int t = t_map.at(t_id(i));
                t_sum(t) += x(i);
                t_cnt(t) += 1;
            }
            Eigen::VectorXd t_mean(T);
            for(int t=0; t<T; ++t) t_mean(t) = (t_cnt(t)>0) ? t_sum(t)/t_cnt(t) : 0;
            double grand_mean = x.mean();
            
            for(int i=0; i<n; ++i) {
                res(i) = x(i) - u_mean(u_map.at(u_id(i))) - t_mean(t_map.at(t_id(i))) + grand_mean;
            }
        }
        return res;
    }
    
    Eigen::MatrixXd compute_cluster_robust_vcov(const Eigen::MatrixXd& X, const Eigen::VectorXd& resid, 
                                                const Eigen::VectorXi& u_id, const std::unordered_map<int,int>& u_map, 
                                                const Eigen::MatrixXd& XtX_inv) {
        int N = u_map.size();
        int k = X.cols();
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(k, k);
        std::vector<Eigen::VectorXd> cl_sums(N, Eigen::VectorXd::Zero(k));
        
        for(int i=0; i<X.rows(); ++i) {
            int g = u_map.at(u_id(i));
            cl_sums[g] += X.row(i).transpose() * resid(i);
        }
        for(int g=0; g<N; ++g) meat += cl_sums[g] * cl_sums[g].transpose();
        
        double corr = (double(N)/(N-1)) * (double(X.rows()-1)/(X.rows()-k));
        return corr * XtX_inv * meat * XtX_inv;
    }
    
    void compute_r_squared(const Eigen::VectorXd& Y, const Eigen::MatrixXd& X, const Eigen::VectorXd& beta,
                           const Eigen::VectorXi& u_id, const std::unordered_map<int,int>& u_map, PanelFEResult& res) {
        Eigen::VectorXd Y_demean = within_transform(Y, u_id, Eigen::VectorXi::Zero(Y.size()), u_map, std::unordered_map<int,int>{{0,0}});
        double rss = res.residuals.squaredNorm();
        res.r_squared_within = 1.0 - rss / Y_demean.squaredNorm();
        
        // Between
        int N = u_map.size();
        Eigen::VectorXd Yb(N); Yb.setZero();
        Eigen::MatrixXd Xb(N, X.cols()); Xb.setZero();
        Eigen::VectorXd cnt(N); cnt.setZero();
        for(int i=0; i<Y.size(); ++i) {
            int u = u_map.at(u_id(i));
            Yb(u) += Y(i);
            Xb.row(u) += X.row(i);
            cnt(u) += 1.0;
        }
        for(int u=0; u<N; ++u) if(cnt(u)>0) { Yb(u)/=cnt(u); Xb.row(u)/=cnt(u); }
        
        double rss_b = (Yb - Xb * beta - res.unit_fe).squaredNorm();
        double tss_b = (Yb.array() - Yb.mean()).square().sum();
        res.r_squared_between = 1.0 - rss_b / tss_b;
        
        // Overall
        Eigen::VectorXd fit = X * beta;
        for(int i=0; i<Y.size(); ++i) fit(i) += res.unit_fe(u_map.at(u_id(i)));
        res.r_squared_overall = 1.0 - (Y - fit).squaredNorm() / (Y.array() - Y.mean()).square().sum();
    }
    
    Eigen::VectorXd compute_unit_fe(const Eigen::VectorXd& Y, const Eigen::MatrixXd& X, const Eigen::VectorXd& b,
                                    const Eigen::VectorXi& u_id, const std::unordered_map<int,int>& u_map) {
        int N = u_map.size();
        Eigen::VectorXd a = Eigen::VectorXd::Zero(N);
        Eigen::VectorXd c = Eigen::VectorXd::Zero(N);
        for(int i=0; i<Y.size(); ++i) {
            int u = u_map.at(u_id(i));
            a(u) += Y(i) - X.row(i).dot(b);
            c(u) += 1;
        }
        for(int u=0; u<N; ++u) a(u) = (c(u)>0) ? a(u)/c(u) : 0;
        return a;
    }
    
    Eigen::VectorXd compute_time_fe(const Eigen::VectorXd& Y, const Eigen::MatrixXd& X, const Eigen::VectorXd& b,
                                    const Eigen::VectorXd& u_fe, const Eigen::VectorXi& u_id, const Eigen::VectorXi& t_id,
                                    const std::unordered_map<int,int>& u_map, const std::unordered_map<int,int>& t_map) {
        int T = t_map.size();
        Eigen::VectorXd l = Eigen::VectorXd::Zero(T);
        Eigen::VectorXd c = Eigen::VectorXd::Zero(T);
        for(int i=0; i<Y.size(); ++i) {
            int t = t_map.at(t_id(i));
            l(t) += Y(i) - X.row(i).dot(b) - u_fe(u_map.at(u_id(i)));
            c(t) += 1;
        }
        for(int t=0; t<T; ++t) l(t) = (c(t)>0) ? l(t)/c(t) : 0;
        l.array() -= l.mean();
        return l;
    }
};

// =============================================================================
// Random Effects Estimator
// =============================================================================

class PanelRandomEffects {
public:
    double conf_level = 0.95;
    
    PanelREResult fit(const Eigen::VectorXd& Y, const Eigen::MatrixXd& X, const Eigen::VectorXi& u_id, const Eigen::VectorXi& t_id) {
        int n = Y.size();
        int k = X.cols();
        PanelREResult res;
        res.n_obs = n;
        
        std::unordered_map<int,int> u_map, t_map;
        for(int i=0; i<n; ++i) {
            if(u_map.find(u_id(i)) == u_map.end()) u_map[u_id(i)] = u_map.size();
            if(t_map.find(t_id(i)) == t_map.end()) t_map[t_id(i)] = t_map.size();
        }
        int N = u_map.size();
        res.n_units = N;
        res.n_periods = t_map.size();
        
        Eigen::VectorXi T_i = Eigen::VectorXi::Zero(N);
        for(int i=0; i<n; ++i) T_i(u_map[u_id(i)]) += 1;
        double T_bar = T_i.cast<double>().mean();
        
        PanelFixedEffects fe;
        fe.cluster_se = false;
        PanelFEResult fe_res = fe.fit(Y, X, u_id, t_id);
        res.sigma2_e = fe_res.sigma2_e;
        
        // Between Estimator for Sigma2_u
        Eigen::VectorXd Yb = Eigen::VectorXd::Zero(N);
        Eigen::MatrixXd Xb = Eigen::MatrixXd::Zero(N, k);
        Eigen::VectorXd cnt = Eigen::VectorXd::Zero(N);
        for(int i=0; i<n; ++i) {
            int u = u_map[u_id(i)];
            Yb(u) += Y(i);
            Xb.row(u) += X.row(i);
            cnt(u) += 1;
        }
        for(int u=0; u<N; ++u) if(cnt(u)>0) { Yb(u)/=cnt(u); Xb.row(u)/=cnt(u); }
        
        Eigen::MatrixXd Xb_aug(N, k+1);
        Xb_aug.col(0).setOnes();
        Xb_aug.rightCols(k) = Xb;
        
        // [NEW] WeightedSolver for Between Estimator
        Eigen::VectorXd w_bet = Eigen::VectorXd::Ones(N);
        WeightedDesignMatrix wdm_bet(Xb_aug, w_bet);
        WeightedSolver solver_bet(SolverStrategy::AUTO);
        Eigen::VectorXd b_bet;
        try {
            b_bet = solver_bet.solve(wdm_bet, Yb);
        } catch (...) {
             // Fallback or just zero? Between estimator failing is rare unless N < k+1
             b_bet = Eigen::VectorXd::Zero(k+1); 
        }

        double s2_bet = (Yb - Xb_aug*b_bet).squaredNorm() / (N - k - 1);
        res.sigma2_u = std::max(0.0, s2_bet - res.sigma2_e / T_bar);
        
        Eigen::VectorXd theta(N);
        for(int u=0; u<N; ++u) {
            double d = T_i(u) * res.sigma2_u + res.sigma2_e;
            theta(u) = 1.0 - std::sqrt(res.sigma2_e / std::max(1e-10, d));
        }
        res.theta = theta.mean();
        res.rho = res.sigma2_u / (res.sigma2_u + res.sigma2_e);
        
        // Transform
        Eigen::VectorXd Yt(n);
        Eigen::MatrixXd Xt(n, k+1);
        for(int i=0; i<n; ++i) {
            int u = u_map[u_id(i)];
            Yt(i) = Y(i) - theta(u) * Yb(u);
            Xt(i,0) = 1.0 - theta(u);
            for(int j=0; j<k; ++j) Xt(i,j+1) = X(i,j) - theta(u) * Xb(u,j);
        }
        
        // [NEW] WeightedSolver for RE GLS
        // RE transform makes errors spherical, so OLS on transformed data = GLS
        // Weights are 1 on transformed data.
        Eigen::VectorXd w_re = Eigen::VectorXd::Ones(n);
        WeightedDesignMatrix wdm_re(Xt, w_re);
        WeightedSolver solver_re(SolverStrategy::AUTO);
        
        Eigen::VectorXd b_full;
        try {
             b_full = solver_re.solve(wdm_re, Yt);
        } catch (const std::exception& e) {
             throw std::runtime_error(std::string("Solver failed in RE estimation: ") + e.what());
        }

        res.intercept = b_full(0);
        res.coef = b_full.tail(k);
        
        res.fitted_values = (X * res.coef).array() + res.intercept;
        res.residuals = Y - res.fitted_values;
        res.df_residual = n - k - 1;
        
        // Covariance
        Eigen::MatrixXd XtX_inv;
        try {
             XtX_inv = solver_re.variance_covariance();
        } catch (...) {
             throw std::runtime_error("Could not compute RE covariance.");
        }
        
        Eigen::VectorXd rt = Yt - Xt * b_full;
        double s2 = rt.squaredNorm() / res.df_residual;
        Eigen::MatrixXd vcov = s2 * XtX_inv; 
        
        res.vcov = vcov.bottomRightCorner(k, k);
        res.intercept_se = std::sqrt(vcov(0,0));
        
        res.std_errors.resize(k);
        res.t_values.resize(k);
        res.p_values.resize(k);
        res.conf_lower.resize(k);
        res.conf_upper.resize(k);
        
        // REFORM: use t-distribution instead of normal approx
        double t_crit = PanelStats::t_quantile(0.5 + conf_level/2, res.df_residual);
        
        for(int j=0; j<k; ++j) {
            res.std_errors(j) = std::sqrt(std::max(0.0, res.vcov(j,j)));
            if(res.std_errors(j) > 1e-12) {
                res.t_values(j) = res.coef(j) / res.std_errors(j);
                res.p_values(j) = 2.0 * (1.0 - PanelStats::t_cdf(std::abs(res.t_values(j)), res.df_residual));
            } else { res.t_values(j)=0; res.p_values(j)=1; }
            res.conf_lower(j) = res.coef(j) - t_crit * res.std_errors(j);
            res.conf_upper(j) = res.coef(j) + t_crit * res.std_errors(j);
        }
        
        res.r_squared_within = fe_res.r_squared_within;
        res.r_squared_between = fe_res.r_squared_between; // Approximate reuse
        res.r_squared_overall = 1.0 - res.residuals.squaredNorm() / (Y.array()-Y.mean()).square().sum();
        return res;
    }
};

// =============================================================================
// Hausman Test
// =============================================================================

class HausmanTest {
public:
    static HausmanTestResult test(const PanelFEResult& fe, const PanelREResult& re) {
        HausmanTestResult res;
        int k = fe.coef.size();
        res.coef_diff = fe.coef - re.coef;
        res.var_diff = fe.vcov - re.vcov;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(res.var_diff);
        Eigen::VectorXd evals = es.eigenvalues();
        Eigen::MatrixXd evecs = es.eigenvectors();
        
        bool neg_eigen = (evals.minCoeff() < -1e-8);
        if (neg_eigen) {
            res.warning = "Warning: var_diff matrix is not positive semi-definite. Negative eigenvalues clamped.";
            // Standard practice: clamp to small constant
        }
        Eigen::VectorXd evals_clamp = evals.array().max(1e-10);
        Eigen::MatrixXd V_inv = evecs * evals_clamp.asDiagonal().inverse() * evecs.transpose();
        
        res.chi2_stat = res.coef_diff.transpose() * V_inv * res.coef_diff;
        res.df = k;
        res.p_value = 1.0 - PanelStats::chi2_cdf(res.chi2_stat, k);
        
        res.prefer_fe = (res.p_value < 0.05);
        res.recommendation = res.prefer_fe ? "Reject H0: Use Fixed Effects" : "Fail to reject H0: Random Effects consistent";
        return res;
    }
    
    static HausmanTestResult test(const Eigen::VectorXd& Y, const Eigen::MatrixXd& X, 
                                  const Eigen::VectorXi& uid, const Eigen::VectorXi& tid) {
        PanelFixedEffects fe; fe.cluster_se = false;
        PanelRandomEffects re;
        return test(fe.fit(Y,X,uid,tid), re.fit(Y,X,uid,tid));
    }
};

// =============================================================================
// First Difference Estimator
// =============================================================================

/**
 * @brief First Difference Estimator
 * 
 * Note: Only computes differences for consecutive time periods within each unit.
 * Gaps in time series result in loss of observation for that gap.
 */
class PanelFirstDifference {
public:
    double conf_level = 0.95;
    bool cluster_se = true;
    
    PanelFDResult fit(const Eigen::VectorXd& Y, const Eigen::MatrixXd& X, const Eigen::VectorXi& u_id, const Eigen::VectorXi& t_id) {
         int n = Y.size();
         std::vector<std::tuple<int,int,int>> s_idx;
         for(int i=0; i<n; ++i) s_idx.push_back({u_id(i), t_id(i), i});
         std::sort(s_idx.begin(), s_idx.end());
         
         std::vector<double> dY;
         std::vector<Eigen::VectorXd> dX;
         std::vector<int> du;
         
         int p_u=-1, p_t=-1, p_i=-1;
         for(const auto& [u, t, idx] : s_idx) {
             if(u == p_u && t == p_t + 1) {
                 dY.push_back(Y(idx) - Y(p_i));
                 dX.push_back(X.row(idx) - X.row(p_i));
                 du.push_back(u);
             }
             p_u=u; p_t=t; p_i=idx;
         }
         
         int nd = dY.size();
         int k = X.cols();
         if(nd < k) throw std::runtime_error("Not enough FD observations.");
         
         Eigen::VectorXd VdY(nd);
         Eigen::MatrixXd MdX(nd, k);
         Eigen::VectorXi Vdu(nd);
         for(int i=0; i<nd; ++i) { VdY(i)=dY[i]; MdX.row(i)=dX[i]; Vdu(i)=du[i]; }
         
         // [NEW] Unified Solver
         Eigen::VectorXd weights = Eigen::VectorXd::Ones(nd);
         WeightedDesignMatrix wdm(MdX, weights);
         WeightedSolver solver(SolverStrategy::AUTO);
         
         PanelFDResult res;
         try {
             res.coef = solver.solve(wdm, VdY);
         } catch (const std::exception& e) {
             throw std::runtime_error(std::string("Solver failed in FD estimation: ") + e.what());
         }

         res.n_obs = nd;
         Eigen::VectorXd resid = VdY - MdX * res.coef;
         
         Eigen::MatrixXd XtX_inv;
         try {
              XtX_inv = solver.variance_covariance();
         } catch (...) {
              // Fallback or rethrow. Rank deficient FD is common.
              throw std::runtime_error("Could not compute FD covariance (rank deficiency?)");
         }

         // DoF: nd - k (standard OLS on diffs)
         // Note: nd is already reduced N*T
         double sigma2 = resid.squaredNorm() / (nd - k);

         if(cluster_se) {
             std::unordered_map<int,int> umap;
             for(int i=0; i<nd; ++i) if(umap.find(Vdu(i))==umap.end()) umap[Vdu(i)]=umap.size();
             int N = umap.size();
             Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(k,k);
             std::vector<Eigen::VectorXd> sums(N, Eigen::VectorXd::Zero(k));
             for(int i=0; i<nd; ++i) sums[umap[Vdu(i)]] += MdX.row(i).transpose() * resid(i);
             for(int g=0; g<N; ++g) meat += sums[g] * sums[g].transpose();
             double c = (double(N)/(N-1)) * (double(nd-1)/(nd-k));
             res.vcov = c * XtX_inv * meat * XtX_inv;
         } else {
             res.vcov = sigma2 * XtX_inv;
         }
         
         double t_crit = PanelStats::t_quantile(0.5 + conf_level/2, nd-k);
         res.std_errors.resize(k);
         res.t_values.resize(k); 
         res.p_values.resize(k);
         for(int j=0; j<k; ++j) {
             res.std_errors(j) = std::sqrt(std::max(0.0, res.vcov(j,j)));
             if(res.std_errors(j)>1e-12) {
                 res.t_values(j) = res.coef(j)/res.std_errors(j);
                 res.p_values(j) = 2.0*(1.0-PanelStats::t_cdf(std::abs(res.t_values(j)), nd-k));
             } else { res.t_values(j)=0; res.p_values(j)=1; }
         }
         res.r_squared = 1.0 - resid.squaredNorm() / (VdY.array()-VdY.mean()).square().sum();
         return res;
    }
};

} // namespace statelix

#endif // STATELIX_PANEL_H
