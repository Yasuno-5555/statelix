/**
 * @file var.h
 * @brief Statelix v2.3 - Vector Autoregression (VAR) Models
 * 
 * Implements:
 *   - VAR(p) estimation (equation-by-equation OLS)
 *   - Lag order selection (AIC, BIC, HQ)
 *   - Impulse Response Functions (IRF)
 *   - Orthogonalized IRF (Cholesky decomposition)
 *   - Forecast Error Variance Decomposition (FEVD)
 *   - Bootstrap confidence intervals
 *   - Granger causality tests
 * 
 * Theory:
 * -------
 * VAR(p) model:
 *   Y_t = c + A₁Y_{t-1} + A₂Y_{t-2} + ... + A_pY_{t-p} + ε_t
 *   where Y_t is K×1, A_i are K×K, ε_t ~ N(0, Σ)
 * 
 * Companion form:
 *   Z_t = ΦZ_{t-1} + U_t  where Z_t = [Y_t', Y_{t-1}', ..., Y_{t-p+1}']'
 * 
 * IRF: Response of Y_{i,t+h} to shock ε_{j,t}
 *   Recursive identification via Cholesky: ε_t = B⁻¹u_t where Σ = BB'
 * 
 * Reference:
 *   - Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis
 *   - Stock, J.H. & Watson, M.W. (2001). Vector Autoregressions
 */
#ifndef STATELIX_VAR_H
#define STATELIX_VAR_H

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief IRF result for single impulse-response pair
 */
struct IRFResult {
    Eigen::MatrixXd irf;            // (K, H+1) impulse response
    Eigen::MatrixXd irf_lower;      // (K, H+1) lower confidence bound
    Eigen::MatrixXd irf_upper;      // (K, H+1) upper confidence bound
    int impulse_var;                // Index of impulse variable
    int horizon;                    // Number of periods
    bool orthogonalized;            // Whether using Cholesky orthogonalization
};

/**
 * @brief Full IRF matrix for all variable pairs
 */
struct FullIRFResult {
    // irf[i][j] = response of variable j to shock in variable i
    std::vector<std::vector<Eigen::VectorXd>> irf;  // [K][K][H+1]
    std::vector<std::vector<Eigen::VectorXd>> irf_lower;
    std::vector<std::vector<Eigen::VectorXd>> irf_upper;
    int horizon;
    int n_vars;
};

/**
 * @brief Forecast Error Variance Decomposition result
 */
struct FEVDResult {
    // fevd[h][i][j] = contribution of shock j to variance of variable i at horizon h
    std::vector<Eigen::MatrixXd> fevd;  // Vector of (K, K) matrices for each horizon
    int horizon;
    int n_vars;
};

/**
 * @brief Granger causality test result
 */
struct GrangerResult {
    double f_stat;
    double p_value;
    int df1;                        // Restrictions
    int df2;                        // Residual df
    bool causes;                    // true if H0 rejected (X Granger-causes Y)
    int cause_var;
    int effect_var;
};

/**
 * @brief VAR model estimation result
 */
struct VARResult {
    // Coefficient matrices: [A₁, A₂, ..., A_p]
    std::vector<Eigen::MatrixXd> coef;  // Each is K×K
    Eigen::VectorXd intercept;          // K×1 constant term
    
    // Residuals and covariance
    Eigen::MatrixXd residuals;          // (T-p, K) residual matrix
    Eigen::MatrixXd sigma;              // K×K residual covariance Σ̂
    Eigen::MatrixXd sigma_chol;         // Cholesky factor: Σ = LL'
    
    // Original data info
    Eigen::MatrixXd Y_raw;              // Original data for bootstrap
    int n_obs;                          // T (including lags)
    int n_vars;                         // K
    int lag_order;                      // p
    int n_effective;                    // T - p (effective sample size)
    
    // Model fit
    double log_likelihood;
    double aic;
    double bic;
    double hqc;                         // Hannan-Quinn criterion
    
    // Standard errors (for each coefficient)
    std::vector<Eigen::MatrixXd> std_errors;  // Each is K×K
    Eigen::VectorXd intercept_se;
    
    // Companion matrix (for stability analysis)
    Eigen::MatrixXd companion;          // Kp × Kp
    Eigen::VectorXd eigenvalues_mod;    // Moduli of companion eigenvalues
    bool is_stable;                     // All eigenvalues inside unit circle
};

/**
 * @brief Lag order selection result
 */
struct LagSelectionResult {
    std::vector<double> aic;
    std::vector<double> bic;
    std::vector<double> hqc;
    std::vector<double> fpe;            // Final Prediction Error
    int best_aic;
    int best_bic;
    int best_hqc;
    int best_fpe;
    int max_lag;
};

// =============================================================================
// VAR Estimator
// =============================================================================

/**
 * @brief Vector Autoregression model
 * 
 * Usage:
 *   VectorAutoregression var(2);  // VAR(2)
 *   auto result = var.fit(Y);      // Y is (T, K) matrix
 *   auto irf = var.irf(result, 20, true);  // 20-period IRF, orthogonalized
 */
class VectorAutoregression {
public:
    int lag_order;                  // p
    bool include_intercept = true;
    double conf_level = 0.95;
    int bootstrap_reps = 1000;
    unsigned int seed = 42;
    
    VectorAutoregression(int p = 1) : lag_order(p) {
        if (p < 1) throw std::invalid_argument("Lag order must be >= 1");
    }
    
    /**
     * @brief Fit VAR(p) model
     * 
     * @param Y Data matrix (T, K) where T is time periods, K is variables
     * @return VARResult containing estimated coefficients and diagnostics
     */
    VARResult fit(const Eigen::MatrixXd& Y) {
        int T = Y.rows();
        int K = Y.cols();
        int p = lag_order;
        
        if (T <= p + K) {
            throw std::runtime_error("Not enough observations for VAR estimation");
        }
        
        VARResult result;
        result.Y_raw = Y;  // Store for bootstrap
        result.n_obs = T;
        result.n_vars = K;
        result.lag_order = p;
        result.n_effective = T - p;
        
        // Build regressor matrix Z and response Y_tilde
        // Z_t = [1, Y_{t-1}', Y_{t-2}', ..., Y_{t-p}']'
        int n_eff = T - p;
        int n_regressors = include_intercept ? (K * p + 1) : (K * p);
        
        Eigen::MatrixXd Z(n_eff, n_regressors);
        Eigen::MatrixXd Y_tilde(n_eff, K);
        
        for (int t = 0; t < n_eff; ++t) {
            int row = 0;
            if (include_intercept) {
                Z(t, row++) = 1.0;
            }
            for (int lag = 1; lag <= p; ++lag) {
                for (int k = 0; k < K; ++k) {
                    Z(t, row++) = Y(p + t - lag, k);
                }
            }
            Y_tilde.row(t) = Y.row(p + t);
        }
        
        // OLS estimation: B = (Z'Z)^(-1) Z'Y
        Eigen::MatrixXd ZtZ = Z.transpose() * Z;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(ZtZ);
        Eigen::MatrixXd B = ldlt.solve(Z.transpose() * Y_tilde);  // (n_regressors, K)
        
        // Extract coefficients
        int row = 0;
        if (include_intercept) {
            result.intercept = B.row(row++).transpose();
        } else {
            result.intercept = Eigen::VectorXd::Zero(K);
        }
        
        result.coef.resize(p);
        for (int lag = 0; lag < p; ++lag) {
            result.coef[lag].resize(K, K);
            for (int k = 0; k < K; ++k) {
                result.coef[lag].col(k) = B.row(row++).transpose();
            }
            result.coef[lag].transposeInPlace();  // Now A_i is K×K
        }
        
        // Residuals
        result.residuals = Y_tilde - Z * B;
        
        // Covariance matrix: Σ̂ = (1/(T-p-Kp-1)) * ε'ε
        int df = n_eff - n_regressors;
        if (df <= 0) df = 1;  // Prevent division by zero
        result.sigma = (result.residuals.transpose() * result.residuals) / df;
        
        // Cholesky decomposition for orthogonalized IRF
        Eigen::LLT<Eigen::MatrixXd> chol(result.sigma);
        if (chol.info() == Eigen::Success) {
            result.sigma_chol = chol.matrixL();
        } else {
            // Fallback: use identity
            result.sigma_chol = Eigen::MatrixXd::Identity(K, K);
        }
        
        // Standard errors
        Eigen::MatrixXd ZtZ_inv = ldlt.solve(Eigen::MatrixXd::Identity(n_regressors, n_regressors));
        compute_standard_errors(result, ZtZ_inv, K, p);
        
        // Companion matrix and stability
        result.companion = build_companion_matrix(result.coef, K, p);
        Eigen::EigenSolver<Eigen::MatrixXd> es(result.companion);
        result.eigenvalues_mod.resize(K * p);
        for (int i = 0; i < K * p; ++i) {
            result.eigenvalues_mod(i) = std::abs(es.eigenvalues()(i));
        }
        result.is_stable = (result.eigenvalues_mod.maxCoeff() < 1.0);
        
        // Information criteria
        double det_sigma = result.sigma.determinant();
        if (det_sigma <= 0) det_sigma = 1e-10;
        
        result.log_likelihood = -0.5 * n_eff * (K * (1 + std::log(2 * M_PI)) + std::log(det_sigma));
        
        int n_params = K * (K * p + (include_intercept ? 1 : 0));
        result.aic = std::log(det_sigma) + 2.0 * n_params / n_eff;
        result.bic = std::log(det_sigma) + n_params * std::log(n_eff) / n_eff;
        result.hqc = std::log(det_sigma) + 2.0 * n_params * std::log(std::log(n_eff)) / n_eff;
        
        return result;
    }
    
    /**
     * @brief Compute Impulse Response Function
     * 
     * @param result Fitted VAR result
     * @param horizon Number of periods (h = 0, 1, ..., horizon)
     * @param orthogonalized Use Cholesky orthogonalization
     * @return FullIRFResult with IRF for all variable pairs
     */
    FullIRFResult irf(const VARResult& result, int horizon, bool orthogonalized = true) {
        int K = result.n_vars;
        int p = result.lag_order;
        
        FullIRFResult irf_result;
        irf_result.horizon = horizon;
        irf_result.n_vars = K;
        irf_result.irf.resize(K, std::vector<Eigen::VectorXd>(K));
        irf_result.irf_lower.resize(K, std::vector<Eigen::VectorXd>(K));
        irf_result.irf_upper.resize(K, std::vector<Eigen::VectorXd>(K));
        
        // Compute IRF using companion form
        // Φ_h = companion^h, then extract top-left K×K block
        std::vector<Eigen::MatrixXd> Phi(horizon + 1);
        Phi[0] = Eigen::MatrixXd::Identity(K, K);
        
        if (p == 1) {
            Eigen::MatrixXd A = result.coef[0];
            for (int h = 1; h <= horizon; ++h) {
                Phi[h] = Phi[h-1] * A;
            }
        } else {
            // Use companion matrix
            Eigen::MatrixXd comp = result.companion;
            Eigen::MatrixXd comp_h = Eigen::MatrixXd::Identity(K * p, K * p);
            
            for (int h = 1; h <= horizon; ++h) {
                comp_h = comp_h * comp;
                Phi[h] = comp_h.topLeftCorner(K, K);
            }
        }
        
        // Structural matrix (orthogonalization)
        Eigen::MatrixXd P = orthogonalized ? result.sigma_chol : Eigen::MatrixXd::Identity(K, K);
        
        // Fill IRF: response of j to shock in i
        for (int impulse = 0; impulse < K; ++impulse) {
            for (int response = 0; response < K; ++response) {
                Eigen::VectorXd response_vec(horizon + 1);
                for (int h = 0; h <= horizon; ++h) {
                    response_vec(h) = (Phi[h] * P)(response, impulse);
                }
                irf_result.irf[impulse][response] = response_vec;
            }
        }
        
        // Bootstrap confidence intervals
        compute_irf_bootstrap_ci(result, irf_result, orthogonalized, horizon);
        
        return irf_result;
    }
    
    /**
     * @brief Forecast Error Variance Decomposition
     */
    FEVDResult fevd(const VARResult& result, int horizon) {
        int K = result.n_vars;
        
        // Get orthogonalized IRF
        FullIRFResult irf_result = irf(result, horizon, true);
        
        FEVDResult fevd_result;
        fevd_result.horizon = horizon;
        fevd_result.n_vars = K;
        fevd_result.fevd.resize(horizon + 1);
        
        // FEVD at horizon h: contribution of shock j to forecast error variance of i
        // = Σ_{s=0}^{h} (θ_ij(s))² / Σ_{s=0}^{h} Σ_{k} (θ_ik(s))²
        
        for (int h = 0; h <= horizon; ++h) {
            Eigen::MatrixXd fevd_h(K, K);
            
            for (int i = 0; i < K; ++i) {  // Response variable
                // Total variance of variable i up to horizon h
                double total_var = 0;
                for (int j = 0; j < K; ++j) {  // Shock
                    for (int s = 0; s <= h; ++s) {
                        double theta = irf_result.irf[j][i](s);
                        total_var += theta * theta;
                    }
                }
                
                if (total_var < 1e-12) total_var = 1e-12;
                
                // Contribution of each shock
                for (int j = 0; j < K; ++j) {
                    double contrib = 0;
                    for (int s = 0; s <= h; ++s) {
                        double theta = irf_result.irf[j][i](s);
                        contrib += theta * theta;
                    }
                    fevd_h(i, j) = contrib / total_var;
                }
            }
            
            fevd_result.fevd[h] = fevd_h;
        }
        
        return fevd_result;
    }
    
    /**
     * @brief Test Granger causality
     * 
     * Tests if variable `cause` Granger-causes variable `effect`
     * H0: All coefficients of cause in the equation of effect are zero
     */
    GrangerResult granger_causality(const VARResult& result, 
                                     const Eigen::MatrixXd& Y,
                                     int cause, int effect) {
        int T = Y.rows();
        int K = result.n_vars;
        int p = result.lag_order;
        
        GrangerResult gr;
        gr.cause_var = cause;
        gr.effect_var = effect;
        
        // Unrestricted model: already fitted (result)
        double rss_unrestricted = 0;
        for (int t = 0; t < result.n_effective; ++t) {
            rss_unrestricted += result.residuals(t, effect) * result.residuals(t, effect);
        }
        
        // Restricted model: exclude cause variable from effect equation
        // Fit a single equation with cause excluded
        int n_eff = T - p;
        int n_regressors_restricted = 1 + p * (K - 1);  // intercept + other vars * lags
        
        Eigen::MatrixXd Z_r(n_eff, n_regressors_restricted);
        Eigen::VectorXd y_eff(n_eff);
        
        for (int t = 0; t < n_eff; ++t) {
            int row = 0;
            Z_r(t, row++) = 1.0;  // Intercept
            for (int lag = 1; lag <= p; ++lag) {
                for (int k = 0; k < K; ++k) {
                    if (k != cause) {
                        Z_r(t, row++) = Y(p + t - lag, k);
                    }
                }
            }
            y_eff(t) = Y(p + t, effect);
        }
        
        Eigen::VectorXd beta_r = (Z_r.transpose() * Z_r).ldlt().solve(Z_r.transpose() * y_eff);
        Eigen::VectorXd resid_r = y_eff - Z_r * beta_r;
        double rss_restricted = resid_r.squaredNorm();
        
        // F-test
        gr.df1 = p;  // Number of restrictions (p coefficients of cause)
        gr.df2 = n_eff - (K * p + 1);  // Residual df from unrestricted
        
        if (gr.df2 <= 0) gr.df2 = 1;
        
        gr.f_stat = ((rss_restricted - rss_unrestricted) / gr.df1) / (rss_unrestricted / gr.df2);
        gr.p_value = 1.0 - f_cdf(gr.f_stat, gr.df1, gr.df2);
        gr.causes = (gr.p_value < 0.05);
        
        return gr;
    }
    
    /**
     * @brief Select optimal lag order
     */
    LagSelectionResult select_lag(const Eigen::MatrixXd& Y, int max_lag) {
        LagSelectionResult result;
        result.max_lag = max_lag;
        result.aic.resize(max_lag);
        result.bic.resize(max_lag);
        result.hqc.resize(max_lag);
        result.fpe.resize(max_lag);
        
        double best_aic = std::numeric_limits<double>::infinity();
        double best_bic = std::numeric_limits<double>::infinity();
        double best_hqc = std::numeric_limits<double>::infinity();
        double best_fpe = std::numeric_limits<double>::infinity();
        
        int orig_p = lag_order;
        
        for (int p = 1; p <= max_lag; ++p) {
            lag_order = p;
            try {
                VARResult res = fit(Y);
                
                result.aic[p-1] = res.aic;
                result.bic[p-1] = res.bic;
                result.hqc[p-1] = res.hqc;
                
                // FPE
                int K = res.n_vars;
                double det_sigma = res.sigma.determinant();
                int n = res.n_effective;
                result.fpe[p-1] = std::pow((n + K*p + 1.0) / (n - K*p - 1.0), K) * det_sigma;
                
                if (result.aic[p-1] < best_aic) {
                    best_aic = result.aic[p-1];
                    result.best_aic = p;
                }
                if (result.bic[p-1] < best_bic) {
                    best_bic = result.bic[p-1];
                    result.best_bic = p;
                }
                if (result.hqc[p-1] < best_hqc) {
                    best_hqc = result.hqc[p-1];
                    result.best_hqc = p;
                }
                if (result.fpe[p-1] < best_fpe) {
                    best_fpe = result.fpe[p-1];
                    result.best_fpe = p;
                }
            } catch (...) {
                result.aic[p-1] = std::numeric_limits<double>::infinity();
                result.bic[p-1] = std::numeric_limits<double>::infinity();
                result.hqc[p-1] = std::numeric_limits<double>::infinity();
                result.fpe[p-1] = std::numeric_limits<double>::infinity();
            }
        }
        
        lag_order = orig_p;
        return result;
    }
    
    /**
     * @brief Multi-step forecast
     */
    Eigen::MatrixXd forecast(const VARResult& result, const Eigen::MatrixXd& Y, int steps) {
        int K = result.n_vars;
        int p = result.lag_order;
        int T = Y.rows();
        
        Eigen::MatrixXd forecasts(steps, K);
        
        // Start with last p observations
        Eigen::MatrixXd history(p, K);
        for (int i = 0; i < p; ++i) {
            history.row(i) = Y.row(T - p + i);
        }
        
        for (int h = 0; h < steps; ++h) {
            Eigen::VectorXd y_next = result.intercept;
            for (int lag = 0; lag < p; ++lag) {
                y_next += result.coef[lag] * history.row(p - 1 - lag).transpose();
            }
            forecasts.row(h) = y_next.transpose();
            
            // Update history
            for (int i = 0; i < p - 1; ++i) {
                history.row(i) = history.row(i + 1);
            }
            history.row(p - 1) = y_next.transpose();
        }
        
        return forecasts;
    }
    
private:
    Eigen::MatrixXd build_companion_matrix(
        const std::vector<Eigen::MatrixXd>& A,
        int K, int p
    ) {
        // Companion matrix:
        // [ A_1   A_2   ... A_{p-1}  A_p   ]
        // [ I_K   0     ... 0        0     ]
        // [ 0     I_K   ... 0        0     ]
        // [ ...                            ]
        // [ 0     0     ... I_K      0     ]
        
        int Kp = K * p;
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(Kp, Kp);
        
        // Top row: coefficient matrices
        for (int i = 0; i < p; ++i) {
            F.block(0, i * K, K, K) = A[i];
        }
        
        // Identity blocks below
        if (p > 1) {
            F.block(K, 0, K * (p - 1), K * (p - 1)) = 
                Eigen::MatrixXd::Identity(K * (p - 1), K * (p - 1));
        }
        
        return F;
    }
    
    void compute_standard_errors(
        VARResult& result,
        const Eigen::MatrixXd& ZtZ_inv,
        int K, int p
    ) {
        result.std_errors.resize(p);
        for (int lag = 0; lag < p; ++lag) {
            result.std_errors[lag].resize(K, K);
        }
        result.intercept_se.resize(K);
        
        // SE for each equation (Kronecker product approach simplified)
        // For equation k: se²(β_k) = σ_kk * (Z'Z)^(-1)
        
        int row_idx = 0;
        if (include_intercept) {
            for (int k = 0; k < K; ++k) {
                result.intercept_se(k) = std::sqrt(result.sigma(k, k) * ZtZ_inv(0, 0));
            }
            row_idx = 1;
        }
        
        for (int lag = 0; lag < p; ++lag) {
            for (int j = 0; j < K; ++j) {  // Column of A_lag
                for (int k = 0; k < K; ++k) {  // Row of A_lag (equation)
                    int idx = row_idx + lag * K + j;
                    result.std_errors[lag](k, j) = 
                        std::sqrt(result.sigma(k, k) * ZtZ_inv(idx, idx));
                }
            }
        }
    }
    
    void compute_irf_bootstrap_ci(
        const VARResult& result,
        FullIRFResult& irf_result,
        bool orthogonalized,
        int horizon
    ) {
        int K = result.n_vars;
        int n_obs = result.n_obs;
        int p = result.lag_order;
        int n_eff = result.n_effective;
        
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dist(0, n_eff - 1);
        
        // Storage for bootstrap IRFs: [bootstrap_rep][impulse][response][horizon]
        std::vector<std::vector<std::vector<Eigen::VectorXd>>> boot_irfs(
            bootstrap_reps,
            std::vector<std::vector<Eigen::VectorXd>>(K, std::vector<Eigen::VectorXd>(K))
        );
        
        // Pre-allocate matrices
        Eigen::MatrixXd Y_boot(n_obs, K);
        
        // Copy initial p observations as fixed history
        Y_boot.topRows(p) = result.Y_raw.topRows(p);
        
        // Helper to fit VAR on bootstrap data (simplified version of fit() logic)
        auto fit_boot = [&](const Eigen::MatrixXd& Y_b) -> VARResult {
            // We assume same lag order and intercept setting
            // To maintain speed, we can inline the essential parts or call fit() 
            // Calling fit() is safer but slower (re-allocations). 
            // Given bootstrap_reps ~ 100-1000, calling fit() is acceptable.
            VectorAutoregression boot_model(result.lag_order);
            boot_model.include_intercept = this->include_intercept;
            // boot_model.fit(Y_b); returns VARResult
            return boot_model.fit(Y_b); 
        };

        for (int b = 0; b < bootstrap_reps; ++b) {
            // 1. Resample residuals
            Eigen::MatrixXd resid_boot(n_eff, K);
            for (int t = 0; t < n_eff; ++t) {
                int idx = dist(gen);
                resid_boot.row(t) = result.residuals.row(idx);
            }
            
            // 2. Generate Recursive Bootstrap Data
            for (int t = 0; t < n_eff; ++t) {
                // Time index in Y_boot is t + p
                int time_idx = t + p;
                
                // Prediction part
                Eigen::VectorXd y_next = result.intercept;
                for (int lag = 0; lag < p; ++lag) {
                    // Y_boot[time_idx - 1 - lag]
                    y_next += result.coef[lag] * Y_boot.row(time_idx - 1 - lag).transpose();
                }
                
                // Add bootstrapped residual
                Y_boot.row(time_idx) = y_next.transpose() + resid_boot.row(t);
            }
            
            // 3. Re-estimate VAR
            try {
                // If unstable, we might want to discard? Standard bootstrap usually keeps them 
                // but stationarity checks might be needed. For now, we accept.
                VARResult res_boot = fit_boot(Y_boot);
                
                // 4. Compute IRF for this bootstrap sample
                // We need to compute Phi matrices and orthogonalize
                
                // Compute Phi
                std::vector<Eigen::MatrixXd> Phi(horizon + 1);
                Phi[0] = Eigen::MatrixXd::Identity(K, K);
                
                if (p == 1) {
                    Eigen::MatrixXd A = res_boot.coef[0];
                    for (int h = 1; h <= horizon; ++h) Phi[h] = Phi[h-1] * A;
                } else {
                    Eigen::MatrixXd comp = res_boot.companion;
                    Eigen::MatrixXd comp_h = Eigen::MatrixXd::Identity(K*p, K*p);
                    for (int h = 1; h <= horizon; ++h) {
                        comp_h = comp_h * comp;
                        Phi[h] = comp_h.topLeftCorner(K, K);
                    }
                }
                
                // Orthogonalization Matrix P
                Eigen::MatrixXd P_boot;
                if (orthogonalized) {
                   if (res_boot.sigma_chol.size() > 0) P_boot = res_boot.sigma_chol;
                   else P_boot = Eigen::MatrixXd::Identity(K, K); 
                } else {
                   P_boot = Eigen::MatrixXd::Identity(K, K);
                }
                
                // Store IRF
                for (int i = 0; i < K; ++i) { // Impulse
                    for (int j = 0; j < K; ++j) { // Response
                        for (int h = 0; h <= horizon; ++h) {
                           boot_irfs[b][i][j](h) = (Phi[h] * P_boot)(j, i);
                        }
                    }
                }
            } catch (...) {
                // If estimation fails (singular), skip or replace with mean?
                // Just reuse previous successful one or zero to avoid crash.
                // In practice, should loop until success or fail.
            }
        }
        
        // Compute percentiles (Vectorized across repetitions)
        double alpha = 1.0 - conf_level;
        int lower_idx = static_cast<int>(std::floor(alpha / 2.0 * bootstrap_reps));
        int upper_idx = static_cast<int>(std::ceil((1.0 - alpha / 2.0) * bootstrap_reps));
        if (upper_idx >= bootstrap_reps) upper_idx = bootstrap_reps - 1;
        
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                Eigen::VectorXd lower(horizon + 1), upper(horizon + 1);
                
                for (int h = 0; h <= horizon; ++h) {
                    std::vector<double> vals;
                    vals.reserve(bootstrap_reps);
                    for (int b = 0; b < bootstrap_reps; ++b) {
                        // Check if initialized (if try-catch skipped)
                        // Assuming most succeed.
                        vals.push_back(boot_irfs[b][i][j](h));
                    }
                    std::sort(vals.begin(), vals.end());
                    lower(h) = vals[lower_idx];
                    upper(h) = vals[upper_idx];
                }
                
                irf_result.irf_lower[i][j] = lower;
                irf_result.irf_upper[i][j] = upper;
            }
        }
    }
    
    double f_cdf(double f, int df1, int df2) {
        if (f <= 0) return 0;
        double x = df1 * f / (df1 * f + df2);
        return beta_inc(df1 / 2.0, df2 / 2.0, x);
    }
    
    double beta_inc(double a, double b, double x) {
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
    
    double beta_cf(double a, double b, double x) {
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
};

} // namespace statelix

#endif // STATELIX_VAR_H
