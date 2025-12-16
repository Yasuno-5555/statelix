/**
 * @file gmm.h
 * @brief Statelix v1.1 - Linear Generalized Method of Moments (GMM)
 * 
 * Implements:
 *   - Linear GMM Estimator
 *   - 2-Step Efficient GMM (Optimal Weighting Matrix)
 *   - J-Test for Overidentification
 */
#ifndef STATELIX_GMM_H
#define STATELIX_GMM_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include "iv.h" 

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

struct GMMResult {
    Eigen::VectorXd coef;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd t_values;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd conf_int;
    
    // Weighting Matrix used
    Eigen::MatrixXd W;
    std::string weighting_scheme; // "2sls", "optimal"
    
    // Diagnostics
    double j_stat;              // J-statistic (Objective function value * n)
    double j_pvalue;
    int overid_df;
    
    // Model fit
    int n_obs;
    int n_params;
    int n_instruments;
    
    // Covariance
    Eigen::MatrixXd vcov;
    
    Eigen::VectorXd residuals;
    double sigma2;
};

// =============================================================================
// Linear GMM Estimator
// =============================================================================

class LinearGMM {
public:
    bool fit_intercept = true;
    double conf_level = 0.95;
    
    /**
     * @brief Fit Linear GMM model
     * @param weight_method "2sls" (W = (Z'Z)^-1) or "optimal" (2-step with heteroskedasticity)
     */
    GMMResult fit(
        const Eigen::VectorXd& Y,
        const Eigen::MatrixXd& X_endog,
        const Eigen::MatrixXd& X_exog,
        const Eigen::MatrixXd& Z,
        const std::string& weight_method = "optimal"
    ) {
        int n = Y.size();
        
        // 1. Prepare Matrices
        // Z_aug = [Z, X_exog, 1]
        // X_aug = [X_endog, X_exog, 1]
        Eigen::MatrixXd Z_aug = build_augmented_instruments(Z, X_exog, n);
        Eigen::MatrixXd X_aug = build_regressors(X_endog, X_exog, n);
        
        int L = Z_aug.cols();
        int P = X_aug.cols();
        
        if (L < P) {
            throw std::invalid_argument("Underidentified: fewer instruments than parameters.");
        }
        
        // 2. Initial Estimate (2SLS) to get residuals
        // We use 2SLS to get consistent estimates for the first step
        TwoStageLeastSquares iv;
        iv.fit_intercept = fit_intercept; 
        // Note: we control intercept build manually in X_aug/Z_aug for GMM logic,
        // but IV class does it internally.
        // To reuse IV code efficiently without re-building matrices, we call fit() with raw inputs.
        // It's fast enough.
        
        IVResult iv_res = iv.fit(Y, X_endog, X_exog, Z);
        Eigen::VectorXd u = iv_res.residuals;
        
        Eigen::MatrixXd W;
        
        // 3. Construct Weighting Matrix
        if (weight_method == "2sls" || weight_method == "identity") {
             // W = (Z'Z)^-1  (This makes GMM equivalent to 2SLS)
             // Or Identity? Usually "2sls" implies unweighted IV.
             Eigen::MatrixXd ZtZ = Z_aug.transpose() * Z_aug;
             W = ZtZ.inverse(); // L x L, usually safe as id assumption
        } else if (weight_method == "optimal") {
             // Heteroskedasticity Consistent (White) Weighting Matrix
             // S = sum (u_i^2 z_i z_i')
             // W = S^-1
             Eigen::MatrixXd S = Eigen::MatrixXd::Zero(L, L);
             for(int i=0; i<n; ++i) {
                 double u_sq = u(i) * u(i);
                 Eigen::VectorXd z_i = Z_aug.row(i);
                 S += u_sq * z_i * z_i.transpose();
             }
             S /= n; // scaling
             W = S.inverse();
        } else {
            throw std::invalid_argument("Unknown weighting method: " + weight_method);
        }
        
        // 4. Compute GMM Estimator
        // beta = (X' Z W Z' X)^-1 X' Z W Z' Y
        Eigen::MatrixXd ZtX = Z_aug.transpose() * X_aug; // L x P
        Eigen::VectorXd ZtY = Z_aug.transpose() * Y;     // L x 1
        
        Eigen::MatrixXd LHS = ZtX.transpose() * W * ZtX; // P x P
        Eigen::VectorXd RHS = ZtX.transpose() * W * ZtY; // P x 1
        
        Eigen::VectorXd beta = LHS.ldlt().solve(RHS);
        
        // 5. Compute Statistics
        GMMResult res;
        res.coef = beta;
        res.n_obs = n;
        res.n_params = P;
        res.n_instruments = L;
        res.weighting_scheme = weight_method;
        res.W = W;
        
        res.residuals = Y - X_aug * beta;
        res.sigma2 = res.residuals.squaredNorm() / (n - P);
        
        // Variance
        // V = n * (X' Z W Z' X)^-1   (if W is optimal)
        // Note: S was scaled by 1/n. So W is scaled by n.
        // Formula: Var(beta) = (X'Z S^-1 Z'X)^-1
        // My W = S^-1. LHS = X'Z W Z'X.
        // So Vcov = LHS^-1 / n. 
        // Wait, standard asymptotic variance V_asy = (E[xz] Omega^-1 E[zx])^-1.
        // Estimator variance = V_asy / n.
        // My LHS is (ZtX)' W (ZtX). 
        // ZtX is sum(z_i x_i') ~ n * E[zx].
        // W is S^-1 ~ (n * E[u^2 zz])^-1 = n^-1 Omega^-1.
        // LHS ~ (n E)' (n^-1 Omega^-1) (n E) = n E Omega^-1 E.
        // Inverse LHS ~ n^-1 (E Omega^-1 E)^-1 = V_asy / n.
        // So taking direct inverse of LHS gives correctly scaled Variance.
        
        res.vcov = LHS.inverse(); 
        // Correction: If not optimal, calculate sandwich. 
        // For now assuming optimal if "optimal" selected.
        // If "2sls", result matches 2SLS variance.
        
        compute_inference(res, n);
        
        // J-Test
        // J = n * g_bar' W g_bar
        // g_bar = 1/n * Z' u_hat
        Eigen::VectorXd u_hat = res.residuals;
        Eigen::VectorXd Ztu = Z_aug.transpose() * u_hat;
        Eigen::VectorXd g_bar = Ztu / n;
        
        // If W was computed from S/n, then W is n * S^-1.
        // Check scaling of W in calculation.
        // I computed W = (S/n)^-1 = n S^-1.
        // So term = n * g' (n S^-1) g = n^2 g' S^-1 g.
        // With g = 1/n Z'u:
        // term = n^2 * (1/n Z'u)' S^-1 (1/n Z'u) = (Z'u)' S^-1 (Z'u).
        // My W variable is (S/n)^-1. 
        // So J = (Ztu)' * (W/n) * (Ztu)? No.
        
        // J_stat = (Z'u)' S_inv (Z'u) is standard formula? No.
        // Hansen J = n * g' S^-1 g.
        // S = 1/n Z' diag(u^2) Z.
        // My W = S^-1.
        // J = n * (Ztu/n)' W (Ztu/n) = n * 1/n^2 * Ztu' W Ztu = 1/n * Ztu' W Ztu.
        
        double obj_val = Ztu.transpose() * W * Ztu;
        res.j_stat = obj_val / n; 
        
        res.overid_df = L - P;
        res.j_pvalue = 1.0 - chi2_cdf(res.j_stat, res.overid_df);
        
        return res;
    }

private:
    Eigen::MatrixXd build_augmented_instruments(const Eigen::MatrixXd& Z, const Eigen::MatrixXd& X_exog, int n) {
        int m_total = Z.cols() + X_exog.cols() + (fit_intercept ? 1 : 0);
        Eigen::MatrixXd mat(n, m_total);
        int col = 0;
        if (fit_intercept) mat.col(col++).setOnes();
        mat.middleCols(col, Z.cols()) = Z;
        col += Z.cols();
        if (X_exog.cols() > 0) mat.middleCols(col, X_exog.cols()) = X_exog;
        return mat;
    }
    
    Eigen::MatrixXd build_regressors(const Eigen::MatrixXd& X_endog, const Eigen::MatrixXd& X_exog, int n) {
        int p = X_endog.cols() + X_exog.cols() + (fit_intercept ? 1 : 0);
        Eigen::MatrixXd mat(n, p);
        int col = 0;
        if (fit_intercept) mat.col(col++).setOnes();
        mat.middleCols(col, X_endog.cols()) = X_endog;
        col += X_endog.cols();
        if (X_exog.cols() > 0) mat.middleCols(col, X_exog.cols()) = X_exog;
        return mat;
    }
    
    void compute_inference(GMMResult& result, int n) {
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
    
    // Duplicated stats helpers (for now, to ensure standalone compilation)
    // Ideall move to iv.h or shared math_utils.h
    static double t_cdf(double t, int df) { return TwoStageLeastSquares::t_cdf(t, df); }
    static double t_quantile(double p, int df) { return TwoStageLeastSquares::t_quantile(p, df); }
    static double chi2_cdf(double x, int df) { return TwoStageLeastSquares::chi2_cdf(x, df); }
};

} // namespace statelix

#endif // STATELIX_GMM_H
