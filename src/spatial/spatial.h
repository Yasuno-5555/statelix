/**
 * @file spatial.h
 * @brief Statelix v2.3 - Spatial Econometrics
 *
 * Implements:
 *   - Spatial Autoregressive Model (SAR/Lag)
 *   - Spatial Error Model (SEM)
 *   - Spatial Durbin Model (SDM)
 *   - Spatial weights matrix utilities
 *   - Moran's I test
 *   - LM tests for spatial dependence
 *
 * Theory:
 * -------
 * SAR (Spatial Lag):
 *   y = ρWy + Xβ + ε        (spatial lag of y)
 *   y = (I - ρW)^{-1}(Xβ + ε)
 *
 * SEM (Spatial Error):
 *   y = Xβ + u
 *   u = λWu + ε             (spatial autocorrelation in errors)
 *
 * SDM (Spatial Durbin):
 *   y = ρWy + Xβ + WXθ + ε  (both spatial lag and spatially lagged X)
 *
 * Estimation:
 *   Maximum Likelihood: Concentrate out β, optimize over ρ/λ
 *   2SLS/GMM: Use spatial lags as instruments
 *
 * Reference:
 *   - Anselin, L. (1988). Spatial Econometrics: Methods and Models
 *   - LeSage, J. & Pace, R.K. (2009). Introduction to Spatial Econometrics
 */
#ifndef STATELIX_SPATIAL_H
#define STATELIX_SPATIAL_H

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace statelix {

// =============================================================================
// Result Structures
// =============================================================================

enum class SpatialModel {
  SAR, // Spatial Autoregressive (Lag)
  SEM, // Spatial Error
  SDM, // Spatial Durbin
  SAC  // Spatial Autocorrelation (both SAR + SEM)
};

struct SpatialResult {
  // Spatial parameters
  double rho; // Spatial lag coefficient (SAR)
  double rho_se;
  double lambda; // Spatial error coefficient (SEM)
  double lambda_se;

  // Regression coefficients
  Eigen::VectorXd beta;
  Eigen::VectorXd beta_se;
  Eigen::VectorXd theta; // Coefficients on WX (SDM only)
  Eigen::VectorXd theta_se;

  // Full parameter vector
  Eigen::VectorXd coef;
  Eigen::VectorXd std_errors;
  Eigen::VectorXd z_values;
  Eigen::VectorXd p_values;

  // Model fit
  double log_likelihood;
  double aic;
  double bic;
  double pseudo_r_squared;
  double sigma2; // Error variance

  // Diagnostics
  Eigen::VectorXd residuals;
  Eigen::VectorXd fitted_values;

  // Direct and indirect effects (for interpretation)
  Eigen::VectorXd direct_effects;
  Eigen::VectorXd indirect_effects;
  Eigen::VectorXd total_effects;

  SpatialModel model_type;
  int n_obs;
  int n_params;
  int iterations;
  bool converged;
};

struct MoranResult {
  double I;      // Moran's I statistic
  double E_I;    // Expected value under H0
  double Var_I;  // Variance under H0
  double z_stat; // Standardized statistic
  double p_value;
  bool spatial_autocorrelation;
};

struct LMSpatialResult {
  // LM tests for spatial dependence
  double lm_lag; // LM test for spatial lag
  double lm_lag_pvalue;
  double lm_error; // LM test for spatial error
  double lm_error_pvalue;
  double rlm_lag; // Robust LM for lag (controlling for error)
  double rlm_lag_pvalue;
  double rlm_error; // Robust LM for error (controlling for lag)
  double rlm_error_pvalue;

  std::string recommendation; // Which model to use
};

// =============================================================================
// Spatial Weights Utilities
// =============================================================================

/**
 * @brief Create spatial weights from distance/contiguity matrix
 */
class SpatialWeights {
public:
  /**
   * @brief Row-standardize a weights matrix
   */
  static Eigen::MatrixXd row_standardize(const Eigen::MatrixXd &W) {
    int n = W.rows();
    Eigen::MatrixXd W_std = W;

    for (int i = 0; i < n; ++i) {
      double row_sum = W.row(i).sum();
      if (row_sum > 1e-10) {
        W_std.row(i) /= row_sum;
      }
    }

    return W_std;
  }

  /**
   * @brief Create contiguity matrix from coordinates (k-nearest neighbors)
   */
  static Eigen::MatrixXd
  knn_weights(const Eigen::MatrixXd &coords, // (n, 2) coordinates
              int k) {
    int n = coords.rows();
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n, n);

    for (int i = 0; i < n; ++i) {
      // Compute distances to all other points
      std::vector<std::pair<double, int>> dists;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          double d = (coords.row(i) - coords.row(j)).norm();
          dists.push_back({d, j});
        }
      }

      // Sort and take k nearest
      std::sort(dists.begin(), dists.end());
      for (int m = 0; m < k && m < (int)dists.size(); ++m) {
        W(i, dists[m].second) = 1.0;
      }
    }

    return row_standardize(W);
  }

  /**
   * @brief Create inverse distance weights
   */
  static Eigen::MatrixXd
  inverse_distance_weights(const Eigen::MatrixXd &coords,
                           double bandwidth = -1, // -1 for no cutoff
                           double power = 1.0) {
    int n = coords.rows();
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n, n);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          double d = (coords.row(i) - coords.row(j)).norm();
          if (bandwidth < 0 || d <= bandwidth) {
            W(i, j) = 1.0 / std::pow(std::max(d, 1e-10), power);
          }
        }
      }
    }

    return row_standardize(W);
  }
};

// =============================================================================
// Spatial Models
// =============================================================================

/**
 * @brief Spatial Regression Models (SAR/SEM/SDM)
 */
class SpatialRegression {
public:
  SpatialModel model = SpatialModel::SAR;
  int max_iter = 100;
  double tol = 1e-6;
  double conf_level = 0.95;

  /**
   * @brief Fit spatial model via Maximum Likelihood
   *
   * @param y Outcome (n,)
   * @param X Covariates (n, k)
   * @param W Spatial weights matrix (n, n), should be row-standardized
   */
  SpatialResult fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                    const Eigen::MatrixXd &W) {
    int n = y.size();
    int k = X.cols();

    SpatialResult result;
    result.model_type = model;
    result.n_obs = n;

    // Compute eigenvalues of W for valid range of rho and log-det optimizations
    // O(N^3) one-time cost
    Eigen::EigenSolver<Eigen::MatrixXd> es(W);
    Eigen::VectorXd lambda = es.eigenvalues().real();

    double lambda_min = lambda.minCoeff();
    double lambda_max = lambda.maxCoeff();
    double rho_min = 1.0 / lambda_min;
    double rho_max = 1.0 / lambda_max;

    // Pre-compute
    Eigen::VectorXd Wy = W * y;

    switch (model) {
    case SpatialModel::SAR:
      fit_sar(y, X, W, Wy, lambda, rho_min, rho_max, result);
      break;
    case SpatialModel::SEM:
      fit_sem(y, X, W, rho_min, rho_max, result);
      break;
    case SpatialModel::SDM:
      fit_sdm(y, X, W, Wy, lambda, rho_min, rho_max, result);
      break;
    default:
      fit_sar(y, X, W, Wy, lambda, rho_min, rho_max, result);
    }

    // Compute effects for interpretation
    if (model == SpatialModel::SAR || model == SpatialModel::SDM) {
      compute_effects(lambda, result);
    }

    return result;
  }

  // ... tests ...

private:
  void fit_sar(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
               const Eigen::MatrixXd &W, const Eigen::VectorXd &Wy,
               const Eigen::VectorXd &lambda, double rho_min, double rho_max,
               SpatialResult &result) {
    int n = y.size();
    int k = X.cols();

    // Optimize rho using pre-computed eigenvalues
    result.rho = optimize_rho_sar_eigen(y, X, W, Wy, lambda, rho_min, rho_max);

    // ... rest same as before ...
    // Final estimates
    Eigen::VectorXd y_star = y - result.rho * Wy;
    result.beta = (X.transpose() * X).ldlt().solve(X.transpose() * y_star);

    result.residuals = y_star - X * result.beta;
    result.sigma2 = result.residuals.squaredNorm() / n;

    result.fitted_values = result.rho * Wy + X * result.beta;

    compute_sar_standard_errors(y, X, W, result);

    // Log-likelihood using eigenvalues
    double log_det = 0;
    for (int i = 0; i < n; ++i)
      log_det += std::log(1.0 - result.rho * lambda(i));

    result.log_likelihood =
        -n / 2.0 * std::log(2 * M_PI * result.sigma2) + log_det -
        result.residuals.squaredNorm() / (2 * result.sigma2);

    result.n_params = k + 2;
    result.aic = -2 * result.log_likelihood + 2 * result.n_params;
    result.bic = -2 * result.log_likelihood + result.n_params * std::log(n);

    // Coefficients
    result.coef.resize(k + 1);
    result.coef(0) = result.rho;
    result.coef.tail(k) = result.beta;

    result.std_errors.resize(k + 1);
    result.std_errors(0) = result.rho_se;
    result.std_errors.tail(k) = result.beta_se;

    result.z_values = result.coef.array() / result.std_errors.array();
    result.p_values.resize(k + 1);
    for (int j = 0; j <= k; ++j) {
      result.p_values(j) = 2 * (1 - normal_cdf(std::abs(result.z_values(j))));
    }

    result.converged = true;
    result.lambda = 0;
  }

  // ... optimize_rho_sar_eigen ... (already optimized)

  void fit_sdm(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
               const Eigen::MatrixXd &W, const Eigen::VectorXd &Wy,
               const Eigen::VectorXd &lambda, double rho_min, double rho_max,
               SpatialResult &result) {
    int n = y.size();
    int k = X.cols();

    Eigen::MatrixXd WX = W * X;
    Eigen::MatrixXd X_aug(n, 2 * k);
    X_aug.leftCols(k) = X;
    X_aug.rightCols(k) = WX;

    // Pass lambda to SAR fit
    Eigen::VectorXd Wy_copy = Wy;
    fit_sar(y, X_aug, W, Wy_copy, lambda, rho_min, rho_max, result);

    Eigen::VectorXd beta_full = result.beta;
    result.beta = beta_full.head(k);
    result.theta = beta_full.tail(k);

    result.beta_se = result.std_errors.segment(1, k);
    result.theta_se = result.std_errors.tail(k);

    result.model_type = SpatialModel::SDM;
  }

  void compute_effects(const Eigen::VectorXd &lambda, SpatialResult &result) {
    int n = lambda.size();
    int k = result.beta.size();

    result.direct_effects.resize(k);
    result.indirect_effects.resize(k);
    result.total_effects.resize(k);

    // Calculate trace of (I - rho*W)^-1 = sum(1/(1-rho*lambda_i))
    double tr_S = 0;
    for (int i = 0; i < n; ++i) {
      tr_S += 1.0 / (1.0 - result.rho * lambda(i));
    }
    double avg_direct = tr_S / n;

    // Average total depends on row-standardization
    // If row-stnd, 1/(1-rho)
    double avg_total = 1.0 / (1.0 - result.rho);

    for (int j = 0; j < k; ++j) {
      result.direct_effects(j) = result.beta(j) * avg_direct;
      result.total_effects(j) = result.beta(j) * avg_total;
      result.indirect_effects(j) =
          result.total_effects(j) - result.direct_effects(j);
    }
  }

public:
  /**
   * @brief Moran's I test for spatial autocorrelation
   */
  MoranResult
  moran_test(const Eigen::VectorXd &z, // Variable to test (e.g., residuals)
             const Eigen::MatrixXd &W) {
    int n = z.size();
    MoranResult result;

    // Center the variable
    double z_mean = z.mean();
    Eigen::VectorXd z_c = z.array() - z_mean;

    // Moran's I = (n/S0) * (z'Wz) / (z'z)
    double S0 = W.sum(); // Sum of all weights
    double num = z_c.transpose() * W * z_c;
    double denom = z_c.squaredNorm();

    result.I = (n / S0) * num / denom;

    // Expected value under H0 (no spatial autocorrelation)
    result.E_I = -1.0 / (n - 1);

    // Variance (normality assumption)
    double S1 = 0, S2 = 0;
    for (int i = 0; i < n; ++i) {
      double row_sum = W.row(i).sum();
      double col_sum = W.col(i).sum();
      S2 += (row_sum + col_sum) * (row_sum + col_sum);
      for (int j = 0; j < n; ++j) {
        S1 += (W(i, j) + W(j, i)) * (W(i, j) + W(j, i));
      }
    }
    S1 /= 2;

    double n2 = n * n;
    double A = n * ((n2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0);
    double B = (n2 - n) * S1 - 2 * n * S2 + 6 * S0 * S0;
    double C = (n - 1) * (n - 2) * (n - 3) * S0 * S0;

    double D = z_c.array().pow(4).sum() / std::pow(z_c.squaredNorm(), 2);

    result.Var_I = (A - D * B) / C - result.E_I * result.E_I;

    result.z_stat =
        (result.I - result.E_I) / std::sqrt(std::max(1e-10, result.Var_I));
    result.p_value = 2 * (1 - normal_cdf(std::abs(result.z_stat)));
    result.spatial_autocorrelation = (result.p_value < 0.05);

    return result;
  }

  /**
   * @brief LM tests for spatial dependence
   */
  LMSpatialResult lm_tests(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                           const Eigen::MatrixXd &W) {
    int n = y.size();
    int k = X.cols();
    LMSpatialResult result;

    // OLS residuals
    Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    Eigen::VectorXd e = y - X * beta;
    double s2 = e.squaredNorm() / n;

    // Compute test statistics
    Eigen::VectorXd We = W * e;
    Eigen::VectorXd Wy = W * y;
    Eigen::MatrixXd WW = W * W;

    double T = (WW + W.transpose() * W).trace();

    // LM-Lag
    Eigen::VectorXd WXb = W * X * beta;
    Eigen::MatrixXd M = Eigen::MatrixXd::Identity(n, n) -
                        X * (X.transpose() * X).ldlt().solve(X.transpose());
    double J = (WXb.transpose() * M * WXb).value() / s2 + T;

    double num_lag = e.dot(Wy) / s2;
    result.lm_lag = num_lag * num_lag / J;
    result.lm_lag_pvalue = 1 - chi2_cdf(result.lm_lag, 1);

    // LM-Error
    double num_err = e.dot(We) / s2;
    result.lm_error = num_err * num_err / T;
    result.lm_error_pvalue = 1 - chi2_cdf(result.lm_error, 1);

    // Robust LM-Lag
    double numer = num_lag - num_err;
    result.rlm_lag = numer * numer / (J - T);
    result.rlm_lag_pvalue = 1 - chi2_cdf(result.rlm_lag, 1);

    // Robust LM-Error
    double T_J = T * J / (J - T);
    result.rlm_error = (num_err - T * num_lag / J) *
                       (num_err - T * num_lag / J) / (T - T * T / J);
    result.rlm_error_pvalue = 1 - chi2_cdf(result.rlm_error, 1);

    // Recommendation
    if (result.rlm_lag_pvalue < 0.05 && result.rlm_error_pvalue >= 0.05) {
      result.recommendation = "SAR (Spatial Lag Model)";
    } else if (result.rlm_error_pvalue < 0.05 &&
               result.rlm_lag_pvalue >= 0.05) {
      result.recommendation = "SEM (Spatial Error Model)";
    } else if (result.rlm_lag_pvalue < 0.05 && result.rlm_error_pvalue < 0.05) {
      result.recommendation = "SAC or SDM (both types present)";
    } else {
      result.recommendation = "OLS (no spatial dependence detected)";
    }

    return result;
  }

private:
  void fit_sar(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
               const Eigen::MatrixXd &W, const Eigen::VectorXd &Wy,
               double rho_min, double rho_max, SpatialResult &result) {
    int n = y.size();
    int k = X.cols();

    // Ord (1975): Use eigenvalues for log-det
    // O(N^3) one-time cost, then O(N) per iteration
    Eigen::VectorXcd eig = W.eigenvalues();
    Eigen::VectorXd lambda =
        eig.real(); // Assume W is similar to symmetric or real eig

    // Optimize rho
    result.rho = optimize_rho_sar_eigen(y, X, W, Wy, lambda, rho_min, rho_max);

    // Final estimates
    Eigen::VectorXd y_star = y - result.rho * Wy;
    result.beta = (X.transpose() * X).ldlt().solve(X.transpose() * y_star);

    result.residuals = y_star - X * result.beta;
    result.sigma2 = result.residuals.squaredNorm() / n;

    result.fitted_values = result.rho * Wy + X * result.beta;

    // Standard errors
    compute_sar_standard_errors(y, X, W, result);

    // Log-likelihood
    double log_det = 0;
    for (int i = 0; i < n; ++i)
      log_det += std::log(1.0 - result.rho * lambda(i));

    result.log_likelihood =
        -n / 2.0 * std::log(2 * M_PI * result.sigma2) + log_det -
        result.residuals.squaredNorm() / (2 * result.sigma2);

    result.n_params = k + 2;
    result.aic = -2 * result.log_likelihood + 2 * result.n_params;
    result.bic = -2 * result.log_likelihood + result.n_params * std::log(n);

    // Coefficients
    result.coef.resize(k + 1);
    result.coef(0) = result.rho;
    result.coef.tail(k) = result.beta;

    result.std_errors.resize(k + 1);
    result.std_errors(0) = result.rho_se;
    result.std_errors.tail(k) = result.beta_se;

    result.z_values = result.coef.array() / result.std_errors.array();
    result.p_values.resize(k + 1);
    for (int j = 0; j <= k; ++j) {
      result.p_values(j) = 2 * (1 - normal_cdf(std::abs(result.z_values(j))));
    }

    result.converged = true;
    result.lambda = 0;
  }

  double optimize_rho_sar_eigen(const Eigen::VectorXd &y,
                                const Eigen::MatrixXd &X,
                                const Eigen::MatrixXd &W,
                                const Eigen::VectorXd &Wy,
                                const Eigen::VectorXd &lambda, double rho_min,
                                double rho_max) {
    int n = y.size();

    double a = std::max(-0.999, rho_min + 0.001);
    double b = std::min(0.999, rho_max - 0.001);
    double gr = (std::sqrt(5) - 1) / 2;

    double c = b - gr * (b - a);
    double d = a + gr * (b - a);

    // Pre-compute projections
    // M = I - X(X'X)^-1 X'
    // We only need e = My and e_L = MWy
    // M projects onto orthogonal complement of X
    // beta_0 = (X'X)^-1 X'y, e_0 = y - X*beta_0
    // beta_L = (X'X)^-1 X'Wy, e_L = Wy - X*beta_L
    // resid(rho) = e_0 - rho * e_L

    Eigen::VectorXd beta_0 =
        (X.transpose() * X).ldlt().solve(X.transpose() * y);
    Eigen::VectorXd e_0 = y - X * beta_0;

    Eigen::VectorXd beta_L =
        (X.transpose() * X).ldlt().solve(X.transpose() * Wy);
    Eigen::VectorXd e_L = Wy - X * beta_L;

    auto conc_ll = [&](double rho) {
      // Check determinant condition first
      double log_det = 0;
      for (int i = 0; i < n; ++i) {
        double val = 1.0 - rho * lambda(i);
        if (val <= 0)
          return -1e9; // Invalid rho
        log_det += std::log(val);
      }

      // Concentrated sigma^2
      Eigen::VectorXd e = e_0 - rho * e_L;
      double s2 = e.squaredNorm() / n;

      return -n / 2.0 * std::log(s2) + log_det;
    };

    double fc = conc_ll(c);
    double fd = conc_ll(d);

    for (int iter = 0; iter < max_iter; ++iter) {
      if (std::abs(b - a) < tol)
        break;

      if (fc > fd) {
        b = d;
        d = c;
        fd = fc;
        c = b - gr * (b - a);
        fc = conc_ll(c);
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + gr * (b - a);
        fd = conc_ll(d);
      }
    }

    return (a + b) / 2;
  }

  void fit_sem(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
               const Eigen::MatrixXd &W, double lambda_min, double lambda_max,
               SpatialResult &result) {
    int n = y.size();
    int k = X.cols();

    // Optimize lambda
    result.lambda = optimize_lambda_sem(y, X, W, lambda_min, lambda_max);

    // GLS estimation given lambda
    Eigen::MatrixXd I_lamW =
        Eigen::MatrixXd::Identity(n, n) - result.lambda * W;
    Eigen::VectorXd y_t = I_lamW * y;
    Eigen::MatrixXd X_t = I_lamW * X;

    result.beta = (X_t.transpose() * X_t).ldlt().solve(X_t.transpose() * y_t);
    result.residuals = y - X * result.beta;
    result.sigma2 = result.residuals.squaredNorm() / n;

    result.fitted_values = X * result.beta;

    // Standard errors
    result.beta_se = result.sigma2 * (X_t.transpose() * X_t)
                                         .ldlt()
                                         .solve(Eigen::MatrixXd::Identity(k, k))
                                         .diagonal()
                                         .cwiseSqrt();
    result.lambda_se = 0.1; // Placeholder - would need Hessian

    // Log-likelihood
    double log_det = std::log(std::abs(I_lamW.determinant()));
    result.log_likelihood =
        -n / 2.0 * std::log(2 * M_PI * result.sigma2) + log_det -
        (I_lamW * result.residuals).squaredNorm() / (2 * result.sigma2);

    result.n_params = k + 2;
    result.aic = -2 * result.log_likelihood + 2 * result.n_params;
    result.bic = -2 * result.log_likelihood + result.n_params * std::log(n);

    result.coef.resize(k + 1);
    result.coef(0) = result.lambda;
    result.coef.tail(k) = result.beta;

    result.std_errors.resize(k + 1);
    result.std_errors(0) = result.lambda_se;
    result.std_errors.tail(k) = result.beta_se;

    result.z_values = result.coef.array() / result.std_errors.array();
    result.p_values.resize(k + 1);
    for (int j = 0; j <= k; ++j) {
      result.p_values(j) = 2 * (1 - normal_cdf(std::abs(result.z_values(j))));
    }

    result.rho = 0;
    result.converged = true;
  }

  double optimize_lambda_sem(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                             const Eigen::MatrixXd &W, double lambda_min,
                             double lambda_max) {
    int n = y.size();

    double a = std::max(-0.999, lambda_min + 0.001);
    double b = std::min(0.999, lambda_max - 0.001);
    double gr = (std::sqrt(5) - 1) / 2;

    double c = b - gr * (b - a);
    double d = a + gr * (b - a);

    auto conc_ll = [&](double lam) {
      Eigen::MatrixXd I_lamW = Eigen::MatrixXd::Identity(n, n) - lam * W;
      Eigen::VectorXd y_t = I_lamW * y;
      Eigen::MatrixXd X_t = I_lamW * X;

      Eigen::VectorXd beta =
          (X_t.transpose() * X_t).ldlt().solve(X_t.transpose() * y_t);
      Eigen::VectorXd resid = y_t - X_t * beta;
      double s2 = resid.squaredNorm() / n;

      double log_det = std::log(std::abs(I_lamW.determinant()));

      return -n / 2.0 * std::log(s2) + log_det;
    };

    double fc = conc_ll(c);
    double fd = conc_ll(d);

    for (int iter = 0; iter < max_iter; ++iter) {
      if (std::abs(b - a) < tol)
        break;

      if (fc > fd) {
        b = d;
        d = c;
        fd = fc;
        c = b - gr * (b - a);
        fc = conc_ll(c);
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + gr * (b - a);
        fd = conc_ll(d);
      }
    }

    return (a + b) / 2;
  }

  void fit_sdm(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
               const Eigen::MatrixXd &W, const Eigen::VectorXd &Wy,
               double rho_min, double rho_max, SpatialResult &result) {
    int n = (int)y.size();
    int k = (int)X.cols();

    // SDM: y = ρWy + Xβ + WXθ + ε
    // Augment X with WX
    Eigen::MatrixXd WX = W * X;
    Eigen::MatrixXd X_aug(n, 2 * k);
    X_aug.leftCols(k) = X;
    X_aug.rightCols(k) = WX;

    // Fit as SAR with augmented X
    Eigen::VectorXd Wy_copy = Wy;
    fit_sar(y, X_aug, W, Wy_copy, rho_min, rho_max, result);

    // Split coefficients
    Eigen::VectorXd beta_full = result.beta;
    result.beta = beta_full.head(k);
    result.theta = beta_full.tail(k);

    result.beta_se = result.std_errors.segment(1, k);
    result.theta_se = result.std_errors.tail(k);

    result.model_type = SpatialModel::SDM;
  }

  void compute_sar_standard_errors(const Eigen::VectorXd &y,
                                   const Eigen::MatrixXd &X,
                                   const Eigen::MatrixXd &W,
                                   SpatialResult &result) {
    int n = y.size();
    int k = X.cols();

    // Approximate standard errors
    result.beta_se = result.sigma2 * (X.transpose() * X)
                                         .ldlt()
                                         .solve(Eigen::MatrixXd::Identity(k, k))
                                         .diagonal()
                                         .cwiseSqrt();

    // Crude rho SE (would need proper Hessian)
    result.rho_se = std::sqrt(result.sigma2 / n) * 0.1;
  }

  void compute_effects(const Eigen::MatrixXd &W, SpatialResult &result) {
    int n = W.rows();
    int k = result.beta.size();

    result.direct_effects.resize(k);
    result.indirect_effects.resize(k);
    result.total_effects.resize(k);

    // S = (I - ρW)^{-1}
    Eigen::MatrixXd I_rhoW = Eigen::MatrixXd::Identity(n, n) - result.rho * W;
    Eigen::MatrixXd S = I_rhoW.ldlt().solve(Eigen::MatrixXd::Identity(n, n));

    for (int j = 0; j < k; ++j) {
      // Total effect = β_j * mean(S)
      result.total_effects(j) = result.beta(j) * S.mean();

      // Direct effect = β_j * mean(diag(S))
      result.direct_effects(j) = result.beta(j) * S.diagonal().mean();

      // Indirect = Total - Direct
      result.indirect_effects(j) =
          result.total_effects(j) - result.direct_effects(j);
    }
  }

  double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
  }

  double chi2_cdf(double x, int df) {
    if (df <= 0 || x <= 0)
      return 0;
    double a = df / 2.0;
    double g = lgamma(a);
    double sum = 1.0 / a;
    double term = sum;
    for (int n = 1; n < 200; ++n) {
      term *= (x / 2.0) / (a + n);
      sum += term;
      if (std::abs(term) < 1e-12 * std::abs(sum))
        break;
    }
    return std::exp(a * std::log(x / 2.0) - x / 2.0 - g) * sum;
  }
};

} // namespace statelix

#endif // STATELIX_SPATIAL_H
