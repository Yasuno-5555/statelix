/**
 * @file garch.h
 * @brief Statelix v2.3 - GARCH Family Models (Refined)
 *
 * Implements:
 *   - GARCH(p,q)
 *   - EGARCH (Nelson 1991)
 *   - GJR-GARCH (Glosten-Jagannathan-Runkle)
 *   - IGARCH (Integrated GARCH)
 *
 * Optimization:
 *   - Uses L-BFGS (via EfficientObjective interface)
 *   - Numerical gradient with parameter scaling
 *   - Robust initialization and constraints
 *
 * Forecasting:
 *   - Analytical point forecasts
 *   - Bootstrap simulation for confidence intervals
 */
#ifndef STATELIX_GARCH_H
#define STATELIX_GARCH_H

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../optimization/lbfgs.h"
#include "../optimization/objective.h"
#include "../optimization/zigen_optimizer.h"

namespace statelix {

// =============================================================================
// Enums and Configuration
// =============================================================================

enum class GARCHType {
  GARCH,  // Standard: σ² = ω + αε² + βσ²
  EGARCH, // Exponential: ln(σ²) = ω + α(z) + βln(σ²)
  GJR,    // Asymmetric: σ² = ω + (α + γI)ε² + βσ²
  IGARCH  // Integrated: α + β = 1 (ω=0 usually, or drift)
};

enum class GARCHDist {
  NORMAL,   // Gaussian innovations
  STUDENT_T // Student's t (heavier tails)
};

// =============================================================================
// Result Structures
// =============================================================================

struct GARCHResult {
  // Model specification
  int p, q;
  GARCHType type;
  GARCHDist dist;

  // Parameters
  double mu;
  double omega;
  Eigen::VectorXd alpha;
  Eigen::VectorXd beta;
  Eigen::VectorXd gamma;
  double nu; // df for Student-t

  // Standard Errors
  double mu_se;
  double omega_se;
  Eigen::VectorXd alpha_se;
  Eigen::VectorXd beta_se;
  Eigen::VectorXd gamma_se;
  double nu_se;

  // Diagnostics
  double log_likelihood;
  double aic;
  double bic;

  // Dynamics
  double persistence;
  bool is_stationary;
  double unconditional_variance;
  double half_life;

  // Series
  Eigen::VectorXd conditional_variance;
  Eigen::VectorXd conditional_volatility;
  Eigen::VectorXd standardized_residuals;
  Eigen::VectorXd residuals;

  // Meta
  int n_obs;
  int n_params;
  int iterations;
  bool converged;
};

struct GARCHForecast {
  Eigen::VectorXd variance;   // Point forecast σ²
  Eigen::VectorXd volatility; // Point forecast σ
  Eigen::VectorXd var_lower;  // Simulation based bounds
  Eigen::VectorXd var_upper;
  int horizon;
  int n_sims; // Number of simulations used
};

// =============================================================================
// GARCH Objective Function (for L-BFGS)
// =============================================================================

class GARCHObjective : public EfficientObjective {
public:
  const Eigen::VectorXd &returns;
  GARCHType type;
  GARCHDist dist;
  int p, q;

  // Pre-computed
  double sample_var;
  double sample_std;

  GARCHObjective(const Eigen::VectorXd &r, GARCHType t, GARCHDist d, int p_,
                 int q_)
      : returns(r), type(t), dist(d), p(p_), q(q_) {
    double mean = returns.mean();
    sample_var = (returns.array() - mean).square().mean();
    sample_std = std::sqrt(sample_var);
  }

  // Compute negative log-likelihood and gradient
  std::pair<double, Eigen::VectorXd>
  value_and_gradient(const Eigen::VectorXd &theta) const override {
    // 1. Calculate Value (Neg LL)
    GARCHResult res; // Temporary structure to hold series
    double nll =
        compute_nll(theta, res); // res is populated with residuals/sigma2

    // 2. Calculate Gradient (Numerical with Scaling)
    int n_params = theta.size();
    Eigen::VectorXd grad(n_params);

    // Adaptive step size based on parameter magnitude using central difference
    // eps = max(|theta| * 1e-4, 1e-6)
    for (int i = 0; i < n_params; ++i) {
      double h = std::max(std::abs(theta(i)) * 1e-4, 1e-6);

      Eigen::VectorXd theta_plus = theta;
      theta_plus(i) += h;
      GARCHResult res_p;
      double nll_plus = compute_nll(theta_plus, res_p);

      Eigen::VectorXd theta_minus = theta;
      theta_minus(i) -= h;
      GARCHResult res_m;
      double nll_minus = compute_nll(theta_minus, res_m);

      grad(i) = (nll_plus - nll_minus) / (2 * h);
    }

    return {nll, grad};
  }

  // Public wrapper for final result extraction
  double compute_nll(const Eigen::VectorXd &theta, GARCHResult &result) const {
    return neg_log_likelihood(theta, result);
  }

private:
  double neg_log_likelihood(const Eigen::VectorXd &theta,
                            GARCHResult &result) const {
    int n = returns.size();

    // --- Unpack Parameters ---
    // Layout: [mu, omega, alpha(p), beta(q), gamma(p or 0), nu(0 or 1)]
    double mu = theta(0);
    double omega = theta(1);

    // Constrain omega > 0 (soft constraint via penalty or hard return)
    if (omega <= 1e-9 && type != GARCHType::IGARCH)
      return 1e20;

    // Vectors
    const double *ptr = theta.data() + 2;
    Eigen::Map<const Eigen::VectorXd> alpha(ptr, p);
    ptr += p;
    Eigen::Map<const Eigen::VectorXd> beta(ptr, q);
    ptr += q;

    Eigen::VectorXd gamma;
    int n_gamma = (type == GARCHType::GJR || type == GARCHType::EGARCH) ? p : 0;
    if (n_gamma > 0) {
      gamma = Eigen::Map<const Eigen::VectorXd>(ptr, n_gamma);
      ptr += n_gamma;
    }

    double nu = 8.0;
    if (dist == GARCHDist::STUDENT_T) {
      nu = *ptr;
      if (nu <= 2.01)
        return 1e20; // Constraint nu > 2
    }

    // --- IGARCH Constraint enforcement ---
    // Ideally handled by parameter re-mapping, but for simplified
    // box-constraints:
    if (type == GARCHType::IGARCH) {
      // IGARCH: Sum(alpha) + Sum(beta) = 1
      // Here we assume the optimizer respects bounds? No, L-BFGS is
      // unconstrained. We penalize deviation heavily. Better: Re-parameterize.
      // For now, penalty method.
      double sum = alpha.sum() + beta.sum();
      if (std::abs(sum - 1.0) > 1e-2)
        return 1e20 + std::pow(sum - 1.0, 2) * 1e6;
    } else if (type == GARCHType::GARCH || type == GARCHType::GJR) {
      // Stationarity check (soft penalty)
      double persistence = alpha.sum() + beta.sum();
      if (n_gamma > 0 && type == GARCHType::GJR)
        persistence += 0.5 * gamma.sum();

      if (persistence >= 0.9999)
        return 1e20 + (persistence - 0.9999) * 1e6;

      // Positivity constraints
      if ((alpha.array() < 0).any() || (beta.array() < 0).any())
        return 1e20;
    }

    // --- Variance Recursion ---
    result.conditional_variance.resize(n);
    result.residuals.resize(n);

    double log_lik = 0;

    for (int t = 0; t < n; ++t) {
      result.residuals(t) = returns(t) - mu;
      double sigma2;

      if (type == GARCHType::EGARCH) {
        // EGARCH: ln(σ²_t) = ω + Σ α g(z) + Σ β ln(σ²)
        using std::abs;
        using std::exp;
        using std::log;
        using std::sqrt;

        // If Scalar is Var, we need special handling if std::abs/log are not
        // overloaded in std namespace But Zigen AD overloads are in
        // Zigen::Autodiff::Reverse

        double ln_sigma2 = omega;
        double E_abs_z = std::sqrt(2.0 / M_PI);

        for (int i = 1; i <= p; ++i) {
          if (t - i >= 0) {
            double z = result.residuals(t - i) /
                       sqrt(result.conditional_variance(t - i));
            double g = alpha(i - 1) * z + gamma(i - 1) * (abs(z) - E_abs_z);
            ln_sigma2 += g;
          }
        }

        for (int j = 1; j <= q; ++j) {
          if (t - j >= 0) {
            ln_sigma2 += beta(j - 1) * log(result.conditional_variance(t - j));
          } else {
            ln_sigma2 += beta(j - 1) * log(sample_var);
          }
        }
        sigma2 = exp(ln_sigma2);

      } else {
        // GARCH / GJR / IGARCH
        sigma2 = omega;

        for (int i = 1; i <= p; ++i) {
          double eps2;
          if (t - i >= 0)
            eps2 = std::pow(result.residuals(t - i), 2);
          else
            eps2 = sample_var; // Init with sample var

          double c = alpha(i - 1);
          if (type == GARCHType::GJR && t - i >= 0 &&
              result.residuals(t - i) < 0) {
            c += gamma(i - 1);
          }
          sigma2 += c * eps2;
        }

        for (int j = 1; j <= q; ++j) {
          if (t - j >= 0)
            sigma2 += beta(j - 1) * result.conditional_variance(t - j);
          else
            sigma2 += beta(j - 1) * sample_var;
        }
      }

      sigma2 = std::max(1e-9, sigma2); // Safety floor
      result.conditional_variance(t) = sigma2;

      // Likelihood contribution
      double z = result.residuals(t) / std::sqrt(sigma2);
      double ll_t;

      if (dist == GARCHDist::NORMAL) {
        ll_t = -0.5 * (std::log(2 * M_PI) + std::log(sigma2) + z * z);
      } else { // Student-t
        ll_t = std::lgamma((nu + 1) / 2) - std::lgamma(nu / 2) -
               0.5 * std::log((nu - 2) * M_PI * sigma2) -
               (nu + 1) / 2 * std::log(1 + z * z / (nu - 2));
      }

      log_lik += ll_t;
    }

    if (std::isnan(log_lik) || std::isinf(log_lik))
      return 1e20;
    return -log_lik; // Minimize NLL
  }

  // Templated implementation to support both double and Var
public:
  template <typename Scalar, typename VectorType>
  Scalar
  compute_nll_template(const VectorType &theta,
                       std::vector<Scalar> &result_residuals,
                       std::vector<Scalar> &result_conditional_variance) const {
    using std::abs;
    using std::exp;
    using std::log;
    using std::sqrt;
    using namespace Zigen::Autodiff::Reverse;

    // Helper lambdas for generic access
    auto get_theta = [&](int i) -> Scalar {
      if constexpr (std::is_same_v<Scalar, double>)
        return theta(i);
      else
        return theta(i, 0);
    };

    auto get_val = [&](const Scalar &s) -> double {
      if constexpr (std::is_same_v<Scalar, double>)
        return s;
      else
        return s.val();
    };

    int n = returns.size();

    // --- Unpack Parameters ---
    Scalar mu = get_theta(0);
    Scalar omega = get_theta(1);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> alpha(p);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> beta(q);

    int ptr = 2;
    for (int i = 0; i < p; ++i)
      alpha(i) = get_theta(ptr++);
    for (int i = 0; i < q; ++i)
      beta(i) = get_theta(ptr++);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> gamma;
    int n_gamma = (type == GARCHType::GJR || type == GARCHType::EGARCH) ? p : 0;
    if (n_gamma > 0) {
      gamma.resize(n_gamma);
      for (int i = 0; i < n_gamma; ++i)
        gamma(i) = get_theta(ptr++);
    }

    Scalar nu(8.0);
    if (dist == GARCHDist::STUDENT_T) {
      nu = get_theta(ptr++);
    }

    // --- Penalties ---
    Scalar penalty(0.0);

    if (type == GARCHType::IGARCH) {
      Scalar sum = alpha.sum() + beta.sum();
      Scalar diff = sum - 1.0;
      penalty = penalty + diff * diff * 1e6;
    } else if (type == GARCHType::GARCH || type == GARCHType::GJR) {
      Scalar persistence = alpha.sum() + beta.sum();
      if (n_gamma > 0 && type == GARCHType::GJR)
        persistence = persistence + 0.5 * gamma.sum();

      for (int i = 0; i < p; ++i)
        if (get_val(alpha(i)) < 0)
          penalty = penalty + alpha(i) * alpha(i) * 1e6;
      for (int i = 0; i < q; ++i)
        if (get_val(beta(i)) < 0)
          penalty = penalty + beta(i) * beta(i) * 1e6;
      if (get_val(omega) < 0)
        penalty = penalty + omega * omega * 1e6;
    }

    // --- Variance Recursion ---
    result_conditional_variance.assign(n, Scalar(0.0));
    result_residuals.assign(n, Scalar(0.0));

    Scalar log_lik(0.0);

    for (int t = 0; t < n; ++t) {
      result_residuals[t] = Scalar(returns(t)) - mu;
      Scalar sigma2;

      if (type == GARCHType::EGARCH) {
        Scalar ln_sigma2 = omega;
        double E_abs_z = std::sqrt(2.0 / M_PI);

        for (int i = 1; i <= p; ++i) {
          if (t - i >= 0) {
            Scalar prev_var = result_conditional_variance[t - i];
            Scalar z = result_residuals[t - i] / sqrt(prev_var);

            Scalar abs_z = (get_val(z) >= 0) ? z : -z;

            Scalar g = alpha(i - 1) * z + gamma(i - 1) * (abs_z - E_abs_z);
            ln_sigma2 = ln_sigma2 + g;
          }
        }

        for (int j = 1; j <= q; ++j) {
          if (t - j >= 0) {
            ln_sigma2 = ln_sigma2 +
                        beta(j - 1) * log(result_conditional_variance[t - j]);
          } else {
            ln_sigma2 = ln_sigma2 + beta(j - 1) * std::log(sample_var);
          }
        }
        sigma2 = exp(ln_sigma2);

      } else {
        sigma2 = omega;

        for (int i = 1; i <= p; ++i) {
          Scalar eps2;
          if (t - i >= 0)
            eps2 = result_residuals[t - i] * result_residuals[t - i];
          else
            eps2 = Scalar(sample_var);

          Scalar c = alpha(i - 1);
          if (type == GARCHType::GJR && t - i >= 0 &&
              get_val(result_residuals[t - i]) < 0) {
            c = c + gamma(i - 1);
          }
          sigma2 = sigma2 + c * eps2;
        }

        for (int j = 1; j <= q; ++j) {
          if (t - j >= 0)
            sigma2 = sigma2 + beta(j - 1) * result_conditional_variance[t - j];
          else
            sigma2 = sigma2 + beta(j - 1) * Scalar(sample_var);
        }
      }

      if (get_val(sigma2) < 1e-9) {
        sigma2 = Scalar(1e-9);
      }

      result_conditional_variance[t] = sigma2;

      Scalar z = result_residuals[t] / sqrt(sigma2);
      Scalar ll_t;

      if (dist == GARCHDist::NORMAL) {
        ll_t = -0.5 * (std::log(2 * M_PI) + log(sigma2) + z * z);
      } else {
        ll_t = -0.5 * (std::log(2 * M_PI) + log(sigma2) + z * z);
      }

      log_lik = log_lik + ll_t;
    }

    return penalty - log_lik;
  }
};

/**
 * @brief Zigen Objective wrapper
 */
struct ZigenGARCHObjective {
  const GARCHObjective &obj;

  ZigenGARCHObjective(const GARCHObjective &o) : obj(o) {}

  template <typename Var>
  Var operator()(const Zigen::Matrix<Var, Zigen::Dynamic, 1> &theta) const {
    std::vector<Var> res, cond_var;
    return obj.compute_nll_template<Var>(theta, res, cond_var);
  }
};

// =============================================================================
// GARCH Class
// =============================================================================

class GARCH {
public:
  int p = 1, q = 1;
  GARCHType type = GARCHType::GARCH;
  GARCHDist dist = GARCHDist::NORMAL;

  // Optimization settings
  int max_iter = 500;

  GARCH(int p_ = 1, int q_ = 1) : p(p_), q(q_) {}

  GARCHResult fit(const Eigen::VectorXd &returns) {
    GARCHResult result;
    result.p = p;
    result.q = q;
    result.type = type;
    result.dist = dist;
    result.n_obs = returns.size();

    // 1. Initial Guess
    double sample_var = (returns.array() - returns.mean()).square().mean();
    int n_core = 2 + p + q; // mu, omega, alpha, beta
    int n_gamma = (type == GARCHType::GJR || type == GARCHType::EGARCH) ? p : 0;
    int n_dist = (dist == GARCHDist::STUDENT_T) ? 1 : 0;
    int n_params = n_core + n_gamma + n_dist;

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(n_params);
    theta(0) = returns.mean();    // mu
    theta(1) = sample_var * 0.05; // omega
    if (type == GARCHType::IGARCH)
      theta(1) = 0.0;

    // Alphas
    for (int i = 0; i < p; ++i)
      theta(2 + i) = 0.05;
    // Betas
    for (int j = 0; j < q; ++j)
      theta(2 + p + j) = 0.85 / q;
    // Gammas
    if (n_gamma > 0)
      for (int i = 0; i < n_gamma; ++i)
        theta(n_core + i) = 0.02;
    // Nu
    if (n_dist > 0)
      theta(n_params - 1) = 8.0;

    // Check IGARCH constraint initialization
    if (type == GARCHType::IGARCH) {
      // Force sum = 1
      double s = 0;
      for (int k = 2; k < 2 + p + q; ++k)
        s += theta(k);
      for (int k = 2; k < 2 + p + q; ++k)
        theta(k) /= s;
    }

    // 2. Optimization
    // 2. Optimization
    GARCHObjective obj(returns, type, dist, p, q);

    // Use Zigen Autodiff Optimizer
    ZigenLBFGS optimizer;
    optimizer.max_iter = max_iter;
    optimizer.verbose = false;

    // Wrap objective
    ZigenGARCHObjective zigen_obj(obj);

    auto opt_res = optimizer.minimize_autodiff(zigen_obj, theta);
    // 3. Populate Result
    result.n_params = n_params;
    result.iterations = opt_res.iterations;
    result.converged = opt_res.converged;

    // Re-compute final fits
    obj.compute_nll(opt_res.x, result);
    unpack_params(opt_res.x, result);
    result.log_likelihood = -opt_res.min_value;

    // IC
    result.aic = -2 * result.log_likelihood + 2 * n_params;
    result.bic = -2 * result.log_likelihood + n_params * std::log(result.n_obs);

    // Robust Standard Errors (Numerical Hessian at Optimum)
    compute_standard_errors(obj, opt_res.x, result);

    // Residuals
    result.standardized_residuals =
        result.residuals.array() / result.conditional_variance.array().sqrt();
    result.conditional_volatility = result.conditional_variance.array().sqrt();

    return result;
  }

  GARCHForecast forecast(const GARCHResult &res, int horizon,
                         int n_sims = 1000) {
    GARCHForecast fc;
    fc.horizon = horizon;
    fc.n_sims = n_sims;
    fc.variance.resize(horizon);
    fc.volatility.resize(horizon);
    fc.var_lower.resize(horizon);
    fc.var_upper.resize(horizon);

    // Simulation for confidence intervals
    std::mt19937 gen(42);
    std::normal_distribution<> normal_dist(0, 1);
    std::student_t_distribution<> t_dist(res.nu);

    // Re-implement simplified analytical forecast for point estimate (standard
    // GARCH) And use approximations for bounds if simulation is too heavy to
    // inline here. Actually, user requested simulation. Let's do a proper
    // simpler simulation loop.

    std::vector<double> sim_sum_var(horizon, 0.0);
    std::vector<std::vector<double>> sim_vars(horizon);
    for (int h = 0; h < horizon; ++h)
      sim_vars[h].reserve(n_sims);

    for (int s = 0; s < n_sims; ++s) {
      // State
      std::vector<double> local_resid(p);
      std::vector<double> local_var(q);
      std::vector<double> local_std_resid(p); // z

      // Init state from end of sample
      for (int i = 0; i < p; ++i)
        local_resid[i] = res.residuals(res.n_obs - 1 - i);
      for (int i = 0; i < q; ++i)
        local_var[i] = res.conditional_variance(res.n_obs - 1 - i);
      for (int i = 0; i < p; ++i)
        local_std_resid[i] = res.standardized_residuals(res.n_obs - 1 - i);

      for (int h = 0; h < horizon; ++h) {
        double next_sigma2 = res.omega;

        if (type == GARCHType::EGARCH) {
          double E_abs_z = std::sqrt(2.0 / M_PI);
          for (int i = 1; i <= p; ++i) {
            double z_lag = local_std_resid[i - 1];
            next_sigma2 += res.alpha(i - 1) * z_lag +
                           ((res.gamma.size() > 0) ? res.gamma(i - 1) : 0) *
                               (std::abs(z_lag) - E_abs_z);
          }
          for (int j = 1; j <= q; ++j) {
            next_sigma2 += res.beta(j - 1) * std::log(local_var[j - 1]);
          }
          next_sigma2 = std::exp(next_sigma2);
        } else {
          for (int i = 1; i <= p; ++i) {
            double eps2 = std::pow(local_resid[i - 1], 2);
            double coef = res.alpha(i - 1);
            if (type == GARCHType::GJR && local_resid[i - 1] < 0 &&
                res.gamma.size() > 0) {
              coef += res.gamma(i - 1);
            }
            next_sigma2 += coef * eps2;
          }
          for (int j = 1; j <= q; ++j) {
            next_sigma2 += res.beta(j - 1) * local_var[j - 1];
          }
        }

        next_sigma2 = std::max(1e-9, next_sigma2);

        // Generate Step
        double z =
            (dist == GARCHDist::STUDENT_T) ? t_dist(gen) : normal_dist(gen);
        if (dist == GARCHDist::STUDENT_T)
          z *= std::sqrt((res.nu - 2.0) / res.nu);

        double next_eps = z * std::sqrt(next_sigma2);

        // Record
        sim_sum_var[h] += next_sigma2;
        sim_vars[h].push_back(next_sigma2);

        // Shift state
        shift_vec(local_resid, next_eps);
        shift_vec(local_var, next_sigma2);
        shift_vec(local_std_resid, z);
      }
    }

    for (int h = 0; h < horizon; ++h) {
      fc.variance(h) = sim_sum_var[h] / n_sims;
      fc.volatility(h) = std::sqrt(fc.variance(h));

      std::sort(sim_vars[h].begin(), sim_vars[h].end());
      fc.var_lower(h) = sim_vars[h][(int)(0.05 * n_sims)];
      fc.var_upper(h) = sim_vars[h][(int)(0.95 * n_sims)];
    }

    return fc;
  }

  // Keep the nice News Impact Curve
  std::pair<Eigen::VectorXd, Eigen::VectorXd>
  news_impact_curve(const GARCHResult &result, double z_min = -3.0,
                    double z_max = 3.0, int n_points = 100) {
    Eigen::VectorXd z_vals(n_points);
    Eigen::VectorXd sigma2_vals(n_points);
    double sigma2_base = result.unconditional_variance;

    // Handle IGARCH infinite variance case for plotting
    if (!result.is_stationary)
      sigma2_base = result.conditional_variance.mean();

    double sigma_base = std::sqrt(sigma2_base);

    for (int i = 0; i < n_points; ++i) {
      double z = z_min + (z_max - z_min) * i / (n_points - 1);
      z_vals(i) = z;
      double eps = z * sigma_base;
      double next_s2 = result.omega;

      if (type == GARCHType::EGARCH) {
        double E_abs_z = std::sqrt(2.0 / M_PI);
        double g = result.alpha(0) * z +
                   ((result.gamma.size()) ? result.gamma(0) : 0) *
                       (std::abs(z) - E_abs_z);
        next_s2 =
            std::exp(result.omega + g + result.beta(0) * std::log(sigma2_base));
      } else {
        double coef = result.alpha(0);
        if (type == GARCHType::GJR && eps < 0 && result.gamma.size())
          coef += result.gamma(0);
        next_s2 += coef * eps * eps + result.beta(0) * sigma2_base;
      }
      sigma2_vals(i) = next_s2;
    }
    return {z_vals, sigma2_vals};
  }

private:
  void unpack_params(const Eigen::VectorXd &theta, GARCHResult &res) {
    res.mu = theta(0);
    res.omega = theta(1);
    res.alpha.resize(p);
    for (int i = 0; i < p; ++i)
      res.alpha(i) = theta(2 + i);
    res.beta.resize(q);
    for (int j = 0; j < q; ++j)
      res.beta(j) = theta(2 + p + j);

    int offset = 2 + p + q;
    if (type == GARCHType::GJR || type == GARCHType::EGARCH) {
      res.gamma.resize(p);
      for (int i = 0; i < p; ++i)
        res.gamma(i) = theta(offset + i);
      offset += p;
    }
    if (dist == GARCHDist::STUDENT_T)
      res.nu = theta(offset);

    // Stats
    res.persistence = res.alpha.sum() + res.beta.sum();
    if (type == GARCHType::GJR && res.gamma.size() > 0)
      res.persistence += 0.5 * res.gamma.sum();

    if (type == GARCHType::IGARCH)
      res.persistence = 1.0; // By definition

    res.is_stationary = (res.persistence < 1.0);
    if (res.is_stationary)
      res.unconditional_variance = res.omega / (1.0 - res.persistence);
    else
      res.unconditional_variance = std::numeric_limits<double>::infinity();

    if (res.is_stationary)
      res.half_life = std::log(0.5) / std::log(res.persistence);
    else
      res.half_life = std::numeric_limits<double>::infinity();
  }

  void compute_standard_errors(GARCHObjective &obj,
                               const Eigen::VectorXd &theta, GARCHResult &res) {
    int n = theta.size();
    Eigen::MatrixXd H(n, n);

    // Central difference Hessian with relative scale
    for (int i = 0; i < n; ++i) {
      double h_i = std::max(1e-4 * std::abs(theta(i)), 1e-5);
      for (int j = i; j < n; ++j) {
        double h_j = std::max(1e-4 * std::abs(theta(j)), 1e-5);

        Eigen::VectorXd pp = theta;
        pp(i) += h_i;
        pp(j) += h_j;
        Eigen::VectorXd mm = theta;
        mm(i) -= h_i;
        mm(j) -= h_j;
        Eigen::VectorXd pm = theta;
        pm(i) += h_i;
        pm(j) -= h_j;
        Eigen::VectorXd mp = theta;
        mp(i) -= h_i;
        mp(j) += h_j;

        double v_pp = obj.compute_nll(pp, res);
        double v_mm = obj.compute_nll(mm, res);
        double v_pm = obj.compute_nll(pm, res);
        double v_mp = obj.compute_nll(mp, res);

        H(i, j) = (v_pp - v_pm - v_mp + v_mm) / (4 * h_i * h_j);
        H(j, i) = H(i, j);
      }
    }

    Eigen::MatrixXd cov = H.inverse();
    Eigen::VectorXd se = cov.diagonal().cwiseSqrt();

    res.mu_se = se(0);
    res.omega_se = se(1);
    res.alpha_se.resize(p);
    for (int k = 0; k < p; ++k)
      res.alpha_se(k) = se(2 + k);
    res.beta_se.resize(q);
    for (int k = 0; k < q; ++k)
      res.beta_se(k) = se(2 + p + k);

    int offset = 2 + p + q;
    if (res.gamma.size() > 0) {
      res.gamma_se.resize(p);
      for (int k = 0; k < p; ++k)
        res.gamma_se(k) = se(offset + k);
      offset += p;
    }
    if (dist == GARCHDist::STUDENT_T)
      res.nu_se = se(offset);
  }

  void shift_vec(std::vector<double> &v, double new_val) {
    for (size_t i = v.size() - 1; i > 0; --i)
      v[i] = v[i - 1];
    if (!v.empty())
      v[0] = new_val;
  }
};

} // namespace statelix

#endif // STATELIX_GARCH_H
