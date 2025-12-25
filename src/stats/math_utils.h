#ifndef STATELIX_MATH_UTILS_H
#define STATELIX_MATH_UTILS_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// Vendor integration
#include <MathUniverse/Sokudo/Backends/ZigenCpuBackend.hpp>
#include <MathUniverse/Sokudo/Core/distributions.hpp>

namespace statelix {
namespace stats {

// Internal aliases for convenience
using Backend = Sokudo::ZigenCpuBackend;
using Vector = Backend::VectorType;

// --- Scalar Math Utility Overloads (Legacy compatibility) ---

inline double beta_inc(double a, double b, double x) {
  return Sokudo::Math::beta_inc(a, b, x);
}

inline double gamma_inc(double a, double x) {
  return Sokudo::Math::gamma_inc(a, x);
}

inline double chi2_cdf(double x, int df) {
  return Sokudo::ChiSquare<double>(double(df)).cdf(x);
}

inline double f_cdf(double f, int df1, int df2) {
  return Sokudo::FDistribution<double>(double(df1), double(df2)).cdf(f);
}

inline double f_pvalue_approx(double f, int df1, int df2) {
  return 1.0 - f_cdf(f, df1, df2);
}

inline double erfinv(double x) { return Sokudo::Math::erfinv(x); }

inline double normal_cdf(double x) {
  return Sokudo::StandardNormal<double>::cdf(x);
}

inline double normal_quantile(double p) {
  return Sokudo::Math::normal_quantile(p);
}

inline double t_cdf(double t, int df) {
  return Sokudo::StudentT<double>(double(df)).cdf(t);
}

inline double t_quantile(double p, int df) {
  // Use StudentT quantile if implemented, or fallback to newton
  // StudentT doesn't have quantile yet, so we use the robust Newton from
  // math_utils.h but updated to use Sokudo's precision
  if (df > 100)
    return normal_quantile(p);

  double t = (p > 0.5) ? normal_quantile(p) : -normal_quantile(1.0 - p);
  Sokudo::StudentT<double> dist{double(df)};

  for (int i = 0; i < 10; ++i) { // Increased iterations for stability
    double cdf_val = dist.cdf(t);
    double pdf_val = dist.pdf(t);
    if (std::abs(pdf_val) < 1e-15)
      break;
    double delta = (cdf_val - p) / pdf_val;
    t -= delta;
    if (std::abs(delta) < 1e-12)
      break;
  }
  return t;
}

// --- Vectorized Math Utility API (Zigen powered) ---

/**
 * @brief Normal CDF for a vector of values
 */
inline Vector normal_cdf_vec(const Vector &x) {
  return Backend::apply_normal_cdf(x);
}

/**
 * @brief Log-likelihood for Normal distribution (vectorized)
 */
inline double normal_log_likelihood(const Vector &x, double mu, double sigma) {
  double n = x.rows();
  double log2pi = std::log(2.0 * M_PI);
  // L = -n/2 * log(2pi) - n * log(sigma) - 1/(2*sigma^2) * sum((x-mu)^2)

  Vector residual = x - Backend::constant(x.rows(), mu);
  double rss = 0;
  const double *ptr = residual.ptr();
#pragma omp parallel for reduction(+ : rss)
  for (size_t i = 0; i < x.rows(); ++i) {
    rss += ptr[i] * ptr[i];
  }

  return -0.5 * n * log2pi - n * std::log(sigma) - rss / (2.0 * sigma * sigma);
}

} // namespace stats
} // namespace statelix

#endif // STATELIX_MATH_UTILS_H
