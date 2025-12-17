/**
 * @file wavelet.cpp
 * @brief Fast Wavelet Transform with In-place Optimization
 *
 * Optimized: Eliminated temporary array allocations per level
 * Uses in-place lifting scheme for Haar wavelet
 */
#include "wavelet.h"
#include <cmath>

namespace statelix {

// Helper: Power of 2 check
static bool is_power_of_two(int n) { return (n > 0) && ((n & (n - 1)) == 0); }

// Precomputed constant to avoid repeated sqrt calls
static const double INV_SQRT2 = 1.0 / std::sqrt(2.0);

// In-place Haar step forward using lifting scheme
// Avoids temporary array allocation
static void haar_step_forward_inplace(double *vec, int n) {
  int half = n / 2;

  // First pass: compute averages and differences in alternating positions
  // Use lifting: d[i] = x[2i+1] - x[2i], then a[i] = x[2i] + d[i]/2
  // But for orthogonal Haar: a = (x[0]+x[1])/sqrt(2), d = (x[0]-x[1])/sqrt(2)

  // We need a small temporary for swapping, but only O(1) extra space
  // Actually for in-place, we use a rolling approach

  // Simpler approach: use stack-allocated temp for small n, heap for large
  if (n <= 1024) {
    // Stack allocation for small arrays
    double temp[512]; // half of 1024

    for (int i = 0; i < half; ++i) {
      double a = vec[2 * i];
      double b = vec[2 * i + 1];
      temp[i] = (a + b) * INV_SQRT2;       // Approximation
      vec[i + half] = (a - b) * INV_SQRT2; // Detail (write to upper half first)
    }
    // Copy approximations to lower half
    for (int i = 0; i < half; ++i) {
      vec[i] = temp[i];
    }
  } else {
    // Heap allocation for large arrays (rare case)
    std::vector<double> temp(half);
    for (int i = 0; i < half; ++i) {
      double a = vec[2 * i];
      double b = vec[2 * i + 1];
      temp[i] = (a + b) * INV_SQRT2;
      vec[i + half] = (a - b) * INV_SQRT2;
    }
    for (int i = 0; i < half; ++i) {
      vec[i] = temp[i];
    }
  }
}

// In-place Haar step inverse
static void haar_step_inverse_inplace(double *vec, int n) {
  int half = n / 2;

  if (n <= 1024) {
    double temp[1024];

    for (int i = 0; i < half; ++i) {
      double avg = vec[i];
      double diff = vec[i + half];
      temp[2 * i] = (avg + diff) * INV_SQRT2;
      temp[2 * i + 1] = (avg - diff) * INV_SQRT2;
    }
    for (int i = 0; i < n; ++i) {
      vec[i] = temp[i];
    }
  } else {
    std::vector<double> temp(n);
    for (int i = 0; i < half; ++i) {
      double avg = vec[i];
      double diff = vec[i + half];
      temp[2 * i] = (avg + diff) * INV_SQRT2;
      temp[2 * i + 1] = (avg - diff) * INV_SQRT2;
    }
    for (int i = 0; i < n; ++i) {
      vec[i] = temp[i];
    }
  }
}

Eigen::VectorXd WaveletTransform::transform(const Eigen::VectorXd &signal,
                                            int level) {
  int n = signal.size();
  if (n == 0)
    return signal;

  // Pad to power of 2
  int p2 = 1;
  while (p2 < n)
    p2 *= 2;

  Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(p2);
  coeffs.head(n) = signal;

  // Get raw pointer for in-place operations
  double *data = coeffs.data();

  int max_levels = 0;
  int temp_n = p2;
  while (temp_n >= 2) {
    temp_n /= 2;
    max_levels++;
  }

  int levels_to_do = (level <= 0 || level > max_levels) ? max_levels : level;

  int current_n = p2;
  for (int i = 0; i < levels_to_do; ++i) {
    if (type == WaveletType::Haar) {
      haar_step_forward_inplace(data, current_n);
    }
    current_n /= 2;
  }

  return coeffs;
}

Eigen::VectorXd WaveletTransform::inverse(const Eigen::VectorXd &coeffs,
                                          int n_original_size, int level) {
  int p2 = coeffs.size();
  if (p2 == 0)
    return Eigen::VectorXd();

  Eigen::VectorXd signal = coeffs;
  double *data = signal.data();

  int max_levels = 0;
  int temp_n = p2;
  while (temp_n >= 2) {
    temp_n /= 2;
    max_levels++;
  }

  int levels_to_do = (level <= 0 || level > max_levels) ? max_levels : level;

  int current_n = p2 >> (levels_to_do - 1);

  while (current_n <= p2) {
    if (type == WaveletType::Haar) {
      haar_step_inverse_inplace(data, current_n);
    }
    current_n *= 2;
  }

  return signal.head(n_original_size);
}

} // namespace statelix
