#include "solver.h"
#include "../optimization/zigen_blas.h"
#include <Eigen/Dense>

namespace statelix {

// WeightedDesignMatrix Implementation

Eigen::MatrixXd WeightedDesignMatrix::compute_gram() const {
  // Check dimensions for Zigen optimization
  // Gram = X^T * W * X = (W^1/2 X)^T * (W^1/2 X)
  // Or X^T * diag(w) * X
  // X is n x p.
  // Result is p x p.
  // Complexity O(n p^2).

  // Thresholds for Zigen
  // Based on benchmark: 100x100 is faster.
  int n = X_.rows();
  int p = X_.cols();

  // Pre-scale X (cheap using Eigen)
  // scaled_X = W * X
  // We can iterate columns, or use diagonal product.
  // X_.transpose() * weights_.asDiagonal() * X_

  // Zigen logic:
  // 1. Scale X (in Eigen is fast enough, O(np))
  // 2. GEMM (Zigen is faster for O(np^2))

  // Heuristic: Use Zigen if p >= 50 && n >= 100
  if (p >= 50 && n >= 100) {
    // Optimization: Check for uniform weights (standard OLS) to avoid copy
    // Assumes weights are exactly 1.0 (set by VectorXd::Ones())
    bool is_uniform = (weights_.array() == 1.0).all();

    if (is_uniform) {
      // Gram = X^T * X
      return optimization::zigen_gemm(X_.transpose(), X_);
    }

    Eigen::MatrixXd scaled_X = weights_.asDiagonal() * X_;
    // Gram = X^T * scaled_X
    // Call Zigen GEMM
    return optimization::zigen_gemm(X_.transpose(), scaled_X);
  } else {
    // Fallback to Eigen
    return X_.transpose() * weights_.asDiagonal() * X_;
  }
}

// Keep other methods inline in header if they are simple or template-heavy?
// compute_XTWy is matrix-vector, Eigen is likely fine or faster (SIMD).
// Solver logic relies on existing decompositions which use Eigen types.

} // namespace statelix
