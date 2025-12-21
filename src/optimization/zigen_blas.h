#ifndef STATELIX_ZIGEN_BLAS_H
#define STATELIX_ZIGEN_BLAS_H

#include <Eigen/Dense>
#include <Zigen/Matrix.hpp>

namespace statelix {
namespace optimization {

/**
 * @brief Computes C = A * B using Zigen's optimized GEMM.
 *
 * Automatically converts Eigen matrices to Zigen matrices, performs
 * multiplication, and converts back. Use this for large matrices where Zigen's
 * OpenMP backend outperforms Eigen.
 */
inline Eigen::MatrixXd zigen_gemm(const Eigen::MatrixXd &A,
                                  const Eigen::MatrixXd &B) {
  // 1. Convert Eigen -> Zigen
  int ra = A.rows();
  int ca = A.cols();
  int rb = B.rows();
  int cb = B.cols();

  // Ensure dimensions match
  if (ca != rb) {
    throw std::invalid_argument("GEMM Dimension mismatch");
  }

  Zigen::Matrix<double, Zigen::Dynamic, Zigen::Dynamic> zA(ra, ca);
  Zigen::Matrix<double, Zigen::Dynamic, Zigen::Dynamic> zB(rb, cb);

  // Optimization: Use Eigen::Map to handle layout conversion (ColMajor ->
  // RowMajor) Zigen stores data in RowMajor layout (default for C++
  // arrays/Storage). Eigen default is ColMajor. Direct assignment handles the
  // conversion efficiently.

  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      zA_map(zA.ptr(), ra, ca);
  zA_map = A;

  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      zB_map(zB.ptr(), rb, cb);
  zB_map = B;

  // 2. Perform GEMM
  // C = A * B
  auto zC = zA * zB;

  // 3. Convert Zigen -> Eigen
  int rc = zC.rows(); // should be ra
  int cc = zC.cols(); // should be cb

  Eigen::MatrixXd C(rc, cc);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      zC_map(zC.ptr(), rc, cc);
  C = zC_map;

  return C;
}

} // namespace optimization
} // namespace statelix

#endif // STATELIX_ZIGEN_BLAS_H
