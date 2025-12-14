#ifndef STATELIX_SPARSE_CORE_H
#define STATELIX_SPARSE_CORE_H

#include <Eigen/Sparse>
#include <vector>

namespace statelix {

// Sparse Matrix wrapper for bindings
// Wraps Eigen::SparseMatrix<double>
class SparseMatrix {
public:
    Eigen::SparseMatrix<double> mat;

    SparseMatrix() = default;
    SparseMatrix(int rows, int cols);

    // Build from CSR components
    // data: values
    // indices: col indices
    // indptr: row pointers (size rows + 1)
    void from_csr(const std::vector<double>& data, 
                  const std::vector<int>& indices, 
                  const std::vector<int>& indptr,
                  int rows, int cols);
                  
    // Sparse-Vector Multiplication
    Eigen::VectorXd dot(const Eigen::VectorXd& x);
    
    // Sparse Solver (Cholesky)
    // Solves Ax = b where A is this matrix
    Eigen::VectorXd solve_cholesky(const Eigen::VectorXd& b);
    
    // Sparse Solver (LU)
    // Solves Ax = b using LU
    Eigen::VectorXd solve_lu(const Eigen::VectorXd& b);
    
    int rows() const { return mat.rows(); }
    int cols() const { return mat.cols(); }
    int nnz() const { return mat.nonZeros(); }
};

} // namespace statelix

#endif // STATELIX_SPARSE_CORE_H
