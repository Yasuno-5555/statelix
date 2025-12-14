/**
 * @file sparse_core.h
 * @brief Statelix v1.1 - Extended Sparse Matrix Support
 * 
 * Provides:
 *   - SparseMatrixT: Templated sparse matrix (CSR/CSC)
 *   - Legacy SparseMatrix: Python binding compatible wrapper
 *   - Solver utilities (Cholesky, LU)
 */
#ifndef STATELIX_SPARSE_CORE_H
#define STATELIX_SPARSE_CORE_H

#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>

namespace statelix {

// =============================================================================
// v1.1 Templated Sparse Matrix
// =============================================================================

/**
 * @brief Templated sparse matrix wrapper
 * @tparam StorageOrder Eigen::ColMajor (CSC) or Eigen::RowMajor (CSR)
 * 
 * CSC (Column-major): Default for Eigen, good for column operations
 * CSR (Row-major): Good for row operations, matches scipy.csr_matrix
 */
template<int StorageOrder = Eigen::ColMajor>
class SparseMatrixT {
public:
    Eigen::SparseMatrix<double, StorageOrder> mat;
    
    SparseMatrixT() = default;
    
    SparseMatrixT(int rows, int cols) {
        mat.resize(rows, cols);
    }
    
    explicit SparseMatrixT(const Eigen::SparseMatrix<double, StorageOrder>& m) 
        : mat(m) {}
    
    // Build from triplets (i, j, v)
    void from_triplets(const std::vector<int>& row_indices,
                       const std::vector<int>& col_indices,
                       const std::vector<double>& values,
                       int rows, int cols) {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(values.size());
        for (size_t k = 0; k < values.size(); ++k) {
            triplets.emplace_back(row_indices[k], col_indices[k], values[k]);
        }
        mat.resize(rows, cols);
        mat.setFromTriplets(triplets.begin(), triplets.end());
    }
    
    // Build from CSR format (row_ptr, col_idx, values)
    void from_csr(const std::vector<double>& data,
                  const std::vector<int>& indices,
                  const std::vector<int>& indptr,
                  int rows, int cols) {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(data.size());
        for (int i = 0; i < rows; ++i) {
            for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
                triplets.emplace_back(i, indices[k], data[k]);
            }
        }
        mat.resize(rows, cols);
        mat.setFromTriplets(triplets.begin(), triplets.end());
    }
    
    // Matrix-vector multiplication
    Eigen::VectorXd operator*(const Eigen::VectorXd& x) const {
        return mat * x;
    }
    
    // Matrix-matrix multiplication (returns dense for simplicity)
    Eigen::MatrixXd operator*(const Eigen::MatrixXd& X) const {
        return mat * X;
    }
    
    // Sparse-Sparse multiplication
    SparseMatrixT<StorageOrder> multiply(const SparseMatrixT<StorageOrder>& other) const {
        return SparseMatrixT<StorageOrder>(mat * other.mat);
    }
    
    // Transpose (changes storage order)
    SparseMatrixT<StorageOrder == Eigen::ColMajor ? Eigen::RowMajor : Eigen::ColMajor>
    transpose() const {
        using TransposedType = Eigen::SparseMatrix<double, 
            StorageOrder == Eigen::ColMajor ? Eigen::RowMajor : Eigen::ColMajor>;
        TransposedType t = mat.transpose();
        SparseMatrixT<StorageOrder == Eigen::ColMajor ? Eigen::RowMajor : Eigen::ColMajor> result;
        result.mat = t;
        return result;
    }
    
    // Solve Ax = b using Cholesky (A must be SPD)
    Eigen::VectorXd solve_cholesky(const Eigen::VectorXd& b) const {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, StorageOrder>> solver;
        solver.compute(mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Cholesky decomposition failed");
        }
        return solver.solve(b);
    }
    
    // Solve Ax = b using LU
    Eigen::VectorXd solve_lu(const Eigen::VectorXd& b) const {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        // SparseLU only supports ColMajor
        Eigen::SparseMatrix<double, Eigen::ColMajor> colMajorMat = mat;
        solver.compute(colMajorMat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("LU decomposition failed");
        }
        return solver.solve(b);
    }
    
    // Accessors
    int rows() const { return mat.rows(); }
    int cols() const { return mat.cols(); }
    int nnz() const { return mat.nonZeros(); }
    double density() const { 
        return static_cast<double>(nnz()) / (rows() * cols()); 
    }
    
    // Access underlying Eigen matrix
    Eigen::SparseMatrix<double, StorageOrder>& eigen() { return mat; }
    const Eigen::SparseMatrix<double, StorageOrder>& eigen() const { return mat; }
};

// Convenient type aliases
using SparseMatrixCSC = SparseMatrixT<Eigen::ColMajor>;
using SparseMatrixCSR = SparseMatrixT<Eigen::RowMajor>;

// =============================================================================
// Legacy SparseMatrix (for Python bindings compatibility)
// =============================================================================

/**
 * @brief Legacy sparse matrix wrapper for Python bindings
 * @deprecated Use SparseMatrixCSC or SparseMatrixCSR in v1.1+
 */
class [[deprecated("Use SparseMatrixCSC/CSR in v1.1+")]] SparseMatrix {
public:
    Eigen::SparseMatrix<double> mat;

    SparseMatrix() = default;
    SparseMatrix(int rows, int cols) { mat.resize(rows, cols); }

    void from_csr(const std::vector<double>& data, 
                  const std::vector<int>& indices, 
                  const std::vector<int>& indptr,
                  int rows, int cols) {
        SparseMatrixCSC csc;
        csc.from_csr(data, indices, indptr, rows, cols);
        mat = csc.mat;
    }
                  
    Eigen::VectorXd dot(const Eigen::VectorXd& x) { return mat * x; }
    
    Eigen::VectorXd solve_cholesky(const Eigen::VectorXd& b) {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Cholesky decomposition failed");
        }
        return solver.solve(b);
    }
    
    Eigen::VectorXd solve_lu(const Eigen::VectorXd& b) {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("LU decomposition failed");
        }
        return solver.solve(b);
    }
    
    int rows() const { return mat.rows(); }
    int cols() const { return mat.cols(); }
    int nnz() const { return mat.nonZeros(); }
};

// =============================================================================
// Utility functions
// =============================================================================

/**
 * @brief Convert dense matrix to sparse (for testing)
 */
template<int StorageOrder = Eigen::ColMajor>
SparseMatrixT<StorageOrder> to_sparse(const Eigen::MatrixXd& dense, 
                                       double threshold = 0.0) {
    SparseMatrixT<StorageOrder> sparse(dense.rows(), dense.cols());
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < dense.rows(); ++i) {
        for (int j = 0; j < dense.cols(); ++j) {
            if (std::abs(dense(i, j)) > threshold) {
                triplets.emplace_back(i, j, dense(i, j));
            }
        }
    }
    sparse.mat.setFromTriplets(triplets.begin(), triplets.end());
    return sparse;
}

/**
 * @brief Create identity sparse matrix
 */
template<int StorageOrder = Eigen::ColMajor>
SparseMatrixT<StorageOrder> sparse_identity(int n) {
    SparseMatrixT<StorageOrder> I(n, n);
    I.mat.setIdentity();
    return I;
}

} // namespace statelix

#endif // STATELIX_SPARSE_CORE_H

