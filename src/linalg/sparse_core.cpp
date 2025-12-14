#include "sparse_core.h"
#include <iostream>

namespace statelix {

SparseMatrix::SparseMatrix(int rows, int cols) {
    mat.resize(rows, cols);
}

void SparseMatrix::from_csr(const std::vector<double>& data, 
                            const std::vector<int>& indices, 
                            const std::vector<int>& indptr,
                            int rows, int cols) {
    // Eigen maps directly? Not easily for Sparse.
    // Eigen sparse matrix construction is generally:
    // 1. Triplets (slowest)
    // 2. Insert (slow)
    // 3. Map (Fastest for existing CSR)
    
    // Eigen Internal Structure for CSR (RowMajor):
    // values, innerIndices, outerIndexPtr
    // We can copy data into Eigen's buffers directly.
    
    mat.resize(rows, cols);
    mat.makeCompressed(); // Typically compressed already but just in case
    mat.resizeNonZeros(data.size());
    
    // Memcpy approach
    // Eigen default is ColMajor (CSC). If we want CSR we need RowMajor template.
    // Wait, class definition was Eigen::SparseMatrix<double>. Default is ColMajor.
    // If input is CSR, we should use Eigen::SparseMatrix<double, Eigen::RowMajor>.
    // But if we use ColMajor internally, we must transpose CSR to CSC manually or let Eigen do it.
    
    // Let's change header to RowMajor? Or just accept conversion cost.
    // For SpMV, RowMajor is slightly worse than CSC for Eigen usually, but data from Python (scipy) is often CSR.
    // Let's implement triplet construction for safety and simplicity first.
    // Or optimized triplet.
    
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(data.size());
    
    for(int i=0; i<rows; ++i) {
        int start = indptr[i];
        int end = indptr[i+1];
        for(int k=start; k<end; ++k) {
            triplets.emplace_back(i, indices[k], data[k]);
        }
    }
    
    mat.setFromTriplets(triplets.begin(), triplets.end());
}

Eigen::VectorXd SparseMatrix::dot(const Eigen::VectorXd& x) {
    return mat * x;
}

Eigen::VectorXd SparseMatrix::solve_cholesky(const Eigen::VectorXd& b) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(mat);
    if(solver.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky Decomposition failed");
    }
    return solver.solve(b);
}

Eigen::VectorXd SparseMatrix::solve_lu(const Eigen::VectorXd& b) {
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(mat);
    if(solver.info() != Eigen::Success) {
        throw std::runtime_error("LU Decomposition failed");
    }
    return solver.solve(b);
}

} // namespace statelix
