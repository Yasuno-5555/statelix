/**
 * @file sparse_core.cpp
 * @brief Statelix v1.1 - Sparse Matrix Implementation
 * 
 * NOTE: Most functionality is now header-only in sparse_core.h.
 * This file exists for backward compatibility with build systems
 * that expect a .cpp file.
 */
#include "sparse_core.h"

// v1.1: SparseMatrixT and legacy SparseMatrix are now header-only.
// This file is kept for CMake compatibility but contains no implementation.

namespace statelix {

// Explicit template instantiations for common types (optional, for faster builds)
// template class SparseMatrixT<Eigen::ColMajor>;
// template class SparseMatrixT<Eigen::RowMajor>;

} // namespace statelix

