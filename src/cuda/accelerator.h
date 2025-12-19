
#ifndef STATELIX_CUDA_ACCELERATOR_H
#define STATELIX_CUDA_ACCELERATOR_H

#include <vector>
#include <cstddef>

namespace statelix {
namespace cuda {

/**
 * @brief Check if a CUDA-capable device is available and accessible.
 * 
 * @return true If at least one CUDA device is detected.
 * @return false Otherwise.
 */
bool is_available();

/**
 * @brief Compute Weighted Gram Matrix (X^T W X) on GPU.
 * 
 * Computes G = X^T W X where X is (n x k) and W is (n) diagonal.
 * 
 * Strategy:
 * 1. Divide N samples into chunks (blocks).
 * 2. Each block computes a partial KxK matrix in shared memory/registers.
 * 3. Partial matrices are written to global memory.
 * 4. Final summation happens on CPU to ensure determinism and simplicity 
 *    (avoiding global atomics).
 * 
 * @param X Pointer to flat X array (row-major, size n*k)
 * @param W Pointer to weights array (size n)
 * @param n Number of samples
 * @param k Number of features
 * @return std::vector<double> Flattened result matrix (k*k), row-major.
 *         Returns empty vector if GPU allocation fails or runtime error occurs.
 */
std::vector<double> compute_weighted_gram(const double* X, const double* W, int n, int k);

} // namespace cuda
} // namespace statelix

#endif // STATELIX_CUDA_ACCELERATOR_H
