
#include "accelerator.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdio.h>

namespace statelix {
namespace cuda {

bool is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in is_available: %d\n", (int)err);
        fflush(stderr);
    }
    return (err == cudaSuccess && count > 0);
}

// ... Kernels ...
#define MAX_K_SHARED 64

__global__ void kernel_weighted_gram_32(
    const double* X, 
    const double* W, 
    double* partial_grams, 
    int n, 
    int k,
    int block_offset
) {
    int row_idx = threadIdx.y;
    int col_idx = threadIdx.x;
    
    if (row_idx >= k || col_idx >= k) return;

    int rows_per_block = 1024; 
    int start_n = blockIdx.x * rows_per_block;
    int end_n = min(start_n + rows_per_block, n);

    double sum = 0.0;
    
    for (int i = start_n; i < end_n; ++i) {
        double w = W[i];
        double x_r = X[i * k + row_idx];
        double x_c = X[i * k + col_idx];
        sum += w * x_r * x_c;
    }
    
    int out_idx = blockIdx.x * (k * k) + row_idx * k + col_idx;
    partial_grams[out_idx] = sum;
}


std::vector<double> compute_weighted_gram(const double* X, const double* W, int n, int k) {
    if (k > 32) {
        printf("Fallback: K=%d > 32\n", k);
        return {}; 
    }

    std::vector<double> result(k * k, 0.0);

    double *d_X = nullptr, *d_W = nullptr, *d_partials = nullptr;
    
    size_t size_X = n * k * sizeof(double);
    size_t size_W = n * sizeof(double);
    
    int rows_per_block = 1024;
    int num_blocks = (n + rows_per_block - 1) / rows_per_block;
    size_t size_partials = num_blocks * k * k * sizeof(double);

    cudaError_t err;
    
    auto cleanup = [&]() {
        if (d_X) cudaFree(d_X);
        if (d_W) cudaFree(d_W);
        if (d_partials) cudaFree(d_partials);
    };

    if ((err = cudaMalloc(&d_X, size_X)) != cudaSuccess) { 
        fprintf(stderr, "Malloc X failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }
    if ((err = cudaMalloc(&d_W, size_W)) != cudaSuccess) { 
        fprintf(stderr, "Malloc W failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }
    if ((err = cudaMalloc(&d_partials, size_partials)) != cudaSuccess) { 
        fprintf(stderr, "Malloc Partials failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }

    if ((err = cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice)) != cudaSuccess) { 
        fprintf(stderr, "Memcpy X failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }
    if ((err = cudaMemcpy(d_W, W, size_W, cudaMemcpyHostToDevice)) != cudaSuccess) { 
        fprintf(stderr, "Memcpy W failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }

    dim3 threads(32, 32); 
    dim3 grid(num_blocks);
    
    kernel_weighted_gram_32<<<grid, threads>>>(d_X, d_W, d_partials, n, k, 0);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) { 
        fprintf(stderr, "Kernel Launch failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { 
        fprintf(stderr, "Sync failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }

    std::vector<double> h_partials(num_blocks * k * k);
    if ((err = cudaMemcpy(h_partials.data(), d_partials, size_partials, cudaMemcpyDeviceToHost)) != cudaSuccess) { 
        fprintf(stderr, "Memcpy Partials failed: %d\n", err); fflush(stderr); cleanup(); return {}; 
    }

    for (int b = 0; b < num_blocks; ++b) {
        int offset = b * k * k;
        for (int i = 0; i < k * k; ++i) {
            result[i] += h_partials[offset + i];
        }
    }

    cleanup();
    return result;
}

} // namespace cuda
} // namespace statelix
