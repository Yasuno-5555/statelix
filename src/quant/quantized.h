#ifndef STATELIX_QUANTIZED_H
#define STATELIX_QUANTIZED_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace statelix {

struct QuantizeParams {
    float scale;
    int32_t zero_point;
};

struct QuantizedTensor {
    std::vector<int8_t> data;
    int rows;
    int cols;
    QuantizeParams params;
};

// Quantize float to int8
inline QuantizedTensor quantize(const std::vector<float>& input, int rows, int cols) {
    float min_val = *std::min_element(input.begin(), input.end());
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Symmetric quantization for simplicity
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    float scale = abs_max / 127.0f;
    int32_t zero_point = 0; // Symmetric
    
    std::vector<int8_t> quantized(input.size());
    for(size_t i = 0; i < input.size(); ++i) {
        int32_t q = static_cast<int32_t>(std::round(input[i] / scale));
        q = std::max(-128, std::min(127, q));
        quantized[i] = static_cast<int8_t>(q);
    }
    
    return {quantized, rows, cols, {scale, zero_point}};
}

// Dequantize int8 to float
inline std::vector<float> dequantize(const QuantizedTensor& tensor) {
    std::vector<float> output(tensor.data.size());
    for(size_t i = 0; i < tensor.data.size(); ++i) {
        output[i] = static_cast<float>(tensor.data[i]) * tensor.params.scale;
    }
    return output;
}

// INT8 Matrix Multiplication: C = A * B
// A: (M x K), B: (K x N), C: (M x N)
// Uses int32 accumulator to prevent overflow
inline std::vector<float> quantized_matmul(
    const QuantizedTensor& A, 
    const QuantizedTensor& B,
    float output_scale = 0.0f // If 0, auto-calculate
) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    if (A.cols != B.rows) {
        throw std::runtime_error("Dimension mismatch in quantized_matmul");
    }
    
    // Output scale = scale_A * scale_B
    float scale_out = A.params.scale * B.params.scale;
    if (output_scale > 0) scale_out = output_scale;
    
    std::vector<float> C(M * N, 0.0f);
    
    // Core int8 matmul with int32 accumulator
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for(int k = 0; k < K; ++k) {
                // A is row-major: A[i, k] = A.data[i * K + k]
                // B is row-major: B[k, j] = B.data[k * N + j]
                int32_t a_val = static_cast<int32_t>(A.data[i * K + k]);
                int32_t b_val = static_cast<int32_t>(B.data[k * N + j]);
                acc += a_val * b_val;
            }
            // Dequantize accumulator
            C[i * N + j] = static_cast<float>(acc) * scale_out;
        }
    }
    
    return C;
}

} // namespace statelix

#endif // STATELIX_QUANTIZED_H
