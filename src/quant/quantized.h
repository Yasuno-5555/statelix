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
    auto [min_it, max_it] = std::minmax_element(input.begin(), input.end());
    float min_val = *min_it;
    float max_val = *max_it;
    
    // Symmetric quantization for simplicity
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    float scale = abs_max / 127.0f;
    int32_t zero_point = 0; // Symmetric
    
    std::vector<int8_t> quantized(input.size());
    for(size_t i = 0; i < input.size(); ++i) {
        int32_t q = static_cast<int32_t>(std::round(input[i] / scale));
        q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
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

// INT8 Matrix Multiplication with Bias: C = A * B + bias
// bias is float vector of size (N) or (1)
inline std::vector<float> quantized_matmul_bias(
    const QuantizedTensor& A, 
    const QuantizedTensor& B,
    const std::vector<float>& bias,
    float output_scale = 0.0f 
) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    if (A.cols != B.rows) {
        throw std::runtime_error("Dimension mismatch: A.cols != B.rows");
    }
    if (A.data.size() != static_cast<size_t>(M * K) || B.data.size() != static_cast<size_t>(K * N)) {
        throw std::runtime_error("Data buffer size mismatch");
    }
    if (!bias.empty() && bias.size() != static_cast<size_t>(N) && bias.size() != 1) {
         throw std::runtime_error("Bias dimension mismatch");
    }
    
    float scale_out = A.params.scale * B.params.scale;
    if (output_scale > 0) scale_out = output_scale;
    
    std::vector<float> C(M * N, 0.0f);
    
    // Core int8 matmul with int32 accumulator
    // Optimization: Loop tiling or SIMD would be here in prod
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for(int k = 0; k < K; ++k) {
                int32_t a_val = static_cast<int32_t>(A.data[i * K + k]);
                int32_t b_val = static_cast<int32_t>(B.data[k * N + j]);
                acc += a_val * b_val;
            }
            
            float val = static_cast<float>(acc) * scale_out;
            
            if (!bias.empty()) {
                val += (bias.size() == 1) ? bias[0] : bias[j];
            }
            C[i * N + j] = val;
        }
    }
    return C;
}

// Deprecated: Wrapper for backward compatibility
inline std::vector<float> quantized_matmul(
    const QuantizedTensor& A, 
    const QuantizedTensor& B,
    float output_scale = 0.0f 
) {
    return quantized_matmul_bias(A, B, {}, output_scale);
}

} // namespace statelix

#endif // STATELIX_QUANTIZED_H
