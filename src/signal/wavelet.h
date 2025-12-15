#ifndef STATELIX_WAVELET_H
#define STATELIX_WAVELET_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

enum class WaveletType {
    Haar,
    Daubechies4 // Placeholder for future
};

// Simple FWT result structure
struct WaveletResult {
    Eigen::VectorXd coeffs; // Concatenated coefficients [Appx_N, Detail_N, Detail_N-1, ..., Detail_1]
    // Or standard Mallat structure: usually returned as single vector in increasing frequency order
    // But helpful to know levels.
    std::vector<int> level_indices; // Start indices for each level
};

class WaveletTransform {
public:
    WaveletType type = WaveletType::Haar;

    // Multi-level Discrete Wavelet Transform
    // Simply returns the transformed vector (inplace or copy)
    // Decomposes down to 'level' depth. If level <= 0, max possible depth.
    Eigen::VectorXd transform(const Eigen::VectorXd& signal, int level = 0);
    
    // Inverse Transform
    // Reconstructs signal from coefficients up to 'level'. 
    // If level <= 0, assumes full decomposition (level=max).
    Eigen::VectorXd inverse(const Eigen::VectorXd& coeffs, int n_original_size, int level = 0);
};

} // namespace statelix

#endif // STATELIX_WAVELET_H
