#include "wavelet.h"
#include <cmath>
#include <iostream>

namespace statelix {

// Helper: Power of 2 check
bool is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Haar Steps
void haar_step_forward(Eigen::VectorXd& vec, int n) {
    int half = n / 2;
    Eigen::VectorXd temp(n);
    
    // sqrt(2) normalization usually
    double s = 1.0 / std::sqrt(2.0); // Or 0.5?
    // Standard orthogonal Haar: (a+b)/sqrt(2), (a-b)/sqrt(2)
    
    for (int i = 0; i < half; ++i) {
        double a = vec(2 * i);
        double b = vec(2 * i + 1);
        
        temp(i) = (a + b) * s;        // Approximation (Low Pass)
        temp(i + half) = (a - b) * s; // Detail (High Pass)
    }
    
    // Copy back to first n elements
    vec.head(n) = temp;
}

void haar_step_inverse(Eigen::VectorXd& vec, int n) {
    int half = n / 2;
    Eigen::VectorXd temp(n);
    
    // Antigravity's fix: Use same scaling factor as forward (1/sqrt(2))
    // Multiplied instead of divided.
    double s = 1.0 / std::sqrt(2.0);

    for (int i = 0; i < half; ++i) {
        double avg = vec(i);
        double diff = vec(i + half);
        
        // Reconstruction: a = (avg+diff)*s, b = (avg-diff)*s
        // (Assuming forward was (a+b)*s)
        temp(2 * i) = (avg + diff) * s;
        temp(2 * i + 1) = (avg - diff) * s; 
    }
    vec.head(n) = temp;
}


Eigen::VectorXd WaveletTransform::transform(const Eigen::VectorXd& signal, int level) {
    int n = signal.size();
    if (n == 0) return signal;

    // Pad to power of 2
    int p2 = 1;
    while (p2 < n) p2 *= 2;
    
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(p2);
    coeffs.head(n) = signal; // Zero pad remainder
    
    int temp_n = p2;
    int max_levels = 0;
    while (temp_n >= 2) { temp_n /= 2; max_levels++; }
    
    int levels_to_do = (level <= 0 || level > max_levels) ? max_levels : level;
    
    int current_n = p2;
    for (int i = 0; i < levels_to_do; ++i) {
        if (type == WaveletType::Haar) {
            haar_step_forward(coeffs, current_n);
        }
        current_n /= 2;
    }

    return coeffs;
}

Eigen::VectorXd WaveletTransform::inverse(const Eigen::VectorXd& coeffs, int n_original_size, int level) {
    int p2 = coeffs.size();
    if (p2 == 0) return Eigen::VectorXd();
    
    Eigen::VectorXd signal = coeffs;
    
    // Determine start level size
    int temp_n = p2;
    int max_levels = 0;
    while (temp_n >= 2) { temp_n /= 2; max_levels++; }
    
    int levels_to_do = (level <= 0 || level > max_levels) ? max_levels : level;
    
    // Start reconstruction from the deepest decomposition level
    // Deepest block size = p2 / 2^levels
    // First step combines it into p2 / 2^(levels-1)
    int current_n = p2 >> (levels_to_do - 1);
    
    while (current_n <= p2) {
        if (type == WaveletType::Haar) {
            haar_step_inverse(signal, current_n);
        }
        current_n *= 2;
    }
    
    // Truncate to original size
    return signal.head(n_original_size);
}

} // namespace statelix
