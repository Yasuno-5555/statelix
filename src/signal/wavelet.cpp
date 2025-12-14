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
    
    double s = std::sqrt(2.0); // Inverse scaling

    for (int i = 0; i < half; ++i) {
        double avg = vec(i);
        double diff = vec(i + half);
        
        // (a+b)/sq2 = avg
        // (a-b)/sq2 = diff
        // a+b = avg*sq2
        // a-b = diff*sq2
        // 2a = (avg+diff)*sq2 -> a = (avg+diff)/sq2 * 2/2? No.
        // a = (avg+diff)*s/2? 
        // Let's re-derive:
        // x = (a+b)/r2, y = (a-b)/r2
        // x+y = 2a/r2 -> a = (x+y)*r2 / 2 = (x+y)/r2
        
        temp(2 * i) = (avg + diff) / s;
        temp(2 * i + 1) = (avg - diff) / s; 
        
        // Wait, if forward used 1/sqrt(2), then inverse uses 1/sqrt(2) too for orthogonal matrix?
        // Inverse of orthogonal matrix is transpose. 
        // [ 1  1 ] / r2
        // [ 1 -1 ] / r2
        // Transpose is same.
        // So factor is same 1/sqrt(2) ?
        // let s = 1/sqrt(2)
        // a_out = (avg + diff) * s ??
        // avg = (a+b)*s
        // diff = (a-b)*s
        // avg+diff = 2a*s -> a = (avg+diff)/(2s) = (avg+diff)/(2/r2) = (avg+diff)*r2/2 = (avg+diff)/r2 ok.
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
    
    int current_n = p2;
    int max_levels = 0;
    int temp_n = p2;
    while (temp_n >= 2) { temp_n /= 2; max_levels++; }
    
    int levels_to_do = (level <= 0 || level > max_levels) ? max_levels : level;

    for (int i = 0; i < levels_to_do; ++i) {
        if (type == WaveletType::Haar) {
            haar_step_forward(coeffs, current_n);
        }
        current_n /= 2;
    }

    return coeffs;
}

Eigen::VectorXd WaveletTransform::inverse(const Eigen::VectorXd& coeffs, int n_original_size) {
    int p2 = coeffs.size();
    Eigen::VectorXd signal = coeffs;
    
    // How many levels dependent on p2
    // We assume fully decomposed for now or need to store level metadata
    // Usually invalid inverse if we don't know "current_n" state.
    // For full decomposition: start from n=2 up to p2
    
    int current_n = 2;
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
