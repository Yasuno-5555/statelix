#ifndef STATELIX_OPTIMIZATION_H
#define STATELIX_OPTIMIZATION_H

#include <cmath>
#include <algorithm>

namespace statelix {
namespace optimization {

    /**
     * @brief Soft Thresholding Operator
     * S(z, gamma) = sign(z) * max(|z| - gamma, 0)
     * 
     * Efficient implementation avoiding unnecessary abs() calls.
     * Defensive against negative gamma.
     */
    inline double soft_threshold(double z, double gamma) {
        double g = std::max(0.0, gamma);
        if (g == 0.0) return z;
        
        if (z > g) return z - g;
        if (z < -g) return z + g;
        return 0.0;
    }

} // namespace optimization
} // namespace statelix

#endif // STATELIX_OPTIMIZATION_H
