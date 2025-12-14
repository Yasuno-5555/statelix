#ifndef STATELIX_OPTIMIZATION_H
#define STATELIX_OPTIMIZATION_H

#include <cmath>
#include <algorithm>

namespace statelix {
namespace optimization {

    // Soft Thresholding Operator
    // S(z, gamma) = sign(z) * max(|z| - gamma, 0)
    inline double soft_threshold(double z, double gamma) {
        if (z > 0 && gamma < std::abs(z)) return z - gamma;
        if (z < 0 && gamma < std::abs(z)) return z + gamma;
        return 0.0;
    }

} // namespace optimization
} // namespace statelix

#endif // STATELIX_OPTIMIZATION_H
