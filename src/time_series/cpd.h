#ifndef STATELIX_CPD_H
#define STATELIX_CPD_H

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace statelix {

enum class CostType {
    L2,         // Mean shift (Constant Variance)
    GAUSSIAN,   // Mean and Variance shift
    POISSON     // Count data shift
};

struct CPDResult {
    std::vector<int> change_points; // Indices of change points (0-based)
    double cost;                    // Total penalized cost
};

class ChangePointDetector {
public:
    CostType cost_type = CostType::L2;
    double penalty = 0.0; // 0.0 = Auto (BIC)
    int min_size = 2;     // Minimum segment length
    
    ChangePointDetector(CostType type = CostType::L2, double pen = 0.0, int min_seg = 2)
        : cost_type(type), penalty(pen), min_size(min_seg) {}

    // Pruned Exact Linear Time (PELT)
    CPDResult fit(const Eigen::VectorXd& data);
};

} // namespace statelix

#endif // STATELIX_CPD_H
