#ifndef STATELIX_CPD_H
#define STATELIX_CPD_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct CPDResult {
    std::vector<int> change_points;
    double cost;
};

class ChangePointDetector {
public:
    // Penalty (beta). Default 2 * log(n) is standard-ish (BIC like)
    double penalty = 1.0; 
    
    // Pruned Exact Linear Time (PELT)
    // Detects changes in Mean.
    CPDResult fit_pelt(const Eigen::VectorXd& data);
};

} // namespace statelix

#endif // STATELIX_CPD_H
