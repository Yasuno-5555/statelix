#ifndef STATELIX_SURVIVAL_H
#define STATELIX_SURVIVAL_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct CoxResult {
    Eigen::VectorXd coef;
    // Basic result for now
};

class CoxPH {
public:
    int max_iter = 50;
    double tol = 1e-6;

    CoxResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& time, const Eigen::VectorXi& status);
};

} // namespace statelix

#endif // STATELIX_SURVIVAL_H
