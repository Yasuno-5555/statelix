#ifndef STATELIX_ICP_H
#define STATELIX_ICP_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct ICPResult {
    Eigen::MatrixXd rotation;    // 3x3 or dim x dim
    Eigen::VectorXd translation; // dim
    double rmse;                 // Root Mean Squared Error
    int iterations;
    bool converged;
};

class ICP {
public:
    int max_iter = 50;
    double tol = 1e-6; // Convergence tolerance for relative RMSE change

    // Aligns Source to Target. Returns transformation R, t such that Target approx R * Source + t
    // Points are rows (N x dim)
    ICPResult align(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target);
};

} // namespace statelix

#endif // STATELIX_ICP_H
