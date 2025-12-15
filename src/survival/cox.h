#ifndef STATELIX_SURVIVAL_H
#define STATELIX_SURVIVAL_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct CoxResult {
    Eigen::VectorXd coef;
    Eigen::VectorXd std_error;
    Eigen::VectorXd z_score;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd covariance;
    double log_likelihood;
    bool converged;
    int iterations;
};

class CoxPH {
public:
    int max_iter = 50;
    double tol = 1e-6;

    // Use Efron approximation by default? User suggested Breslow or Efron. We'll implement Breslow as requested.
    CoxResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& time, const Eigen::VectorXi& status);
    
private:
    // Helper to calculate p-value from z-score
    double calculate_p_value(double z);
};

} // namespace statelix

#endif // STATELIX_SURVIVAL_H
