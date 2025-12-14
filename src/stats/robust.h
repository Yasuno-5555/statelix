#ifndef STATELIX_ROBUST_H
#define STATELIX_ROBUST_H

#include <Eigen/Dense>
#include <cmath>
#include "../optimization/lbfgs.h"

namespace statelix {

struct HuberResult {
    Eigen::VectorXd coef;
    double delta;     // Huber threshold
    int iterations;
    bool converged;
};

// Huber Regression
// Uses L-BFGS to minimize Huber loss
class HuberRegression {
public:
    double delta = 1.35; // Huber threshold (default: ~95% Gaussian)
    int max_iter = 100;
    
    // Internal objective functor for L-BFGS
    struct HuberObjective {
        const Eigen::MatrixXd& X;
        const Eigen::VectorXd& y;
        double delta;
        
        double operator()(const Eigen::VectorXd& beta, Eigen::VectorXd& grad) {
            int n = X.rows();
            int p = X.cols();
            
            Eigen::VectorXd residuals = y - X * beta;
            
            double loss = 0.0;
            grad.setZero(p);
            
            for(int i = 0; i < n; ++i) {
                double r = residuals(i);
                double abs_r = std::abs(r);
                
                if (abs_r <= delta) {
                    // Quadratic region
                    loss += 0.5 * r * r;
                    grad -= r * X.row(i).transpose();
                } else {
                    // Linear region
                    loss += delta * (abs_r - 0.5 * delta);
                    double sign_r = (r > 0) ? 1.0 : -1.0;
                    grad -= delta * sign_r * X.row(i).transpose();
                }
            }
            
            return loss;
        }
    };
    
    HuberResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        int p = X.cols();
        Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
        
        HuberObjective objective{X, y, delta};
        
        LBFGS<HuberObjective> solver;
        solver.max_iter = max_iter;
        solver.epsilon = 1e-5;
        
        OptimizerResult res = solver.minimize(objective, beta0);
        
        return {res.x, delta, res.iterations, res.converged};
    }
};

// Quantile Regression using ADMM (Placeholder for future)
// min sum(rho_tau(y - X*beta)) where rho_tau is tilted absolute loss

} // namespace statelix

#endif // STATELIX_ROBUST_H
