#include "cox.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace statelix {

// Cox Proportional Hazards (Simple Partial Likelihood Newton-Raphson)
CoxResult CoxPH::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& time, const Eigen::VectorXi& status) {
    int n = X.rows();
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);

    // 1. Sort data by time descending (standard for Cox calculation usually)
    // Or indices.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return time[a] > time[b]; // Descending time
    });

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd theta = (X * coef).array().exp();
        
        // Gradient and Hessian
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(p, p);

        // Accumulators for risk set
        double risk_sum = 0.0;
        Eigen::VectorXd risk_weighted_X = Eigen::VectorXd::Zero(p); 
        Eigen::MatrixXd risk_weighted_XX = Eigen::MatrixXd::Zero(p, p); 

        // Breslow approximation (handling ties loosely by order)
        for (int i : idx) {
            double theta_i = theta(i);
            Eigen::VectorXd x_i = X.row(i);
            
            // Add subject i to risk set (since we iterate descending, risk set grows or shrinks? 
            // Wait. At largest time t_1, risk set is just subject 1? 
            // No, standard is risk set = {j : t_j >= t_i}.
            // So iterating descending means we start with t_max (risk set size 1? or N?)
            // Actually, if we sort descending, the first subject has largest time.
            // Everyone else died earlier.
            // Risk set R(t_i) includes all j such that t_j >= t_i.
            // So if we iterate t_i from max to min:
            // At max time t_1, R(t_1) contains only {1} (if no ties).
            // At next time t_2, R(t_2) contains {1, 2}.
            // So we accumulate.
            
            risk_sum += theta_i;
            risk_weighted_X += theta_i * x_i;
            risk_weighted_XX += theta_i * (x_i * x_i.transpose());

            if (status(i) == 1) { // Event occurred
                // l_i = x_i * beta - log(risk_sum)
                // grad_i = x_i - (risk_weighted_X / risk_sum)
                Eigen::VectorXd x_bar = risk_weighted_X / risk_sum;
                grad += x_i - x_bar;

                // hess_i = -( (risk_weighted_XX / risk_sum) - x_bar * x_bar^T )
                hess -= (risk_weighted_XX / risk_sum) - (x_bar * x_bar.transpose());
            }
        }

        // Hessian might be singular
        // Beta_new = Beta_old - H^-1 * G
        Eigen::VectorXd step = hess.ldlt().solve(grad);
        
        if (step.norm() < tol) {
            break;
        }
        
        coef -= step; // Subtract because Newton-Raphson maximizes log-likelihood (Grad/Hess negative def)?
        // Usually beta_new = beta - H^-1 * Grad (if finding root of grad=0)
        // Check sign: maximizing L. H is negative definite. 
        // beta -= H^-1 * G  -> beta -= (-POS)^-1 * G -> beta += POS^-1 * G.
        
    }

    return {coef};
}

} // namespace statelix
