#include "lbfgs.h"
#include <iostream>
#include <cmath>

namespace statelix {

OptimizerResult LBFGS::minimize(DifferentiableFunction& func, const Eigen::VectorXd& x0) {
    Eigen::VectorXd x = x0;
    Eigen::VectorXd g(x.size()); // Pre-allocate
    double f = func(x, g);       // g is updated here

    // History Storage (deque for efficient pop_front)
    std::deque<Eigen::VectorXd> s_list; 
    std::deque<Eigen::VectorXd> y_list;
    std::deque<double> rho_list;

    // Pre-allocate temporaries to avoid malloc inside loop
    Eigen::VectorXd x_old(x.size());
    Eigen::VectorXd g_old(x.size());
    Eigen::VectorXd d(x.size());
    Eigen::VectorXd s(x.size());
    Eigen::VectorXd y(x.size());

    int iter = 0;
    bool converged = false;

    if (g.norm() < epsilon) {
        return {x, f, 0, true};
    }

    for (iter = 0; iter < max_iter; ++iter) {
        // 1. Compute search direction d = -H * g
        // Pass output buffer d to avoid allocation
        compute_direction(g, s_list, y_list, rho_list, d);

        // Store old state for History Update
        x_old = x;
        g_old = g;

        // 2. Line Search (UPDATES x, f, g IN-PLACE)
        // returns step_size, but critical side effects are in x, f, g
        double step = line_search(func, x, d, f, g); 

        // Check for convergence or stall
        if (g.norm() < epsilon || step < 1e-12) {
            converged = (g.norm() < epsilon); // Only truly converged if gradient is small
            break;
        }

        // 3. Update History
        s.noalias() = x - x_old; // noalias() prevents temporary creation
        y.noalias() = g - g_old;

        double ys = y.dot(s);

        // Curvature condition: y^T s > 0
        if (ys > 1e-10) {
            // Manage History Size
            if (s_list.size() >= m) {
                s_list.pop_front();
                y_list.pop_front();
                rho_list.pop_front();
            }
            s_list.push_back(s);
            y_list.push_back(y);
            rho_list.push_back(1.0 / ys);
        }
        
        // Safety Break for tiny steps
        if (s.norm() < 1e-9) {
            converged = true; 
            break;
        }
    }

    return {x, f, iter, converged};
}

void LBFGS::compute_direction(const Eigen::VectorXd& g,
                              const std::deque<Eigen::VectorXd>& s_list,
                              const std::deque<Eigen::VectorXd>& y_list,
                              const std::deque<double>& rho_list,
                              Eigen::VectorXd& r) { // Output parameter
    
    int k = s_list.size();
    if (k == 0) {
        r = -g; 
        return;
    }

    // q is initialized as g
    Eigen::VectorXd q = g; 
    
    // Alpha buffer
    std::vector<double> alpha(k);

    // First loop (backward)
    for (int i = k - 1; i >= 0; --i) {
        alpha[i] = rho_list[i] * s_list[i].dot(q);
        q.noalias() -= alpha[i] * y_list[i];
    }

    // Scaling (gamma)
    double gamma = 1.0;
    const auto& y_last = y_list.back();
    const auto& s_last = s_list.back();
    
    double y_dot_y = y_last.dot(y_last);
    double s_dot_y = s_last.dot(y_last); 
    
    if (y_dot_y > 1e-10) {
        gamma = s_dot_y / y_dot_y;
    }

    r.noalias() = gamma * q; 

    // Second loop (forward)
    for (int i = 0; i < k; ++i) {
        double beta = rho_list[i] * y_list[i].dot(r);
        r.noalias() += s_list[i] * (alpha[i] - beta);
    }

    r = -r; // Descent direction
}

double LBFGS::line_search(DifferentiableFunction& func, 
                          Eigen::VectorXd& x,       // Ref to current x (will be updated)
                          const Eigen::VectorXd& d, // Search direction
                          double& f,                // Ref to current f (will be updated)
                          Eigen::VectorXd& g) {     // Ref to current g (will be updated)
    
    // Backtracking Armijo
    double alpha = 1.0;
    const double c1 = ftol;
    const double beta = 0.5; // Backtracking factor
    
    double f_old = f;
    double g_dot_d = g.dot(d); // directional derivative

    // Sanity check for descent
    if (g_dot_d >= 0) {
        // Not a descent direction
        return 0.0;
    }

    Eigen::VectorXd x_new(x.size()); 
    Eigen::VectorXd g_new(x.size()); 
    double f_new;

    int max_ls_iter = 20;
    for (int i = 0; i < max_ls_iter; ++i) {
        x_new.noalias() = x + alpha * d;
        f_new = func(x_new, g_new); // Evaluates at x_new, fills g_new
        
        // Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * g^T d
        if (f_new <= f_old + c1 * alpha * g_dot_d) {
            // Success! Update the Caller's State
            x = x_new;
            g = g_new;
            f = f_new;
            return alpha;
        }
        
        alpha *= beta;
    }
    
    // Fallback
    return 0.0;
}

} // namespace statelix
