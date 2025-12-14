#ifndef STATELIX_LBFGS_H
#define STATELIX_LBFGS_H

#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <iostream>
#include <cmath>

namespace statelix {

// OptimizerResult structure
struct OptimizerResult {
    Eigen::VectorXd x;
    double min_value;
    int iterations;
    bool converged;
};

// Abstract Base Class for Dynamic Polymorphism (used by Python Bindings)
class DifferentiableFunction {
public:
    virtual ~DifferentiableFunction() = default;
    virtual double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) = 0;
};

// Templated L-BFGS Class
// Functor must support: double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& g)
template <typename Functor>
class LBFGS {
public:
    int max_iter = 100;
    int m = 10; // History size
    double epsilon = 1e-5;
    double ftol = 1e-4; // Armijo
    double gtol = 0.9;  // Wolfe c2 (Future use)

    OptimizerResult minimize(Functor& func, const Eigen::VectorXd& x0) {
        Eigen::VectorXd x = x0;
        Eigen::VectorXd g(x.size());
        double f = func(x, g);

        // History Storage (deque)
        std::deque<Eigen::VectorXd> s_list; 
        std::deque<Eigen::VectorXd> y_list;
        std::deque<double> rho_list;

        // Temporaries
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
            // 1. Compute search direction
            compute_direction(g, s_list, y_list, rho_list, d);

            x_old = x;
            g_old = g;

            // 2. Line Search (Updates x, f, g in-place)
            double step = line_search(func, x, d, f, g);

            // Check
            if (g.norm() < epsilon || step < 1e-12) {
                converged = (g.norm() < epsilon);
                break;
            }

            // 3. Update History
            s.noalias() = x - x_old;
            y.noalias() = g - g_old;
            double ys = y.dot(s);

            if (ys > 1e-10) {
                if (s_list.size() >= m) {
                    s_list.pop_front();
                    y_list.pop_front();
                    rho_list.pop_front();
                }
                s_list.push_back(s);
                y_list.push_back(y);
                rho_list.push_back(1.0 / ys);
            }
            
            if (s.norm() < 1e-9) break;
        }

        return {x, f, iter, converged};
    }

private:
    void compute_direction(const Eigen::VectorXd& g,
                          const std::deque<Eigen::VectorXd>& s_list,
                          const std::deque<Eigen::VectorXd>& y_list,
                          const std::deque<double>& rho_list,
                          Eigen::VectorXd& r) {
        
        int k = s_list.size();
        if (k == 0) {
            r = -g; 
            return;
        }

        Eigen::VectorXd q = g; 
        std::vector<double> alpha(k);

        for (int i = k - 1; i >= 0; --i) {
            alpha[i] = rho_list[i] * s_list[i].dot(q);
            q.noalias() -= alpha[i] * y_list[i];
        }

        double gamma = 1.0;
        double y_dot_y = y_list.back().dot(y_list.back());
        double s_dot_y = s_list.back().dot(y_list.back());
        
        if (y_dot_y > 1e-10) gamma = s_dot_y / y_dot_y;

        r.noalias() = gamma * q;

        for (int i = 0; i < k; ++i) {
            double beta = rho_list[i] * y_list[i].dot(r);
            r.noalias() += s_list[i] * (alpha[i] - beta);
        }
        r = -r;
    }

    double line_search(Functor& func, 
                       Eigen::VectorXd& x,
                       const Eigen::VectorXd& d, 
                       double& f, 
                       Eigen::VectorXd& g) {
        
        double alpha = 1.0;
        const double c1 = ftol;
        const double beta = 0.5;
        
        double f_old = f;
        double g_dot_d = g.dot(d);

        if (g_dot_d >= 0) return 0.0;

        Eigen::VectorXd x_new(x.size()); 
        Eigen::VectorXd g_new(x.size()); 
        double f_new;

        for (int i = 0; i < 20; ++i) {
            x_new.noalias() = x + alpha * d;
            f_new = func(x_new, g_new);
            
            if (f_new <= f_old + c1 * alpha * g_dot_d) {
                x = x_new;
                g = g_new;
                f = f_new;
                return alpha;
            }
            alpha *= beta;
        }
        return 0.0;
    }
};

} // namespace statelix

#endif // STATELIX_LBFGS_H
