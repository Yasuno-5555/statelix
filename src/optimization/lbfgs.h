/**
 * @file lbfgs.h
 * @brief Statelix v1.1 - L-BFGS Quasi-Newton Optimizer
 * 
 * Supports both:
 *   1. Legacy functor interface (backward compatible)
 *   2. New Objective interface (recommended)
 */
#ifndef STATELIX_LBFGS_H
#define STATELIX_LBFGS_H

#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>
#include "objective.h"

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

// =============================================================================
// NEW API: LBFGSOptimizer for Objective interface
// =============================================================================

/**
 * @brief L-BFGS Optimizer using new Objective interface
 * 
 * This is the recommended API for v1.1+.
 * For non-smooth objectives (e.g., with L1 penalty), use ProximalGradient instead.
 */
class LBFGSOptimizer {
public:
    int max_iter = 100;
    int history_size = 10;
    double epsilon = 1e-5;    // Gradient norm convergence
    double ftol = 1e-4;       // Armijo condition
    double min_step = 1e-12;  // Minimum step size
    bool verbose = false;
    
    /**
     * @brief Minimize an Objective function
     * @param objective Objective to minimize (must be smooth)
     * @param x0 Initial parameter vector
     * @return OptimizerResult containing solution
     */
    OptimizerResult minimize(Objective& objective, const Eigen::VectorXd& x0) {
        // Wrap Objective in internal functor and delegate
        ObjectiveFunctor functor(objective);
        LBFGS<ObjectiveFunctor> solver;
        solver.max_iter = max_iter;
        solver.m = history_size;
        solver.epsilon = epsilon;
        solver.ftol = ftol;
        return solver.minimize(functor, x0);
    }
    
    /**
     * @brief Minimize an EfficientObjective function (more efficient)
     */
    OptimizerResult minimize(EfficientObjective& objective, const Eigen::VectorXd& x0) {
        EfficientObjectiveFunctor functor(objective);
        LBFGS<EfficientObjectiveFunctor> solver;
        solver.max_iter = max_iter;
        solver.m = history_size;
        solver.epsilon = epsilon;
        solver.ftol = ftol;
        return solver.minimize(functor, x0);
    }

private:
    // Internal adapter: Objective -> Functor
    struct ObjectiveFunctor {
        Objective& obj;
        ObjectiveFunctor(Objective& o) : obj(o) {}
        
        double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
            grad = obj.gradient(x);
            return obj.value(x);
        }
    };
    
    // Internal adapter: EfficientObjective -> Functor (single pass)
    struct EfficientObjectiveFunctor {
        EfficientObjective& obj;
        EfficientObjectiveFunctor(EfficientObjective& o) : obj(o) {}
        
        double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
            auto [val, g] = obj.value_and_gradient(x);
            grad = g;
            return val;
        }
    };
};

// =============================================================================
// Proximal Gradient Descent (ISTA / FISTA)
// For non-smooth penalties like L1
// =============================================================================

/**
 * @brief Proximal Gradient Optimizer with FISTA Acceleration
 * 
 * Minimizes: F(x) = f(x) + g(x)
 * where f is smooth (has Lipschitz gradient) and g has a proximal operator.
 * 
 * Algorithm Selection:
 * -------------------
 * - ISTA (Iterative Shrinkage-Thresholding): O(1/k) convergence
 * - FISTA (Fast ISTA, Beck & Teboulle 2009): O(1/k²) convergence
 * 
 * Design Decisions:
 * -----------------
 * 1. **Step Size Strategy**: Backtracking line search with Armijo-like condition
 *    - Initial step = 1/L (L = Lipschitz constant, estimated via backtracking)
 *    - Decay factor = 0.5 (halve step size on failure)
 *    - This ensures convergence without knowing L in advance
 * 
 * 2. **FISTA Momentum**: Nesterov-style acceleration
 *    - t_{k+1} = (1 + sqrt(1 + 4t_k²)) / 2
 *    - y_k = x_k + ((t_k - 1) / t_{k+1}) * (x_k - x_{k-1})
 *    - Achieves optimal O(1/k²) rate for first-order methods
 * 
 * 3. **Restart Strategy**: NOT implemented (potential enhancement)
 *    - Gradient restart: restart when g(y_k)' * (x_k - x_{k-1}) > 0
 *    - Function restart: restart when F(x_k) > F(x_{k-1})
 *    - Restarts can help with ill-conditioned problems
 *    - TODO: Add adaptive restart in future version
 * 
 * Reference: Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-
 *            thresholding algorithm for linear inverse problems. SIAM J.
 *            Imaging Sciences, 2(1), 183-202.
 */
class ProximalGradient {
public:
    int max_iter = 1000;
    double tol = 1e-6;
    double initial_step = 1.0;
    double step_decay = 0.5;      // Backtrack multiplier (halve on failure)
    bool use_fista = true;        // O(1/k²) vs O(1/k)
    bool verbose = false;
    
    /**
     * @brief Minimize f(x) + g(x)
     * @param smooth_objective The smooth part f (provides gradient)
     * @param penalizer The non-smooth part g (provides prox operator)
     * @param x0 Initial parameter vector
     */
    OptimizerResult minimize(
        Objective& smooth_objective,
        Penalizer& penalizer,
        const Eigen::VectorXd& x0
    ) {
        int n = x0.size();
        Eigen::VectorXd x = x0;
        Eigen::VectorXd x_prev = x0;
        Eigen::VectorXd y = x0;  // FISTA momentum term
        
        double t = 1.0;
        double step = initial_step;
        
        double f_prev = smooth_objective.value(x) + penalizer.penalty(x);
        
        int iter = 0;
        bool converged = false;
        
        for (iter = 0; iter < max_iter; ++iter) {
            // Gradient at momentum point y
            Eigen::VectorXd grad = smooth_objective.gradient(y);
            
            // Line search for step size
            bool step_found = false;
            for (int ls = 0; ls < 20; ++ls) {
                // Gradient step
                Eigen::VectorXd z = y - step * grad;
                
                // Proximal step
                Eigen::VectorXd x_new = penalizer.prox(z, step);
                
                // Check sufficient decrease (composite objective)
                double f_new = smooth_objective.value(x_new) + penalizer.penalty(x_new);
                double f_y = smooth_objective.value(y) + penalizer.penalty(y);
                
                // Generalized Armijo-like condition for proximal gradient
                Eigen::VectorXd diff = x_new - y;
                double expected_decrease = grad.dot(diff) + 
                    (0.5 / step) * diff.squaredNorm() + 
                    penalizer.penalty(x_new) - penalizer.penalty(y);
                
                if (f_new <= f_y + expected_decrease + 1e-10) {
                    x_prev = x;
                    x = x_new;
                    step_found = true;
                    break;
                }
                step *= step_decay;
            }
            
            if (!step_found) {
                // Step size too small
                break;
            }
            
            // Check convergence
            double change = (x - x_prev).norm();
            if (change < tol) {
                converged = true;
                break;
            }
            
            // FISTA momentum update
            if (use_fista) {
                double t_new = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
                y = x + ((t - 1.0) / t_new) * (x - x_prev);
                t = t_new;
            } else {
                y = x;
            }
        }
        
        double final_value = smooth_objective.value(x) + penalizer.penalty(x);
        return {x, final_value, iter, converged};
    }
};

// =============================================================================
// Convenience functions
// =============================================================================

/**
 * @brief Minimize an objective using the appropriate method
 * 
 * Automatically chooses L-BFGS for smooth objectives,
 * Proximal Gradient for non-smooth.
 */
inline OptimizerResult minimize(
    Objective& objective,
    const Eigen::VectorXd& x0,
    Penalizer* penalizer = nullptr,
    int max_iter = 100
) {
    if (penalizer == nullptr || penalizer->is_smooth()) {
        // Use L-BFGS for smooth problems
        LBFGSOptimizer solver;
        solver.max_iter = max_iter;
        
        if (penalizer) {
            // Add smooth penalty to objective
            auto reg = std::make_shared<RegularizedObjective>(
                std::shared_ptr<Objective>(&objective, [](Objective*){}),  // non-owning
                std::shared_ptr<Penalizer>(penalizer, [](Penalizer*){})
            );
            return solver.minimize(*reg, x0);
        }
        return solver.minimize(objective, x0);
    } else {
        // Use Proximal Gradient for non-smooth penalties
        ProximalGradient solver;
        solver.max_iter = max_iter;
        return solver.minimize(objective, *penalizer, x0);
    }
}

} // namespace statelix

#endif // STATELIX_LBFGS_H

