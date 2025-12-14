/**
 * @file numerical_tests.h
 * @brief Statelix v1.1 - Numerical Testing Utilities
 * 
 * Provides:
 *   - Gradient checking (numerical vs analytical)
 *   - Hessian symmetry verification
 *   - Condition number estimation
 *   - Parameter recovery tests
 * 
 * Purpose:
 * --------
 * These utilities verify that optimization and inference
 * routines are numerically correct. Critical for:
 *   - HMC (gradients must be exact)
 *   - L-BFGS (gradients affect convergence)
 *   - GLM (Fisher information correctness)
 */
#ifndef STATELIX_NUMERICAL_TESTS_H
#define STATELIX_NUMERICAL_TESTS_H

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "../optimization/objective.h"

namespace statelix {
namespace test {

// =============================================================================
// Gradient Checking
// =============================================================================

/**
 * @brief Result of gradient check
 */
struct GradientCheckResult {
    bool passed;
    double max_error;               // Max absolute error
    double max_relative_error;      // Max relative error
    int worst_index;                // Index with largest error
    Eigen::VectorXd analytical;     // Analytical gradient
    Eigen::VectorXd numerical;      // Numerical gradient
    Eigen::VectorXd errors;         // Absolute errors per dimension
};

/**
 * @brief Check analytical gradient against numerical gradient
 * 
 * Uses central differences: ∂f/∂x_i ≈ (f(x + εe_i) - f(x - εe_i)) / 2ε
 * 
 * @param objective The objective with gradient to check
 * @param x Point at which to check gradient
 * @param epsilon Step size for finite differences
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 */
inline GradientCheckResult check_gradient(
    Objective& objective,
    const Eigen::VectorXd& x,
    double epsilon = 1e-5,
    double rtol = 1e-4,
    double atol = 1e-6
) {
    GradientCheckResult result;
    int n = x.size();
    
    result.analytical = objective.gradient(x);
    result.numerical.resize(n);
    result.errors.resize(n);
    
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd x_plus = x, x_minus = x;
        x_plus(i) += epsilon;
        x_minus(i) -= epsilon;
        
        double f_plus = objective.value(x_plus);
        double f_minus = objective.value(x_minus);
        
        result.numerical(i) = (f_plus - f_minus) / (2.0 * epsilon);
        result.errors(i) = std::abs(result.analytical(i) - result.numerical(i));
    }
    
    result.max_error = result.errors.maxCoeff(&result.worst_index);
    
    // Compute relative error
    double denom = std::max(result.analytical.norm(), result.numerical.norm());
    result.max_relative_error = (denom > 1e-10) ? 
        result.errors.maxCoeff() / denom : result.max_error;
    
    // Check pass/fail
    result.passed = true;
    for (int i = 0; i < n; ++i) {
        double scale = std::max(std::abs(result.analytical(i)), 
                                std::abs(result.numerical(i)));
        double tol = atol + rtol * scale;
        if (result.errors(i) > tol) {
            result.passed = false;
            break;
        }
    }
    
    return result;
}

/**
 * @brief Check gradient at multiple random points
 */
inline bool check_gradient_multi(
    Objective& objective,
    int dim,
    int n_points = 10,
    double epsilon = 1e-5,
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < n_points; ++i) {
        Eigen::VectorXd x(dim);
        for (int j = 0; j < dim; ++j) {
            x(j) = dist(rng);
        }
        
        auto result = check_gradient(objective, x, epsilon);
        if (!result.passed) {
            return false;
        }
    }
    
    return true;
}

// =============================================================================
// Hessian Checking
// =============================================================================

/**
 * @brief Check Hessian symmetry and correctness
 */
struct HessianCheckResult {
    bool is_symmetric;
    bool gradient_consistent;       // H matches gradient of gradient
    double symmetry_error;
    double gradient_error;
    Eigen::MatrixXd hessian;
};

/**
 * @brief Compute numerical Hessian
 */
inline Eigen::MatrixXd numerical_hessian(
    Objective& objective,
    const Eigen::VectorXd& x,
    double epsilon = 1e-5
) {
    int n = x.size();
    Eigen::MatrixXd H(n, n);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Eigen::VectorXd x_pp = x, x_pm = x, x_mp = x, x_mm = x;
            x_pp(i) += epsilon; x_pp(j) += epsilon;
            x_pm(i) += epsilon; x_pm(j) -= epsilon;
            x_mp(i) -= epsilon; x_mp(j) += epsilon;
            x_mm(i) -= epsilon; x_mm(j) -= epsilon;
            
            H(i, j) = (objective.value(x_pp) - objective.value(x_pm) 
                     - objective.value(x_mp) + objective.value(x_mm)) 
                     / (4.0 * epsilon * epsilon);
        }
    }
    
    return H;
}

/**
 * @brief Check Hessian properties
 */
inline HessianCheckResult check_hessian(
    Objective& objective,
    const Eigen::VectorXd& x,
    double epsilon = 1e-5,
    double tol = 1e-4
) {
    HessianCheckResult result;
    
    // Compute numerical Hessian
    result.hessian = numerical_hessian(objective, x, epsilon);
    
    // Check symmetry
    Eigen::MatrixXd H_sym = (result.hessian + result.hessian.transpose()) / 2.0;
    result.symmetry_error = (result.hessian - H_sym).norm();
    result.is_symmetric = (result.symmetry_error < tol);
    
    // Check gradient consistency (if objective provides Hessian)
    if (objective.has_hessian()) {
        Eigen::MatrixXd H_analytical = objective.hessian(x);
        result.gradient_error = (H_analytical - result.hessian).norm();
        result.gradient_consistent = (result.gradient_error < tol);
    } else {
        result.gradient_error = 0.0;
        result.gradient_consistent = true;  // N/A
    }
    
    return result;
}

// =============================================================================
// Condition Number
// =============================================================================

/**
 * @brief Estimate condition number of Hessian
 */
inline double condition_number(const Eigen::MatrixXd& H) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    if (solver.info() != Eigen::Success) {
        return std::numeric_limits<double>::infinity();
    }
    
    const auto& eigenvalues = solver.eigenvalues();
    double max_ev = eigenvalues.cwiseAbs().maxCoeff();
    double min_ev = eigenvalues.cwiseAbs().minCoeff();
    
    if (min_ev < 1e-15) {
        return std::numeric_limits<double>::infinity();
    }
    
    return max_ev / min_ev;
}

// =============================================================================
// Parameter Recovery Tests
// =============================================================================

/**
 * @brief Parameter recovery test result
 */
struct RecoveryTestResult {
    bool passed;
    Eigen::VectorXd true_params;
    Eigen::VectorXd estimated_params;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd z_scores;       // (estimated - true) / se
    int n_within_2se;               // Parameters within 2 SE of true
    double rmse;
};

/**
 * @brief Generate test data and check if method recovers true parameters
 * 
 * @param generate_data Function that generates (X, y) given parameters
 * @param fit_method Function that estimates parameters from (X, y)
 * @param true_params True parameter values to recover
 * @param n_samples Sample size
 * @param seed Random seed
 */
template<typename DataGenerator, typename Fitter>
RecoveryTestResult recovery_test(
    DataGenerator generate_data,
    Fitter fit_method,
    const Eigen::VectorXd& true_params,
    int n_samples = 1000,
    unsigned int seed = 42
) {
    RecoveryTestResult result;
    result.true_params = true_params;
    
    // Generate synthetic data
    std::mt19937 rng(seed);
    auto [X, y] = generate_data(true_params, n_samples, rng);
    
    // Fit model
    auto [estimated, std_errors] = fit_method(X, y);
    
    result.estimated_params = estimated;
    result.std_errors = std_errors;
    
    // Compute z-scores
    result.z_scores.resize(true_params.size());
    result.n_within_2se = 0;
    
    for (int i = 0; i < true_params.size(); ++i) {
        double z = (estimated(i) - true_params(i)) / std_errors(i);
        result.z_scores(i) = z;
        
        if (std::abs(z) <= 2.0) {
            result.n_within_2se++;
        }
    }
    
    // RMSE
    result.rmse = (estimated - true_params).norm() / std::sqrt(true_params.size());
    
    // Pass if most parameters are within 2 SE (expect ~95%)
    double coverage = static_cast<double>(result.n_within_2se) / true_params.size();
    result.passed = (coverage >= 0.85);  // Some slack for small samples
    
    return result;
}

// =============================================================================
// Reproducibility Test
// =============================================================================

/**
 * @brief Check that method produces identical results with same seed
 */
template<typename Method>
bool test_reproducibility(
    Method method,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    int n_trials = 3
) {
    std::vector<Eigen::VectorXd> results;
    
    for (int i = 0; i < n_trials; ++i) {
        auto result = method(X, y);
        results.push_back(result);
    }
    
    // Check all results are identical
    for (int i = 1; i < n_trials; ++i) {
        if ((results[i] - results[0]).norm() > 1e-10) {
            return false;
        }
    }
    
    return true;
}

// =============================================================================
// Verbose Gradient Check (for debugging)
// =============================================================================

/**
 * @brief Print detailed gradient check report
 */
inline void print_gradient_check(
    Objective& objective,
    const Eigen::VectorXd& x,
    double epsilon = 1e-5
) {
    auto result = check_gradient(objective, x, epsilon);
    
    std::cout << "=== Gradient Check Report ===\n";
    std::cout << "Dimension: " << x.size() << "\n";
    std::cout << "Epsilon: " << epsilon << "\n";
    std::cout << "Status: " << (result.passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "Max error: " << std::scientific << result.max_error << "\n";
    std::cout << "Max relative error: " << result.max_relative_error << "\n";
    std::cout << "Worst index: " << result.worst_index << "\n\n";
    
    std::cout << std::setw(6) << "Index" 
              << std::setw(15) << "Analytical"
              << std::setw(15) << "Numerical"
              << std::setw(15) << "Error" << "\n";
    std::cout << std::string(51, '-') << "\n";
    
    for (int i = 0; i < x.size(); ++i) {
        std::cout << std::setw(6) << i
                  << std::setw(15) << std::scientific << result.analytical(i)
                  << std::setw(15) << result.numerical(i)
                  << std::setw(15) << result.errors(i) << "\n";
    }
}

} // namespace test
} // namespace statelix

#endif // STATELIX_NUMERICAL_TESTS_H
