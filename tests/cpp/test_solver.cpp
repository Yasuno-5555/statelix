#include <iostream>
#include <Eigen/Dense>
#include "../src/linear_model/solver.h"

using namespace statelix;
using namespace Eigen;

void test_simple_ols() {
    std::cout << "Testing Simple OLS (Weights = 1)..." << std::endl;
    MatrixXd X(4, 2);
    X << 1, 1,
         1, 2,
         1, 3,
         1, 4;
    VectorXd y(4);
    y << 6, 5, 7, 10;
    // OLS: y = 3.5 + 1.4*x (approx)
    
    VectorXd w = VectorXd::Ones(4);
    WeightedDesignMatrix wdm(X, w);
    WeightedSolver solver(SolverStrategy::CHOLESKY);
    
    VectorXd beta = solver.solve(wdm, y);
    std::cout << "Beta: " << beta.transpose() << std::endl;
    
    // Manual check: (X'X)^-1 X'y
    VectorXd beta_true = (X.transpose() * X).inverse() * X.transpose() * y;
    std::cout << "True: " << beta_true.transpose() << std::endl;
    
    double error = (beta - beta_true).norm();
    if (error < 1e-10) std::cout << "[PASS] Coefficients match." << std::endl;
    else std::cout << "[FAIL] Error: " << error << std::endl;
}

void test_weighted() {
    std::cout << "\nTesting Weighted OLS..." << std::endl;
    MatrixXd X(3, 2);
    X << 1, 1,
         1, 2,
         1, 3;
    VectorXd y(3);
    y << 1, 2, 3;
    VectorXd w(3);
    w << 0.1, 0.5, 10.0; // High weight on last point (1,3 -> 3) which fits perfect line y=x
    
    WeightedDesignMatrix wdm(X, w);
    WeightedSolver solver(SolverStrategy::AUTO);
    
    VectorXd beta = solver.solve(wdm, y);
    std::cout << "Beta: " << beta.transpose() << std::endl;
    
    MatrixXd W = w.asDiagonal();
    VectorXd beta_true = (X.transpose() * W * X).inverse() * X.transpose() * W * y;
    std::cout << "True: " << beta_true.transpose() << std::endl;
    
    double error = (beta - beta_true).norm();
    if (error < 1e-10) std::cout << "[PASS] Weighted coefficients match." << std::endl;
    else std::cout << "[FAIL] Error: " << error << std::endl;
}

int main() {
    try {
        test_simple_ols();
        test_weighted();
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
