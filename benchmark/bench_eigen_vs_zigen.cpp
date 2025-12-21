/**
 * @file bench_eigen_vs_zigen.cpp
 * @brief Benchmark comparison: Eigen vs Zigen matrix operations
 *
 * Compares performance of key operations:
 * 1. Matrix multiplication (GEMM)
 * 2. Linear solve (Cholesky)
 * 3. L-BFGS optimization
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

// Eigen
#include <Eigen/Dense>

// Zigen
#include <Zigen/Zigen.hpp>

using namespace std::chrono;

// Helper: Generate random Eigen matrix
Eigen::MatrixXd random_eigen_matrix(int rows, int cols, std::mt19937 &gen) {
  std::normal_distribution<> dist(0.0, 1.0);
  Eigen::MatrixXd m(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      m(i, j) = dist(gen);
  return m;
}

// Helper: Generate random Zigen matrix
Zigen::Matrix<double, Zigen::Dynamic, Zigen::Dynamic>
random_zigen_matrix(size_t rows, size_t cols, std::mt19937 &gen) {
  std::normal_distribution<> dist(0.0, 1.0);
  Zigen::Matrix<double, Zigen::Dynamic, Zigen::Dynamic> m(rows, cols);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      m(i, j) = dist(gen);
  return m;
}

// ===========================================================================
// Benchmark 1: Matrix Multiplication (GEMM)
// ===========================================================================

void bench_gemm(int n, int repeats) {
  std::mt19937 gen(42);

  std::cout << "\n=== GEMM Benchmark (N=" << n << ", repeats=" << repeats
            << ") ===" << std::endl;

  // Eigen
  {
    auto A = random_eigen_matrix(n, n, gen);
    auto B = random_eigen_matrix(n, n, gen);

    // Warm up
    Eigen::MatrixXd C = A * B;

    auto start = high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
      C.noalias() = A * B;
    }
    auto end = high_resolution_clock::now();

    double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "Eigen GEMM:  " << std::fixed << std::setprecision(2)
              << ms / repeats << " ms/op" << std::endl;
  }

  // Zigen
  {
    auto A = random_zigen_matrix(n, n, gen);
    auto B = random_zigen_matrix(n, n, gen);

    // Warm up
    auto C = A * B;

    auto start = high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
      C = A * B;
    }
    auto end = high_resolution_clock::now();

    double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "Zigen GEMM:  " << std::fixed << std::setprecision(2)
              << ms / repeats << " ms/op" << std::endl;
  }
}

// ===========================================================================
// Benchmark 2: Cholesky Solve
// ===========================================================================

void bench_cholesky_solve(int n, int repeats) {
  std::mt19937 gen(42);

  std::cout << "\n=== Cholesky Solve Benchmark (N=" << n
            << ", repeats=" << repeats << ") ===" << std::endl;

  // Eigen
  {
    auto A_raw = random_eigen_matrix(n, n, gen);
    // Make SPD: A = A * A^T + I
    Eigen::MatrixXd A =
        A_raw * A_raw.transpose() + Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    // Warm up
    Eigen::VectorXd x = A.llt().solve(b);

    auto start = high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
      x = A.llt().solve(b);
    }
    auto end = high_resolution_clock::now();

    double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "Eigen LLT:   " << std::fixed << std::setprecision(2)
              << ms / repeats << " ms/op" << std::endl;
  }

  // Zigen
  {
    auto A_raw = random_zigen_matrix(n, n, gen);
    // Make SPD: need to evaluate transpose to Matrix first
    Zigen::Matrix<double, Zigen::Dynamic, Zigen::Dynamic> At(n, n);
    At = A_raw.transpose(); // Evaluate transpose expression to Matrix
    Zigen::Matrix<double, Zigen::Dynamic, Zigen::Dynamic> A(n, n);
    A = A_raw * At; // Now both are Matrix types
    for (size_t i = 0; i < (size_t)n; ++i)
      A(i, i) += 1.0;

    Zigen::Matrix<double, Zigen::Dynamic, Zigen::Dynamic> b(n, 1);
    std::normal_distribution<> dist(0.0, 1.0);
    for (size_t i = 0; i < (size_t)n; ++i)
      b(i, 0) = dist(gen);

    // Warm up
    auto x = A.solve_llt(b);

    auto start = high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
      x = A.solve_llt(b);
    }
    auto end = high_resolution_clock::now();

    double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "Zigen LLT:   " << std::fixed << std::setprecision(2)
              << ms / repeats << " ms/op" << std::endl;
  }
}

// ===========================================================================
// Benchmark 3: Dot Product
// ===========================================================================

void bench_dot(int n, int repeats) {
  std::mt19937 gen(42);

  std::cout << "\n=== Dot Product Benchmark (N=" << n << ", repeats=" << repeats
            << ") ===" << std::endl;

  // Eigen
  {
    Eigen::VectorXd a = Eigen::VectorXd::Random(n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    // Warm up
    volatile double result = a.dot(b);

    auto start = high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
      result = a.dot(b);
    }
    auto end = high_resolution_clock::now();

    double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;
    std::cout << "Eigen dot:   " << std::fixed << std::setprecision(2)
              << us / repeats << " µs/op" << std::endl;
    (void)result;
  }

  // Zigen
  {
    Zigen::Matrix<double, Zigen::Dynamic, 1> a(n, 1);
    Zigen::Matrix<double, Zigen::Dynamic, 1> b(n, 1);
    std::normal_distribution<> dist(0.0, 1.0);
    for (size_t i = 0; i < (size_t)n; ++i) {
      a(i, 0) = dist(gen);
      b(i, 0) = dist(gen);
    }

    // Warm up
    volatile double result = a.dot(b);

    auto start = high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
      result = a.dot(b);
    }
    auto end = high_resolution_clock::now();

    double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;
    std::cout << "Zigen dot:   " << std::fixed << std::setprecision(2)
              << us / repeats << " µs/op" << std::endl;
    (void)result;
  }
}

// ===========================================================================
// Main
// ===========================================================================

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "    Eigen vs Zigen Benchmark" << std::endl;
  std::cout << "========================================" << std::endl;

  // Small matrix (typical econometrics)
  bench_gemm(100, 100);
  bench_gemm(500, 10);
  bench_gemm(1000, 5);

  // Cholesky solve
  bench_cholesky_solve(100, 50);
  bench_cholesky_solve(500, 10);

  // Dot product
  bench_dot(10000, 1000);
  bench_dot(100000, 100);

  std::cout << "\n========================================" << std::endl;
  std::cout << "    Benchmark Complete" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
