#include "stats/poly_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  std::cout << "--- Polynomial (Risan) Integration Test ---" << std::endl;

  // Test data: y = x^2
  std::vector<double> x = {0, 1, 2};
  std::vector<double> y = {0, 1, 4};

  // Lagrange Interpolation
  auto p_lagrange = statelix::stats::poly::lagrange_interpolate(x, y);
  double val1 = statelix::stats::poly::evaluate(p_lagrange, 1.5);
  std::cout << "Lagrange P(1.5): " << val1 << " (Expected: 2.25)" << std::endl;
  assert(std::abs(val1 - 2.25) < 1e-9);

  // Newton Interpolation
  auto p_newton = statelix::stats::poly::newton_interpolate(x, y);
  double val2 = statelix::stats::poly::evaluate(p_newton, 1.5);
  std::cout << "Newton P(1.5): " << val2 << " (Expected: 2.25)" << std::endl;
  assert(std::abs(val2 - 2.25) < 1e-9);

  std::cout << "Polynomial Test Passed!" << std::endl;
  return 0;
}
