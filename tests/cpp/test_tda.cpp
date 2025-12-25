#include "stats/tda.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::cout << "--- TDA (Keirin) Integration Test ---" << std::endl;

  // A simple square point cloud
  std::vector<std::vector<double>> points = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

  // At small epsilon, beta0=4 (4 components)
  size_t b0_small = statelix::stats::tda::compute_betti_number(points, 0.1, 0);
  std::cout << "Beta 0 (eps=0.1): " << b0_small << " (Expected: 4)"
            << std::endl;
  assert(b0_small == 4);

  // At epsilon=1.1, it's a hollow square (beta0=1, beta1=1)
  size_t b0_large = statelix::stats::tda::compute_betti_number(points, 1.1, 0);
  size_t b1_large = statelix::stats::tda::compute_betti_number(points, 1.1, 1);
  std::cout << "Beta 0 (eps=1.1): " << b0_large << " (Expected: 1)"
            << std::endl;
  std::cout << "Beta 1 (eps=1.1): " << b1_large << " (Expected: 1)"
            << std::endl;
  assert(b0_large == 1);
  assert(b1_large == 1);

  // Persistence
  auto pairs = statelix::stats::tda::compute_persistence(points, 2.0);
  std::cout << "Persistence pairs count: " << pairs.size() << std::endl;
  assert(!pairs.empty());

  std::cout << "TDA Test Passed!" << std::endl;
  return 0;
}
