#include "../../src/stats/math_utils.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace statelix::stats;

void bench_erfinv() {
  const int N = 1000000;
  std::vector<double> inputs;
  for (int i = 0; i < N; ++i) {
    inputs.push_back((double)i / N * 1.99 - 0.995);
  }

  auto start = std::chrono::high_resolution_clock::now();
  double sum = 0;
  for (double x : inputs) {
    sum += erfinv(x);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "erfinv (1M calls): " << std::fixed << std::setprecision(6)
            << diff.count() << " s (sum=" << sum << ")" << std::endl;
}

void bench_t_quantile() {
  const int N = 10000;
  auto start = std::chrono::high_resolution_clock::now();
  double sum = 0;
  for (int i = 1; i <= N; ++i) {
    sum += t_quantile((double)i / (N + 1), 10);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "t_quantile (10k calls, df=10): " << diff.count()
            << " s (sum=" << sum << ")" << std::endl;
}

void bench_beta_inc() {
  const int N = 100000;
  auto start = std::chrono::high_resolution_clock::now();
  double sum = 0;
  for (int i = 1; i <= N; ++i) {
    sum += beta_inc(2.0, 5.0, (double)i / (N + 1));
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "beta_inc (100k calls, a=2, b=5): " << diff.count()
            << " s (sum=" << sum << ")" << std::endl;
}

int main() {
  std::cout << "--- Math Library Benchmark ---" << std::endl;
  bench_erfinv();
  bench_t_quantile();
  bench_beta_inc();
  return 0;
}
