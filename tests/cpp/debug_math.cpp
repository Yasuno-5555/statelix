#include "../../src/stats/math_utils.h"
#include <cmath>
#include <iostream>

using namespace statelix::stats;

int main() {
  double p_vals[] = {0.0001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.9999};
  int dfs[] = {1, 2, 5, 10, 50, 100};

  for (int df : dfs) {
    for (double p : p_vals) {
      double q = t_quantile(p, df);
      double cdf = t_cdf(q, df);
      std::cout << "df=" << df << ", p=" << p << " => q=" << q
                << ", cdf(q)=" << cdf;
      if (std::isnan(q))
        std::cout << " [NAN!]";
      std::cout << std::endl;
    }
  }
  return 0;
}
