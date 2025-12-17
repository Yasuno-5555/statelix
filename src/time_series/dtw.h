#ifndef STATELIX_DTW_H
#define STATELIX_DTW_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct DTWResult {
  double distance;
  Eigen::MatrixXd cost_matrix; // Optional: only populated if compute_path=true
  std::vector<std::pair<int, int>> path; // Indices (i, j)
};

class DTW {
public:
  int sakoe_chiba_band = -1; // -1 = auto (10% of max length)
  bool compute_path = true;  // Set false for speed when path not needed

  // Euclidean DTW with Sakoe-Chiba band optimization
  DTWResult compute(const Eigen::VectorXd &s1, const Eigen::VectorXd &s2);
};

} // namespace statelix

#endif // STATELIX_DTW_H
