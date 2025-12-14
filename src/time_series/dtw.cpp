#include "dtw.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace statelix {

DTWResult DTW::compute(const Eigen::VectorXd& s1, const Eigen::VectorXd& s2) {
    int n = s1.size();
    int m = s2.size();

    // Cost matrix (accumulated)
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n + 1, m + 1);
    
    // Initialize with infinity
    D.fill(std::numeric_limits<double>::infinity());
    D(0, 0) = 0.0;

    // DP Calculation
    // O(N*M) - Extremely slow in Python loop, fast in C++
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            double dist = std::pow(s1(i-1) - s2(j-1), 2); // Squared Euclidean
            // dist = std::abs(s1(i-1) - s2(j-1)); // Manhattan (optional)
            
            double cost = dist + std::min({
                D(i-1, j),   // Insertion
                D(i, j-1),   // Deletion
                D(i-1, j-1)  // Match
            });
            
            D(i, j) = cost;
        }
    }

    double final_dist = std::sqrt(D(n, m)); // Return Euclidean-like distance

    // Backtracking to find path
    std::vector<std::pair<int, int>> path;
    int i = n;
    int j = m;
    path.push_back({i-1, j-1});

    while (i > 1 || j > 1) {
        if (i == 1) {
            j--;
        } else if (j == 1) {
            i--;
        } else {
            if (D(i-1, j-1) <= D(i-1, j) && D(i-1, j-1) <= D(i, j-1)) {
                i--; j--;
            } else if (D(i-1, j) <= D(i, j-1)) {
                i--;
            } else {
                j--;
            }
        }
        path.push_back({i-1, j-1});
    }
    std::reverse(path.begin(), path.end());

    return {final_dist, D, path};
}

} // namespace statelix
