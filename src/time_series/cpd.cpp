#include "cpd.h"
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>

namespace statelix {

// Cost function: Change in Mean (Normal distribution, constant variance assumption)
// Cost = Sum Squared Residuals
// To make it O(1), we use precomputed sums.
// Cost(a, b) = Sum_{i=a}^{b} (y_i - mean)^2
//            = Sum y^2 - (Sum y)^2 / n
class MeanShiftCost {
    const Eigen::VectorXd& data;
    std::vector<double> sum_x;
    std::vector<double> sum_x2;

public:
    MeanShiftCost(const Eigen::VectorXd& d) : data(d) {
        int n = data.size();
        sum_x.resize(n + 1, 0.0);
        sum_x2.resize(n + 1, 0.0);
        
        for (int i = 0; i < n; ++i) {
            sum_x[i+1] = sum_x[i] + data[i];
            sum_x2[i+1] = sum_x2[i] + data[i] * data[i];
        }
    }

    double operator()(int start, int end) { // interval [start, end)
        if (end <= start) return 0.0;
        int n = end - start;
        double s = sum_x[end] - sum_x[start];
        double s2 = sum_x2[end] - sum_x2[start];
        
        return s2 - (s * s) / n;
    }
};

CPDResult ChangePointDetector::fit_pelt(const Eigen::VectorXd& data) {
    int n = data.size();
    if (n < 2) return {{}, 0.0};

    // Auto-penalty: BIC-like penalty
    // 2 * log(n) * p is classic BIC. For mean shift, p=1. 
    // Some suggest 3.0-4.0 * log(n) to avoid over-segmentation.
    // We use 3.0 * log(n) as a balanced default.
    double beta = (penalty <= 0) ? 3.0 * std::log(n) : penalty;

    MeanShiftCost cost_func(data);

    // F[t] = optimal cost to segment data[0...t-1]
    std::vector<double> F(n + 1);
    F[0] = 0.0; // Cost of empty segment is 0. Penalty applied when segment created.
    
    // R = set of candidate changepoints (start indices of new segments)
    // Initially {0} means we can start a segment at 0.
    std::vector<int> R = {0};

    // Backpointers
    std::vector<int> cp(n + 1);

    for (int t_star = 1; t_star <= n; ++t_star) {
        double min_val = std::numeric_limits<double>::infinity();
        int best_tau = -1;

        // Iterate over candidates
        for (int tau : R) {
            double c = cost_func(tau, t_star);
            double val = F[tau] + c + beta;
            
            if (val < min_val) {
                min_val = val;
                best_tau = tau;
            }
        }
        
        F[t_star] = min_val;
        cp[t_star] = best_tau;

        // Pruning Step (Killick et al. 2012)
        // Keep tau if F[tau] + C(tau, t_star) + beta < F[t_star]
        // Note: Strict inequality removes the optimal tau from R, 
        // implying we commit to t_star for that path.
        std::vector<int> new_R;
        new_R.reserve(R.size()); // Heuristic size
        
        for (int tau : R) {
            double future_cost = F[tau] + cost_func(tau, t_star) + beta;
            // Prune if future_cost > F[t_star]. So Keep if <=
            if (future_cost <= F[t_star] + 1e-9) { // Add epsilon for float stability
                new_R.push_back(tau);
            }
        }
        // Add current t_star as start of next segment candidate
        new_R.push_back(t_star);
        R = std::move(new_R);
    }

    // Reconstruction
    std::vector<int> change_points;
    int curr = n;
    while (curr > 0) {
        int prev = cp[curr];
        if (prev != 0) change_points.push_back(prev);
        curr = prev;
    }
    std::reverse(change_points.begin(), change_points.end());

    return {change_points, F[n]};
}

} // namespace statelix
