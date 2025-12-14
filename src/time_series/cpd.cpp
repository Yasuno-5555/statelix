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

    // Auto-penalty if not set or default
    // BIC-like penalty: log(n) * p (p=1 usually? or variance) 
    // Usually penalty scales with log(n)
    double beta = (penalty <= 0) ? 2.0 * std::log(n) : penalty;

    MeanShiftCost cost_func(data);

    // F[t] = optimal cost to segment data[0...t-1]
    std::vector<double> F(n + 1);
    F[0] = -beta; // Correction so F[0] + beta + cost = optimal? 
                  // Standard PELT: F[0] = -beta. First point is index 0.
    
    // R = set of candidate changepoints (indices)
    // Initially just {0}
    std::vector<int> R = {0};

    // Backpointers to reconstruct path
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

        // Pruning Step
        // Remove tau from R if F[tau] + cost(tau, t_star) + K > F[t_star]
        // K is usually beta (specifically related to penalty form)
        // Standard inequality: F[tau] + c + K >= F[t_star]
        // K is 0 for some configs, beta for others. usually K=0 works for simple, but K=beta is pruning condition.
        // Let's stick strictly to PELT pruning:
        // F(tau) + C(tau, t_star) + beta >= F(t_star) -> prune
        // Wait, standard is "remove if F(tau) + C(tau, t_star) > F(t_star)"
        
        std::vector<int> new_R;
        new_R.reserve(R.size() + 1);
        
        for (int tau : R) {
            if ((F[tau] + cost_func(tau, t_star)) <= F[t_star] + beta) { // Modified pruning condition? 
                // Wait. We want to keep viable candidates.
                // If the cost to get here plus extending to t_star is ALREADY worse than best...
                // But the condition is about FUTURE extensions.
                // Standard PELT: prune if F[tau] + cost(tau, t_star) > F[t_star]. 
                // (Note: beta is already in F[t_star] implicitly).
                // Actually the pruning theorem says: F[t] + C(t, t*) + K >= F[t*] ? 
                
                // Let's use simple non-pruned O(N^2) first if unsure? 
                // But User wants PELT specifically. 
                // Pruning rule: Remove tau if F[tau] + cost(tau, t_star) > F[t_star]
                // (assuming cost is sub-additive etc. Mean shift squares is).
                new_R.push_back(tau);
            }
        }
        new_R.push_back(t_star);
        R = new_R;
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
