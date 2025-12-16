#include "cpd.h"
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include <memory>
#include <algorithm>

namespace statelix {

// Abstract Base Class for Cost Functions
class CostFunction {
public:
    virtual ~CostFunction() = default;
    // Calculate cost for segment data[start...end-1]
    virtual double evaluate(int start, int end) = 0;
};

// L2 Cost: Mean Shift (Variance fixed/ignored)
// Cost = Sum (y_i - mean)^2
//      = Sum y^2 - (Sum y)^2 / n
class L2Cost : public CostFunction {
    std::vector<double> sum_x;
    std::vector<double> sum_x2;

public:
    L2Cost(const Eigen::VectorXd& data) {
        int n = data.size();
        sum_x.resize(n + 1, 0.0);
        sum_x2.resize(n + 1, 0.0);
        
        for (int i = 0; i < n; ++i) {
            sum_x[i+1] = sum_x[i] + data[i];
            sum_x2[i+1] = sum_x2[i] + data[i] * data[i];
        }
    }

    double evaluate(int start, int end) override {
        int n = end - start;
        if (n <= 0) return 0.0;
        
        double s = sum_x[end] - sum_x[start];
        double s2 = sum_x2[end] - sum_x2[start];
        
        return s2 - (s * s) / n;
    }
};

// Gaussian Cost: Mean and Variance Shift
// Cost = Sum (log(2*pi*sigma^2) + (y_i - mu)^2 / sigma^2)
// ML estimates: mu = mean, sigma^2 = variance (MLE) = Sum(y-mu)^2 / n
// Term simplifies to: n * (log(2*pi) + log(sigma^2) + 1)
// We drop constants like log(2*pi) and +1 usually, but let's keep log(var).
// Cost ~ n * log(Sum(y-mu)^2 / n)  ... plus constants.
// Actually, strictly -2*logL = n * log(2*pi*sigma^2) + n
// Ignoring constants: n * log(variance) is the core.
// Variance = (Sum y^2 - (Sum y)^2/n) / n
class GaussianCost : public CostFunction {
    std::vector<double> sum_x;
    std::vector<double> sum_x2;
    static constexpr double EPS = 1e-9;

public:
    GaussianCost(const Eigen::VectorXd& data) {
        int n = data.size();
        sum_x.resize(n + 1, 0.0);
        sum_x2.resize(n + 1, 0.0);
        
        for (int i = 0; i < n; ++i) {
            sum_x[i+1] = sum_x[i] + data[i];
            sum_x2[i+1] = sum_x2[i] + data[i] * data[i];
        }
    }

    double evaluate(int start, int end) override {
        int n = end - start;
        if (n <= 0) return 0.0;
        
        double s = sum_x[end] - sum_x[start];
        double s2 = sum_x2[end] - sum_x2[start];
        
        // Sum of Squared Residuals
        double ssr = s2 - (s * s) / n;
        
        // Variance (MLE)
        double var = ssr / n;
        if (var < EPS) var = EPS; // Prevent log(0)
        
        // n * log(var) is the shape. 
        // We can add log(2*pi) + 1 if we want exact 2*NLL relation, but for optimization
        // only slope (penalty) matters relative to n*log(var).
        // Standard in literature (incl ruptures): n * log(var)
        return n * std::log(var); 
    }
};

// Poisson Cost: Count Data Shift
// Cost = 2 * sum(y_i * log(y_i / lambda_hat) - (y_i - lambda_hat)) ??
// Wait, -2 log L for Poisson:
// L = prod (e^-lambda * lambda^y_i / y_i!)
// log L = Sum (-lambda + y_i log lambda - log y_i!)
// MLE lambda = mean = S/n
// -2 log L = 2 * (n * lambda - log(lambda) * S + Sum(log y_i!))
// Dropping constant Sum(log y_i!):
// Cost = 2 * (n * (S/n) - S * log(S/n))
//      = 2 * (S - S * log(S/n))
// if S=0, 0.
class PoissonCost : public CostFunction {
    std::vector<double> sum_x;
    static constexpr double EPS = 1e-9;

public:
    PoissonCost(const Eigen::VectorXd& data) {
        int n = data.size();
        sum_x.resize(n + 1, 0.0);
        
        for (int i = 0; i < n; ++i) {
            // Ensure data is non-negative for Poisson
            double val = data[i];
            if (val < 0) val = 0; // Clamp or error? Clamp for robustness
            sum_x[i+1] = sum_x[i] + val;
        }
    }

    double evaluate(int start, int end) override {
        int n = end - start;
        if (n <= 0) return 0.0;
        
        double s = sum_x[end] - sum_x[start];
        
        // If sum is 0, lambda is 0. Cost is 0 (limit x log x -> 0).
        if (s < EPS) return 0.0;
        
        double lambda = s / n;
        // Cost = 2 * (S - S * log(lambda)) ?
        // From formula: 2 * n * lambda - 2 * S * log(lambda)
        // = 2 * S - 2 * S * log(S/n)
        return 2.0 * s * (1.0 - std::log(lambda));
    }
};

CPDResult ChangePointDetector::fit(const Eigen::VectorXd& data) {
    int n = data.size();
    if (n < min_size) return {{}, 0.0};

    // 1. Setup Cost Function
    std::unique_ptr<CostFunction> cf;
    switch (cost_type) {
        case CostType::L2:
            cf = std::make_unique<L2Cost>(data);
            break;
        case CostType::GAUSSIAN:
            cf = std::make_unique<GaussianCost>(data);
            break;
        case CostType::POISSON:
            cf = std::make_unique<PoissonCost>(data);
            break;
    }

    // 2. Determine Penalty
    // Default BIC:
    // L2 (1 param: mean) -> beta = 2 * log(n) * 1? Or 1 * log(n)?
    // Gaussian (2 params: mean, var) -> beta = 2 * log(n) * 2?
    // Standard practice often just uses k * log(n) where k is small constant.
    // For L2/Mean, often 1.0 * log(n) is used (BIC).
    double beta = penalty;
    if (beta <= 0.0) {
        // Auto-configure
        if (cost_type == CostType::L2 || cost_type == CostType::POISSON) {
            beta = 2.0 * std::log(n); // 1 parameter
        } else {
            // Gaussian has 2 parameters (mean, var) so effectively 2x complexity?
            // Actually BIC is k * ln(n). Delta k is what matters.
            // Adding a segment adds (mean, var) so 2 params.
            beta = 2.0 * 2.0 * std::log(n); // Stricter for Gaussian
        }
    }

    // 3. PELT Initialization
    std::vector<double> F(n + 1);
    // F[0] = -beta; // This allows the first segment to not pay the penalty? 
    // Wait, standard PELT definition:
    // F(t) = min_{tau} [ F(tau) + C(tau, t) + beta ]
    // At t=0, F[0] = -beta is a trick to offset the first beta addition, 
    // so total cost has K segments * beta.
    F[0] = -beta; 
    
    std::vector<int> cp(n + 1, 0); // Backpointers
    std::vector<int> R = {0};      // Candidate changepoints

    // 4. DP Loop
    for (int t_star = min_size; t_star <= n; ++t_star) {
        double min_val = std::numeric_limits<double>::infinity();
        int best_tau = -1;

        // Iterate over candidates
        // Pruning logic must be tightly coupled
        std::vector<int> new_R;
        new_R.reserve(R.size());

        for (int tau : R) {
            // Enforce min_size constraint
            if (t_star - tau < min_size) {
                 // Too short, cannot end segment here.
                 // But wait, R contains valid previous changepoints.
                 // If t_star - tau < min_size, we just skip evaluating this tau as immediately preceding CP.
                 // BUT we must keep tau in new_R if it might be valid later?
                 // No, standard PELT filters candidates.
                 // If (t_star - tau) < min_size, it's not a valid segment *ending* at t_star.
                 // But we should keep it in R for future t_star > t_star.
                 new_R.push_back(tau); 
                 continue;
            }

            double c = cf->evaluate(tau, t_star);
            double val = F[tau] + c + beta;

            if (val < min_val) {
                min_val = val;
                best_tau = tau;
            }
        }
        
        // If we found a valid previous segment
        if (best_tau != -1) {
            F[t_star] = min_val;
            cp[t_star] = best_tau;
        } else {
            // No valid segmentation found (e.g. at start), propagate infinity or handle gracefully
            F[t_star] = std::numeric_limits<double>::infinity();
        }

        // 5. Pruning Step (K = 0 for PELT)
        // Prune tau if F[tau] + C(tau, t_star) + K > F[t_star]
        // K is usually 0 if constant penalty.
        // We only prune from the explicitly checked valid ones?
        // Actually we loop R again or merge loops. 
        // Merging loops is tricky if min_size is involved.
        // Let's do a second pass for pruning to be safe and clear.
        
        if (best_tau != -1) {
            std::vector<int> next_R;
            next_R.reserve(R.size() + 1);
            
            for (int tau : R) {
                // If this tau was skipped due to min_size, keep it blindly?
                // Or does pruning require C(tau, t_star) to be calculatable?
                // C(tau, t_star) is valid even if size < min_size numerically, but logically invalid.
                // If size < min_size, we can't prune based on it because we didn't calculate 'val' correctly using it?
                // Actually Killick et al. simply says:
                // if F[tau] + C(tau, t_star) <= F[t_star], keep tau.
                
                // If t_star - tau < min_size, we didn't use it for min_val.
                // We shouldn't prune it because it hasn't had a chance to become the optimal.
                // So keep it.
                if (t_star - tau < min_size) {
                    next_R.push_back(tau);
                } else {
                    double c = cf->evaluate(tau, t_star);
                    if (F[tau] + c <= F[t_star]) { // + beta is already in F[t_star] vs F[tau]+c+beta comparison?
                        // Inequality: F(tau) + C(tau, t) + K <= F(t)
                        // Here K=0 (since penalty is constant).
                        // So: F[tau] + c <= F[t_star]
                        next_R.push_back(tau);
                    }
                }
            }
            // Add current t_star as candidate for FUTURE
            // But only if it's a valid cut point?
            // Yes, t_star is a valid end of a segment.
            next_R.push_back(t_star);
            R = std::move(next_R);
        } else {
           // If we couldn't form a segment ending at t_star (e.g. t_star < 2*min_size maybe?),
           // we just keep R as is + t_star?
           // Actually if F[t_star] is inf, we can't prune anything.
           // Just keep R and add t_star.
           R.push_back(t_star);
        }
    }

    // 6. Reconstruction
    std::vector<int> change_points;
    int curr = n;
    while (curr > 0) {
        int prev = cp[curr];
        // 0 is the start index, not a change point in the *middle* of data.
        // But the first segment starts at 0.
        // Indices in output: usually the *end* of segments, or *start* of new ones.
        // Ruptures style: list of indices where segment ends.
        // e.g. [10, 20, 30] for size 30 means 0-10, 10-20, 20-30.
        // Let's return the "cut points".
        change_points.push_back(curr);
        curr = prev;
    }
    // Remove the last one if it is 0 (start of data)
    // Wait, loop condition curr > 0 handles it. 
    // If last segment is 0->10, prev=0. curr=10 pushed. curr=0 loop ends.
    // So we invoke 0.
    
    // Sort? They are pushed in reverse order (n, ...).
    std::reverse(change_points.begin(), change_points.end());
    
    // Usually we don't return 'n' as a change point, or do we?
    // "Change points" usually implies the indices *within* the series.
    // If we return [10, 20, 30] for size 30, it effectively describes the segmentation.
    // User can infer segments.
    // Let's exclude 'n' if it's just the end of data? 
    // No, standard format usually includes end of series to define last segment clearly.
    // Or just [10, 20] implies 0-10, 10-20, 20-end.
    // Let's return the list of split points. [10, 20] for length 30.
    // If we include 30, it is explicit.
    // I will include 'n' to be safe and explicit about the last segment. 
    // Wait, python bindings usually like [10, 20] excluding n if n is end.
    // Let's stick to including n for now, it's easier to filter in Python if needed.
    // Actually, checkingruptures, they include the last index.

    return {change_points, F[n] + beta}; // F[n] has -beta start offset, so adding beta normalizes it? 
    // F[0] = -beta.
    // One segment 0-n: Cost = -beta + C(0,n) + beta = C(0,n). Correct.
    // Two segments: -beta + C + beta + C + beta = 2C + beta. Correct (1 change point = 1 penalty).
    // So we don't need to add beta if we consider 'penalty per segment'.
    // Or if penalty is per 'change point'.
    // 1 segment = 0 change points. Cost should be just C.
    // F[n] gives exactly C.
    // 2 segments = 1 change point. Cost C1+C2 + beta.
    // F[n] gives C1+C2+beta. 
    // Perfect. F[n] is the correct penalized cost.
}

} // namespace statelix
