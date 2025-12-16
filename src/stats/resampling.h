/**
 * @file resampling.h
 * @brief Statelix v2.3 - Resampling Methods
 * 
 * Implements:
 *   - Bootstrap (i.i.d.)
 *   - Block Bootstrap (Time Series)
 *   - Jackknife
 *   - Permutation Tests
 * 
 * Features:
 *   - Parallel execution via OpenMP
 *   - Template-based generic statistics support
 */

#ifndef STATELIX_RESAMPLING_H
#define STATELIX_RESAMPLING_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>
#include <future>
#include <thread>

// Optional OpenMP support
#ifdef _OPENMP
#include <omp.h>
#endif

namespace statelix {

class Resampler {
public:
    unsigned int seed = 42;
    bool parallel = true;
    int n_jobs = -1;  // -1 means use all available cores

    Resampler() {}
    
    // =========================================================================
    // Bootstrap
    // =========================================================================
    
    /**
     * @brief I.I.D. Bootstrap
     * 
     * @param data Matrix (n_samples, n_features) or Vector
     * @param func Function that takes bootstrapped data and returns a statistic (VectorXd)
     * @param n_reps Number of bootstrap repetitions
     * @return Matrix of statistics (n_reps, stat_dim)
     */
    template <typename Derived, typename Func>
    Eigen::MatrixXd bootstrap(
        const Eigen::MatrixBase<Derived>& data,
        Func func,
        int n_reps = 1000
    ) {
        int n = data.rows();
        int cols = data.cols();
        
        // Run once to determine output size
        Eigen::VectorXd sample_stat = func(data);
        int stat_dim = sample_stat.size();
        
        Eigen::MatrixXd results(n_reps, stat_dim);
        
        // Prepare seeds for parallel execution
        std::vector<unsigned int> seeds(n_reps);
        std::mt19937 master_gen(seed);
        for(int i=0; i<n_reps; ++i) seeds[i] = master_gen();
        
        int n_threads = determine_threads();
        
        #pragma omp parallel for num_threads(n_threads) if(parallel)
        for (int b = 0; b < n_reps; ++b) {
            std::mt19937 gen(seeds[b]);
            std::uniform_int_distribution<> dist(0, n - 1);
            
            // Resample
            // Note: For large data, we might want to pass indices to func instead to avoid copy,
            // but for generality we copy data now.
            typename Derived::PlainObject resampled_data(n, cols);
            for (int i = 0; i < n; ++i) {
                resampled_data.row(i) = data.row(dist(gen));
            }
            
            results.row(b) = func(resampled_data);
        }
        
        return results;
    }
    
    // =========================================================================
    // Time Series Bootstrap (Block Bootstrap)
    // =========================================================================
    
    /**
     * @brief Moving Block Bootstrap (MBB) for time series
     * 
     * Preserves local dependency structure.
     */
    template <typename Derived, typename Func>
    Eigen::MatrixXd block_bootstrap(
        const Eigen::MatrixBase<Derived>& data,
        Func func,
        int block_size,
        int n_reps = 1000
    ) {
        int n = data.rows();
        int cols = data.cols();
        
        // Number of possible overlapping blocks
        int n_blocks = n - block_size + 1;
        if (n_blocks <= 0) throw std::invalid_argument("Block size too large for data");
        
        // How many blocks needed to reconstruct time series of length approx n
        int k = (n + block_size - 1) / block_size;
        
        Eigen::VectorXd sample_stat = func(data);
        int stat_dim = sample_stat.size();
        Eigen::MatrixXd results(n_reps, stat_dim);
        
        std::vector<unsigned int> seeds(n_reps);
        std::mt19937 master_gen(seed);
        for(int i=0; i<n_reps; ++i) seeds[i] = master_gen();
        
        int n_threads = determine_threads();
        
        #pragma omp parallel for num_threads(n_threads) if(parallel)
        for (int b = 0; b < n_reps; ++b) {
            std::mt19937 gen(seeds[b]);
            std::uniform_int_distribution<> dist(0, n_blocks - 1);
            
            typename Derived::PlainObject resampled_data(n, cols);
            
            int current_idx = 0;
            for (int i = 0; i < k; ++i) {
                // Pick a start index for the block
                int start_idx = dist(gen);
                
                // Copy block
                int copy_len = std::min(block_size, n - current_idx);
                resampled_data.block(current_idx, 0, copy_len, cols) = 
                    data.block(start_idx, 0, copy_len, cols);
                
                current_idx += copy_len;
                if (current_idx >= n) break;
            }
            
            results.row(b) = func(resampled_data);
        }
        
        return results;
    }
    
    // =========================================================================
    // Jackknife
    // =========================================================================
    
    /**
     * @brief Delete-1 Jackknife
     * 
     * @return Matrix of statistics (n_samples, stat_dim)
     */
    template <typename Derived, typename Func>
    Eigen::MatrixXd jackknife(
        const Eigen::MatrixBase<Derived>& data,
        Func func
    ) {
        int n = data.rows();
        int cols = data.cols();
        
        // Run once
        Eigen::VectorXd sample_stat = func(data);
        int stat_dim = sample_stat.size();
        Eigen::MatrixXd results(n, stat_dim);
        
        int n_threads = determine_threads();
        
        #pragma omp parallel for num_threads(n_threads) if(parallel)
        for (int i = 0; i < n; ++i) {
            // Construct Leave-One-Out dataset
            typename Derived::PlainObject loo_data(n - 1, cols);
            if (i > 0) loo_data.topRows(i) = data.topRows(i);
            if (i < n - 1) loo_data.bottomRows(n - 1 - i) = data.bottomRows(n - 1 - i);
            
            results.row(i) = func(loo_data);
        }
        
        return results;
    }
    
    // =========================================================================
    // Utilities
    // =========================================================================
    
    /**
     * @brief Compute Bias-Corrected Accelerated (BCa) Confidence Intervals
     * 
     * Requires both Bootstrap Results and Jackknife Results (for acceleration)
     */
    static Eigen::MatrixXd computed_bca_interval(
        const Eigen::MatrixXd& boot_dist, // (B, p)
        const Eigen::VectorXd& theta_hat, // (p,) original estimate
        const Eigen::MatrixXd& jack_dist, // (n, p) jackknife estimates
        double alpha = 0.05
    ) {
        int B = boot_dist.rows();
        int p = boot_dist.cols();
        int n = jack_dist.rows();
        
        Eigen::MatrixXd intervals(p, 2);
        
        for (int j = 0; j < p; ++j) {
            // Bias correction z0
            int less = 0;
            for(int b=0; b<B; ++b) {
                if (boot_dist(b, j) < theta_hat(j)) less++;
            }
            double z0 = inverse_normal_cdf(static_cast<double>(less) / B);
            
            // Acceleration a via Jackknife
            double theta_jack_mean = jack_dist.col(j).mean();
            double num = 0, den = 0;
            for(int i=0; i<n; ++i) {
                double diff = theta_jack_mean - jack_dist(i, j);
                num += diff * diff * diff;
                den += diff * diff;
            }
            double a = num / (6.0 * std::pow(den, 1.5));
            if (std::abs(den) < 1e-12) a = 0; // Handle zero variance
            
            // Interval endpoints
            double z_alpha = inverse_normal_cdf(alpha / 2.0);
            double z_1alpha = inverse_normal_cdf(1.0 - alpha / 2.0);
            
            double pct1 = normal_cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)));
            double pct2 = normal_cdf(z0 + (z0 + z_1alpha) / (1.0 - a * (z0 + z_1alpha)));
            
            // Get quantiles from bootstrap distribution
            std::vector<double> dist_j(B);
            for(int b=0; b<B; ++b) dist_j[b] = boot_dist(b, j);
            std::sort(dist_j.begin(), dist_j.end());
            
            int idx1 = static_cast<int>(pct1 * (B - 1));
            int idx2 = static_cast<int>(pct2 * (B - 1));
            
            intervals(j, 0) = dist_j[std::max(0, std::min(B-1, idx1))];
            intervals(j, 1) = dist_j[std::max(0, std::min(B-1, idx2))];
        }
        
        return intervals;
    }
    
private:
    int determine_threads() const {
        if (!parallel) return 1;
        if (n_jobs > 0) return n_jobs;
        return std::thread::hardware_concurrency();
    }
    
    static double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    
    static double inverse_normal_cdf(double p) {
        // Approximate implementation
        if (p <= 0.0) return -1e9;
        if (p >= 1.0) return 1e9;
        
        double a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
        double a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
        double b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
        double b4 = 66.8013118877197, b5 = -13.2806815528857, c1 = -7.78489400243029e-03;
        double c2 = -0.322396458041136, c3 = -2.40075827716184, c4 = -2.54973253934373;
        double c5 = 4.37466414146497, c6 = 2.93816398269878, d1 = 7.78469570904146e-03;
        double d2 = 0.32246712907004, d3 = 2.445134137143, d4 = 3.75440866190742;
        double p_low = 0.02425, p_high = 1 - 0.02425;
        
        double q, r;
        if (p < p_low) {
            q = std::sqrt(-2 * std::log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        } else if (p <= p_high) {
            q = p - 0.5;
            r = q * q;
            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
        } else {
            q = std::sqrt(-2 * std::log(1 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                    ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        }
    }
};

} // namespace statelix

#endif // STATELIX_RESAMPLING_H
