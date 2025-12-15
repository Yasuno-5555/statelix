#include "kmeans.h"
#include <random>
#include <limits>
#include <iostream>

namespace statelix {

KMeansResult fit_kmeans(
    const Eigen::MatrixXd& X,
    int k,
    int max_iter,
    double tol,
    int random_state
) {
    const int n_samples = static_cast<int>(X.rows());
    const int n_features = static_cast<int>(X.cols());

    if (n_samples < k) {
        throw std::invalid_argument("n_samples must be >= k");
    }

    // Random initialization of centroids (Randomly pick k samples)
    Eigen::MatrixXd centroids(k, n_features);
    std::mt19937 gen(static_cast<unsigned int>(random_state));
    std::uniform_int_distribution<> sample_dis(0, n_samples - 1);

    std::vector<int> indices;
    indices.reserve(k);
    while (static_cast<int>(indices.size()) < k) {
        int idx = sample_dis(gen);
        bool found = false;
        for (int i : indices) {
            if (i == idx) {
                found = true;
                break;
            }
        }
        if (!found) {
            indices.push_back(idx);
            centroids.row(static_cast<int>(indices.size()) - 1) = X.row(idx);
        }
    }

    Eigen::VectorXi labels(n_samples);
    double inertia = 0.0;
    int iter = 0;
    bool converged = false;

    // Precompute squared norms of data points for efficient distance calculation
    // ||x - c||² = ||x||² - 2*x·c + ||c||²
    Eigen::VectorXd X_sq_norms = X.rowwise().squaredNorm();

    for (; iter < max_iter; ++iter) {
        // ========== VECTORIZED ASSIGNMENT STEP ==========
        // Compute all pairwise squared distances: dists(i, j) = ||X[i] - centroids[j]||²
        // Using: ||x - c||² = ||x||² + ||c||² - 2*x·c
        
        Eigen::VectorXd centroid_sq_norms = centroids.rowwise().squaredNorm();
        
        // X * centroids^T gives (n_samples x k) matrix of dot products
        Eigen::MatrixXd dot_products = X * centroids.transpose();
        
        // dists(i, j) = X_sq_norms(i) + centroid_sq_norms(j) - 2 * dot_products(i, j)
        Eigen::MatrixXd dists = (-2.0 * dot_products).rowwise() + centroid_sq_norms.transpose();
        dists.colwise() += X_sq_norms;

        // Find argmin for each row and compute inertia
        double current_inertia = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            Eigen::Index min_idx;
            double min_dist = dists.row(i).minCoeff(&min_idx);
            labels(i) = static_cast<int>(min_idx);
            current_inertia += min_dist;
        }

        // ========== UPDATE STEP ==========
        Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(k, n_features);
        Eigen::VectorXd counts = Eigen::VectorXd::Zero(k);

        for (int i = 0; i < n_samples; ++i) {
            new_centroids.row(labels(i)) += X.row(i);
            counts(labels(i)) += 1.0;
        }

        for (int j = 0; j < k; ++j) {
            if (counts(j) > 0) {
                new_centroids.row(j) /= counts(j);
            } else {
                // Handle empty cluster: Re-initialize to a random sample
                int random_idx = sample_dis(gen);
                new_centroids.row(j) = X.row(random_idx);
            }
        }

        // Check convergence: sum of squared centroid shifts
        double shift = (new_centroids - centroids).squaredNorm();
        centroids = new_centroids;
        inertia = current_inertia;

        if (shift < tol) {
            converged = true;
            ++iter;  // Count this iteration
            break;
        }
    }

    return {centroids, labels, inertia, iter, converged};
}

} // namespace statelix

