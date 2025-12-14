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
    int n_samples = X.rows();
    int n_features = X.cols();

    if (n_samples < k) {
        throw std::invalid_argument("n_samples must be >= k");
    }

    // Random initialization of centroids (Randomly pick k samples)
    Eigen::MatrixXd centroids(k, n_features);
    std::mt19937 gen(random_state);
    std::uniform_int_distribution<> dis(0, n_samples - 1);

    std::vector<int> indices;
    while (indices.size() < (size_t)k) {
        int idx = dis(gen);
        bool found = false;
        for (int i : indices) if (i == idx) found = true;
        if (!found) {
            indices.push_back(idx);
            centroids.row(indices.size() - 1) = X.row(idx);
        }
    }

    Eigen::VectorXi labels(n_samples);
    double inertia = 0.0;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        // Assignment step
        bool changed = false;
        double current_inertia = 0.0;
        
        // Temporarily store old centroids to check convergence
        Eigen::MatrixXd old_centroids = centroids;

        // Assign labels
        for (int i = 0; i < n_samples; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;

            for (int j = 0; j < k; ++j) {
                double dist = (X.row(i) - centroids.row(j)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            labels(i) = best_cluster;
            current_inertia += min_dist;
        }

        // Update step
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
                // to prevent getting stuck with a dead centroid.
                std::uniform_int_distribution<> dis(0, n_samples - 1);
                int random_idx = dis(gen);
                new_centroids.row(j) = X.row(random_idx);
            }
        }

        double shift = (new_centroids - centroids).squaredNorm();
        centroids = new_centroids;
        inertia = current_inertia;

        if (shift < tol) {
            break;
        }
    }

    if (iter == max_iter) {
        // Warning: Did not converge
        // In C++, we just return what we have, but could log/warn.
    }

    return {centroids, labels, inertia, iter};
}

} // namespace statelix
