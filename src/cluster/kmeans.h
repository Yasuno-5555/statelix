#pragma once
#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct KMeansResult {
    Eigen::MatrixXd centroids;
    Eigen::VectorXi labels;
    double inertia;
    int n_iter;
    bool converged;  ///< True if converged before max_iter
};

// K-Means clustering using Lloyd's algorithm
KMeansResult fit_kmeans(
    const Eigen::MatrixXd& X,
    int k,
    int max_iter = 300,
    double tol = 1e-4,
    int random_state = 42
);

} // namespace statelix
