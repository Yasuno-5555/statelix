#include "icp.h"
#include "../search/kdtree.h"
#include <iostream>
#include <cmath>

namespace statelix {

// Helper: Calculate centroids
Eigen::VectorXd compute_centroid(const Eigen::MatrixXd& points) {
    return points.colwise().mean();
}

ICPResult ICP::align(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target) {
    int n = source.rows();
    int dim = source.cols();

    // 1. Build KD-Tree on Target for fast lookup
    KDTree tree;
    tree.fit(target);

    // Initial Transformation
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::VectorXd t = Eigen::VectorXd::Zero(dim);

    // Current State of Source
    Eigen::MatrixXd current_source = source;
    
    double prev_rmse = std::numeric_limits<double>::max();
    bool converged = false;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        // A. Find Correspondences
        // For each point in current_source, find closest point in target.
        // We accumulate centered points to run Kabsch SVD directly.
        
        Eigen::MatrixXd P(n, dim); // Correspondences in Source (Current State)
        Eigen::MatrixXd Q(n, dim); // Correspondences in Target (Nearest Neighbors)
        
        double current_error_sq_sum = 0.0;
        
        // This loop can be parallelized via OpenMP if needed
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd p_point = current_source.row(i);
            
            // Query 1-NN
            auto result = tree.query(p_point, 1);
            int nn_idx = result.indices[0]; // Index in target
            double dist = result.distances[0];
            
            P.row(i) = p_point;
            Q.row(i) = target.row(nn_idx);
            
            current_error_sq_sum += dist * dist;
        }
        
        double rmse = std::sqrt(current_error_sq_sum / n);

        // Check Convergence
        if (std::abs(prev_rmse - rmse) < tol) {
            converged = true;
            break;
        }
        prev_rmse = rmse;
        
        // B. Estimate Rigid Transformation (Kabsch Algorithm)
        // 1. Centroids
        Eigen::VectorXd centroid_P = compute_centroid(P);
        Eigen::VectorXd centroid_Q = compute_centroid(Q);
        
        // 2. Center the points
        Eigen::MatrixXd P_centered = P.rowwise() - centroid_P.transpose();
        Eigen::MatrixXd Q_centered = Q.rowwise() - centroid_Q.transpose();
        
        // 3. Covariance Matrix H = P_centered^T * Q_centered
        Eigen::MatrixXd H = P_centered.transpose() * Q_centered;
        
        // 4. SVD of H = U S V^T
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        
        // 5. Rotation R_new = V * U^T
        Eigen::MatrixXd R_step = V * U.transpose();
        
        // Check for reflection (determinant should be +1)
        if (R_step.determinant() < 0) {
            V.col(dim - 1) *= -1;
            R_step = V * U.transpose();
        }
        
        // 6. Translation t_new = centroid_Q - R_step * centroid_P
        Eigen::VectorXd t_step = centroid_Q - R_step * centroid_P;
        
        // C. Update Global Transformation
        // New Global R = R_step * Old Global R
        // New Global t = R_step * Old Global t + t_step
        R = R_step * R;
        t = R_step * t + t_step;
        
        // D. Apply Transformation to Source for next iteration
        // current_source = (R_step * current_source^T)^T + t_step^T ??
        // Actually easier: just transform current_source directly using R_step and t_step
        // P_new = R_step * P + t_step
        for (int i = 0; i < n; ++i) {
            current_source.row(i) = (R_step * current_source.row(i).transpose()).transpose() + t_step.transpose();
        }
    }

    return {R, t, prev_rmse, iter, converged};
}

} // namespace statelix
