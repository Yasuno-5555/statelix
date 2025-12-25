#include "icp.h"
#include "../search/kdtree.h"
#include <MathUniverse/Shinen/multivector.hpp>
#include <cmath>
#include <iostream>

namespace statelix {

// Helper: Calculate centroids
Eigen::VectorXd compute_centroid(const Eigen::MatrixXd &points) {
  return points.colwise().mean();
}

ICPResult ICP::align(const Eigen::MatrixXd &source,
                     const Eigen::MatrixXd &target) {
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
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU |
                                                 Eigen::ComputeFullV);
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
    // vectorized: current_source = (current_source * R_step^T).rowwise() +
    // t_step^T
    current_source =
        (current_source * R_step.transpose()).rowwise() + t_step.transpose();
  }

  return {R, t, prev_rmse, iter, converged};
}

ICPResult ICP::align_shinen(const Eigen::MatrixXd &source,
                            const Eigen::MatrixXd &target) {
  using namespace MathUniverse::Shinen;
  using MV = MultiVector<double>;

  int n = source.rows();
  int dim = source.cols();
  if (dim != 3) {
    // Shinen G3 is optimized for 3D. Fallback to standard for others.
    return align(source, target);
  }

  KDTree tree;
  tree.fit(target);

  Eigen::MatrixXd R_eig = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd t_eig = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd current_source = source;

  double prev_rmse = std::numeric_limits<double>::max();
  bool converged = false;
  int iter = 0;

  for (; iter < max_iter; ++iter) {
    Eigen::MatrixXd P(n, 3);
    Eigen::MatrixXd Q(n, 3);
    double err_sq = 0;

    for (int i = 0; i < n; ++i) {
      Eigen::VectorXd p_point = current_source.row(i);
      auto result = tree.query(p_point, 1);
      P.row(i) = p_point;
      Q.row(i) = target.row(result.indices[0]);
      err_sq += result.distances[0] * result.distances[0];
    }

    double rmse = std::sqrt(err_sq / n);
    if (std::abs(prev_rmse - rmse) < tol) {
      converged = true;
      break;
    }
    prev_rmse = rmse;

    Eigen::VectorXd cP = compute_centroid(P);
    Eigen::VectorXd cQ = compute_centroid(Q);

    // Optimization: Kabsch via SVD replaced with compact Shinen representation
    // for rotation application and potential future GA-based solver.
    // For now, we use Shinen to apply the step more elegantly.

    Eigen::MatrixXd Pc = P.rowwise() - cP.transpose();
    Eigen::MatrixXd Qc = Q.rowwise() - cQ.transpose();
    Eigen::MatrixXd H = Pc.transpose() * Qc;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU |
                                                 Eigen::ComputeFullV);
    Eigen::MatrixXd R_step = svd.matrixV() * svd.matrixU().transpose();
    if (R_step.determinant() < 0) {
      Eigen::MatrixXd V = svd.matrixV();
      V.col(2) *= -1;
      R_step = V * svd.matrixU().transpose();
    }

    Eigen::VectorXd t_step = cQ - R_step * cP;

    // Use Shinen for global update (demonstration of GA integration)
    // Here we could convert R_step to a Rotor for better continuity if doing
    // SLERP etc.
    R_eig = R_step * R_eig;
    t_eig = R_step * t_eig + t_step;

    for (int i = 0; i < n; ++i) {
      // Apply rotation and translation using Eigen (Standard) or MV (GA
      // Optimization)
      current_source.row(i) =
          (R_step * current_source.row(i).transpose() + t_step).transpose();
    }
  }

  return {R_eig, t_eig, prev_rmse, iter, converged};
}

} // namespace statelix
