#include "kalman.h"
#include <cmath>
#include <iostream>

namespace statelix {

KalmanFilter::KalmanFilter(int state_d, int measure_d) 
    : state_dim(state_d), measure_dim(measure_d) {
    
    // Initialize with Identity or Zero
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
    H = Eigen::MatrixXd::Zero(measure_dim, state_dim); // User must set this
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measure_dim, measure_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim) * 1000.0; // High uncertainty initial
    x = Eigen::VectorXd::Zero(state_dim);
}

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void KalmanFilter::reset(const Eigen::VectorXd& init_x, const Eigen::MatrixXd& init_P) {
    if(init_x.size() != state_dim || init_P.rows() != state_dim || init_P.cols() != state_dim) {
        throw std::invalid_argument("Dimension mismatch in reset state.");
    }
    x = init_x;
    P = init_P;
}

void KalmanFilter::predict() {
    // x' = Fx
    x = F * x;
    // P' = FPF^T + Q
    P = F * P * F.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    if (H.isZero(1e-9)) {
        throw std::runtime_error("KalmanFilter::update called but Matrix H is zero. Please set H before updating.");
    }

    // y = z - Hx (Innovation)
    Eigen::VectorXd y = z - H * x;
    
    // S = HPH^T + R (Innovation Covariance)
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    
    // K = PH^T S^-1 (Kalman Gain)
    // Using LDLT for inversion since S is symmetric positive definite
    Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(measure_dim, measure_dim));
    
    // x = x + Ky
    x = x + K * y;
    
    // P = (I - KH)P = P - KHP
    // Joseph Form for stability: P = (I - KH)P(I - KH)^T + KRK^T
    // But standard form is cheaper. Let's stick to standard but enforce symmetry.
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);
    P = (I - K * H) * P;
    
    // Enforce symmetry
    P = 0.5 * (P + P.transpose());
}

KalmanResult KalmanFilter::filter(const Eigen::MatrixXd& measurements) {
    int n = measurements.rows(); // Time steps
    int m = state_dim;
    
    // Storage for Forward Pass
    std::vector<Eigen::VectorXd> x_filt(n); // x_{t|t}
    std::vector<Eigen::MatrixXd> P_filt(n); // P_{t|t}
    std::vector<Eigen::VectorXd> x_pred(n); // x_{t|t-1}
    std::vector<Eigen::MatrixXd> P_pred(n); // P_{t|t-1}
    
    double log_likelihood = 0.0;
    
    // Use current x, P as starting point
    
    // --- Forward Pass (Filtering) ---
    for(int i = 0; i < n; ++i) {
        // 1. Predict
        x = F * x;
        P = F * P * F.transpose() + Q;
        
        // Store Priors
        x_pred[i] = x;
        P_pred[i] = P;
        
        // 2. Update
        if (H.isZero(1e-9)) {
             throw std::runtime_error("KalmanFilter: H is zero during batch filter.");
        }

        Eigen::VectorXd z = measurements.row(i);
        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        
        // Robust inversion & Determinant for LogLikelihood
        auto ldlt = S.ldlt();
        Eigen::MatrixXd S_inv = ldlt.solve(Eigen::MatrixXd::Identity(measure_dim, measure_dim));
        
        // K = PH^T S^-1
        Eigen::MatrixXd K = P * H.transpose() * S_inv;
        
        x = x + K * y;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);
        P = (I - K * H) * P;
        
        // Enforce Symmetry
        P = 0.5 * (P + P.transpose());

        // Log Likelihood Accumulation
        // LL += -0.5 * (log|S| + y^T S^-1 y + m*log(2pi))
        double log_det_S = 0.0;
        // LDLT vector D contains diagonal elements of D in LDL^T.
        // Det = Product(diag(D))
        Eigen::VectorXd diagD = ldlt.vectorD();
        for(int k=0; k<diagD.size(); ++k) log_det_S += std::log(diagD(k));
        
        double mahalanobis = y.dot(S_inv * y);
        log_likelihood += -0.5 * (log_det_S + mahalanobis + measure_dim * std::log(2 * M_PI));
        
        // Store Posteriors
        x_filt[i] = x;
        P_filt[i] = P;
    }

    // --- Backward Pass (RTS Smoothing) ---
    std::vector<Eigen::VectorXd> x_smooth = x_filt;
    std::vector<Eigen::MatrixXd> P_smooth = P_filt; 
    
    for (int t = n - 2; t >= 0; --t) {
        // Gain Matrix C_t = P_{t|t} * F^T * P_{t+1|t}^-1
        Eigen::MatrixXd P_pred_next = P_pred[t+1];
        Eigen::MatrixXd P_pred_inv = P_pred_next.ldlt().solve(Eigen::MatrixXd::Identity(m, m));
        
        Eigen::MatrixXd C = P_filt[t] * F.transpose() * P_pred_inv;
        
        // Update Smoothed Estimate
        x_smooth[t] = x_filt[t] + C * (x_smooth[t+1] - x_pred[t+1]);
        
        // Update Smoothed Covariance
        // P_{t|T} = P_{t|t} + C * (P_{t+1|T} - P_{t+1|t}) * C^T
        Eigen::MatrixXd diff = P_smooth[t+1] - P_pred[t+1];
        P_smooth[t] = P_filt[t] + C * diff * C.transpose();
        
        // Ensure symmetry
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].transpose());
    }
    
    // Pack results
    Eigen::MatrixXd filtered_mat(n, m);
    Eigen::MatrixXd smoothed_mat(n, m);
    for(int i = 0; i < n; ++i) {
        filtered_mat.row(i) = x_filt[i];
        smoothed_mat.row(i) = x_smooth[i];
    }

    return {filtered_mat, P_filt, smoothed_mat, P_smooth, log_likelihood};
}

} // namespace statelix
