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

void KalmanFilter::predict() {
    // x' = Fx
    x = F * x;
    // P' = FPF^T + Q
    P = F * P * F.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    // y = z - Hx (Innovation)
    Eigen::VectorXd y = z - H * x;
    
    // S = HPH^T + R (Innovation Covariance)
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    
    // K = PH^T S^-1 (Kalman Gain)
    // Using LDLT or LLT for inversion since S is symmetric positive definite
    Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(measure_dim, measure_dim));
    
    // x = x + Ky
    x = x + K * y;
    
    // P = (I - KH)P = P - KHP
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);
    P = (I - K * H) * P;
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
    
    // --- Forward Pass (Filtering) ---
    for(int i = 0; i < n; ++i) {
        // 1. Predict
        x = F * x;
        P = F * P * F.transpose() + Q;
        
        // Store Priors
        x_pred[i] = x;
        P_pred[i] = P;
        
        // 2. Update
        Eigen::VectorXd z = measurements.row(i);
        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        
        // Robust inversion
        Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(measure_dim, measure_dim));
        
        x = x + K * y;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);
        P = (I - K * H) * P;
        
        // Store Posteriors
        x_filt[i] = x;
        P_filt[i] = P;
    }

    // --- Backward Pass (RTS Smoothing) ---
    // x_{t|T} = x_{t|t} + C_t * (x_{t+1|T} - x_{t+1|t})
    // P_{t|T} = P_{t|t} + C_t * (P_{t+1|T} - P_{t+1|t}) * C_t^T
    // C_t = P_{t|t} * F^T * P_{t+1|t}^-1
    
    std::vector<Eigen::VectorXd> x_smooth = x_filt;
    // P_smooth is not strictly needed for just states but good for confidence intervals.
    // We only return states in the struct currently, but computation requires P.
    std::vector<Eigen::MatrixXd> P_smooth = P_filt; 
    
    for (int t = n - 2; t >= 0; --t) {
        // Gain Matrix C_t
        // P_{t+1|t}^-1
        Eigen::MatrixXd P_pred_next = P_pred[t+1];
        // Use Pseudo-inverse or LDLT solve for stability
        Eigen::MatrixXd P_pred_inv = P_pred_next.ldlt().solve(Eigen::MatrixXd::Identity(m, m));
        
        Eigen::MatrixXd C = P_filt[t] * F.transpose() * P_pred_inv;
        
        // Update Smoothed Estimate
        x_smooth[t] = x_filt[t] + C * (x_smooth[t+1] - x_pred[t+1]);
        
        // Update Smoothed Covariance (optional for states but needed for loop if we used P in C, but we use P_filt[t])
        // C depends on P_filt[t], not P_smooth[t+1].
        // Equation check: C_t depends on P_{t|t} (known) and P_{t+1|t} (known).
        // Update for P_smooth[t] depends on P_smooth[t+1].
        // Even if we don't return P_smooth, we might need it? No, x_smooth only depends on x_smooth[t+1].
        // So we only need to update x_smooth.
        // BUT, if we want accurate intervals later, we should compute P.
        // Let's compute P for correctness future-proofing.
        P_smooth[t] = P_filt[t] + C * (P_smooth[t+1] - P_pred[t+1]) * C.transpose();
    }
    
    // Pack results
    Eigen::MatrixXd filtered_mat(n, m);
    Eigen::MatrixXd smoothed_mat(n, m);
    for(int i = 0; i < n; ++i) {
        filtered_mat.row(i) = x_filt[i];
        smoothed_mat.row(i) = x_smooth[i];
    }

    return {filtered_mat, smoothed_mat, log_likelihood};
}

} // namespace statelix
