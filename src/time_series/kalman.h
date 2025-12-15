#ifndef STATELIX_KALMAN_H
#define STATELIX_KALMAN_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct KalmanResult {
    Eigen::MatrixXd states;          // Estimated states (n_samples x state_dim)
    std::vector<Eigen::MatrixXd> cov_states; // Estimated covariances P_{t|t}
    
    Eigen::MatrixXd smoothed_states; // RTS Smoothed states
    std::vector<Eigen::MatrixXd> cov_smoothed; // Smoothed covariances P_{t|T}

    double log_likelihood;
};

class KalmanFilter {
public:
    // Dimensions
    int state_dim;
    int measure_dim;

    // Model Matrices
    Eigen::MatrixXd F; // State Transition (state x state)
    Eigen::MatrixXd H; // Measurement (measure x state)
    Eigen::MatrixXd Q; // Process Noise (state x state)
    Eigen::MatrixXd R; // Measurement Noise (measure x measure)
    
    // State
    Eigen::MatrixXd P; // Estimate Covariance (state x state)
    Eigen::VectorXd x; // State Estimate (state)

    KalmanFilter(int state_d, int measure_d);
    
    // Reset state to initial conditions (useful before running batch filter)
    void reset(const Eigen::VectorXd& init_x, const Eigen::MatrixXd& init_P);

    // Single step
    void predict();
    void update(const Eigen::VectorXd& z);

    // Batch processing
    KalmanResult filter(const Eigen::MatrixXd& measurements);
};

} // namespace statelix

#endif // STATELIX_KALMAN_H
