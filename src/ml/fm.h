#ifndef STATELIX_FM_H
#define STATELIX_FM_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "../optimization/lbfgs.h"
#include "../quant/quantized.h"

namespace statelix {

enum class FMTask { Regression, Classification };

// Factorization Machine Model
// Uses L-BFGS for optimization
class FactorizationMachine {
public:
    int n_factors = 8;
    int max_iter = 100;
    double learning_rate = 0.01; // Not used for L-BFGS, but placeholder
    double reg_w = 0.0; // L2 regularization for w
    double reg_v = 0.0; // L2 regularization for V
    FMTask task = FMTask::Regression;

    // Parameters (Stored in flattened vector for L-BFGS)
    // Structure: [w0 (1), w (n_features), V (n_features * n_factors)]
    Eigen::VectorXd params;
    int n_features = 0;

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X);

private:
    double predict_one(const Eigen::VectorXd& x, const Eigen::VectorXd& p, 
                       double w0, const Eigen::VectorXd& w, const Eigen::MatrixXd& V);

    // Functor for L-BFGS
    // Functor for L-BFGS
    struct FMObjective {
        const Eigen::MatrixXd& X;
        const Eigen::VectorXd& y;
        FactorizationMachine* fm;
        
        // This operator is called by L-BFGS
        double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& grad);
    };

    // --- Quantization Support ---
    struct QuantizedFMModel; // Forward declaration
    QuantizedFMModel quantize_model() const;
};

// INT8 Quantized FM Model for Inference
struct QuantizedFMModel {
    double w0;
    QuantizedTensor w_q; // (1 x d) - viewed as matrix for matmul
    QuantizedTensor V_q; // (d x k)
    QuantizedTensor V_sq_q; // (d x k) Element-wise squared V
    FMTask task;
    
    // Predict using quantized inputs
    // Input X_q must be (N x d)
    std::vector<float> predict(const QuantizedTensor& X_q);
};

} // namespace statelix

#endif // STATELIX_FM_H
