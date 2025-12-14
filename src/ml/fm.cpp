#include "fm.h"
#include <iostream>

namespace statelix {

// Helper to unpack parameters
// params: [w0 (1) | w (d) | V (d*k)]
void unpack_params(const Eigen::VectorXd& params, int d, int k, 
                   double& w0, Eigen::VectorXd& w, Eigen::MatrixXd& V) {
    w0 = params(0);
    w = params.segment(1, d);
    
    // V is stored as vector, map back to matrix d x k
    // We assume row-major or col-major consistent. 
    // Let's say d rows, k cols. Stored flat.
    // Map with k columns.
    const Eigen::VectorXd& v_flat = params.segment(1 + d, d * k);
    // Note: Eigen Map is column-major by default. We need to be careful.
    // Let's copy to be safe and simple first.
    V = Eigen::MatrixXd::Map(v_flat.data(), d, k); 
}

double FactorizationMachine::predict_one(const Eigen::VectorXd& x, const Eigen::VectorXd& p, 
                                        double w0, const Eigen::VectorXd& w, const Eigen::MatrixXd& V) {
    // Linear term
    double pred = w0 + x.dot(w);
    
    // Interaction term: (1/2) * sum_f ((sum_i v_if x_i)^2 - sum_i v_if^2 x_i^2)
    // Optimized calculation O(k * n)
    // Let's transpose logic:
    // term1 = (x^T * V) .array().square()  -> 1 x k vector
    // term2 = (x.array().square().matrix().transpose() * V.array().square().matrix()) 
    // This is vector heavy. Let's do loop for clarity and speed for small k.
    
    // V is (d x k). x is (d x 1).
    // XV = x^T * V -> (1 x k)
    Eigen::VectorXd XV = x.transpose() * V;
    
    double interaction = 0.0;
    for(int f=0; f < n_factors; ++f) {
        double s1 = XV(f);
        double s2 = x.array().square().matrix().dot(V.col(f).array().square().matrix());
        interaction += (s1 * s1 - s2);
    }
    
    return pred + 0.5 * interaction;
}

Eigen::VectorXd FactorizationMachine::predict(const Eigen::MatrixXd& X) {
    int n = X.rows();
    Eigen::VectorXd preds(n);
    
    double w0;
    Eigen::VectorXd w;
    Eigen::MatrixXd V;
    unpack_params(params, n_features, n_factors, w0, w, V);
    
    for(int i=0; i<n; ++i) {
        double y_pred = predict_one(X.row(i), params, w0, w, V);
        if (task == FMTask::Classification) {
            y_pred = 1.0 / (1.0 + std::exp(-y_pred));
        }
        preds(i) = y_pred;
    }
    return preds;
}

double FactorizationMachine::FMObjective::operator()(const Eigen::VectorXd& p, Eigen::VectorXd& grad) {
    int n = X.rows();
    int d = fm->n_features;
    int k = fm->n_factors;
    
    double w0;
    Eigen::VectorXd w;
    Eigen::MatrixXd V;
    unpack_params(p, d, k, w0, w, V);
    
    grad.setZero(p.size());
    double total_loss = 0.0;
    
    // Gradient pointers
    double& g_w0 = grad(0);
    auto g_w = grad.segment(1, d);
    // Map V gradient part
    Eigen::Map<Eigen::MatrixXd> g_V(grad.data() + 1 + d, d, k);
    
    for(int i=0; i<n; ++i) {
        Eigen::VectorXd x = X.row(i);
        double y_pred = fm->predict_one(x, p, w0, w, V);
        double y_true = y(i);
        
        double error = 0.0;
        double dL_dp = 0.0; // derivative of loss w.r.t prediction
        
        if (fm->task == FMTask::Regression) {
            // MSE: Loss = 0.5*(y - y_est)^2
            error = y_pred - y_true;
            total_loss += 0.5 * error * error;
            dL_dp = error;
        } else {
            // LogLoss: y in {0,1}. -[y log(h) + (1-y)log(1-h)]
            // h = sigmoid(pred)
            // grad w.r.t pred = h - y
            double h = 1.0 / (1.0 + std::exp(-y_pred));
            if (y_true > 0.5) total_loss -= std::log(h + 1e-15);
            else total_loss -= std::log(1.0 - h + 1e-15);
            dL_dp = h - y_true;
        }
        
        // Gradients
        // w0
        g_w0 += dL_dp;
        
        // w
        g_w += dL_dp * x;
        
        // V
        // d(interaction)/dv_if = x_i * (sum_j v_jf x_j) - v_if * x_i^2
        //                      = x_i * (x^T * v_f) - v_if * x_i^2
        for(int f=0; f<k; ++f) {
            double xv_f = x.dot(V.col(f)); // sum_j v_jf x_j
            for(int j=0; j<d; ++j) {
                // gradient for v_jf
                double term = x(j) * xv_f - V(j,f) * x(j) * x(j);
                g_V(j, f) += dL_dp * term;
            }
        }
    }
    
    // Regularization (L2)
    if (fm->reg_w > 0) {
        total_loss += 0.5 * fm->reg_w * w.squaredNorm();
        g_w += fm->reg_w * w;
    }
    if (fm->reg_v > 0) {
        total_loss += 0.5 * fm->reg_v * V.squaredNorm();
        g_V += fm->reg_v * V;
    }
    
    return total_loss;
}

void FactorizationMachine::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    n_features = X.cols();
    int param_size = 1 + n_features + n_features * n_factors;
    
    // Initialize parameters
    // w0 = 0, w = 0, V ~ Normal(0, 0.01)
    params.setZero(param_size);
    
    // Random init for V
    std::srand(42);
    for(int i = 1 + n_features; i < param_size; ++i) {
        params(i) = ((double)std::rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    // Create objective
    FMObjective objective{X, y, this};
    
    // Create Solver (L-BFGS Template)
    LBFGS<FMObjective> solver;
    solver.max_iter = max_iter;
    solver.epsilon = 1e-5;
    
    // Optimize
    OptimizerResult res = solver.minimize(objective, params);
    
    // Store result
    params = res.x;
    std::cout << "FM Converged: " << res.converged << " Iter: " << res.iterations << " Loss: " << res.min_value << std::endl;
}

} // namespace statelix
