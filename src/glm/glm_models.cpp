#include "glm_models.h"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace statelix {

// Helper: Probit functions
namespace {
    double phi(double x) {
        return std::exp(-0.5 * x * x) / std::sqrt(2 * M_PI);
    }
    double Phi(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
}

// 1. Logistic Regression
LogisticResult LogisticRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    bool converged = false;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        Eigen::VectorXd eta = X * coef;
        Eigen::VectorXd mu = 1.0 / (1.0 + (-eta.array()).exp());

        // Weights
        Eigen::VectorXd W_diag = mu.array() * (1.0 - mu.array());
        
        // Avoid division by zero
        for(int i=0; i<W_diag.size(); ++i) {
             if(W_diag(i) < 1e-8) W_diag(i) = 1e-8; 
        }

        // Adjusted response
        Eigen::VectorXd z = eta.array() + (y - mu).array() / W_diag.array();

        // Solve Weighted Least Squares: (X'WX) d = X'Wz -> new_coef
        // X'WX = X' * diag(W) * X
        Eigen::MatrixXd XTW = X.transpose() * W_diag.asDiagonal();
        Eigen::MatrixXd XTWX = XTW * X;
        Eigen::VectorXd XTWz = XTW * z;

        Eigen::VectorXd new_coef = XTWX.ldlt().solve(XTWz);

        if ((new_coef - coef).norm() < tol) {
            coef = new_coef;
            converged = true;
            break;
        }
        coef = new_coef;
    }
    return {coef, iter, converged};
}

Eigen::VectorXd LogisticRegression::predict_prob(const Eigen::MatrixXd& X, const Eigen::VectorXd& coef) {
    return 1.0 / (1.0 + (-X * coef).array().exp());
}

// 2. Poisson Regression
PoissonResult PoissonRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    bool converged = false;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        Eigen::VectorXd eta = X * coef;
        Eigen::VectorXd mu = eta.array().exp();

        // Weights W = mu
        Eigen::VectorXd W_diag = mu;
         for(int i=0; i<W_diag.size(); ++i) {
             if(W_diag(i) < 1e-8) W_diag(i) = 1e-8; 
        }

        Eigen::VectorXd z = eta.array() + (y - mu).array() / mu.array();

        Eigen::MatrixXd XTW = X.transpose() * W_diag.asDiagonal();
        Eigen::MatrixXd XTWX = XTW * X;
        Eigen::VectorXd XTWz = XTW * z;

        Eigen::VectorXd new_coef = XTWX.ldlt().solve(XTWz);

        if ((new_coef - coef).norm() < tol) {
            coef = new_coef;
            converged = true;
            break;
        }
        coef = new_coef;
    }
    return {coef, iter, converged};
}

// 3. Negative Binomial Regression
// TODO: Implement theta estimation (currently fixed at 1.0)
NegBinResult NegBinRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    double theta = 1.0; 
    bool converged = false;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        Eigen::VectorXd eta = X * coef;
        Eigen::VectorXd mu = eta.array().exp();

        // Weights W = mu / (1 + mu/theta)
        Eigen::VectorXd W_diag = mu.array() / (1.0 + mu.array() / theta);
         for(int i=0; i<W_diag.size(); ++i) {
             if(W_diag(i) < 1e-8) W_diag(i) = 1e-8; 
        }

        Eigen::VectorXd z = eta.array() + (y - mu).array() / mu.array();

        Eigen::MatrixXd XTW = X.transpose() * W_diag.asDiagonal();
        Eigen::MatrixXd XTWX = XTW * X;
        Eigen::VectorXd XTWz = XTW * z;

        Eigen::VectorXd new_coef = XTWX.ldlt().solve(XTWz);

        if ((new_coef - coef).norm() < tol) {
            coef = new_coef;
            converged = true;
            break;
        }
        coef = new_coef;
    }
    return {coef, theta, iter, converged};
}

// 4. Gamma Regression
GammaResult GammaRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p); // Init with linear regression usually better
    bool converged = false;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        Eigen::VectorXd eta = X * coef;
        Eigen::VectorXd mu = eta.array().exp();

        // Weights W = 1/mu^2
        Eigen::VectorXd W_diag = 1.0 / mu.array().square();
         for(int i=0; i<W_diag.size(); ++i) {
             if(W_diag(i) < 1e-8) W_diag(i) = 1e-8; 
        }

        Eigen::VectorXd z = eta.array() + (y - mu).array() / mu.array();

        Eigen::MatrixXd XTW = X.transpose() * W_diag.asDiagonal();
        Eigen::MatrixXd XTWX = XTW * X;
        Eigen::VectorXd XTWz = XTW * z;

        Eigen::VectorXd new_coef = XTWX.ldlt().solve(XTWz);

        if ((new_coef - coef).norm() < tol) {
            coef = new_coef;
            converged = true;
            break;
        }
        coef = new_coef;
    }
    return {coef, iter, converged};
}

// 5. Probit Regression
ProbitResult ProbitRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int p = X.cols();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    bool converged = false;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        Eigen::VectorXd eta = X * coef;
        Eigen::VectorXd z(eta.size());
        Eigen::VectorXd W_diag(eta.size());
        
        for (int j = 0; j < eta.size(); j++) {
            double prob = Phi(eta[j]); // p
            double pdf = phi(eta[j]);  // phi
            
            // Clip probability
            if(prob < 1e-8) prob = 1e-8;
            if(prob > 1.0 - 1e-8) prob = 1.0 - 1e-8;
            
            // Clip Gradient pdf
            if(pdf < 1e-8) pdf = 1e-8;

            // Working Weights: W = pdf^2 / (p * (1-p))
            W_diag[j] = (pdf * pdf) / (prob * (1.0 - prob));
            
            // Working Response: z = eta + (y - p) / pdf
            // Note: If pdf is tiny, this explodes. Trust clipping.
            z[j] = eta[j] + (y[j] - prob) / pdf;
        }

        Eigen::MatrixXd XTW = X.transpose() * W_diag.asDiagonal();
        Eigen::MatrixXd XTWX = XTW * X;
        Eigen::VectorXd XTWz = XTW * z;

        Eigen::VectorXd new_coef = XTWX.ldlt().solve(XTWz);

        if ((new_coef - coef).norm() < tol) {
            coef = new_coef;
            converged = true;
            break;
        }
        coef = new_coef;
    }
    return {coef, iter, converged};
}

} // namespace statelix
