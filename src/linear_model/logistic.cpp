/**
 * @file logistic.cpp
 * @brief Logistic Regression using GLMSolver (IRLS)
 * 
 * Refactored to use statelix's unified GLM framework.
 */
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include "../glm/glm_solver.h"
#include "../glm/glm_base.h"

namespace statelix {

struct LogisticResult {
    Eigen::VectorXd coef;
    double intercept = 0.0;
    int iterations;
    bool converged;
    double deviance = 0.0;
    double aic = 0.0;
};

class LogisticRegression {
public:
    int max_iter = 100;
    double tol = 1e-6;
    bool fit_intercept = true;
    
    LogisticResult fit(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y
    ) {
        // Use GLMSolver with Binomial family and Logit link
        DenseGLMSolver solver;
        solver.family = std::make_unique<BinomialFamily>();
        solver.link = std::make_unique<LogitLink>();
        solver.fit_intercept = fit_intercept;
        solver.max_iter = max_iter;
        solver.tol = tol;
        
        GLMResult glm_result = solver.fit(X, y);
        
        // Convert to LogisticResult
        LogisticResult result;
        result.coef = glm_result.coef;
        result.intercept = glm_result.intercept;
        result.iterations = glm_result.iterations;
        result.converged = glm_result.converged;
        result.deviance = glm_result.deviance;
        result.aic = glm_result.aic;
        
        return result;
    }
    
    Eigen::VectorXd predict_prob(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& coef,
        double intercept = 0.0
    ) {
        Eigen::VectorXd eta = X * coef;
        if (fit_intercept) {
            eta.array() += intercept;
        }
        
        // Apply logistic function
        Eigen::VectorXd probs(eta.size());
        for (int i = 0; i < eta.size(); ++i) {
            if (eta(i) >= 0) {
                double ez = std::exp(-eta(i));
                probs(i) = 1.0 / (1.0 + ez);
            } else {
                double ez = std::exp(eta(i));
                probs(i) = ez / (1.0 + ez);
            }
        }
        return probs;
    }
};

} // namespace statelix
