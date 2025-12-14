/**
 * @file glm_legacy.h
 * @brief Statelix v1.1 - Legacy GLM API Wrappers
 * 
 * DEPRECATED: These classes wrap the new GLMSolver for backward compatibility.
 * They will be REMOVED in v1.2.
 * 
 * Migration guide:
 *   OLD: LogisticRegression().fit(X, y)
 *   NEW: GLMSolver<> s;
 *        s.family = make_unique<BinomialFamily>();
 *        s.link = make_unique<LogitLink>();
 *        s.fit(X, y);
 */
#ifndef STATELIX_GLM_LEGACY_H
#define STATELIX_GLM_LEGACY_H

#include "glm_base.h"
#include "glm_solver.h"
#include <memory>

// Suppression for deprecated warnings during internal use
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace statelix {

// =============================================================================
// Legacy Result Structures (thin wrappers)
// =============================================================================

struct LogisticResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

struct PoissonResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

struct NegBinResult {
    Eigen::VectorXd coef;
    double theta;
    int iterations;
    bool converged;
};

struct GammaResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

struct ProbitResult {
    Eigen::VectorXd coef;
    int iterations;
    bool converged;
};

// =============================================================================
// Legacy Classes (deprecated)
// =============================================================================

/**
 * @deprecated Use GLMSolver with BinomialFamily + LogitLink
 */
class [[deprecated("Use GLMSolver with BinomialFamily + LogitLink in v1.1+")]]
LogisticRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;
    
    LogisticResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        GLMSolver<> solver;
        solver.family = std::make_unique<BinomialFamily>();
        solver.link = std::make_unique<LogitLink>();
        solver.max_iter = max_iter;
        solver.tol = tol;
        
        auto result = solver.fit(X, y);
        return {result.coef, result.iterations, result.converged};
    }
    
    Eigen::VectorXd predict_prob(const Eigen::MatrixXd& X, 
                                  const Eigen::VectorXd& coef) {
        LogitLink link;
        Eigen::VectorXd eta = X * coef;
        return link.inverse(eta);
    }
};

/**
 * @deprecated Use GLMSolver with PoissonFamily + LogLink
 */
class [[deprecated("Use GLMSolver with PoissonFamily + LogLink in v1.1+")]]
PoissonRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    PoissonResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        GLMSolver<> solver;
        solver.family = std::make_unique<PoissonFamily>();
        solver.link = std::make_unique<LogLink>();
        solver.max_iter = max_iter;
        solver.tol = tol;
        
        auto result = solver.fit(X, y);
        return {result.coef, result.iterations, result.converged};
    }
};

/**
 * @deprecated Use GLMSolver with NegativeBinomialFamily + LogLink
 */
class [[deprecated("Use GLMSolver with NegativeBinomialFamily + LogLink in v1.1+")]]
NegBinRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    NegBinResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        GLMSolver<> solver;
        auto nb_family = std::make_unique<NegativeBinomialFamily>(1.0);
        double theta = nb_family->theta;
        solver.family = std::move(nb_family);
        solver.link = std::make_unique<LogLink>();
        solver.max_iter = max_iter;
        solver.tol = tol;
        
        auto result = solver.fit(X, y);
        return {result.coef, theta, result.iterations, result.converged};
    }
};

/**
 * @deprecated Use GLMSolver with GammaFamily
 */
class [[deprecated("Use GLMSolver with GammaFamily in v1.1+")]]
GammaRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    GammaResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        GLMSolver<> solver;
        solver.family = std::make_unique<GammaFamily>();
        solver.link = std::make_unique<LogLink>();  // Log is common default
        solver.max_iter = max_iter;
        solver.tol = tol;
        
        auto result = solver.fit(X, y);
        return {result.coef, result.iterations, result.converged};
    }
};

/**
 * @deprecated Use GLMSolver with BinomialFamily + ProbitLink
 */
class [[deprecated("Use GLMSolver with BinomialFamily + ProbitLink in v1.1+")]]
ProbitRegression {
public:
    int max_iter = 50;
    double tol = 1e-6;

    ProbitResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        GLMSolver<> solver;
        solver.family = std::make_unique<BinomialFamily>();
        solver.link = std::make_unique<ProbitLink>();
        solver.max_iter = max_iter;
        solver.tol = tol;
        
        auto result = solver.fit(X, y);
        return {result.coef, result.iterations, result.converged};
    }
};

} // namespace statelix

#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif

#endif // STATELIX_GLM_LEGACY_H
