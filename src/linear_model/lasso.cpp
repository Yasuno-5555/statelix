#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace statelix {

struct LassoResult {
    Eigen::VectorXd coef;    // original-scale coefficients (length p)
    double intercept; // intercept (original scale)
    int iterations;
    bool converged;
};

inline double soft_threshold(double z, double gamma) {
    if (z > gamma) return z - gamma;
    if (z < -gamma) return z + gamma;
    return 0.0;
}

LassoResult fit_lasso_cd(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    double lambda,
    int max_iter = 1000,
    double tol = 1e-6,
    bool standardize = true
) {
    int n = X.rows();
    int p = X.cols();

    // Means and stds
    Eigen::VectorXd x_mean = X.colwise().mean();
    Eigen::VectorXd x_std(p);
    Eigen::MatrixXd Xs = X; // standardized copy or original if not standardize
    for (int j = 0; j < p; ++j) {
        Eigen::VectorXd col = X.col(j);
        double mean = x_mean(j);
        Eigen::VectorXd centered = col.array() - mean;
        double sd = std::sqrt((centered.array().square().sum()) / (n - 1.0));
        if (sd < 1e-12) sd = 1.0; // avoid division by zero (constant column)
        x_std(j) = sd;
        if (standardize) {
            Xs.col(j) = centered / sd;
        } else {
            Xs.col(j) = col;
        }
    }

    double y_mean = y.mean();
    Eigen::VectorXd y_center = y.array() - y_mean;

    // For standardized Xs, we do Lasso on centered y
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd residual = y_center - Xs * beta;

    int iter;
    bool converged = false;
    for (iter = 0; iter < max_iter; ++iter) {
        double maxchg = 0.0;
        for (int j = 0; j < p; ++j) {
            Eigen::VectorXd xj = Xs.col(j);
            double xj_norm2 = xj.squaredNorm();
            if (xj_norm2 < 1e-12) continue; // skip nearly-constant post-std column

            double rho = xj.dot(residual) + beta(j) * xj_norm2;
            double z = xj_norm2;

            double newb = soft_threshold(rho, lambda) / z;
            double change = newb - beta(j);
            if (std::abs(change) != 0.0) {
                residual -= xj * change;
                beta(j) = newb;
                if (std::abs(change) > maxchg) maxchg = std::abs(change);
            }
        }
        if (maxchg < tol) {
            converged = true;
            break;
        }
    }

    // Convert back to original scale if standardized
    Eigen::VectorXd coef_orig(p);
    if (standardize) {
        for (int j = 0; j < p; ++j) {
            coef_orig(j) = beta(j) / x_std(j);
        }
    } else {
        coef_orig = beta;
    }

    double intercept = y_mean;
    if (standardize) {
        intercept -= x_mean.dot(coef_orig);
    } else {
        intercept -= x_mean.dot(coef_orig); // same formula works
    }

    LassoResult res;
    res.coef = coef_orig;
    res.intercept = intercept;
    res.iterations = iter + 1;
    res.converged = converged;
    return res;
}

} // namespace statelix
