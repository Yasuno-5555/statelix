#include <Eigen/Dense>
#include <cmath>

namespace statelix {

Eigen::VectorXd ridge_solver(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double lambda) {
    int p = X.cols();
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::MatrixXd A = XtX + lambda * Eigen::MatrixXd::Identity(p,p);
    Eigen::VectorXd Xty = X.transpose() * y;
    return A.ldlt().solve(Xty);
}

// Lasso via coordinate descent (standardized X assumed)
Eigen::VectorXd lasso_cd(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double lambda, int maxit=1000, double tol=1e-6) {
    int n = X.rows(), p = X.cols();
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd residual = y - X * beta;
    for (int iter=0; iter<maxit; ++iter) {
        double maxchg = 0;
        for (int j=0; j<p; ++j) {
            Eigen::VectorXd xj = X.col(j);
            double rho = xj.dot(residual) + beta(j) * xj.squaredNorm();
            double z = xj.squaredNorm();
            double newb = 0.0;
            if (rho > lambda/2.0) newb = (rho - lambda/2.0) / z;
            else if (rho < -lambda/2.0) newb = (rho + lambda/2.0) / z;
            else newb = 0.0;
            double change = newb - beta(j);
            if (std::abs(change) > 0) {
                residual -= xj * change;
                maxchg = std::max(maxchg, std::abs(change));
                beta(j) = newb;
            }
        }
        if (maxchg < tol) break;
    }
    return beta;
}

} // namespace statelix
