#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd ridge_solver(const MatrixXd& X, const VectorXd& y, double lambda) {
    int p = X.cols();
    MatrixXd XtX = X.transpose() * X;
    MatrixXd A = XtX + lambda * MatrixXd::Identity(p,p);
    VectorXd Xty = X.transpose() * y;
    return A.ldlt().solve(Xty);
}

// Lasso via coordinate descent (standardized X assumed)
VectorXd lasso_cd(const MatrixXd& X, const VectorXd& y, double lambda, int maxit=1000, double tol=1e-6) {
    int n = X.rows(), p = X.cols();
    VectorXd beta = VectorXd::Zero(p);
    VectorXd residual = y - X * beta;
    for (int iter=0; iter<maxit; ++iter) {
        double maxchg = 0;
        for (int j=0; j<p; ++j) {
            VectorXd xj = X.col(j);
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

namespace py = pybind11;
PYBIND11_MODULE(statelix_penalty, m) {
    m.def("ridge_solver", &ridge_solver);
    m.def("lasso_cd", &lasso_cd);
}
