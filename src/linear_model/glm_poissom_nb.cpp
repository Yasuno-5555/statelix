// Poisson via IRLS (log link). Negative binomial: estimate theta via moment or WLS loop.
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct GlmResult { VectorXd coef; MatrixXd vcov; };

GlmResult fit_poisson(const MatrixXd& X, const VectorXd& y, int maxit=25, double tol=1e-8) {
    int n = X.rows(), p = X.cols();
    VectorXd beta = VectorXd::Zero(p);
    VectorXd eta = X * beta;
    VectorXd mu = eta.array().exp();

    for (int iter=0; iter<maxit; ++iter) {
        VectorXd W = mu; // variance = mu for Poisson, weight = mu (for IRLS with log link)
        for (int i=0;i<n;i++) if (W(i) < 1e-12) W(i) = 1e-12;
        VectorXd z = eta + (y - mu).array() / mu.array();
        MatrixXd XWX = X.transpose() * W.asDiagonal() * X;
        VectorXd XWz = X.transpose() * W.asDiagonal() * z;
        VectorXd beta_new = XWX.ldlt().solve(XWz);
        if ((beta_new - beta).norm() < tol) { beta = beta_new; break; }
        beta = beta_new;
        eta = X * beta;
        mu = eta.array().exp();
    }
    VectorXd W = mu;
    MatrixXd Fisher = X.transpose() * W.asDiagonal() * X;
    MatrixXd vcov = Fisher.ldlt().solve(MatrixXd::Identity(p,p));
    return {beta, vcov};
}

namespace py = pybind11;
PYBIND11_MODULE(statelix_glm, m) {
    py::class_<GlmResult>(m, "GlmResult").def_readonly("coef",&GlmResult::coef).def_readonly("vcov",&GlmResult::vcov);
    m.def("fit_poisson", &fit_poisson, "Fit Poisson GLM");
}
