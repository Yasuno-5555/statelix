// Poisson via IRLS (log link). Negative binomial: estimate theta via moment or WLS loop.
#include <Eigen/Dense>

namespace statelix {

struct PoissonGlmResult { 
    Eigen::VectorXd coef; 
    Eigen::MatrixXd vcov; 
};

PoissonGlmResult fit_poisson_simple(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int maxit=25, double tol=1e-8) {
    int n = X.rows(), p = X.cols();
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd eta = X * beta;
    Eigen::VectorXd mu = eta.array().exp();

    for (int iter=0; iter<maxit; ++iter) {
        Eigen::VectorXd W = mu; // variance = mu for Poisson, weight = mu (for IRLS with log link)
        for (int i=0;i<n;i++) if (W(i) < 1e-12) W(i) = 1e-12;
        Eigen::VectorXd z = eta + (y - mu).array() / mu.array();
        Eigen::MatrixXd XWX = X.transpose() * W.asDiagonal() * X;
        Eigen::VectorXd XWz = X.transpose() * W.asDiagonal() * z;
        Eigen::VectorXd beta_new = XWX.ldlt().solve(XWz);
        if ((beta_new - beta).norm() < tol) { beta = beta_new; break; }
        beta = beta_new;
        eta = X * beta;
        mu = eta.array().exp();
    }
    Eigen::VectorXd W = mu;
    Eigen::MatrixXd Fisher = X.transpose() * W.asDiagonal() * X;
    Eigen::MatrixXd vcov = Fisher.ldlt().solve(Eigen::MatrixXd::Identity(p,p));
    return {beta, vcov};
}

} // namespace statelix
