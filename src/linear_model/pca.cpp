#include <Eigen/Dense>

namespace statelix {

struct PCAResult { 
    Eigen::MatrixXd components; 
    Eigen::VectorXd explained_variance; 
};

PCAResult run_pca(const Eigen::MatrixXd& X, int k) {
    // center X
    Eigen::MatrixXd Xc = X.rowwise() - X.colwise().mean();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd S = svd.singularValues();
    Eigen::VectorXd var = (S.array().square() / (X.rows()-1));
    Eigen::MatrixXd comps = V.leftCols(k);
    return {comps, var.head(k)};
}

} // namespace statelix
