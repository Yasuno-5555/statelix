#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct PCAResult { MatrixXd components; VectorXd explained_variance; };

PCAResult run_pca(const MatrixXd& X, int k) {
    // center X
    MatrixXd Xc = X.rowwise() - X.colwise().mean();
    Eigen::JacobiSVD<MatrixXd> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd V = svd.matrixV();
    VectorXd S = svd.singularValues();
    VectorXd var = (S.array().square() / (X.rows()-1));
    MatrixXd comps = V.leftCols(k);
    return {comps, var.head(k)};
}

namespace py = pybind11;
PYBIND11_MODULE(statelix_pca, m) {
    py::class_<PCAResult>(m,"PCAResult").def_readonly("components",&PCAResult::components).def_readonly("explained_variance",&PCAResult::explained_variance);
    m.def("run_pca",&run_pca);
}
