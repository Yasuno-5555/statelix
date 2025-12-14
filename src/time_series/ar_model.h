#pragma once
#include <Eigen/Dense>

namespace statelix {

struct ARResult {
    Eigen::VectorXd params; // [const, phi_1, ..., phi_p]
    double sigma2;
    int p; // order
};

// Fit AR(p) model using OLS
ARResult fit_ar(const Eigen::VectorXd& series, int p);

} // namespace statelix
