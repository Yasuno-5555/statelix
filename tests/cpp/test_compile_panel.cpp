#include "../../src/panel/dynamic_panel.h"
#include <vector>
#include <iostream>

int main() {
    statelix::panel::DynamicPanelGMM estimator;
    estimator.max_lags = 2;
    
    Eigen::VectorXd y(5);
    y.setZero();
    Eigen::MatrixXd X(5, 2);
    X.setZero();
    std::vector<int> ids = {1, 1, 1, 2, 2};
    std::vector<int> time = {1, 2, 3, 1, 2};
    
    // Instantiation check
    auto result = estimator.estimate(y, X, ids, time);
    
    std::cout << "Coeff: " << result.coefficients.transpose() << std::endl;
    return 0;
}
