#include "optimization/lbfgs.h"
#include <iostream>

class StubObjective : public statelix::Objective {
public:
    double value(const Eigen::VectorXd& x) override { return x.squaredNorm(); }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) override { return 2*x; }
};

int main() {
    statelix::LBFGSOptimizer opt;
    StubObjective obj;
    Eigen::VectorXd x(2);
    x << 1, 2;
    auto res = opt.minimize(obj, x);
    std::cout << "Min: " << res.min_value << std::endl;
    return 0;
}
