#ifndef STATELIX_ELASTIC_NET_H
#define STATELIX_ELASTIC_NET_H

#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct ElasticNetResult {
    Eigen::VectorXd coef;
    double intercept;
    int iterations;
    double duality_gap;
};

class ElasticNet {
public:
    double alpha = 1.0;      // lambda (overall penalty scale)
    double l1_ratio = 0.5;   // alpha in glmnet notation (1=Lasso, 0=Ridge)
    int max_iter = 1000;
    double tol = 1e-4;
    bool fit_intercept = true;

    ElasticNetResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

} // namespace statelix

#endif // STATELIX_ELASTIC_NET_H
