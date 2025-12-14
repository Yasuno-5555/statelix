#pragma once
#include <Eigen/Dense>
#include <vector>

namespace statelix {

struct AnoVaResult {
    double f_statistic;
    double p_value;
    double ss_between;
    double ss_within;
    double ss_total;
    int df_between;
    int df_within;
    int df_total;
    double ms_between;
    double ms_within;
};

// One-way ANOVA
// Takes a list of vectors (groups)
AnoVaResult f_oneway(const std::vector<Eigen::VectorXd>& groups);

// One-way ANOVA using X (data) and y (groups - integer)
AnoVaResult f_oneway_flat(const Eigen::VectorXd& data, const Eigen::VectorXi& groups);

} // namespace statelix
