/**
 * @file objective.cpp
 * @brief Implementation of Objective interface components
 */
#include "objective.h"
#include "penalizer.h"

namespace statelix {

std::pair<double, Eigen::VectorXd> 
RegularizedObjective::value_and_gradient(const Eigen::VectorXd& x) const {
    double val = base_objective->value(x);
    Eigen::VectorXd grad = base_objective->gradient(x);
    
    if (penalizer) {
        val += penalizer->penalty(x);
        grad += penalizer->gradient(x);
    }
    
    return {val, grad};
}

} // namespace statelix
