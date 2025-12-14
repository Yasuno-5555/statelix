#ifndef STATELIX_MCMC_H
#define STATELIX_MCMC_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>

namespace statelix {

struct MCMCResult {
    Eigen::MatrixXd samples; // N x dim
    Eigen::VectorXd log_probs;
    double acceptance_rate;
};

// LogProbFunction concept:
// double operator()(const Eigen::VectorXd& x)

template <typename LogProbFunction>
class MetropolisHastings {
public:
    int n_samples = 1000;
    int burn_in = 100;
    double step_size = 0.5; // Scale of proposal distribution (Gaussian)
    
    MCMCResult sample(LogProbFunction& log_prob_func, const Eigen::VectorXd& x0) {
        int dim = x0.size();
        
        Eigen::MatrixXd samples(n_samples, dim);
        Eigen::VectorXd log_probs(n_samples);
        
        Eigen::VectorXd current_x = x0;
        double current_log_prob = log_prob_func(current_x);
        
        int accepted = 0;
        
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, step_size);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        
        // Burn-in
        for (int i = 0; i < burn_in; ++i) {
            Eigen::VectorXd proposal = current_x;
            for(int d=0; d<dim; ++d) proposal(d) += dist(rng);
            
            double proposal_log_prob = log_prob_func(proposal);
            
            // Acceptance Ratio (Symmetric Proposal -> cancel out q(x|x') terms)
            // alpha = min(1, P(x')/P(x)) = exp(logP(x') - logP(x))
            double log_alpha = proposal_log_prob - current_log_prob;
            
            if (std::log(uniform(rng)) < log_alpha) {
                current_x = proposal;
                current_log_prob = proposal_log_prob;
            }
        }
        
        // Sampling
        for (int i = 0; i < n_samples; ++i) {
            Eigen::VectorXd proposal = current_x;
            for(int d=0; d<dim; ++d) proposal(d) += dist(rng);
            
            double proposal_log_prob = log_prob_func(proposal);
            
            double log_alpha = proposal_log_prob - current_log_prob;
            
            if (std::log(uniform(rng)) < log_alpha) {
                current_x = proposal;
                current_log_prob = proposal_log_prob;
                accepted++;
            }
            
            samples.row(i) = current_x;
            log_probs(i) = current_log_prob;
        }
        
        return {samples, log_probs, (double)accepted / n_samples};
    }
};

} // namespace statelix

#endif // STATELIX_MCMC_H
