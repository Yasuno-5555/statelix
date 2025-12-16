#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../bayes/bayesian_regression.h"

namespace py = pybind11;

PYBIND11_MODULE(bayes, m) {
    m.doc() = "Statelix Bayesian Econometrics";

    // --- HMC Result ---
    py::class_<statelix::HMCResult>(m, "HMCResult", py::module_local())
        .def_readonly("samples", &statelix::HMCResult::samples)
        .def_readonly("log_probs", &statelix::HMCResult::log_probs)
        .def_readonly("acceptance_rate", &statelix::HMCResult::acceptance_rate)
        .def_readonly("n_divergences", &statelix::HMCResult::n_divergences)
        .def_readonly("mean", &statelix::HMCResult::mean)
        .def_readonly("std_dev", &statelix::HMCResult::std_dev)
        .def_readonly("quantiles", &statelix::HMCResult::quantiles)
        .def_readonly("ess", &statelix::HMCResult::ess)
        .def_readonly("r_hat", &statelix::HMCResult::r_hat);

    // --- VI Result ---
    py::class_<statelix::VIResult>(m, "VIResult", py::module_local())
        .def_readonly("mean", &statelix::VIResult::mean)
        .def_readonly("variance", &statelix::VIResult::variance)
        .def_readonly("elbo", &statelix::VIResult::elbo)
        .def_readonly("converged", &statelix::VIResult::converged);

    // --- Bayesian Linear Regression ---
    py::class_<statelix::bayes::BayesianLinearRegression>(m, "BayesianLinearRegression", py::module_local())
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>(), py::arg("X"), py::arg("y"))
        .def_readwrite("prior_beta_std", &statelix::bayes::BayesianLinearRegression::prior_beta_std)
        .def_readwrite("prior_sigma_scale", &statelix::bayes::BayesianLinearRegression::prior_sigma_scale)
        .def("fit", &statelix::bayes::BayesianLinearRegression::fit_map, "Fit MAP estimate")
        .def("sample", &statelix::bayes::BayesianLinearRegression::sample,
             py::arg("n_samples")=1000, py::arg("warmup")=500, "Run HMC sampling")
        .def("fit_vi", &statelix::bayes::BayesianLinearRegression::fit_vi,
             py::arg("max_iter")=1000, "Run Variational Inference")
        .def_readonly("map_theta", &statelix::bayes::BayesianLinearRegression::map_theta);
}
