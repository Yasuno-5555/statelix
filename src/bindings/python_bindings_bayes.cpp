#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../bayes/bayesian_regression.h"
#include "../bayes/hmc.h"               // Added generic HMC header
#include "../bayes/native_objectives.h" // Added for LogisticObjective

namespace py = pybind11;
using namespace statelix;

// Wrapper for HMC target density provided by Python
// Supports f(x) -> (log_prob, grad)
class PyEfficientObjective : public statelix::EfficientObjective {
public:
  py::object func; // Python callable

  PyEfficientObjective(py::object f) : func(f) {}

  std::pair<double, Eigen::VectorXd>
  value_and_gradient(const Eigen::VectorXd &x) const override {
    // Must acquire GIL to call Python
    py::gil_scoped_acquire acquire;
    try {
      // Call Python function
      // Expects return: (log_prob, grad_vector)
      py::tuple res = func(x);

      if (res.size() != 2) {
        throw std::runtime_error("Callback must return (log_prob, gradient)");
      }

      double log_prob = res[0].cast<double>();
      Eigen::VectorXd grad = res[1].cast<Eigen::VectorXd>();

      // HMC minimizes Energy = -LogProb
      return {-log_prob, -grad};

    } catch (py::error_already_set &e) {
      // Propagate Python error as C++ runtime error to stop sampling safely
      throw std::runtime_error("Python callback error: " +
                               std::string(e.what()));
    }
  }
};

PYBIND11_MODULE(bayes, m) {
  m.doc() = "Statelix Bayesian Econometrics";

  // --- HMC Config ---
  py::class_<statelix::HMCConfig>(m, "HMCConfig")
      .def(py::init<>())
      .def_readwrite("n_samples", &statelix::HMCConfig::n_samples)
      .def_readwrite("warmup", &statelix::HMCConfig::warmup)
      .def_readwrite("step_size", &statelix::HMCConfig::step_size)
      .def_readwrite("n_leapfrog", &statelix::HMCConfig::n_leapfrog)
      .def_readwrite("adapt_step_size", &statelix::HMCConfig::adapt_step_size)
      .def_readwrite("target_accept", &statelix::HMCConfig::target_accept)
      .def_readwrite("seed", &statelix::HMCConfig::seed);

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
  py::class_<statelix::bayes::BayesianLinearRegression>(
      m, "BayesianLinearRegression", py::module_local())
      .def(py::init<Eigen::MatrixXd, Eigen::VectorXd>(), py::arg("X"),
           py::arg("y"))
      .def_readwrite("prior_beta_std",
                     &statelix::bayes::BayesianLinearRegression::prior_beta_std)
      .def_readwrite(
          "prior_sigma_scale",
          &statelix::bayes::BayesianLinearRegression::prior_sigma_scale)
      .def("fit", &statelix::bayes::BayesianLinearRegression::fit_map,
           "Fit MAP estimate")
      .def("sample", &statelix::bayes::BayesianLinearRegression::sample,
           py::arg("n_samples") = 1000, py::arg("warmup") = 500,
           "Run HMC sampling")
      .def("fit_vi", &statelix::bayes::BayesianLinearRegression::fit_vi,
           py::arg("max_iter") = 1000, "Run Variational Inference")
      .def_readonly("map_theta",
                    &statelix::bayes::BayesianLinearRegression::map_theta);

  // --- HMC Sampler Binding ---
  m.def(
      "hmc_sample",
      [](py::function log_prob_func, // Returns (log_prob, grad)
         const Eigen::VectorXd &theta0, const statelix::HMCConfig &config) {
        // Create C++ wrapper for Python objective
        PyEfficientObjective objective(log_prob_func);

        statelix::HamiltonianMonteCarlo hmc;
        hmc.config = config;
        return hmc.sample(objective, theta0);
      },
      py::arg("log_prob_func"), py::arg("theta0"), py::arg("config"),
      "Run HMC sampling. log_prob_func(theta) -> (log_p, grad_vector)");

  // v2.3 Native HMC for Logistic Regression (GIL Released)
  m.def(
      "hmc_sample_logistic",
      [](const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
         const statelix::HMCConfig &config, double prior_std) {
        // Create Native Objective (No Python GIL needed for calcs)
        statelix::LogisticObjective obj(X, y, prior_std);

        statelix::HamiltonianMonteCarlo hmc;
        hmc.config = config;

        // Run sampling with GIL RELEASED
        // This is safe because LogisticObjective uses only C++ Eigen types
        {
          py::gil_scoped_release release;
          // Initialize at zero
          Eigen::VectorXd theta0 = Eigen::VectorXd::Zero(X.cols());
          return hmc.sample(obj, theta0);
        }
      },
      py::arg("X"), py::arg("y"), py::arg("config"),
      py::arg("prior_std") = 10.0,
      "Run HMC for Logistic Regression using high-performance C++ backend (No "
      "GIL).");
}
