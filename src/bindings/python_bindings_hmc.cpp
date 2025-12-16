#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../bayes/hmc.h"
#include "../optimization/objective.h"

namespace py = pybind11;
using namespace statelix;

// Trampoline for EfficientObjective
class PyEfficientObjective : public EfficientObjective {
public:
    using EfficientObjective::EfficientObjective; 

    // Helper type for macro
    using PairRet = std::pair<double, Eigen::VectorXd>;

    std::pair<double, Eigen::VectorXd> value_and_gradient(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERRIDE_PURE(
            PairRet,             // Return type (no comma)
            EfficientObjective,                 
            value_and_gradient,                 
            x                                   
        );
    }
    
    double value(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERRIDE(double, EfficientObjective, value, x);
    }

    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERRIDE(Eigen::VectorXd, EfficientObjective, gradient, x);
    }
    
    int dimension() const override {
         PYBIND11_OVERRIDE(int, EfficientObjective, dimension, );
    }
};

PYBIND11_MODULE(hmc, m) {
    m.doc() = "Statelix HMC (Bayes)";

    py::class_<HMCResult>(m, "HMCResult")
        .def_readonly("samples", &HMCResult::samples)
        .def_readonly("log_probs", &HMCResult::log_probs)
        .def_readonly("acceptance_rate", &HMCResult::acceptance_rate)
        .def_readonly("n_divergences", &HMCResult::n_divergences); 

    py::class_<HMCConfig>(m, "HMCConfig")
        .def(py::init<>())
        .def_readwrite("n_samples", &HMCConfig::n_samples)
        .def_readwrite("warmup", &HMCConfig::warmup)
        .def_readwrite("step_size", &HMCConfig::step_size)
        .def_readwrite("n_leapfrog", &HMCConfig::n_leapfrog)
        .def_readwrite("seed", &HMCConfig::seed);

    py::class_<EfficientObjective, PyEfficientObjective>(m, "EfficientObjective")
        .def(py::init<>())
        .def("value", &EfficientObjective::value)
        .def("gradient", &EfficientObjective::gradient)
        .def("value_and_gradient", &EfficientObjective::value_and_gradient)
        .def("dimension", &EfficientObjective::dimension);

    py::class_<HamiltonianMonteCarlo>(m, "HamiltonianMonteCarlo")
        .def(py::init<const HMCConfig&>())
        .def("sample", static_cast<HMCResult (HamiltonianMonteCarlo::*)(EfficientObjective&, const Eigen::VectorXd&)>(&HamiltonianMonteCarlo::sample), 
             py::arg("objective"), py::arg("theta0"));
}
