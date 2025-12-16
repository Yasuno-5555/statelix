#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../causal/psm.h"

namespace py = pybind11;

PYBIND11_MODULE(psm, m) {
    m.doc() = "Statelix PSM (Causal Inference)";

    py::enum_<statelix::MatchingMethod>(m, "MatchingMethod")
        .value("NEAREST_NEIGHBOR", statelix::MatchingMethod::NEAREST_NEIGHBOR)
        .value("CALIPER", statelix::MatchingMethod::CALIPER)
        .value("RADIUS", statelix::MatchingMethod::RADIUS)
        .value("KERNEL", statelix::MatchingMethod::KERNEL)
        .value("COVARIATE", statelix::MatchingMethod::COVARIATE)
        .export_values();

    py::class_<statelix::MatchingResult>(m, "MatchingResult")
        .def_readonly("att", &statelix::MatchingResult::att)
        .def_readonly("att_se", &statelix::MatchingResult::att_se)
        .def_readonly("atc", &statelix::MatchingResult::atc)
        .def_readonly("ate", &statelix::MatchingResult::ate)
        .def_readonly("n_matched_treated", &statelix::MatchingResult::n_matched_treated)
        .def_readonly("n_matched_control", &statelix::MatchingResult::n_matched_control);

     py::class_<statelix::PropensityScoreResult>(m, "PropensityScoreResult")
        .def_readonly("scores", &statelix::PropensityScoreResult::scores)
        .def_readonly("coef", &statelix::PropensityScoreResult::coef)
        .def_readonly("n_treated", &statelix::PropensityScoreResult::n_treated)
        .def_readonly("n_control", &statelix::PropensityScoreResult::n_control);

    py::class_<statelix::PropensityScoreMatching>(m, "PropensityScoreMatching")
        .def(py::init<>())
        .def_readwrite("method", &statelix::PropensityScoreMatching::method)
        .def_readwrite("n_neighbors", &statelix::PropensityScoreMatching::n_neighbors)
        .def_readwrite("with_replacement", &statelix::PropensityScoreMatching::with_replacement)
        .def_readwrite("caliper", &statelix::PropensityScoreMatching::caliper)
        .def("estimate_propensity", &statelix::PropensityScoreMatching::estimate_propensity, 
             py::arg("D"), py::arg("X"))
        .def("match", &statelix::PropensityScoreMatching::match, 
             py::arg("Y"), py::arg("D"), py::arg("X"), py::arg("ps"))
        .def("ipw", &statelix::PropensityScoreMatching::ipw, 
             py::arg("Y"), py::arg("D"), py::arg("ps"))
        .def("aipw", &statelix::PropensityScoreMatching::aipw, 
             py::arg("Y"), py::arg("D"), py::arg("X"), py::arg("ps"));
}
