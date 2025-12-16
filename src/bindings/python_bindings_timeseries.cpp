#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../time_series/var.h"
#include "../time_series/garch.h"
#include "../time_series/cpd.h"
// #include "../time_series/cointegration.h" // Optional for now

namespace py = pybind11;

PYBIND11_MODULE(time_series, m) {
    m.doc() = "Statelix Time Series Module";

    // --- VAR ---
    py::class_<statelix::VARResult>(m, "VARResult_Type", py::module_local())
        .def_readonly("coef", &statelix::VARResult::coef)
        .def_readonly("intercept", &statelix::VARResult::intercept)
        .def_readonly("aic", &statelix::VARResult::aic)
        .def_readonly("bic", &statelix::VARResult::bic)
        .def_readonly("hqc", &statelix::VARResult::hqc)
        .def_readonly("log_likelihood", &statelix::VARResult::log_likelihood)
        .def_readonly("is_stable", &statelix::VARResult::is_stable);

    py::class_<statelix::GrangerResult>(m, "GrangerResult_Type", py::module_local())
        .def_readonly("f_stat", &statelix::GrangerResult::f_stat)
        .def_readonly("p_value", &statelix::GrangerResult::p_value)
        .def_readonly("causes", &statelix::GrangerResult::causes);
        
    py::class_<statelix::VectorAutoregression>(m, "VAR_Model", py::module_local())
        .def(py::init<int>(), py::arg("p")=1)
        .def("fit", &statelix::VectorAutoregression::fit, py::arg("Y"), "Fit VAR model")
        .def("granger_causality", &statelix::VectorAutoregression::granger_causality, 
             py::arg("result"), py::arg("Y"), py::arg("cause"), py::arg("effect"), "Test Granger Causality")
        // .def("irf", ... )
        ;

    // --- GARCH ---
    py::enum_<statelix::GARCHType>(m, "GARCHType", py::module_local())
        .value("GARCH", statelix::GARCHType::GARCH)
        .value("EGARCH", statelix::GARCHType::EGARCH)
        .value("GJR", statelix::GARCHType::GJR)
        .export_values();

    py::enum_<statelix::GARCHDist>(m, "GARCHDist", py::module_local())
        .value("NORMAL", statelix::GARCHDist::NORMAL)
        .value("STUDENT_T", statelix::GARCHDist::STUDENT_T)
        .export_values();

    py::class_<statelix::GARCHResult>(m, "GARCHResult_Type", py::module_local())
        .def_readonly("omega", &statelix::GARCHResult::omega)
        .def_readonly("alpha", &statelix::GARCHResult::alpha)
        .def_readonly("beta", &statelix::GARCHResult::beta)
        .def_readonly("mu", &statelix::GARCHResult::mu)
        .def_readonly("log_likelihood", &statelix::GARCHResult::log_likelihood)
        .def_readonly("converged", &statelix::GARCHResult::converged);

    py::class_<statelix::GARCH>(m, "GARCH_Model", py::module_local())
        .def(py::init<int, int>(), py::arg("p")=1, py::arg("q")=1)
        .def_readwrite("type", &statelix::GARCH::type)
        .def_readwrite("dist", &statelix::GARCH::dist)
        .def("fit", &statelix::GARCH::fit, py::arg("returns"));

    // --- CPD ---

    
    py::enum_<statelix::CostType>(m, "CostType", py::module_local())
        .value("L2", statelix::CostType::L2)
        .value("GAUSSIAN", statelix::CostType::GAUSSIAN)
        .value("POISSON", statelix::CostType::POISSON)
        .export_values();

    py::class_<statelix::CPDResult>(m, "CPDResult", py::module_local())
        .def_readonly("change_points", &statelix::CPDResult::change_points)
        .def_readonly("cost", &statelix::CPDResult::cost);

    py::class_<statelix::ChangePointDetector>(m, "ChangePointDetector", py::module_local())
        .def(py::init<statelix::CostType, double, int>(), 
             py::arg("cost_type")=statelix::CostType::L2, 
             py::arg("penalty")=0.0, 
             py::arg("min_size")=2)
        .def_readwrite("cost_type", &statelix::ChangePointDetector::cost_type)
        .def_readwrite("penalty", &statelix::ChangePointDetector::penalty)
        .def_readwrite("min_size", &statelix::ChangePointDetector::min_size)
        .def("fit", &statelix::ChangePointDetector::fit, py::arg("data"));
}

