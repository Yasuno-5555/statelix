#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../panel/dynamic_panel.h"

namespace py = pybind11;
using namespace statelix::panel;

PYBIND11_MODULE(panel, m) {
    m.doc() = "Statelix Panel (Econometrics)";

    py::class_<DynamicPanelResult>(m, "DynamicPanelResult")
        .def_readonly("coefficients", &DynamicPanelResult::coefficients)
        .def_readonly("std_errors", &DynamicPanelResult::std_errors)
        .def_readonly("z_values", &DynamicPanelResult::z_values)
        .def_readonly("p_values", &DynamicPanelResult::p_values)
        .def_readonly("sargan_test", &DynamicPanelResult::sargan_test)
        .def_readonly("sargan_pvalue", &DynamicPanelResult::sargan_pvalue)
        .def_readonly("ar1_test", &DynamicPanelResult::ar1_test)
        .def_readonly("ar1_pvalue", &DynamicPanelResult::ar1_pvalue)
        .def_readonly("ar2_test", &DynamicPanelResult::ar2_test)
        .def_readonly("ar2_pvalue", &DynamicPanelResult::ar2_pvalue)
        .def_readonly("n_obs", &DynamicPanelResult::n_obs)
        .def_readonly("n_instruments", &DynamicPanelResult::n_instruments)
        .def_readonly("converged", &DynamicPanelResult::converged);

    py::class_<DynamicPanelGMM>(m, "DynamicPanelGMM")
        .def(py::init<>())
        .def_readwrite("two_step", &DynamicPanelGMM::two_step)
        .def_readwrite("robust_se", &DynamicPanelGMM::robust_se)
        .def_readwrite("max_lags", &DynamicPanelGMM::max_lags)
        .def("estimate", &DynamicPanelGMM::estimate,
             py::arg("y"), py::arg("X"), py::arg("ids"), py::arg("time"));
}
