#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../panel/dynamic_panel.h"
#include "../panel/panel.h" // [Fix] Include Panel FE/RE/FD definitions

namespace py = pybind11;
using namespace statelix::panel;

PYBIND11_MODULE(panel, m) {
    m.doc() = "Statelix Panel (Econometrics)";

    // Dynamic Panel (v2.3)
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

    // Fixed Effects
    py::class_<statelix::PanelFEResult>(m, "PanelFEResult")
        .def_readonly("coef", &statelix::PanelFEResult::coef)
        .def_readonly("std_errors", &statelix::PanelFEResult::std_errors)
        .def_readonly("t_values", &statelix::PanelFEResult::t_values)
        .def_readonly("p_values", &statelix::PanelFEResult::p_values)
        .def_readonly("df_residual", &statelix::PanelFEResult::df_residual)
        .def_readonly("r_squared_within", &statelix::PanelFEResult::r_squared_within);

    py::class_<statelix::PanelFixedEffects>(m, "FixedEffects")
        .def(py::init<>())
        .def_readwrite("cluster_se", &statelix::PanelFixedEffects::cluster_se)
        .def_readwrite("two_way", &statelix::PanelFixedEffects::two_way)
        .def("fit", &statelix::PanelFixedEffects::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    // Random Effects
    py::class_<statelix::PanelREResult>(m, "PanelREResult")
        .def_readonly("coef", &statelix::PanelREResult::coef)
        .def_readonly("std_errors", &statelix::PanelREResult::std_errors)
        .def_readonly("theta", &statelix::PanelREResult::theta)
        .def_readonly("rho", &statelix::PanelREResult::rho)
        .def_readonly("sigma2_u", &statelix::PanelREResult::sigma2_u)
        .def_readonly("sigma2_e", &statelix::PanelREResult::sigma2_e);

    py::class_<statelix::PanelRandomEffects>(m, "RandomEffects")
        .def(py::init<>())
        .def("fit", &statelix::PanelRandomEffects::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    // First Difference
    py::class_<statelix::PanelFDResult>(m, "PanelFDResult")
        .def_readonly("coef", &statelix::PanelFDResult::coef)
        .def_readonly("std_errors", &statelix::PanelFDResult::std_errors)
        .def_readonly("r_squared", &statelix::PanelFDResult::r_squared);

    py::class_<statelix::PanelFirstDifference>(m, "FirstDifference")
        .def(py::init<>())
        .def("fit", &statelix::PanelFirstDifference::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    // Hausman Test
    py::class_<statelix::HausmanTestResult>(m, "HausmanTestResult")
        .def_readonly("chi2_stat", &statelix::HausmanTestResult::chi2_stat)
        .def_readonly("df", &statelix::HausmanTestResult::df)
        .def_readonly("p_value", &statelix::HausmanTestResult::p_value)
        .def_readonly("prefer_fe", &statelix::HausmanTestResult::prefer_fe)
        .def_readonly("recommendation", &statelix::HausmanTestResult::recommendation)
        .def_readonly("warning", &statelix::HausmanTestResult::warning);

    py::class_<statelix::HausmanTest>(m, "HausmanTest")
        .def_static("test", py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&, const Eigen::VectorXi&, const Eigen::VectorXi&>(&statelix::HausmanTest::test),
                    py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"),
                    "Run Hausman Test by estimating FE and RE models internally")
        .def_static("compare", py::overload_cast<const statelix::PanelFEResult&, const statelix::PanelREResult&>(&statelix::HausmanTest::test),
                    py::arg("fe_result"), py::arg("re_result"),
                    "Run Hausman Test comparing existing FE and RE results");
}
