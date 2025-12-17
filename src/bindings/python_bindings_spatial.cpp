#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../spatial/icp.h"
#include "../spatial/spatial.h"

namespace py = pybind11;

PYBIND11_MODULE(spatial, m) {
  m.doc() = "Statelix Spatial Econometrics";

  py::enum_<statelix::SpatialModel>(m, "SpatialModel")
      .value("SAR", statelix::SpatialModel::SAR)
      .value("SEM", statelix::SpatialModel::SEM)
      .value("SDM", statelix::SpatialModel::SDM)
      .value("SAC", statelix::SpatialModel::SAC)
      .export_values();

  py::class_<statelix::SpatialResult>(m, "SpatialResult")
      .def_readonly("rho", &statelix::SpatialResult::rho)
      .def_readonly("rho_se", &statelix::SpatialResult::rho_se)
      .def_readonly("lambda", &statelix::SpatialResult::lambda)
      .def_readonly("beta", &statelix::SpatialResult::beta)
      .def_readonly("beta_se", &statelix::SpatialResult::beta_se)
      .def_readonly("coef", &statelix::SpatialResult::coef)
      .def_readonly("std_errors", &statelix::SpatialResult::std_errors)
      .def_readonly("p_values", &statelix::SpatialResult::p_values)
      .def_readonly("z_values", &statelix::SpatialResult::z_values)
      .def_readonly("aic", &statelix::SpatialResult::aic)
      .def_readonly("bic", &statelix::SpatialResult::bic)
      .def_readonly("pseudo_r_squared",
                    &statelix::SpatialResult::pseudo_r_squared)
      .def_readonly("log_likelihood", &statelix::SpatialResult::log_likelihood)
      .def_readonly("direct_effects", &statelix::SpatialResult::direct_effects)
      .def_readonly("indirect_effects",
                    &statelix::SpatialResult::indirect_effects)
      .def_readonly("total_effects", &statelix::SpatialResult::total_effects);

  py::class_<statelix::MoranResult>(m, "MoranResult")
      .def_readonly("statistic", &statelix::MoranResult::I)
      .def_readonly("p_value", &statelix::MoranResult::p_value)
      .def_readonly("z_score", &statelix::MoranResult::z_stat)
      .def_readonly("spatial_autocorrelation",
                    &statelix::MoranResult::spatial_autocorrelation);

  py::class_<statelix::LMSpatialResult>(m, "LMSpatialResult")
      .def_readonly("lm_lag", &statelix::LMSpatialResult::lm_lag)
      .def_readonly("lm_lag_pvalue", &statelix::LMSpatialResult::lm_lag_pvalue)
      .def_readonly("lm_error", &statelix::LMSpatialResult::lm_error)
      .def_readonly("lm_error_pvalue",
                    &statelix::LMSpatialResult::lm_error_pvalue)
      .def_readonly("rlm_lag", &statelix::LMSpatialResult::rlm_lag)
      .def_readonly("rlm_error", &statelix::LMSpatialResult::rlm_error)
      .def_readonly("recommendation",
                    &statelix::LMSpatialResult::recommendation);

  py::class_<statelix::SpatialWeights>(m, "SpatialWeights")
      .def_static("row_standardize", &statelix::SpatialWeights::row_standardize,
                  py::arg("W"))
      .def_static("knn_weights", &statelix::SpatialWeights::knn_weights,
                  py::arg("X"), py::arg("k") = 5)
      .def_static("inverse_distance_weights",
                  &statelix::SpatialWeights::inverse_distance_weights,
                  py::arg("coords"), py::arg("threshold") = -1.0,
                  py::arg("alpha") = 1.0);

  py::class_<statelix::SpatialRegression>(m, "SpatialRegression")
      .def(py::init<>())
      .def_readwrite("model", &statelix::SpatialRegression::model)
      .def_readwrite("max_iter", &statelix::SpatialRegression::max_iter)
      .def("fit", &statelix::SpatialRegression::fit, py::arg("y"), py::arg("X"),
           py::arg("W"))
      .def("moran_test", &statelix::SpatialRegression::moran_test, py::arg("z"),
           py::arg("W"))
      .def("lm_tests", &statelix::SpatialRegression::lm_tests, py::arg("y"),
           py::arg("X"), py::arg("W"));
}
