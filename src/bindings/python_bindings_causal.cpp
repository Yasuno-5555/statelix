#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../causal/iv.h"
#include "../causal/psm.h"
#include "../causal/gmm.h"
#include "../causal/did.h"
#include "../causal/rdd.h"

namespace py = pybind11;

PYBIND11_MODULE(causal, m) {
    m.doc() = "Statelix Causal Inference Module (IV, PSM, DiD, RDD) - Updated";

    // =========================================================================
    // Instrumental Variables (IV)
    // =========================================================================
    py::class_<statelix::IVResult>(m, "IVResult")
        .def_readonly("coef", &statelix::IVResult::coef)
        .def_readonly("std_errors", &statelix::IVResult::std_errors)
        .def_readonly("t_values", &statelix::IVResult::t_values)
        .def_readonly("p_values", &statelix::IVResult::p_values)
        .def_readonly("conf_int", &statelix::IVResult::conf_int)
        .def_readonly("first_stage_coef", &statelix::IVResult::first_stage_coef)
        .def_readonly("first_stage_f", &statelix::IVResult::first_stage_f)
        .def_readonly("first_stage_f_pvalue", &statelix::IVResult::first_stage_f_pvalue)
        .def_readonly("weak_instruments", &statelix::IVResult::weak_instruments)
        .def_readonly("sargan_stat", &statelix::IVResult::sargan_stat)
        .def_readonly("sargan_pvalue", &statelix::IVResult::sargan_pvalue)
        .def_readonly("overid_test_valid", &statelix::IVResult::overid_test_valid)
        .def_readonly("r_squared", &statelix::IVResult::r_squared)
        .def_readonly("residual_std_error", &statelix::IVResult::residual_std_error)
        .def_readonly("n_obs", &statelix::IVResult::n_obs)
        .def_readonly("fitted_values", &statelix::IVResult::fitted_values)
        .def_readonly("residuals", &statelix::IVResult::residuals);

    py::class_<statelix::TwoStageLeastSquares>(m, "TwoStageLeastSquares")
        .def(py::init<>())
        .def_readwrite("fit_intercept", &statelix::TwoStageLeastSquares::fit_intercept)
        .def_readwrite("robust_se", &statelix::TwoStageLeastSquares::robust_se)
        .def_readwrite("conf_level", &statelix::TwoStageLeastSquares::conf_level)
        .def("fit", py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&>(&statelix::TwoStageLeastSquares::fit),
             py::arg("Y"), py::arg("X_endog"), py::arg("X_exog"), py::arg("Z"), "Fit 2SLS with exogenous controls")
        .def("fit", py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&>(&statelix::TwoStageLeastSquares::fit),
             py::arg("Y"), py::arg("X_endog"), py::arg("Z"), "Fit 2SLS without exogenous controls");

    // =========================================================================
    // Propensity Score Matching (PSM)
    // =========================================================================
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

    // =========================================================================
    // Linear GMM
    // =========================================================================
    py::class_<statelix::GMMResult>(m, "GMMResult")
        .def_readonly("coef", &statelix::GMMResult::coef)
        .def_readonly("std_errors", &statelix::GMMResult::std_errors)
        .def_readonly("t_values", &statelix::GMMResult::t_values)
        .def_readonly("p_values", &statelix::GMMResult::p_values)
        .def_readonly("conf_int", &statelix::GMMResult::conf_int)
        .def_readonly("W", &statelix::GMMResult::W)
        .def_readonly("weighting_scheme", &statelix::GMMResult::weighting_scheme)
        .def_readonly("j_stat", &statelix::GMMResult::j_stat)
        .def_readonly("j_pvalue", &statelix::GMMResult::j_pvalue)
        .def_readonly("overid_df", &statelix::GMMResult::overid_df)
        .def_readonly("n_obs", &statelix::GMMResult::n_obs)
        .def_readonly("n_params", &statelix::GMMResult::n_params)
        .def_readonly("n_instruments", &statelix::GMMResult::n_instruments)
        .def_readonly("vcov", &statelix::GMMResult::vcov)
        .def_readonly("residuals", &statelix::GMMResult::residuals)
        .def_readonly("sigma2", &statelix::GMMResult::sigma2);
        
    py::class_<statelix::LinearGMM>(m, "LinearGMM")
        .def(py::init<>())
        .def_readwrite("fit_intercept", &statelix::LinearGMM::fit_intercept)
        .def_readwrite("conf_level", &statelix::LinearGMM::conf_level)
        .def("fit", &statelix::LinearGMM::fit,
             py::arg("Y"), py::arg("X_endog"), py::arg("X_exog"), py::arg("Z"),
             py::arg("weight_method") = "optimal");

    // =========================================================================
    // Difference-in-Differences (DiD)
    // =========================================================================
    py::class_<statelix::DIDResult>(m, "DIDResult")
        .def_readonly("att", &statelix::DIDResult::att)
        .def_readonly("att_std_error", &statelix::DIDResult::att_std_error)
        .def_readonly("t_stat", &statelix::DIDResult::t_stat)
        .def_readonly("pvalue", &statelix::DIDResult::pvalue)
        .def_readonly("conf_lower", &statelix::DIDResult::conf_lower)
        .def_readonly("conf_upper", &statelix::DIDResult::conf_upper)
        .def_readonly("coef", &statelix::DIDResult::coef)
        .def_readonly("std_errors", &statelix::DIDResult::std_errors)
        .def_readonly("pre_trend_diff", &statelix::DIDResult::pre_trend_diff)
        .def_readonly("pre_trend_pvalue", &statelix::DIDResult::pre_trend_pvalue)
        .def_readonly("parallel_trends_valid", &statelix::DIDResult::parallel_trends_valid)
        .def_readonly("r_squared", &statelix::DIDResult::r_squared)
        .def_readonly("n_obs", &statelix::DIDResult::n_obs)
        .def_readonly("n_treated", &statelix::DIDResult::n_treated)
        .def_readonly("n_control", &statelix::DIDResult::n_control);

    py::class_<statelix::DifferenceInDifferences>(m, "DifferenceInDifferences")
        .def(py::init<>())
        .def_readwrite("conf_level", &statelix::DifferenceInDifferences::conf_level)
        .def_readwrite("robust_se", &statelix::DifferenceInDifferences::robust_se)
        .def("fit", &statelix::DifferenceInDifferences::fit,
             py::arg("Y"), py::arg("treated"), py::arg("post"))
        .def("fit_with_pretest", &statelix::DifferenceInDifferences::fit_with_pretest,
             py::arg("Y"), py::arg("treated"), py::arg("time"), py::arg("treatment_time"));

    // =========================================================================
    // Two-Way Fixed Effects (TWFE)
    // =========================================================================
    py::class_<statelix::TWFEResult>(m, "TWFEResult")
        .def_readonly("delta", &statelix::TWFEResult::delta)
        .def_readonly("delta_std_error", &statelix::TWFEResult::delta_std_error)
        .def_readonly("t_stat", &statelix::TWFEResult::t_stat)
        .def_readonly("pvalue", &statelix::TWFEResult::pvalue)
        .def_readonly("conf_lower", &statelix::TWFEResult::conf_lower)
        .def_readonly("conf_upper", &statelix::TWFEResult::conf_upper)
        .def_readonly("unit_fe", &statelix::TWFEResult::unit_fe)
        .def_readonly("time_fe", &statelix::TWFEResult::time_fe)
        .def_readonly("has_staggered_adoption", &statelix::TWFEResult::has_staggered_adoption)
        .def_readonly("r_squared_within", &statelix::TWFEResult::r_squared_within)
        .def_readonly("r_squared_overall", &statelix::TWFEResult::r_squared_overall)
        .def_readonly("n_obs", &statelix::TWFEResult::n_obs)
        .def_readonly("clustered_se", &statelix::TWFEResult::clustered_se);

    py::class_<statelix::TwoWayFixedEffects>(m, "TwoWayFixedEffects")
        .def(py::init<>())
        .def_readwrite("conf_level", &statelix::TwoWayFixedEffects::conf_level)
        .def_readwrite("cluster_se", &statelix::TwoWayFixedEffects::cluster_se)
        .def("fit", &statelix::TwoWayFixedEffects::fit,
             py::arg("Y"), py::arg("D"), py::arg("unit_id"), py::arg("time_id"),
             py::arg("X_controls") = Eigen::MatrixXd());

    // =========================================================================
    // Regression Discontinuity (RDD)
    // =========================================================================
    py::enum_<statelix::RDDKernel>(m, "RDDKernel")
        .value("TRIANGULAR", statelix::RDDKernel::TRIANGULAR)
        .value("UNIFORM", statelix::RDDKernel::UNIFORM)
        .value("EPANECHNIKOV", statelix::RDDKernel::EPANECHNIKOV)
        .export_values();
        
    py::class_<statelix::RDDResult>(m, "RDDResult")
        .def_readonly("tau", &statelix::RDDResult::tau)
        .def_readonly("tau_se", &statelix::RDDResult::tau_se)
        .def_readonly("tau_pvalue", &statelix::RDDResult::tau_pvalue)
        .def_readonly("tau_ci_lower", &statelix::RDDResult::tau_ci_lower)
        .def_readonly("tau_ci_upper", &statelix::RDDResult::tau_ci_upper)
        .def_readonly("tau_bc", &statelix::RDDResult::tau_bc)
        .def_readonly("bandwidth", &statelix::RDDResult::bandwidth)
        .def_readonly("n_left", &statelix::RDDResult::n_left)
        .def_readonly("n_right", &statelix::RDDResult::n_right)
        .def_readonly("is_fuzzy", &statelix::RDDResult::is_fuzzy)
        .def_readonly("first_stage_jump", &statelix::RDDResult::first_stage_jump);

    py::class_<statelix::SharpRDD>(m, "SharpRDD")
        .def(py::init<>())
        .def_readwrite("bandwidth", &statelix::SharpRDD::bandwidth)
        .def_readwrite("polynomial_order", &statelix::SharpRDD::polynomial_order)
        .def_readwrite("kernel", &statelix::SharpRDD::kernel)
        .def_readwrite("bias_correction", &statelix::SharpRDD::bias_correction)
        .def("fit", &statelix::SharpRDD::fit,
             py::arg("Y"), py::arg("X"), py::arg("cutoff"), 
             py::arg("cluster_id") = Eigen::VectorXi());
             
    py::class_<statelix::FuzzyRDD>(m, "FuzzyRDD")
        .def(py::init<>())
        .def_readwrite("bandwidth", &statelix::FuzzyRDD::bandwidth)
        .def("fit", &statelix::FuzzyRDD::fit,
             py::arg("Y"), py::arg("D"), py::arg("X"), py::arg("cutoff"));
}
