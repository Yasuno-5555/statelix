#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

// Core Headers
#include "../linear_model/ols.h"
#include "../cluster/kmeans.h"
#include "../stats/anova.h"
#include "../time_series/ar_model.h"
#include "../glm/glm_models.h"
#include "../survival/cox.h"
#include "../optimization/lbfgs.h"

// v1.1 Headers
#include "../graph/louvain.h"
#include "../graph/pagerank.h"
#include "../causal/iv.h"
#include "../causal/did.h"
#include "../bayes/hmc.h"

// v2.3 Headers
#include "../panel/panel.h"
#include "../time_series/var.h"
#include "../stats/tests.h"
#include "../causal/rdd.h"
#include "../causal/psm.h"
#include "../panel/dynamic_panel.h"
#include "../glm/discrete_choice.h"
#include "../linear_model/poisson_full.h"
#include "../time_series/cointegration.h"

namespace py = pybind11;

// Wrapper for HMC target density provided by Python
class PyEfficientObjective : public statelix::EfficientObjective {
public:
    py::object func; // Python callable

    PyEfficientObjective(py::object f) : func(f) {}

    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& x) const override {
        py::gil_scoped_acquire acquire;
        try {
            py::tuple res = func(x);
            if (res.size() != 2) throw std::runtime_error("Callback must return (log_prob, gradient)");
            double log_prob = res[0].cast<double>();
            Eigen::VectorXd grad = res[1].cast<Eigen::VectorXd>();
            return {-log_prob, -grad};
        } catch (py::error_already_set& e) {
            throw std::runtime_error("Python callback error: " + std::string(e.what()));
        }
    }
    
    // Minimal implementation of required pure virtuals if any others exist
    double value(const Eigen::VectorXd& x) const override {
        return value_and_gradient(x).first;
    }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        return value_and_gradient(x).second;
    }
};

// Helper function for HMC
std::vector<Eigen::VectorXd> hmc_wrapper(py::object target_func, Eigen::VectorXd init_params, int n_samples, double step_size, int n_steps) {
    PyEfficientObjective objective(target_func);
    statelix::HMCConfig config;
    config.step_size = step_size;
    config.n_leapfrog_steps = n_steps;
    config.n_samples = n_samples;
    config.n_burnin = n_samples / 2;
    
    statelix::HMC sampler(config);
    auto result = sampler.sample(objective, init_params);
    return result.samples;
}

PYBIND11_MODULE(statelix_core, m) {
    m.doc() = "Statelix Core C++ Module (v2.3) - CLEAN BUILD";

    // ... (rest of module)

    py::class_<statelix::OLSResult>(m, "OLSResult")
        .def_readonly("coef", &statelix::OLSResult::coef)
        .def_readonly("std_errors", &statelix::OLSResult::std_errors)
        .def_readonly("t_values", &statelix::OLSResult::t_values)
        .def_readonly("p_values", &statelix::OLSResult::p_values)
        .def_readonly("r_squared", &statelix::OLSResult::r_squared)
        .def_readonly("adj_r_squared", &statelix::OLSResult::adj_r_squared);

    m.def("fit_ols_qr", &statelix::fit_ols_qr, py::arg("X"), py::arg("y"));
    m.def("fit_ols_full", &statelix::fit_ols_full, py::arg("X"), py::arg("y"), py::arg("fit_intercept")=true, py::arg("conf_level")=0.95);

    // --- GLM ---
    py::module_ glm = m.def_submodule("glm", "Generalized Linear Models");
    py::class_<statelix::LogisticResult>(glm, "LogisticResult")
        .def_readonly("coef", &statelix::LogisticResult::coef)
        .def_readonly("converged", &statelix::LogisticResult::converged);
    py::class_<statelix::LogisticRegression>(glm, "LogisticRegression")
        .def(py::init<>())
        .def("fit", &statelix::LogisticRegression::fit)
        .def("predict_prob", &statelix::LogisticRegression::predict_prob);

    // Poisson Full
    py::class_<statelix::poisson_detail::PoissonResult>(glm, "PoissonResultFull")
        .def_readonly("coef", &statelix::poisson_detail::PoissonResult::coef)
        .def_readonly("std_errors", &statelix::poisson_detail::PoissonResult::std_errors)
        .def_readonly("p_values", &statelix::poisson_detail::PoissonResult::p_values);
    m.def("fit_poisson_full", &statelix::poisson_detail::fit_poisson,
          py::arg("X"), py::arg("y"), py::arg("fit_intercept")=true, py::arg("offset")=Eigen::VectorXd(),
          py::arg("max_iter")=50, py::arg("tol")=1e-8, py::arg("conf_level")=0.95);

    // Discrete Choice
    py::class_<statelix::OrderedLogitResult>(glm, "OrderedLogitResult")
        .def_readonly("coef", &statelix::OrderedLogitResult::coef)
        .def_readonly("thresholds", &statelix::OrderedLogitResult::thresholds);
    py::class_<statelix::OrderedLogit>(glm, "OrderedLogit")
        .def(py::init<>())
        .def("fit", &statelix::OrderedLogit::fit, py::arg("y"), py::arg("X"));

    // --- Panel ---
    py::module_ panel = m.def_submodule("panel", "Panel data models");

    // Dynamic Panel (v2.3)
    py::class_<statelix::panel::DynamicPanelResult>(panel, "DynamicPanelResult")
        .def_readonly("coefficients", &statelix::panel::DynamicPanelResult::coefficients)
        .def_readonly("std_errors", &statelix::panel::DynamicPanelResult::std_errors)
        .def_readonly("sargan_test", &statelix::panel::DynamicPanelResult::sargan_test)
        .def_readonly("n_obs", &statelix::panel::DynamicPanelResult::n_obs);

    py::class_<statelix::panel::DynamicPanelGMM>(panel, "DynamicPanelGMM")
        .def(py::init<>())
        .def_readwrite("max_lags", &statelix::panel::DynamicPanelGMM::max_lags)
        .def_readwrite("two_step", &statelix::panel::DynamicPanelGMM::two_step)
        .def("estimate", &statelix::panel::DynamicPanelGMM::estimate,
             py::arg("y"), py::arg("X"), py::arg("ids"), py::arg("time"));

    // Fixed & Random Effects
    py::class_<statelix::PanelFEResult>(panel, "PanelFEResult")
        .def_readonly("coef", &statelix::PanelFEResult::coef);
    py::class_<statelix::PanelFixedEffects>(panel, "FixedEffects")
        .def(py::init<>())
        .def("fit", &statelix::PanelFixedEffects::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    py::class_<statelix::PanelREResult>(panel, "PanelREResult")
        .def_readonly("coef", &statelix::PanelREResult::coef);
    py::class_<statelix::PanelRandomEffects>(panel, "RandomEffects")
        .def(py::init<>())
        .def("fit", &statelix::PanelRandomEffects::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    // --- Bayes (HMC) ---
    py::module_ bayes = m.def_submodule("bayes", "Bayesian inference");
    
    // Helper function for HMC
    bayes.def("hmc_sample", &hmc_wrapper, 
        py::arg("target_func"), py::arg("init_params"), py::arg("n_samples"), py::arg("step_size"), py::arg("n_steps"),
        "Run HMC sampling on a Python defined objective function");

    // --- Causal ---
    py::module_ causal = m.def_submodule("causal", "Causal inference");

    // PSM (Propensity Score Matching)
    py::enum_<statelix::MatchingMethod>(causal, "MatchingMethod")
        .value("NEAREST_NEIGHBOR", statelix::MatchingMethod::NEAREST_NEIGHBOR)
        .value("CALIPER", statelix::MatchingMethod::CALIPER)
        .value("KERNEL", statelix::MatchingMethod::KERNEL)
        .export_values();

    py::class_<statelix::MatchingResult>(causal, "MatchingResult")
        .def_readonly("att", &statelix::MatchingResult::att)
        .def_readonly("att_se", &statelix::MatchingResult::att_se)
        .def_readonly("n_matched_treated", &statelix::MatchingResult::n_matched_treated);

     py::class_<statelix::PropensityScoreResult>(causal, "PropensityScoreResult")
        .def_readonly("scores", &statelix::PropensityScoreResult::scores);

    py::class_<statelix::PropensityScoreMatching>(causal, "PropensityScoreMatching")
        .def(py::init<>())
        .def_readwrite("method", &statelix::PropensityScoreMatching::method)
        .def_readwrite("n_neighbors", &statelix::PropensityScoreMatching::n_neighbors)
        .def("estimate_propensity", &statelix::PropensityScoreMatching::estimate_propensity, py::arg("D"), py::arg("X"))
        .def("match", &statelix::PropensityScoreMatching::match, py::arg("Y"), py::arg("D"), py::arg("X"), py::arg("ps"))
        .def("ipw", &statelix::PropensityScoreMatching::ipw, py::arg("Y"), py::arg("D"), py::arg("ps"));

    // --- Time Series (VAR, Cointegration) ---
    py::module_ ts = m.def_submodule("time_series", "Time series");
    
    py::class_<statelix::VARResult>(ts, "VARResult")
        .def_readonly("coef", &statelix::VARResult::coef)
        .def_readonly("aic", &statelix::VARResult::aic);
    py::class_<statelix::VectorAutoregression>(ts, "VAR")
        .def(py::init<int>(), py::arg("p")=1)
        .def("fit", &statelix::VectorAutoregression::fit, py::arg("Y"));

    py::class_<statelix::ADFResult>(ts, "ADFResult")
        .def_readonly("adf_statistic", &statelix::ADFResult::adf_statistic)
        .def_readonly("p_value", &statelix::ADFResult::p_value);
    py::class_<statelix::ADF>(ts, "ADF")
        .def(py::init<>())
        .def("test", &statelix::ADF::test, py::arg("y"));

    // --- Stats / Tests ---
    py::module_ tests = m.def_submodule("tests", "Statistical tests");
    
    py::class_<statelix::tests::DurbinWatsonResult>(tests, "DurbinWatsonResult")
        .def_readonly("dw_statistic", &statelix::tests::DurbinWatsonResult::dw_statistic);
    tests.def("durbin_watson", &statelix::tests::durbin_watson, py::arg("residuals"), py::arg("n_obs"), py::arg("k"));

    tests.def("breusch_godfrey", &statelix::tests::breusch_godfrey, py::arg("X"), py::arg("residuals"), py::arg("order")=1);

    py::class_<statelix::tests::ConditionNumberResult>(tests, "ConditionNumberResult")
        .def_readonly("condition_number", &statelix::tests::ConditionNumberResult::condition_number);
    tests.def("condition_number", &statelix::tests::condition_number, py::arg("X"));

}
