#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../linear_model/ols.h"
#include "../cluster/kmeans.h"
#include "../stats/anova.h"
#include "../time_series/ar_model.h"
#include "../glm/glm_models.h"
#include "../survival/cox.h"
#include "../linear_model/ridge.h"
#include "../linear_model/elastic_net.h"
#include "../time_series/dtw.h"
#include "../search/kdtree.h"

namespace py = pybind11;

PYBIND11_MODULE(statelix_core, m) {
    m.doc() = "Statelix Core C++ Module";

    // --- OLS Bindings ---
    py::class_<statelix::OLSResult>(m, "OLSResult")
        .def_readonly("coef", &statelix::OLSResult::coef, "Regression coefficients")
        .def_readonly("intercept", &statelix::OLSResult::intercept, "Intercept term")
        .def_readonly("std_errors", &statelix::OLSResult::std_errors, "Standard errors of coefficients")
        .def_readonly("t_values", &statelix::OLSResult::t_values, "t-statistics")
        .def_readonly("p_values", &statelix::OLSResult::p_values, "p-values for coefficients")
        .def_readonly("conf_int", &statelix::OLSResult::conf_int, "Confidence intervals")
        .def_readonly("residuals", &statelix::OLSResult::residuals, "Residuals")
        .def_readonly("fitted_values", &statelix::OLSResult::fitted_values, "Fitted values")
        .def_readonly("r_squared", &statelix::OLSResult::r_squared, "R-squared")
        .def_readonly("adj_r_squared", &statelix::OLSResult::adj_r_squared, "Adjusted R-squared")
        .def_readonly("f_statistic", &statelix::OLSResult::f_statistic, "F-statistic")
        .def_readonly("f_pvalue", &statelix::OLSResult::f_pvalue, "p-value of F-statistic")
        .def_readonly("residual_std_error", &statelix::OLSResult::residual_std_error, "Residual standard error")
        .def_readonly("aic", &statelix::OLSResult::aic, "AIC")
        .def_readonly("bic", &statelix::OLSResult::bic, "BIC")
        .def_readonly("log_likelihood", &statelix::OLSResult::log_likelihood, "Log-likelihood")
        .def_readonly("vcov", &statelix::OLSResult::vcov, "Variance-covariance matrix")
        .def_readonly("n_obs", &statelix::OLSResult::n_obs, "Number of observations")
        .def_readonly("n_params", &statelix::OLSResult::n_params, "Number of parameters");

    py::class_<statelix::PredictionInterval>(m, "PredictionInterval")
        .def_readonly("predictions", &statelix::PredictionInterval::predictions)
        .def_readonly("lower_bound", &statelix::PredictionInterval::lower_bound)
        .def_readonly("upper_bound", &statelix::PredictionInterval::upper_bound);

    m.def("fit_ols_qr", &statelix::fit_ols_qr, "Fit simple OLS with QR", py::arg("X"), py::arg("y"));
    m.def("fit_ols_full", &statelix::fit_ols_full, "Fit full OLS model", 
          py::arg("X"), py::arg("y"), py::arg("fit_intercept")=true, py::arg("conf_level")=0.95);
    m.def("predict_ols", &statelix::predict_ols, "Predict OLS", py::arg("result"), py::arg("X_new"), py::arg("fit_intercept")=true);
    m.def("predict_with_interval", &statelix::predict_with_interval, "Predict with Interval", 
          py::arg("result"), py::arg("X_new"), py::arg("fit_intercept")=true, py::arg("conf_level")=0.95);


    // --- K-Means Bindings ---
    py::class_<statelix::KMeansResult>(m, "KMeansResult")
        .def_readonly("centroids", &statelix::KMeansResult::centroids)
        .def_readonly("labels", &statelix::KMeansResult::labels)
        .def_readonly("inertia", &statelix::KMeansResult::inertia)
        .def_readonly("n_iter", &statelix::KMeansResult::n_iter);

    m.def("fit_kmeans", &statelix::fit_kmeans, "Fit K-Means clustering",
          py::arg("X"), py::arg("k"), py::arg("max_iter")=300, py::arg("tol")=1e-4, py::arg("random_state")=42);

    // --- ANOVA Bindings ---
    py::class_<statelix::AnoVaResult>(m, "AnoVaResult")
        .def_readonly("f_statistic", &statelix::AnoVaResult::f_statistic)
        .def_readonly("p_value", &statelix::AnoVaResult::p_value)
        .def_readonly("ss_between", &statelix::AnoVaResult::ss_between)
        .def_readonly("ss_within", &statelix::AnoVaResult::ss_within)
        .def_readonly("ss_total", &statelix::AnoVaResult::ss_total)
        .def_readonly("df_between", &statelix::AnoVaResult::df_between)
        .def_readonly("df_within", &statelix::AnoVaResult::df_within);

    m.def("f_oneway", &statelix::f_oneway_flat, "One-Way ANOVA (flat data + groups)",
          py::arg("data"), py::arg("groups"));
    // Note: f_oneway with vector<VectorXd> might be tricky to bind directly unless pybind supports it simply (it usually does as list of arrays)
    // But f_oneway_flat is safer/standard for dataframe-like inputs.

    // --- AR Model Bindings ---
    py::class_<statelix::ARResult>(m, "ARResult")
        .def_readonly("params", &statelix::ARResult::params)
        .def_readonly("sigma2", &statelix::ARResult::sigma2)
        .def_readonly("p", &statelix::ARResult::p);

    m.def("fit_ar", &statelix::fit_ar, "Fit Autoregressive Model",
          py::arg("series"), py::arg("p"));

    // --- GLM: Logistic ---
    py::class_<statelix::LogisticResult>(m, "LogisticResult")
        .def_readonly("coef", &statelix::LogisticResult::coef)
        .def_readonly("iterations", &statelix::LogisticResult::iterations)
        .def_readonly("converged", &statelix::LogisticResult::converged);

    py::class_<statelix::LogisticRegression>(m, "LogisticRegression")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::LogisticRegression::max_iter)
        .def("fit", &statelix::LogisticRegression::fit)
        .def("predict_prob", &statelix::LogisticRegression::predict_prob);

    // --- GLM: Poisson ---
    py::class_<statelix::PoissonResult>(m, "PoissonResult")
        .def_readonly("coef", &statelix::PoissonResult::coef)
        .def_readonly("iterations", &statelix::PoissonResult::iterations)
        .def_readonly("converged", &statelix::PoissonResult::converged);

    py::class_<statelix::PoissonRegression>(m, "PoissonRegression")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::PoissonRegression::max_iter)
        .def("fit", &statelix::PoissonRegression::fit);
    
    // --- GLM: Negative Binomial ---
    py::class_<statelix::NegBinResult>(m, "NegBinResult")
        .def_readonly("coef", &statelix::NegBinResult::coef)
        .def_readonly("theta", &statelix::NegBinResult::theta)
        .def_readonly("iterations", &statelix::NegBinResult::iterations);

    py::class_<statelix::NegBinRegression>(m, "NegBinRegression")
        .def(py::init<>())
        .def("fit", &statelix::NegBinRegression::fit);

    // --- GLM: Gamma ---
    py::class_<statelix::GammaResult>(m, "GammaResult")
        .def_readonly("coef", &statelix::GammaResult::coef)
        .def_readonly("iterations", &statelix::GammaResult::iterations);

    py::class_<statelix::GammaRegression>(m, "GammaRegression")
        .def(py::init<>())
        .def("fit", &statelix::GammaRegression::fit);

    // --- GLM: Probit ---
    py::class_<statelix::ProbitResult>(m, "ProbitResult")
        .def_readonly("coef", &statelix::ProbitResult::coef)
        .def_readonly("iterations", &statelix::ProbitResult::iterations);

    py::class_<statelix::ProbitRegression>(m, "ProbitRegression")
        .def(py::init<>())
        .def("fit", &statelix::ProbitRegression::fit);

    // --- Regularized: Ridge ---
    py::class_<statelix::RidgeResult>(m, "RidgeResult")
        .def_readonly("coef", &statelix::RidgeResult::coef);

    py::class_<statelix::RidgeRegression>(m, "RidgeRegression")
        .def(py::init<>())
        .def_readwrite("alpha", &statelix::RidgeRegression::alpha)
        .def("fit", &statelix::RidgeRegression::fit);

    // --- Survival: CoxPH ---
    py::class_<statelix::CoxResult>(m, "CoxResult")
        .def_readonly("coef", &statelix::CoxResult::coef);

    py::class_<statelix::CoxPH>(m, "CoxPH")
        .def(py::init<>())
        .def("fit", &statelix::CoxPH::fit);

    // --- Regularized: ElasticNet ---
    py::class_<statelix::ElasticNetResult>(m, "ElasticNetResult")
        .def_readonly("coef", &statelix::ElasticNetResult::coef)
        .def_readonly("intercept", &statelix::ElasticNetResult::intercept)
        .def_readonly("iterations", &statelix::ElasticNetResult::iterations)
        .def_readonly("duality_gap", &statelix::ElasticNetResult::duality_gap);

    py::class_<statelix::ElasticNet>(m, "ElasticNet")
        .def(py::init<>())
        .def_readwrite("alpha", &statelix::ElasticNet::alpha)
        .def_readwrite("l1_ratio", &statelix::ElasticNet::l1_ratio)
        .def_readwrite("max_iter", &statelix::ElasticNet::max_iter)
        .def("fit", &statelix::ElasticNet::fit);

    // --- Time Series: DTW ---
    py::class_<statelix::DTWResult>(m, "DTWResult")
        .def_readonly("distance", &statelix::DTWResult::distance)
        .def_readonly("path", &statelix::DTWResult::path);

    py::class_<statelix::DTW>(m, "DTW")
        .def(py::init<>())
        .def("compute", &statelix::DTW::compute);

    // --- Search: KDTree ---
    py::class_<statelix::KNNSearchResult>(m, "KNNSearchResult")
        .def_readonly("indices", &statelix::KNNSearchResult::indices)
        .def_readonly("distances", &statelix::KNNSearchResult::distances);

    py::class_<statelix::KDTree>(m, "KDTree")
        .def(py::init<>())
        .def("fit", &statelix::KDTree::fit)
        .def("query", &statelix::KDTree::query);
}
