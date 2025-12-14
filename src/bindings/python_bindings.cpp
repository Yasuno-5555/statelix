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
#include "../time_series/cpd.h"
#include "../time_series/kalman.h"
#include "../spatial/icp.h"
#include "../signal/wavelet.h"
#include "../ml/gbdt.h"
#include "../optimization/lbfgs.h"
#include "../bayes/mcmc.h"
#include "../bayes/vi.h"
#include "../ml/fm.h"
#include "../linalg/sparse_core.h"
#include "../stats/robust.h"
#include "../quant/quantized.h"
#include "../optim/branch_bound.h"

namespace py = pybind11;

// --- Trampolines ---

class PyDifferentiableFunction : public statelix::DifferentiableFunction {
public:
    using statelix::DifferentiableFunction::DifferentiableFunction;
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) override {
        py::gil_scoped_acquire gil;
        py::function overload = py::get_overload(this, static_cast<const statelix::DifferentiableFunction*>(this), "__call__");
        if (overload) {
            auto result_obj = overload(x);
            auto result = result_obj.cast<py::tuple>();
            double val = result[0].cast<double>();
            grad = result[1].cast<Eigen::VectorXd>(); 
            return val;
        }
        return 0.0;
    }
};

struct LogProbFunction {
    virtual ~LogProbFunction() = default;
    virtual double operator()(const Eigen::VectorXd& x) = 0;
};

class PyLogProbFunction : public LogProbFunction {
public:
    using LogProbFunction::LogProbFunction;
    double operator()(const Eigen::VectorXd& x) override {
        PYBIND11_OVERRIDE_PURE(double, LogProbFunction, operator(), x);
    }
};

// --- Module Definition ---

PYBIND11_MODULE(statelix_core, m) {
    m.doc() = "Statelix Core C++ Module";

    // --- OLS ---
    py::class_<statelix::OLSResult>(m, "OLSResult")
        .def_readonly("coef", &statelix::OLSResult::coef)
        .def_readonly("intercept", &statelix::OLSResult::intercept)
        .def_readonly("std_errors", &statelix::OLSResult::std_errors)
        .def_readonly("t_values", &statelix::OLSResult::t_values)
        .def_readonly("p_values", &statelix::OLSResult::p_values)
        .def_readonly("conf_int", &statelix::OLSResult::conf_int)
        .def_readonly("residuals", &statelix::OLSResult::residuals)
        .def_readonly("fitted_values", &statelix::OLSResult::fitted_values)
        .def_readonly("r_squared", &statelix::OLSResult::r_squared)
        .def_readonly("adj_r_squared", &statelix::OLSResult::adj_r_squared)
        .def_readonly("f_statistic", &statelix::OLSResult::f_statistic)
        .def_readonly("f_pvalue", &statelix::OLSResult::f_pvalue)
        .def_readonly("residual_std_error", &statelix::OLSResult::residual_std_error)
        .def_readonly("aic", &statelix::OLSResult::aic)
        .def_readonly("bic", &statelix::OLSResult::bic)
        .def_readonly("log_likelihood", &statelix::OLSResult::log_likelihood)
        .def_readonly("vcov", &statelix::OLSResult::vcov)
        .def_readonly("n_obs", &statelix::OLSResult::n_obs)
        .def_readonly("n_params", &statelix::OLSResult::n_params);

    py::class_<statelix::PredictionInterval>(m, "PredictionInterval")
        .def_readonly("predictions", &statelix::PredictionInterval::predictions)
        .def_readonly("lower_bound", &statelix::PredictionInterval::lower_bound)
        .def_readonly("upper_bound", &statelix::PredictionInterval::upper_bound);

    m.def("fit_ols_qr", &statelix::fit_ols_qr, py::arg("X"), py::arg("y"));
    m.def("fit_ols_full", &statelix::fit_ols_full, py::arg("X"), py::arg("y"), py::arg("fit_intercept")=true, py::arg("conf_level")=0.95);
    m.def("predict_ols", &statelix::predict_ols, py::arg("result"), py::arg("X_new"), py::arg("fit_intercept")=true);
    m.def("predict_with_interval", &statelix::predict_with_interval, py::arg("result"), py::arg("X_new"), py::arg("fit_intercept")=true, py::arg("conf_level")=0.95);

    // --- K-Means ---
    py::class_<statelix::KMeansResult>(m, "KMeansResult")
        .def_readonly("centroids", &statelix::KMeansResult::centroids)
        .def_readonly("labels", &statelix::KMeansResult::labels)
        .def_readonly("inertia", &statelix::KMeansResult::inertia)
        .def_readonly("n_iter", &statelix::KMeansResult::n_iter);
    m.def("fit_kmeans", &statelix::fit_kmeans, py::arg("X"), py::arg("k"), py::arg("max_iter")=300, py::arg("tol")=1e-4, py::arg("random_state")=42);

    // --- ANOVA ---
    py::class_<statelix::AnoVaResult>(m, "AnoVaResult")
        .def_readonly("f_statistic", &statelix::AnoVaResult::f_statistic)
        .def_readonly("p_value", &statelix::AnoVaResult::p_value)
        .def_readonly("ss_between", &statelix::AnoVaResult::ss_between)
        .def_readonly("ss_within", &statelix::AnoVaResult::ss_within)
        .def_readonly("ss_total", &statelix::AnoVaResult::ss_total)
        .def_readonly("df_between", &statelix::AnoVaResult::df_between)
        .def_readonly("df_within", &statelix::AnoVaResult::df_within);
    m.def("f_oneway", &statelix::f_oneway_flat, py::arg("data"), py::arg("groups"));

    // --- AR ---
    py::class_<statelix::ARResult>(m, "ARResult")
        .def_readonly("params", &statelix::ARResult::params)
        .def_readonly("sigma2", &statelix::ARResult::sigma2)
        .def_readonly("p", &statelix::ARResult::p);
    m.def("fit_ar", &statelix::fit_ar, py::arg("series"), py::arg("p"));

    // --- GLM ---
    py::class_<statelix::LogisticResult>(m, "LogisticResult").def_readonly("coef", &statelix::LogisticResult::coef).def_readonly("iterations", &statelix::LogisticResult::iterations).def_readonly("converged", &statelix::LogisticResult::converged);
    py::class_<statelix::LogisticRegression>(m, "LogisticRegression").def(py::init<>()).def_readwrite("max_iter", &statelix::LogisticRegression::max_iter).def("fit", &statelix::LogisticRegression::fit).def("predict_prob", &statelix::LogisticRegression::predict_prob);
    py::class_<statelix::PoissonResult>(m, "PoissonResult").def_readonly("coef", &statelix::PoissonResult::coef).def_readonly("iterations", &statelix::PoissonResult::iterations).def_readonly("converged", &statelix::PoissonResult::converged);
    py::class_<statelix::PoissonRegression>(m, "PoissonRegression").def(py::init<>()).def_readwrite("max_iter", &statelix::PoissonRegression::max_iter).def("fit", &statelix::PoissonRegression::fit);
    py::class_<statelix::NegBinResult>(m, "NegBinResult").def_readonly("coef", &statelix::NegBinResult::coef).def_readonly("theta", &statelix::NegBinResult::theta).def_readonly("iterations", &statelix::NegBinResult::iterations);
    py::class_<statelix::NegBinRegression>(m, "NegBinRegression").def(py::init<>()).def("fit", &statelix::NegBinRegression::fit);
    py::class_<statelix::GammaResult>(m, "GammaResult").def_readonly("coef", &statelix::GammaResult::coef).def_readonly("iterations", &statelix::GammaResult::iterations);
    py::class_<statelix::GammaRegression>(m, "GammaRegression").def(py::init<>()).def("fit", &statelix::GammaRegression::fit);
    py::class_<statelix::ProbitResult>(m, "ProbitResult").def_readonly("coef", &statelix::ProbitResult::coef).def_readonly("iterations", &statelix::ProbitResult::iterations);
    py::class_<statelix::ProbitRegression>(m, "ProbitRegression").def(py::init<>()).def("fit", &statelix::ProbitRegression::fit);

    // --- Regularized ---
    py::class_<statelix::RidgeResult>(m, "RidgeResult").def_readonly("coef", &statelix::RidgeResult::coef);
    py::class_<statelix::RidgeRegression>(m, "RidgeRegression").def(py::init<>()).def_readwrite("alpha", &statelix::RidgeRegression::alpha).def("fit", &statelix::RidgeRegression::fit);
    py::class_<statelix::CoxResult>(m, "CoxResult").def_readonly("coef", &statelix::CoxResult::coef);
    py::class_<statelix::CoxPH>(m, "CoxPH").def(py::init<>()).def("fit", &statelix::CoxPH::fit);
    py::class_<statelix::ElasticNetResult>(m, "ElasticNetResult").def_readonly("coef", &statelix::ElasticNetResult::coef).def_readonly("intercept", &statelix::ElasticNetResult::intercept).def_readonly("iterations", &statelix::ElasticNetResult::iterations).def_readonly("duality_gap", &statelix::ElasticNetResult::duality_gap);
    py::class_<statelix::ElasticNet>(m, "ElasticNet").def(py::init<>()).def_readwrite("alpha", &statelix::ElasticNet::alpha).def_readwrite("l1_ratio", &statelix::ElasticNet::l1_ratio).def_readwrite("max_iter", &statelix::ElasticNet::max_iter).def("fit", &statelix::ElasticNet::fit);

    // --- Time Series ---
    py::class_<statelix::DTWResult>(m, "DTWResult").def_readonly("distance", &statelix::DTWResult::distance).def_readonly("path", &statelix::DTWResult::path);
    py::class_<statelix::DTW>(m, "DTW").def(py::init<>()).def("compute", &statelix::DTW::compute);
    py::class_<statelix::KNNSearchResult>(m, "KNNSearchResult").def_readonly("indices", &statelix::KNNSearchResult::indices).def_readonly("distances", &statelix::KNNSearchResult::distances);
    py::class_<statelix::KDTree>(m, "KDTree").def(py::init<>()).def("fit", &statelix::KDTree::fit).def("query", &statelix::KDTree::query);
    py::class_<statelix::CPDResult>(m, "CPDResult").def_readonly("change_points", &statelix::CPDResult::change_points).def_readonly("cost", &statelix::CPDResult::cost);
    py::class_<statelix::ChangePointDetector>(m, "ChangePointDetector").def(py::init<>()).def_readwrite("penalty", &statelix::ChangePointDetector::penalty).def("fit_pelt", &statelix::ChangePointDetector::fit_pelt);
    py::class_<statelix::KalmanResult>(m, "KalmanResult").def_readonly("states", &statelix::KalmanResult::states).def_readonly("smoothed_states", &statelix::KalmanResult::smoothed_states).def_readonly("log_likelihood", &statelix::KalmanResult::log_likelihood);
    py::class_<statelix::KalmanFilter>(m, "KalmanFilter").def(py::init<int, int>(), py::arg("state_dim"), py::arg("measure_dim")).def_readwrite("F", &statelix::KalmanFilter::F).def_readwrite("H", &statelix::KalmanFilter::H).def_readwrite("Q", &statelix::KalmanFilter::Q).def_readwrite("R", &statelix::KalmanFilter::R).def_readwrite("P", &statelix::KalmanFilter::P).def_readwrite("x", &statelix::KalmanFilter::x).def("predict", &statelix::KalmanFilter::predict).def("update", &statelix::KalmanFilter::update).def("filter", &statelix::KalmanFilter::filter);

    // --- Spatial/Signal ---
    py::class_<statelix::ICPResult>(m, "ICPResult").def_readonly("rotation", &statelix::ICPResult::rotation).def_readonly("translation", &statelix::ICPResult::translation).def_readonly("rmse", &statelix::ICPResult::rmse).def_readonly("converged", &statelix::ICPResult::converged);
    py::class_<statelix::ICP>(m, "ICP").def(py::init<>()).def_readwrite("max_iter", &statelix::ICP::max_iter).def_readwrite("tol", &statelix::ICP::tol).def("align", &statelix::ICP::align);
    py::enum_<statelix::WaveletType>(m, "WaveletType").value("Haar", statelix::WaveletType::Haar).value("Daubechies4", statelix::WaveletType::Daubechies4).export_values();
    py::class_<statelix::WaveletTransform>(m, "WaveletTransform").def(py::init<>()).def_readwrite("type", &statelix::WaveletTransform::type).def("transform", &statelix::WaveletTransform::transform).def("inverse", &statelix::WaveletTransform::inverse);

    // --- ML ---
    py::class_<statelix::GradientBoostingRegressor>(m, "GradientBoostingRegressor").def(py::init<>()).def_readwrite("n_estimators", &statelix::GradientBoostingRegressor::n_estimators).def_readwrite("learning_rate", &statelix::GradientBoostingRegressor::learning_rate).def_readwrite("max_depth", &statelix::GradientBoostingRegressor::max_depth).def_readwrite("subsample", &statelix::GradientBoostingRegressor::subsample).def("fit", &statelix::GradientBoostingRegressor::fit).def("predict", &statelix::GradientBoostingRegressor::predict);
    py::enum_<statelix::FMTask>(m, "FMTask").value("Regression", statelix::FMTask::Regression).value("Classification", statelix::FMTask::Classification).export_values();
    py::class_<statelix::FactorizationMachine>(m, "FactorizationMachine").def(py::init<>()).def_readwrite("n_factors", &statelix::FactorizationMachine::n_factors).def_readwrite("max_iter", &statelix::FactorizationMachine::max_iter).def_readwrite("learning_rate", &statelix::FactorizationMachine::learning_rate).def_readwrite("reg_w", &statelix::FactorizationMachine::reg_w).def_readwrite("reg_v", &statelix::FactorizationMachine::reg_v).def_readwrite("task", &statelix::FactorizationMachine::task).def("fit", &statelix::FactorizationMachine::fit).def("predict", &statelix::FactorizationMachine::predict);

    // --- Optimization: L-BFGS ---
    py::class_<statelix::DifferentiableFunction, PyDifferentiableFunction>(m, "DifferentiableFunction").def(py::init<>()).def("__call__", &statelix::DifferentiableFunction::operator());
    py::class_<statelix::OptimizerResult>(m, "OptimizerResult").def_readonly("x", &statelix::OptimizerResult::x).def_readonly("min_value", &statelix::OptimizerResult::min_value).def_readonly("iterations", &statelix::OptimizerResult::iterations).def_readonly("converged", &statelix::OptimizerResult::converged);
    using PythonLBFGS = statelix::LBFGS<statelix::DifferentiableFunction>;
    py::class_<PythonLBFGS>(m, "LBFGS").def(py::init<>()).def_readwrite("max_iter", &PythonLBFGS::max_iter).def_readwrite("m", &PythonLBFGS::m).def_readwrite("epsilon", &PythonLBFGS::epsilon).def("minimize", &PythonLBFGS::minimize);

    // --- Bayes: MCMC ---
    py::class_<LogProbFunction, PyLogProbFunction>(m, "LogProbFunction").def(py::init<>()).def("__call__", &LogProbFunction::operator());
    py::class_<statelix::MCMCResult>(m, "MCMCResult").def_readonly("samples", &statelix::MCMCResult::samples).def_readonly("log_probs", &statelix::MCMCResult::log_probs).def_readonly("acceptance_rate", &statelix::MCMCResult::acceptance_rate);
    using PythonMCMC = statelix::MetropolisHastings<LogProbFunction>;
    py::class_<PythonMCMC>(m, "MetropolisHastings").def(py::init<>()).def_readwrite("n_samples", &PythonMCMC::n_samples).def_readwrite("burn_in", &PythonMCMC::burn_in).def_readwrite("step_size", &PythonMCMC::step_size).def("sample", &PythonMCMC::sample);

    // --- Stats: Robust (Huber) ---
    py::class_<statelix::HuberResult>(m, "HuberResult")
        .def_readonly("coef", &statelix::HuberResult::coef)
        .def_readonly("delta", &statelix::HuberResult::delta)
        .def_readonly("iterations", &statelix::HuberResult::iterations)
        .def_readonly("converged", &statelix::HuberResult::converged);
    py::class_<statelix::HuberRegression>(m, "HuberRegression")
        .def(py::init<>())
        .def_readwrite("delta", &statelix::HuberRegression::delta)
        .def_readwrite("max_iter", &statelix::HuberRegression::max_iter)
        .def("fit", &statelix::HuberRegression::fit);

    // --- Linalg: Sparse Core ---
    py::class_<statelix::SparseMatrix>(m, "SparseMatrix")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def("from_csr", &statelix::SparseMatrix::from_csr, py::arg("data"), py::arg("indices"), py::arg("indptr"), py::arg("rows"), py::arg("cols"))
        .def("dot", &statelix::SparseMatrix::dot)
        .def("solve_cholesky", &statelix::SparseMatrix::solve_cholesky)
        .def("solve_lu", &statelix::SparseMatrix::solve_lu)
        .def_property_readonly("rows", &statelix::SparseMatrix::rows)
        .def_property_readonly("cols", &statelix::SparseMatrix::cols)
        .def_property_readonly("nnz", &statelix::SparseMatrix::nnz);

    // --- Quant: Quantized Inference ---
    py::class_<statelix::QuantizedTensor>(m, "QuantizedTensor")
        .def_readonly("rows", &statelix::QuantizedTensor::rows)
        .def_readonly("cols", &statelix::QuantizedTensor::cols);
    m.def("quantize", [](const std::vector<float>& input, int rows, int cols) {
        return statelix::quantize(input, rows, cols);
    }, py::arg("input"), py::arg("rows"), py::arg("cols"));
    m.def("dequantize", &statelix::dequantize, py::arg("tensor"));
    m.def("quantized_matmul", &statelix::quantized_matmul, py::arg("A"), py::arg("B"), py::arg("output_scale")=0.0f);

    // --- Optim: Branch & Bound ---
    py::class_<statelix::BnBResult>(m, "BnBResult")
        .def_readonly("solution", &statelix::BnBResult::solution)
        .def_readonly("objective", &statelix::BnBResult::objective)
        .def_readonly("nodes_explored", &statelix::BnBResult::nodes_explored)
        .def_readonly("feasible", &statelix::BnBResult::feasible);
    py::class_<statelix::BranchAndBound>(m, "BranchAndBound")
        .def(py::init<>())
        .def_readwrite("max_nodes", &statelix::BranchAndBound::max_nodes)
        .def("solve", &statelix::BranchAndBound::solve);
}
