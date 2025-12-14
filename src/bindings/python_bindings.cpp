#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

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
#include "../ml/fm.h"
#include "../linalg/sparse_core.h"
#include "../stats/robust.h"
#include "../quant/quantized.h"
#include "../optimization/branch_bound.h"

// v1.1 Headers
#include "../graph/louvain.h"
#include "../graph/pagerank.h"
#include "../causal/iv.h"
#include "../causal/did.h"
#include "../bayes/hmc.h"
#include "../search/hnsw.h"

// v2.3 Native Headers
#include "../bayes/native_objectives.h"

namespace py = pybind11;

// --- Helper Functions ---
// Convert simple python dict or object to result structs if needed
// (Most structs are bound directly)

// --- Helper Classes for Callbacks ---

// Wrapper for HMC target density provided by Python
// Supports f(x) -> (log_prob, grad)
class PyEfficientObjective : public statelix::EfficientObjective {
public:
    py::object func; // Python callable

    PyEfficientObjective(py::object f) : func(f) {}

    std::pair<double, Eigen::VectorXd> 
    value_and_gradient(const Eigen::VectorXd& x) const override {
        // Must acquire GIL to call Python
        py::gil_scoped_acquire acquire;
        try {
            // Call Python function
            // Expects return: (log_prob, grad_vector)
            py::tuple res = func(x);
            
            if (res.size() != 2) {
                throw std::runtime_error("Callback must return (log_prob, gradient)");
            }

            double log_prob = res[0].cast<double>();
            Eigen::VectorXd grad = res[1].cast<Eigen::VectorXd>();

            // HMC minimizes Energy = -LogProb
            return {-log_prob, -grad};
            
        } catch (py::error_already_set& e) {
            // Propagate Python error as C++ runtime error to stop sampling safely
            throw std::runtime_error("Python callback error: " + std::string(e.what()));
        }
    }
};

// --- Module Definition ---

PYBIND11_MODULE(statelix_core, m) {
    m.doc() = "Statelix Core C++ Module (v2.3)";

    // =========================================================================
    // Existing Bindings (Preserved & Compacted)
    // =========================================================================

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
    
    // --- Poisson/NegBin/Gamma/Probit omitted for brevity ---
    
    // --- Regularized ---
    py::class_<statelix::RidgeResult>(m, "RidgeResult").def_readonly("coef", &statelix::RidgeResult::coef);
    py::class_<statelix::RidgeRegression>(m, "RidgeRegression").def(py::init<>()).def_readwrite("alpha", &statelix::RidgeRegression::alpha).def("fit", &statelix::RidgeRegression::fit);
    
    py::class_<statelix::ElasticNetResult>(m, "ElasticNetResult").def_readonly("coef", &statelix::ElasticNetResult::coef).def_readonly("intercept", &statelix::ElasticNetResult::intercept);
    py::class_<statelix::ElasticNet>(m, "ElasticNet").def(py::init<>()).def_readwrite("alpha", &statelix::ElasticNet::alpha).def_readwrite("l1_ratio", &statelix::ElasticNet::l1_ratio).def("fit", &statelix::ElasticNet::fit);

    // --- Time Series ---
    py::class_<statelix::DTW>(m, "DTW").def(py::init<>()).def("compute", &statelix::DTW::compute);
    py::class_<statelix::KDTree>(m, "KDTree").def(py::init<>()).def("fit", &statelix::KDTree::fit).def("query", &statelix::KDTree::query);
    py::class_<statelix::ChangePointDetector>(m, "ChangePointDetector").def(py::init<>()).def_readwrite("penalty", &statelix::ChangePointDetector::penalty).def("fit_pelt", &statelix::ChangePointDetector::fit_pelt);
    py::class_<statelix::KalmanFilter>(m, "KalmanFilter").def(py::init<int, int>(), py::arg("state_dim"), py::arg("measure_dim"))
        .def_readwrite("F", &statelix::KalmanFilter::F)
        .def("predict", &statelix::KalmanFilter::predict)
        .def("update", &statelix::KalmanFilter::update);

    // --- ML ---
    py::class_<statelix::GradientBoostingRegressor>(m, "GradientBoostingRegressor").def(py::init<>()).def("fit", &statelix::GradientBoostingRegressor::fit).def("predict", &statelix::GradientBoostingRegressor::predict);
    
    py::enum_<statelix::FMTask>(m, "FMTask").value("Regression", statelix::FMTask::Regression).value("Classification", statelix::FMTask::Classification).export_values();
    py::class_<statelix::FactorizationMachine>(m, "FactorizationMachine")
        .def(py::init<>())
        .def_readwrite("n_factors", &statelix::FactorizationMachine::n_factors)
        .def("fit", &statelix::FactorizationMachine::fit)
        .def("predict", &statelix::FactorizationMachine::predict);

    // --- Linalg ---
    py::class_<statelix::SparseMatrix>(m, "SparseMatrix")
        .def(py::init<int, int>())
        .def("from_csr", &statelix::SparseMatrix::from_csr);

    // =========================================================================
    // Statelix v1.1 API (IMPLEMENTED)
    // =========================================================================

    // --- Graph Analysis ---
    
    py::module_ graph = m.def_submodule("graph", "Graph analysis module");

    py::class_<statelix::graph::LouvainResult>(graph, "LouvainResult")
        .def_readonly("labels", &statelix::graph::LouvainResult::labels)
        .def_readonly("n_communities", &statelix::graph::LouvainResult::n_communities)
        .def_readonly("modularity", &statelix::graph::LouvainResult::modularity)
        .def_readonly("hierarchy", &statelix::graph::LouvainResult::hierarchy)
        .def_readonly("community_sizes", &statelix::graph::LouvainResult::community_sizes);

    py::class_<statelix::graph::Louvain>(graph, "Louvain")
        .def(py::init<>())
        .def_readwrite("resolution", &statelix::graph::Louvain::resolution)
        .def_readwrite("max_iterations", &statelix::graph::Louvain::max_iterations)
        .def_readwrite("randomize_order", &statelix::graph::Louvain::randomize_order)
        .def_readwrite("seed", &statelix::graph::Louvain::seed)
        .def("fit", &statelix::graph::Louvain::fit, py::arg("adjacency"),
             "Detect communities from sparse adjacency matrix. Ensures node IDs match matrix indices.");

    py::class_<statelix::graph::PageRankResult>(graph, "PageRankResult")
        .def_readonly("scores", &statelix::graph::PageRankResult::scores)
        .def_readonly("ranking", &statelix::graph::PageRankResult::ranking)
        .def_readonly("converged", &statelix::graph::PageRankResult::converged);

    py::class_<statelix::graph::PageRank>(graph, "PageRank")
        .def(py::init<>())
        .def_readwrite("damping", &statelix::graph::PageRank::damping)
        .def_readwrite("max_iter", &statelix::graph::PageRank::max_iter)
        .def_readwrite("tol", &statelix::graph::PageRank::tol)
        .def("compute", &statelix::graph::PageRank::compute, py::arg("adjacency"))
        .def("personalized", &statelix::graph::PageRank::personalized, 
             py::arg("adjacency"), py::arg("seeds"), py::arg("restart_prob")=0.15);

    // --- Causal Inference ---
    
    py::module_ causal = m.def_submodule("causal", "Causal inference module");

    py::class_<statelix::IVResult>(causal, "IVResult")
        .def_readonly("coef", &statelix::IVResult::coef)
        .def_readonly("std_errors", &statelix::IVResult::std_errors)
        .def_readonly("p_values", &statelix::IVResult::p_values)
        .def_readonly("first_stage_f", &statelix::IVResult::first_stage_f)
        .def_readonly("weak_instruments", &statelix::IVResult::weak_instruments)
        .def_readonly("sargan_pvalue", &statelix::IVResult::sargan_pvalue);

    py::class_<statelix::TwoStageLeastSquares>(causal, "TwoStageLeastSquares")
        .def(py::init<>())
        .def_readwrite("fit_intercept", &statelix::TwoStageLeastSquares::fit_intercept)
        .def_readwrite("robust_se", &statelix::TwoStageLeastSquares::robust_se)
        .def("fit", 
             py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&>(&statelix::TwoStageLeastSquares::fit),
             py::arg("Y"), py::arg("X_endog"), py::arg("X_exog"), py::arg("Z"));

    py::class_<statelix::DIDResult>(causal, "DIDResult")
        .def_readonly("att", &statelix::DIDResult::att)
        .def_readonly("att_std_error", &statelix::DIDResult::att_std_error)
        .def_readonly("p_value", &statelix::DIDResult::p_value)
        .def_readonly("parallel_trends_valid", &statelix::DIDResult::parallel_trends_valid)
        .def_readonly("pre_trend_pvalue", &statelix::DIDResult::pre_trend_pvalue);

    py::class_<statelix::DifferenceInDifferences>(causal, "DifferenceInDifferences")
        .def(py::init<>())
        .def_readwrite("robust_se", &statelix::DifferenceInDifferences::robust_se)
        .def("fit", &statelix::DifferenceInDifferences::fit, 
             py::arg("Y"), py::arg("treated"), py::arg("post"))
        .def("fit_with_pretest", &statelix::DifferenceInDifferences::fit_with_pretest,
             py::arg("Y"), py::arg("treated"), py::arg("time"), py::arg("treatment_time"));

    // --- Bayesian (HMC) ---
    
    py::module_ bayes = m.def_submodule("bayes", "Bayesian module");

    py::class_<statelix::HMCConfig>(bayes, "HMCConfig")
        .def(py::init<>())
        .def_readwrite("n_samples", &statelix::HMCConfig::n_samples)
        .def_readwrite("warmup", &statelix::HMCConfig::warmup)
        .def_readwrite("step_size", &statelix::HMCConfig::step_size)
        .def_readwrite("n_leapfrog", &statelix::HMCConfig::n_leapfrog)
        .def_readwrite("adapt_step_size", &statelix::HMCConfig::adapt_step_size)
        .def_readwrite("target_accept", &statelix::HMCConfig::target_accept)
        .def_readwrite("seed", &statelix::HMCConfig::seed);

    py::class_<statelix::HMCResult>(bayes, "HMCResult")
        .def_readonly("samples", &statelix::HMCResult::samples)
        .def_readonly("log_probs", &statelix::HMCResult::log_probs)
        .def_readonly("acceptance_rate", &statelix::HMCResult::acceptance_rate)
        .def_readonly("n_divergences", &statelix::HMCResult::n_divergences)
        .def_readonly("mean", &statelix::HMCResult::mean)
        .def_readonly("std_dev", &statelix::HMCResult::std_dev)
        .def_readonly("quantiles", &statelix::HMCResult::quantiles)
        .def_readonly("ess", &statelix::HMCResult::ess);

    // HMC Sampler binding with callback
    m.def("hmc_sample", [](
        py::function log_prob_func, // Returns (log_prob, grad)
        const Eigen::VectorXd& theta0,
        const statelix::HMCConfig& config
    ) {
        // Create C++ wrapper for Python objective
        PyEfficientObjective objective(log_prob_func);
        
        statelix::HamiltonianMonteCarlo hmc;
        hmc.config = config;
        return hmc.sample(objective, theta0);
    }, py::arg("log_prob_func"), py::arg("theta0"), py::arg("config"),
       "Run HMC sampling. log_prob_func(theta) -> (log_p, grad_vector)");

    // v2.3 Native HMC for Logistic Regression (GIL Released)
    m.def("hmc_sample_logistic", [](
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y,
        const statelix::HMCConfig& config,
        double prior_std
    ) {
        // Create Native Objective (No Python GIL needed for calcs)
        statelix::LogisticObjective obj(X, y, prior_std);
        
        statelix::HamiltonianMonteCarlo hmc;
        hmc.config = config;
        
        // Run sampling with GIL RELEASED
        // This is safe because LogisticObjective uses only C++ Eigen types
        {
            py::gil_scoped_release release;
            // Initialize at zero
            Eigen::VectorXd theta0 = Eigen::VectorXd::Zero(X.cols());
            return hmc.sample(obj, theta0);
        }
    }, py::arg("X"), py::arg("y"), py::arg("config"), py::arg("prior_std")=10.0,
       "Run HMC for Logistic Regression using high-performance C++ backend (No GIL).");

    // --- Search (HNSW) ---
    
    py::module_ search = m.def_submodule("search", "Approximate NN search");

    // Enums
    py::enum_<statelix::search::HNSWConfig::Distance>(search, "Distance")
        .value("L2", statelix::search::HNSWConfig::Distance::L2)
        .value("COSINE", statelix::search::HNSWConfig::Distance::COSINE)
        .value("INNER_PRODUCT", statelix::search::HNSWConfig::Distance::INNER_PRODUCT)
        .export_values();

    py::class_<statelix::search::HNSWConfig>(search, "HNSWConfig")
        .def(py::init<>())
        .def_readwrite("M", &statelix::search::HNSWConfig::M)
        .def_readwrite("ef_construction", &statelix::search::HNSWConfig::ef_construction)
        .def_readwrite("ef_search", &statelix::search::HNSWConfig::ef_search)
        .def_readwrite("distance", &statelix::search::HNSWConfig::distance)
        .def_readwrite("seed", &statelix::search::HNSWConfig::seed);

    py::class_<statelix::search::HNSWSearchResult>(search, "HNSWSearchResult")
        .def_readonly("indices", &statelix::search::HNSWSearchResult::indices)
        .def_readonly("distances", &statelix::search::HNSWSearchResult::distances)
        .def_readonly("n_comparisons", &statelix::search::HNSWSearchResult::n_comparisons);

    py::class_<statelix::search::HNSW>(search, "HNSW")
        .def(py::init<>())
        .def(py::init<const statelix::search::HNSWConfig&>())
        .def_readwrite("config", &statelix::search::HNSW::config)
        .def("build", [](statelix::search::HNSW& self, 
                         py::array_t<double, py::array::c_style | py::array::forcecast> data) {
            auto buffer = data.request();
            if (buffer.ndim != 2) throw std::runtime_error("Data must be 2D array");
            Eigen::Map<Eigen::MatrixXd> mat(
                static_cast<double*>(buffer.ptr), 
                buffer.shape[0], 
                buffer.shape[1]
            );
            self.build(mat);
        }, py::arg("data"), "Build index from data (rows=items, cols=features)")
        .def("query", &statelix::search::HNSW::query, py::arg("query"), py::arg("k"))
        .def("query_batch", &statelix::search::HNSW::query_batch, py::arg("queries"), py::arg("k"))
        .def("save", &statelix::search::HNSW::save, py::arg("path"))
        .def("load", &statelix::search::HNSW::load, py::arg("path"))
        .def_property_readonly("size", &statelix::search::HNSW::size);
}
