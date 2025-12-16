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

// v2.3 Econometrics Headers
#include "../panel/panel.h"
#include "../time_series/var.h"
#include "../time_series/garch.h"
#include "../time_series/cointegration.h"
#include "../stats/tests.h"
#include "../causal/rdd.h"
#include "../causal/psm.h"
#include "../panel/dynamic_panel.h"
#include "../glm/discrete_choice.h"
#include "../glm/zero_inflated.h"

// v2.3 Phase 4: Advanced Models
#include "../causal/synthetic_control.h"
#include "../panel/dynamic_panel.h"
#include "../spatial/spatial.h"
#include "../stats/resampling.h"
#include "../stats/quantile_regression.h"

// Advanced Poisson (God Tier)
#include "../linear_model/poisson_full.h"

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
    
    // Advanced Poisson (Full version)
    py::class_<statelix::poisson_detail::PoissonResult>(m, "PoissonResultFull")
        .def_readonly("coef", &statelix::poisson_detail::PoissonResult::coef)
        .def_readonly("intercept", &statelix::poisson_detail::PoissonResult::intercept)
        .def_readonly("std_errors", &statelix::poisson_detail::PoissonResult::std_errors)
        .def_readonly("z_values", &statelix::poisson_detail::PoissonResult::z_values)
        .def_readonly("p_values", &statelix::poisson_detail::PoissonResult::p_values)
        .def_readonly("conf_int", &statelix::poisson_detail::PoissonResult::conf_int)
        .def_readonly("fitted_values", &statelix::poisson_detail::PoissonResult::fitted_values)
        .def_readonly("linear_predictors", &statelix::poisson_detail::PoissonResult::linear_predictors)
        .def_readonly("deviance_residuals", &statelix::poisson_detail::PoissonResult::deviance_residuals)
        .def_readonly("log_likelihood", &statelix::poisson_detail::PoissonResult::log_likelihood)
        .def_readonly("deviance", &statelix::poisson_detail::PoissonResult::deviance)
        .def_readonly("aic", &statelix::poisson_detail::PoissonResult::aic)
        .def_readonly("bic", &statelix::poisson_detail::PoissonResult::bic)
        .def_readonly("pseudo_r_squared", &statelix::poisson_detail::PoissonResult::pseudo_r_squared)
        .def_readonly("vcov", &statelix::poisson_detail::PoissonResult::vcov)
        .def_readonly("iterations", &statelix::poisson_detail::PoissonResult::iterations)
        .def_readonly("converged", &statelix::poisson_detail::PoissonResult::converged);

    m.def("fit_poisson_full", &statelix::poisson_detail::fit_poisson,
          py::arg("X"), py::arg("y"), py::arg("fit_intercept")=true, py::arg("offset")=Eigen::VectorXd(),
          py::arg("max_iter")=50, py::arg("tol")=1e-8, py::arg("conf_level")=0.95);
    
    m.def("predict_poisson_full", &statelix::poisson_detail::predict_poisson,
          py::arg("result"), py::arg("X_new"), py::arg("fit_intercept")=true, py::arg("offset")=Eigen::VectorXd(),
          py::arg("return_log")=false);
    
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
    
    // Updated Kalman Result
    py::class_<statelix::KalmanResult>(m, "KalmanResult")
        .def_readonly("states", &statelix::KalmanResult::states)
        .def_readonly("cov_states", &statelix::KalmanResult::cov_states)
        .def_readonly("smoothed_states", &statelix::KalmanResult::smoothed_states)
        .def_readonly("cov_smoothed", &statelix::KalmanResult::cov_smoothed)
        .def_readonly("log_likelihood", &statelix::KalmanResult::log_likelihood);

    py::class_<statelix::KalmanFilter>(m, "KalmanFilter").def(py::init<int, int>(), py::arg("state_dim"), py::arg("measure_dim"))
        .def_readwrite("F", &statelix::KalmanFilter::F)
        .def_readwrite("H", &statelix::KalmanFilter::H)
        .def_readwrite("Q", &statelix::KalmanFilter::Q)
        .def_readwrite("R", &statelix::KalmanFilter::R)
        .def_readwrite("P", &statelix::KalmanFilter::P)
        .def_readwrite("x", &statelix::KalmanFilter::x)
        .def("predict", &statelix::KalmanFilter::predict)
        .def("update", &statelix::KalmanFilter::update)
        .def("reset", &statelix::KalmanFilter::reset, py::arg("init_x"), py::arg("init_P"))
        .def("filter", &statelix::KalmanFilter::filter);

    // --- ML ---
    py::class_<statelix::GradientBoostingRegressor>(m, "GradientBoostingRegressor").def(py::init<>()).def("fit", &statelix::GradientBoostingRegressor::fit).def("predict", &statelix::GradientBoostingRegressor::predict);
    
    py::enum_<statelix::FMTask>(m, "FMTask").value("Regression", statelix::FMTask::Regression).value("Classification", statelix::FMTask::Classification).export_values();
    py::class_<statelix::FactorizationMachine>(m, "FactorizationMachine")
        .def(py::init<>())
        .def_readwrite("n_factors", &statelix::FactorizationMachine::n_factors)
        .def("fit", &statelix::FactorizationMachine::fit)
        .def("predict", &statelix::FactorizationMachine::predict);

    // --- Survival ---
    py::class_<statelix::CoxResult>(m, "CoxResult")
        .def_readonly("coef", &statelix::CoxResult::coef)
        .def_readonly("std_error", &statelix::CoxResult::std_error)
        .def_readonly("z_score", &statelix::CoxResult::z_score)
        .def_readonly("p_values", &statelix::CoxResult::p_values)
        .def_readonly("covariance", &statelix::CoxResult::covariance)
        .def_readonly("log_likelihood", &statelix::CoxResult::log_likelihood)
        .def_readonly("converged", &statelix::CoxResult::converged)
        .def_readonly("iterations", &statelix::CoxResult::iterations);

    py::class_<statelix::CoxPH>(m, "CoxPH")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::CoxPH::max_iter)
        .def_readwrite("tol", &statelix::CoxPH::tol)
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

    py::class_<statelix::TWFEResult>(causal, "TWFEResult")
        .def_readonly("delta", &statelix::TWFEResult::delta)
        .def_readonly("delta_std_error", &statelix::TWFEResult::delta_std_error)
        .def_readonly("p_value", &statelix::TWFEResult::p_value)
        .def_readonly("unit_fe", &statelix::TWFEResult::unit_fe)
        .def_readonly("time_fe", &statelix::TWFEResult::time_fe)
        .def_readonly("has_staggered", &statelix::TWFEResult::has_staggered_adoption)
        .def_readonly("twfe_biased_warning", &statelix::TWFEResult::twfe_potentially_biased);

    py::class_<statelix::TwoWayFixedEffects>(causal, "TwoWayFixedEffects")
        .def(py::init<>())
        .def_readwrite("cluster_se", &statelix::TwoWayFixedEffects::cluster_se)
        .def("fit", &statelix::TwoWayFixedEffects::fit,
             py::arg("Y"), py::arg("D"), py::arg("unit_id"), py::arg("time_id"), 
             py::arg("X_controls")=Eigen::MatrixXd());

    // --- Bayesian (HMC) ---
    
    // Ray: HMC Enabled
    // v2.3 Bayesian Module
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
        .def_readonly("ess", &statelix::HMCResult::ess)
        .def_readonly("r_hat", &statelix::HMCResult::r_hat);

    // HMC Sampler binding with callback
    bayes.def("hmc_sample", [](
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
    bayes.def("hmc_sample_logistic", [](
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

    // =========================================================================
    // v2.3 Econometrics API
    // =========================================================================

    // --- Panel Data ---
    
    py::module_ panel = m.def_submodule("panel", "Panel data analysis module");

    py::class_<statelix::PanelFEResult>(panel, "PanelFEResult")
        .def_readonly("coef", &statelix::PanelFEResult::coef)
        .def_readonly("std_errors", &statelix::PanelFEResult::std_errors)
        .def_readonly("t_values", &statelix::PanelFEResult::t_values)
        .def_readonly("p_values", &statelix::PanelFEResult::p_values)
        .def_readonly("conf_lower", &statelix::PanelFEResult::conf_lower)
        .def_readonly("conf_upper", &statelix::PanelFEResult::conf_upper)
        .def_readonly("unit_fe", &statelix::PanelFEResult::unit_fe)
        .def_readonly("time_fe", &statelix::PanelFEResult::time_fe)
        .def_readonly("sigma2_e", &statelix::PanelFEResult::sigma2_e)
        .def_readonly("sigma2_u", &statelix::PanelFEResult::sigma2_u)
        .def_readonly("r_squared_within", &statelix::PanelFEResult::r_squared_within)
        .def_readonly("r_squared_between", &statelix::PanelFEResult::r_squared_between)
        .def_readonly("r_squared_overall", &statelix::PanelFEResult::r_squared_overall)
        .def_readonly("vcov", &statelix::PanelFEResult::vcov)
        .def_readonly("n_obs", &statelix::PanelFEResult::n_obs)
        .def_readonly("n_units", &statelix::PanelFEResult::n_units)
        .def_readonly("n_periods", &statelix::PanelFEResult::n_periods)
        .def_readonly("f_stat", &statelix::PanelFEResult::f_stat)
        .def_readonly("f_pvalue", &statelix::PanelFEResult::f_pvalue)
        .def_readonly("residuals", &statelix::PanelFEResult::residuals)
        .def_readonly("fitted_values", &statelix::PanelFEResult::fitted_values);

    // Dynamic Panel (v2.3)
    py::class_<statelix::panel::DynamicPanelResult>(panel, "DynamicPanelResult")
        .def_readonly("coefficients", &statelix::panel::DynamicPanelResult::coefficients);
        // ... (other fields)

    py::class_<statelix::panel::DynamicPanelGMM>(panel, "DynamicPanelGMM")
        .def(py::init<>())
        .def_readwrite("max_lags", &statelix::panel::DynamicPanelGMM::max_lags)
        .def("estimate", &statelix::panel::DynamicPanelGMM::estimate,
             py::arg("y"), py::arg("X"), py::arg("ids"), py::arg("time"));

    py::class_<statelix::PanelFixedEffects>(panel, "FixedEffects")
        .def(py::init<>())
        .def_readwrite("conf_level", &statelix::PanelFixedEffects::conf_level)
        .def_readwrite("cluster_se", &statelix::PanelFixedEffects::cluster_se)
        .def_readwrite("two_way", &statelix::PanelFixedEffects::two_way)
        .def("fit", &statelix::PanelFixedEffects::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    py::class_<statelix::PanelREResult>(panel, "PanelREResult")
        .def_readonly("coef", &statelix::PanelREResult::coef)
        .def_readonly("std_errors", &statelix::PanelREResult::std_errors)
        .def_readonly("intercept", &statelix::PanelREResult::intercept)
        .def_readonly("theta", &statelix::PanelREResult::theta)
        .def_readonly("rho", &statelix::PanelREResult::rho)
        .def_readonly("sigma2_e", &statelix::PanelREResult::sigma2_e)
        .def_readonly("sigma2_u", &statelix::PanelREResult::sigma2_u)
        .def_readonly("r_squared_within", &statelix::PanelREResult::r_squared_within)
        .def_readonly("r_squared_overall", &statelix::PanelREResult::r_squared_overall);

    py::class_<statelix::PanelRandomEffects>(panel, "RandomEffects")
        .def(py::init<>())
        .def_readwrite("conf_level", &statelix::PanelRandomEffects::conf_level)
        .def("fit", &statelix::PanelRandomEffects::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    py::class_<statelix::HausmanTestResult>(panel, "HausmanTestResult")
        .def_readonly("chi2_stat", &statelix::HausmanTestResult::chi2_stat)
        .def_readonly("df", &statelix::HausmanTestResult::df)
        .def_readonly("p_value", &statelix::HausmanTestResult::p_value)
        .def_readonly("prefer_fe", &statelix::HausmanTestResult::prefer_fe)
        .def_readonly("recommendation", &statelix::HausmanTestResult::recommendation)
        .def_readonly("warning", &statelix::HausmanTestResult::warning);

    py::class_<statelix::HausmanTest>(panel, "HausmanTest")
        .def_static("test", py::overload_cast<const statelix::PanelFEResult&, const statelix::PanelREResult&>(&statelix::HausmanTest::test))
        .def_static("test_from_data", py::overload_cast<const Eigen::VectorXd&, const Eigen::MatrixXd&, const Eigen::VectorXi&, const Eigen::VectorXi&>(&statelix::HausmanTest::test),
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    py::class_<statelix::PanelFDResult>(panel, "PanelFDResult")
        .def_readonly("coef", &statelix::PanelFDResult::coef)
        .def_readonly("std_errors", &statelix::PanelFDResult::std_errors)
        .def_readonly("r_squared", &statelix::PanelFDResult::r_squared);

    py::class_<statelix::PanelFirstDifference>(panel, "FirstDifference")
        .def(py::init<>())
        .def_readwrite("cluster_se", &statelix::PanelFirstDifference::cluster_se)
        .def("fit", &statelix::PanelFirstDifference::fit,
             py::arg("Y"), py::arg("X"), py::arg("unit_id"), py::arg("time_id"));

    // --- VAR ---
    
    py::module_ ts = m.def_submodule("time_series", "Time series econometrics");

    py::class_<statelix::VARResult>(ts, "VARResult")
        .def_readonly("coef", &statelix::VARResult::coef)
        .def_readonly("intercept", &statelix::VARResult::intercept)
        .def_readonly("residuals", &statelix::VARResult::residuals)
        .def_readonly("sigma", &statelix::VARResult::sigma)
        .def_readonly("n_obs", &statelix::VARResult::n_obs)
        .def_readonly("n_vars", &statelix::VARResult::n_vars)
        .def_readonly("lag_order", &statelix::VARResult::lag_order)
        .def_readonly("log_likelihood", &statelix::VARResult::log_likelihood)
        .def_readonly("aic", &statelix::VARResult::aic)
        .def_readonly("bic", &statelix::VARResult::bic)
        .def_readonly("is_stable", &statelix::VARResult::is_stable)
        .def_readonly("eigenvalues_mod", &statelix::VARResult::eigenvalues_mod)
        .def_readonly("companion", &statelix::VARResult::companion);

    py::class_<statelix::FullIRFResult>(ts, "FullIRFResult")
        .def_readonly("irf", &statelix::FullIRFResult::irf)
        .def_readonly("irf_lower", &statelix::FullIRFResult::irf_lower)
        .def_readonly("irf_upper", &statelix::FullIRFResult::irf_upper)
        .def_readonly("horizon", &statelix::FullIRFResult::horizon)
        .def_readonly("n_vars", &statelix::FullIRFResult::n_vars);

    py::class_<statelix::FEVDResult>(ts, "FEVDResult")
        .def_readonly("fevd", &statelix::FEVDResult::fevd)
        .def_readonly("horizon", &statelix::FEVDResult::horizon);

    py::class_<statelix::GrangerResult>(ts, "GrangerResult")
        .def_readonly("f_stat", &statelix::GrangerResult::f_stat)
        .def_readonly("p_value", &statelix::GrangerResult::p_value)
        .def_readonly("causes", &statelix::GrangerResult::causes)
        .def_readonly("cause_var", &statelix::GrangerResult::cause_var)
        .def_readonly("effect_var", &statelix::GrangerResult::effect_var);

    py::class_<statelix::LagSelectionResult>(ts, "LagSelectionResult")
        .def_readonly("aic", &statelix::LagSelectionResult::aic)
        .def_readonly("bic", &statelix::LagSelectionResult::bic)
        .def_readonly("best_aic", &statelix::LagSelectionResult::best_aic)
        .def_readonly("best_bic", &statelix::LagSelectionResult::best_bic);

    py::class_<statelix::VectorAutoregression>(ts, "VAR")
        .def(py::init<int>(), py::arg("p")=1)
        .def_readwrite("lag_order", &statelix::VectorAutoregression::lag_order)
        .def_readwrite("include_intercept", &statelix::VectorAutoregression::include_intercept)
        .def_readwrite("conf_level", &statelix::VectorAutoregression::conf_level)
        .def_readwrite("bootstrap_reps", &statelix::VectorAutoregression::bootstrap_reps)
        .def("fit", &statelix::VectorAutoregression::fit, py::arg("Y"))
        .def("irf", &statelix::VectorAutoregression::irf, 
             py::arg("result"), py::arg("horizon"), py::arg("orthogonalized")=true)
        .def("fevd", &statelix::VectorAutoregression::fevd,
             py::arg("result"), py::arg("horizon"))
        .def("granger_causality", &statelix::VectorAutoregression::granger_causality,
             py::arg("result"), py::arg("Y"), py::arg("cause"), py::arg("effect"))
        .def("select_lag", &statelix::VectorAutoregression::select_lag,
             py::arg("Y"), py::arg("max_lag"))
        .def("forecast", &statelix::VectorAutoregression::forecast,
             py::arg("result"), py::arg("Y"), py::arg("steps"));

    // --- GARCH ---

    py::enum_<statelix::GARCHType>(ts, "GARCHType")
        .value("GARCH", statelix::GARCHType::GARCH)
        .value("EGARCH", statelix::GARCHType::EGARCH)
        .value("GJR", statelix::GARCHType::GJR)
        .value("IGARCH", statelix::GARCHType::IGARCH)
        .export_values();

    py::enum_<statelix::GARCHDist>(ts, "GARCHDist")
        .value("NORMAL", statelix::GARCHDist::NORMAL)
        .value("STUDENT_T", statelix::GARCHDist::STUDENT_T)
        .export_values();

    py::class_<statelix::GARCHResult>(ts, "GARCHResult")
        .def_readonly("p", &statelix::GARCHResult::p)
        .def_readonly("q", &statelix::GARCHResult::q)
        .def_readonly("mu", &statelix::GARCHResult::mu)
        .def_readonly("omega", &statelix::GARCHResult::omega)
        .def_readonly("alpha", &statelix::GARCHResult::alpha)
        .def_readonly("beta", &statelix::GARCHResult::beta)
        .def_readonly("gamma", &statelix::GARCHResult::gamma)
        .def_readonly("nu", &statelix::GARCHResult::nu)
        .def_readonly("log_likelihood", &statelix::GARCHResult::log_likelihood)
        .def_readonly("aic", &statelix::GARCHResult::aic)
        .def_readonly("bic", &statelix::GARCHResult::bic)
        .def_readonly("persistence", &statelix::GARCHResult::persistence)
        .def_readonly("is_stationary", &statelix::GARCHResult::is_stationary)
        .def_readonly("unconditional_variance", &statelix::GARCHResult::unconditional_variance)
        .def_readonly("half_life", &statelix::GARCHResult::half_life)
        .def_readonly("conditional_variance", &statelix::GARCHResult::conditional_variance)
        .def_readonly("conditional_volatility", &statelix::GARCHResult::conditional_volatility)
        .def_readonly("standardized_residuals", &statelix::GARCHResult::standardized_residuals)
        .def_readonly("converged", &statelix::GARCHResult::converged);

    py::class_<statelix::GARCHForecast>(ts, "GARCHForecast")
        .def_readonly("variance", &statelix::GARCHForecast::variance)
        .def_readonly("volatility", &statelix::GARCHForecast::volatility)
        .def_readonly("horizon", &statelix::GARCHForecast::horizon);

    /*
    py::class_<statelix::GARCH>(ts, "GARCH")
        .def(py::init<int, int>(), py::arg("p")=1, py::arg("q")=1)
        .def_readwrite("type", &statelix::GARCH::type)
        .def_readwrite("dist", &statelix::GARCH::dist)
        .def_readwrite("max_iter", &statelix::GARCH::max_iter)
        .def("fit", &statelix::GARCH::fit, py::arg("returns"))
        .def("forecast", &statelix::GARCH::forecast, py::arg("result"), py::arg("horizon"))
        .def("news_impact_curve", &statelix::GARCH::news_impact_curve,
             py::arg("result"), py::arg("z_min")=-3.0, py::arg("z_max")=3.0, py::arg("n_points")=100);
    */

    // --- Econometric Tests ---
    
    py::module_ tests = m.def_submodule("tests", "Econometric specification tests");

    py::class_<statelix::tests::DurbinWatsonResult>(tests, "DurbinWatsonResult")
        .def_readonly("dw_statistic", &statelix::tests::DurbinWatsonResult::dw_statistic)
        .def_readonly("d_lower", &statelix::tests::DurbinWatsonResult::d_lower)
        .def_readonly("d_upper", &statelix::tests::DurbinWatsonResult::d_upper)
        .def_readonly("conclusion", &statelix::tests::DurbinWatsonResult::conclusion);

    tests.def("durbin_watson", &statelix::tests::durbin_watson,
              py::arg("residuals"), py::arg("n_obs"), py::arg("k"));

    py::class_<statelix::tests::BreuschGodfreyResult>(tests, "BreuschGodfreyResult")
        .def_readonly("lm_statistic", &statelix::tests::BreuschGodfreyResult::lm_statistic)
        .def_readonly("f_statistic", &statelix::tests::BreuschGodfreyResult::f_statistic)
        .def_readonly("lm_pvalue", &statelix::tests::BreuschGodfreyResult::lm_pvalue)
        .def_readonly("f_pvalue", &statelix::tests::BreuschGodfreyResult::f_pvalue)
        .def_readonly("serial_correlation", &statelix::tests::BreuschGodfreyResult::serial_correlation);

    tests.def("breusch_godfrey", &statelix::tests::breusch_godfrey,
              py::arg("X"), py::arg("residuals"), py::arg("order")=1);

    py::class_<statelix::tests::WhiteTestResult>(tests, "WhiteTestResult")
        .def_readonly("chi2_statistic", &statelix::tests::WhiteTestResult::chi2_statistic)
        .def_readonly("f_statistic", &statelix::tests::WhiteTestResult::f_statistic)
        .def_readonly("chi2_pvalue", &statelix::tests::WhiteTestResult::chi2_pvalue)
        .def_readonly("f_pvalue", &statelix::tests::WhiteTestResult::f_pvalue)
        .def_readonly("heteroskedastic", &statelix::tests::WhiteTestResult::heteroskedastic);

    tests.def("white_test", &statelix::tests::white_test,
              py::arg("X"), py::arg("residuals"));

    py::class_<statelix::tests::BreuschPaganResult>(tests, "BreuschPaganResult")
        .def_readonly("lm_statistic", &statelix::tests::BreuschPaganResult::lm_statistic)
        .def_readonly("lm_pvalue", &statelix::tests::BreuschPaganResult::lm_pvalue)
        .def_readonly("heteroskedastic", &statelix::tests::BreuschPaganResult::heteroskedastic);

    tests.def("breusch_pagan", &statelix::tests::breusch_pagan,
              py::arg("X"), py::arg("residuals"));

    py::class_<statelix::tests::RamseyResetResult>(tests, "RamseyResetResult")
        .def_readonly("f_statistic", &statelix::tests::RamseyResetResult::f_statistic)
        .def_readonly("p_value", &statelix::tests::RamseyResetResult::p_value)
        .def_readonly("misspecified", &statelix::tests::RamseyResetResult::misspecified);

    tests.def("ramsey_reset", &statelix::tests::ramsey_reset,
              py::arg("X"), py::arg("y"), py::arg("fitted"), py::arg("power")=3);

    py::class_<statelix::tests::ChowTestResult>(tests, "ChowTestResult")
        .def_readonly("f_statistic", &statelix::tests::ChowTestResult::f_statistic)
        .def_readonly("p_value", &statelix::tests::ChowTestResult::p_value)
        .def_readonly("structural_break", &statelix::tests::ChowTestResult::structural_break);

    tests.def("chow_test", &statelix::tests::chow_test,
              py::arg("X"), py::arg("y"), py::arg("break_point"));

    py::class_<statelix::tests::JarqueBeraResult>(tests, "JarqueBeraResult")
        .def_readonly("jb_statistic", &statelix::tests::JarqueBeraResult::jb_statistic)
        .def_readonly("skewness", &statelix::tests::JarqueBeraResult::skewness)
        .def_readonly("kurtosis", &statelix::tests::JarqueBeraResult::kurtosis)
        .def_readonly("p_value", &statelix::tests::JarqueBeraResult::p_value)
        .def_readonly("normal", &statelix::tests::JarqueBeraResult::normal);

    tests.def("jarque_bera", &statelix::tests::jarque_bera, py::arg("residuals"));

    py::class_<statelix::tests::ConditionNumberResult>(tests, "ConditionNumberResult")
        .def_readonly("condition_number", &statelix::tests::ConditionNumberResult::condition_number)
        .def_readonly("multicollinearity", &statelix::tests::ConditionNumberResult::multicollinearity)
        .def_readonly("severity", &statelix::tests::ConditionNumberResult::severity);

    tests.def("condition_number", &statelix::tests::condition_number, py::arg("X"));

    tests.def("robust_vcov", &statelix::tests::robust_vcov,
              py::arg("X"), py::arg("residuals"), py::arg("type")="HC1",
              "Compute heteroskedasticity-robust variance-covariance. Types: HC0, HC1, HC2, HC3");

    tests.def("newey_west_vcov", &statelix::tests::newey_west_vcov,
              py::arg("X"), py::arg("residuals"), py::arg("max_lag")=-1,
              "Compute Newey-West HAC standard errors");

    tests.def("cluster_robust_vcov", &statelix::tests::cluster_robust_vcov,
              py::arg("X"), py::arg("residuals"), py::arg("cluster_id"),
              "Compute cluster-robust standard errors");

    // =========================================================================
    // v2.3 Additional Econometrics
    // =========================================================================

    // --- Cointegration ---
    
    // Define submodules for Phase 3/4
    // causal module already defined above
    py::module_ glm = m.def_submodule("glm", "Generalized Linear Models");

    py::enum_<statelix::TrendType>(ts, "TrendType")
        .value("NONE", statelix::TrendType::NONE)
        .value("CONSTANT", statelix::TrendType::CONSTANT)
        .value("TREND", statelix::TrendType::TREND)
        .export_values();

    py::class_<statelix::ADFResult>(ts, "ADFResult")
        .def_readonly("adf_statistic", &statelix::ADFResult::adf_statistic)
        .def_readonly("p_value", &statelix::ADFResult::p_value)
        .def_readonly("lags_used", &statelix::ADFResult::lags_used)
        .def_readonly("cv_1pct", &statelix::ADFResult::cv_1pct)
        .def_readonly("cv_5pct", &statelix::ADFResult::cv_5pct)
        .def_readonly("has_unit_root", &statelix::ADFResult::has_unit_root)
        .def_readonly("conclusion", &statelix::ADFResult::conclusion);

    py::class_<statelix::ADF>(ts, "ADF")
        .def(py::init<>())
        .def_readwrite("max_lags", &statelix::ADF::max_lags)
        .def_readwrite("trend", &statelix::ADF::trend)
        .def_readwrite("auto_lag", &statelix::ADF::auto_lag)
        .def("test", &statelix::ADF::test, py::arg("y"));

    py::class_<statelix::KPSSResult>(ts, "KPSSResult")
        .def_readonly("kpss_statistic", &statelix::KPSSResult::kpss_statistic)
        .def_readonly("cv_5pct", &statelix::KPSSResult::cv_5pct)
        .def_readonly("is_stationary", &statelix::KPSSResult::is_stationary);

    py::class_<statelix::KPSS>(ts, "KPSS")
        .def(py::init<>())
        .def_readwrite("trend", &statelix::KPSS::trend)
        .def("test", &statelix::KPSS::test, py::arg("y"));

    py::class_<statelix::EngleGrangerResult>(ts, "EngleGrangerResult")
        .def_readonly("coef", &statelix::EngleGrangerResult::coef)
        .def_readonly("adf_statistic", &statelix::EngleGrangerResult::adf_statistic)
        .def_readonly("p_value", &statelix::EngleGrangerResult::p_value)
        .def_readonly("cointegrated", &statelix::EngleGrangerResult::cointegrated);

    py::class_<statelix::EngleGranger>(ts, "EngleGranger")
        .def(py::init<>())
        .def("test", &statelix::EngleGranger::test, py::arg("y"), py::arg("x"));

    py::class_<statelix::JohansenResult>(ts, "JohansenResult")
        .def_readonly("eigenvalues", &statelix::JohansenResult::eigenvalues)
        .def_readonly("eigenvectors", &statelix::JohansenResult::eigenvectors)
        .def_readonly("trace_stats", &statelix::JohansenResult::trace_stats)
        .def_readonly("trace_cv_5pct", &statelix::JohansenResult::trace_cv_5pct)
        .def_readonly("trace_rank", &statelix::JohansenResult::trace_rank)
        .def_readonly("max_rank", &statelix::JohansenResult::max_rank)
        .def_readonly("recommended_rank", &statelix::JohansenResult::recommended_rank);

    py::class_<statelix::Johansen>(ts, "Johansen")
        .def(py::init<>())
        .def_readwrite("lag_order", &statelix::Johansen::lag_order)
        .def_readwrite("trend", &statelix::Johansen::trend)
        .def("test", &statelix::Johansen::test, py::arg("Y"))
        .def("estimate_vecm", &statelix::Johansen::estimate_vecm, py::arg("Y"), py::arg("rank"));

    // --- RDD ---

    py::enum_<statelix::RDDKernel>(causal, "RDDKernel")
        .value("TRIANGULAR", statelix::RDDKernel::TRIANGULAR)
        .value("UNIFORM", statelix::RDDKernel::UNIFORM)
        .value("EPANECHNIKOV", statelix::RDDKernel::EPANECHNIKOV)
        .export_values();

    py::class_<statelix::RDDResult>(causal, "RDDResult")
        .def_readonly("tau", &statelix::RDDResult::tau)
        .def_readonly("tau_se", &statelix::RDDResult::tau_se)
        .def_readonly("tau_pvalue", &statelix::RDDResult::tau_pvalue)
        .def_readonly("tau_ci_lower", &statelix::RDDResult::tau_ci_lower)
        .def_readonly("tau_ci_upper", &statelix::RDDResult::tau_ci_upper)
        .def_readonly("tau_bc", &statelix::RDDResult::tau_bc)
        .def_readonly("bandwidth", &statelix::RDDResult::bandwidth)
        .def_readonly("n_left", &statelix::RDDResult::n_left)
        .def_readonly("n_right", &statelix::RDDResult::n_right)
        .def_readonly("is_fuzzy", &statelix::RDDResult::is_fuzzy);

    py::class_<statelix::SharpRDD>(causal, "SharpRDD")
        .def(py::init<>())
        .def_readwrite("bandwidth", &statelix::SharpRDD::bandwidth)
        .def_readwrite("polynomial_order", &statelix::SharpRDD::polynomial_order)
        .def_readwrite("kernel", &statelix::SharpRDD::kernel)
        .def_readwrite("bias_correction", &statelix::SharpRDD::bias_correction)
        .def("fit", &statelix::SharpRDD::fit, py::arg("Y"), py::arg("X"), py::arg("cutoff"), 
             py::arg("cluster_id")=Eigen::VectorXi())
        .def("compute_ik_bandwidth", &statelix::SharpRDD::compute_ik_bandwidth,
             py::arg("Y"), py::arg("X_centered"));

    py::class_<statelix::FuzzyRDD>(causal, "FuzzyRDD")
        .def(py::init<>())
        .def_readwrite("bandwidth", &statelix::FuzzyRDD::bandwidth)
        .def("fit", &statelix::FuzzyRDD::fit, py::arg("Y"), py::arg("D"), py::arg("X"), py::arg("cutoff"));

    // --- PSM ---

    py::enum_<statelix::MatchingMethod>(causal, "MatchingMethod")
        .value("NEAREST_NEIGHBOR", statelix::MatchingMethod::NEAREST_NEIGHBOR)
        .value("CALIPER", statelix::MatchingMethod::CALIPER)
        .value("KERNEL", statelix::MatchingMethod::KERNEL)
        .export_values();

    py::class_<statelix::PropensityScoreResult>(causal, "PropensityScoreResult")
        .def_readonly("scores", &statelix::PropensityScoreResult::scores)
        .def_readonly("coef", &statelix::PropensityScoreResult::coef)
        .def_readonly("n_treated", &statelix::PropensityScoreResult::n_treated)
        .def_readonly("n_control", &statelix::PropensityScoreResult::n_control)
        .def_readonly("overlap_min", &statelix::PropensityScoreResult::overlap_min)
        .def_readonly("overlap_max", &statelix::PropensityScoreResult::overlap_max);

    py::class_<statelix::MatchingResult>(causal, "MatchingResult")
        .def_readonly("att", &statelix::MatchingResult::att)
        .def_readonly("att_se", &statelix::MatchingResult::att_se)
        .def_readonly("att_pvalue", &statelix::MatchingResult::att_pvalue)
        .def_readonly("att_ci_lower", &statelix::MatchingResult::att_ci_lower)
        .def_readonly("att_ci_upper", &statelix::MatchingResult::att_ci_upper)
        .def_readonly("n_matched_treated", &statelix::MatchingResult::n_matched_treated)
        .def_readonly("mean_std_diff_before", &statelix::MatchingResult::mean_std_diff_before)
        .def_readonly("mean_std_diff_after", &statelix::MatchingResult::mean_std_diff_after);

    py::class_<statelix::IPWResult>(causal, "IPWResult")
        .def_readonly("att", &statelix::IPWResult::att)
        .def_readonly("att_se", &statelix::IPWResult::att_se)
        .def_readonly("ate", &statelix::IPWResult::ate)
        .def_readonly("n_trimmed", &statelix::IPWResult::n_trimmed);

    py::class_<statelix::AIPWResult>(causal, "AIPWResult")
        .def_readonly("att", &statelix::AIPWResult::att)
        .def_readonly("att_se", &statelix::AIPWResult::att_se)
        .def_readonly("efficiency_gain", &statelix::AIPWResult::efficiency_gain);

    py::class_<statelix::PropensityScoreMatching>(causal, "PropensityScoreMatching")
        .def(py::init<>())
        .def_readwrite("method", &statelix::PropensityScoreMatching::method)
        .def_readwrite("n_neighbors", &statelix::PropensityScoreMatching::n_neighbors)
        .def_readwrite("with_replacement", &statelix::PropensityScoreMatching::with_replacement)
        .def_readwrite("caliper", &statelix::PropensityScoreMatching::caliper)
        .def("estimate_propensity", &statelix::PropensityScoreMatching::estimate_propensity,
             py::arg("D"), py::arg("X"))
        .def("match", &statelix::PropensityScoreMatching::match,
             py::arg("Y"), py::arg("D"), py::arg("X"), py::arg("ps"))
        .def("ipw", &statelix::PropensityScoreMatching::ipw, py::arg("Y"), py::arg("D"), py::arg("ps"))
        .def("aipw", &statelix::PropensityScoreMatching::aipw, 
             py::arg("Y"), py::arg("D"), py::arg("X"), py::arg("ps"));

    // -------------------------------------------------------------------------
    // Panel Data API
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // Panel Data API
    // -------------------------------------------------------------------------
    py::module_ panel_mod = m.def_submodule("panel", "Panel data models");

    py::class_<statelix::panel::DynamicPanelResult>(panel_mod, "DynamicPanelResult")
        .def_readonly("coefficients", &statelix::panel::DynamicPanelResult::coefficients)
        .def_readonly("std_errors", &statelix::panel::DynamicPanelResult::std_errors)
        .def_readonly("sargan_test", &statelix::panel::DynamicPanelResult::sargan_test)
        .def_readonly("n_obs", &statelix::panel::DynamicPanelResult::n_obs);

    py::class_<statelix::panel::DynamicPanelGMM>(panel_mod, "DynamicPanelGMM")
        .def(py::init<>())
        .def_readwrite("two_step", &statelix::panel::DynamicPanelGMM::two_step)
        .def("estimate", &statelix::panel::DynamicPanelGMM::estimate,
             py::arg("y"), py::arg("X"), py::arg("ids"), py::arg("time"));

    // --- Discrete Choice ---

    py::class_<statelix::OrderedLogitResult>(glm, "OrderedLogitResult")
        .def_readonly("coef", &statelix::OrderedLogitResult::coef)
        .def_readonly("coef_se", &statelix::OrderedLogitResult::coef_se)
        .def_readonly("thresholds", &statelix::OrderedLogitResult::thresholds)
        .def_readonly("log_likelihood", &statelix::OrderedLogitResult::log_likelihood)
        .def_readonly("aic", &statelix::OrderedLogitResult::aic)
        .def_readonly("pseudo_r_squared", &statelix::OrderedLogitResult::pseudo_r_squared)
        .def_readonly("predicted_probs", &statelix::OrderedLogitResult::predicted_probs)
        .def_readonly("predicted_class", &statelix::OrderedLogitResult::predicted_class)
        .def_readonly("converged", &statelix::OrderedLogitResult::converged);

    py::class_<statelix::OrderedLogit>(glm, "OrderedLogit")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::OrderedLogit::max_iter)
        .def_readwrite("use_probit", &statelix::OrderedLogit::use_probit)
        .def("fit", &statelix::OrderedLogit::fit, py::arg("y"), py::arg("X"))
        .def("marginal_effects", &statelix::OrderedLogit::marginal_effects, py::arg("X"), py::arg("result"));

    py::class_<statelix::MultinomialLogitResult>(glm, "MultinomialLogitResult")
        .def_readonly("coef", &statelix::MultinomialLogitResult::coef)
        .def_readonly("coef_se", &statelix::MultinomialLogitResult::coef_se)
        .def_readonly("log_likelihood", &statelix::MultinomialLogitResult::log_likelihood)
        .def_readonly("aic", &statelix::MultinomialLogitResult::aic)
        .def_readonly("pseudo_r_squared", &statelix::MultinomialLogitResult::pseudo_r_squared)
        .def_readonly("predicted_probs", &statelix::MultinomialLogitResult::predicted_probs)
        .def_readonly("hit_rate", &statelix::MultinomialLogitResult::hit_rate)
        .def_readonly("converged", &statelix::MultinomialLogitResult::converged);

    py::class_<statelix::MultinomialLogit>(glm, "MultinomialLogit")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::MultinomialLogit::max_iter)
        .def("fit", &statelix::MultinomialLogit::fit, py::arg("y"), py::arg("X"))
        .def("predict", &statelix::MultinomialLogit::predict, py::arg("X"), py::arg("result"));

    // --- Zero-Inflated Models ---

    py::class_<statelix::ZIPResult>(glm, "ZIPResult")
        .def_readonly("count_coef", &statelix::ZIPResult::count_coef)
        .def_readonly("count_intercept", &statelix::ZIPResult::count_intercept)
        .def_readonly("inflate_coef", &statelix::ZIPResult::inflate_coef)
        .def_readonly("inflate_intercept", &statelix::ZIPResult::inflate_intercept)
        .def_readonly("log_likelihood", &statelix::ZIPResult::log_likelihood)
        .def_readonly("aic", &statelix::ZIPResult::aic)
        .def_readonly("vuong_stat", &statelix::ZIPResult::vuong_stat)
        .def_readonly("fitted_mean", &statelix::ZIPResult::fitted_mean)
        .def_readonly("fitted_zero_prob", &statelix::ZIPResult::fitted_zero_prob)
        .def_readonly("zero_pct", &statelix::ZIPResult::zero_pct)
        .def_readonly("converged", &statelix::ZIPResult::converged);

    py::class_<statelix::ZeroInflatedPoisson>(glm, "ZeroInflatedPoisson")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::ZeroInflatedPoisson::max_iter)
        .def("fit", &statelix::ZeroInflatedPoisson::fit, py::arg("y"), py::arg("X"), 
             py::arg("Z")=Eigen::MatrixXd());

    py::class_<statelix::ZINBResult>(glm, "ZINBResult")
        .def_readonly("count_coef", &statelix::ZINBResult::count_coef)
        .def_readonly("inflate_coef", &statelix::ZINBResult::inflate_coef)
        .def_readonly("alpha", &statelix::ZINBResult::alpha)
        .def_readonly("log_likelihood", &statelix::ZINBResult::log_likelihood)
        .def_readonly("zero_pct", &statelix::ZINBResult::zero_pct);

    py::class_<statelix::ZeroInflatedNegBin>(glm, "ZeroInflatedNegBin")
        .def(py::init<>())
        .def("fit", &statelix::ZeroInflatedNegBin::fit, py::arg("y"), py::arg("X"),
             py::arg("Z")=Eigen::MatrixXd());

    py::class_<statelix::HurdleResult>(glm, "HurdleResult")
        .def_readonly("zero_coef", &statelix::HurdleResult::zero_coef)
        .def_readonly("count_coef", &statelix::HurdleResult::count_coef)
        .def_readonly("log_likelihood", &statelix::HurdleResult::log_likelihood);

    py::class_<statelix::HurdlePoisson>(glm, "HurdlePoisson")
        .def(py::init<>())
        .def("fit", &statelix::HurdlePoisson::fit, py::arg("y"), py::arg("X"));

    // =========================================================================
    // v2.3 Phase 4: Advanced Econometric Models
    // =========================================================================

    // --- Synthetic Control ---

    py::class_<statelix::SyntheticControlResult>(causal, "SyntheticControlResult")
        .def_readonly("weights", &statelix::SyntheticControlResult::weights)
        .def_readonly("gaps", &statelix::SyntheticControlResult::gaps)
        .def_readonly("y_synthetic", &statelix::SyntheticControlResult::y_synthetic)
        .def_readonly("att", &statelix::SyntheticControlResult::att)
        .def_readonly("pre_rmspe", &statelix::SyntheticControlResult::pre_rmspe)
        .def_readonly("post_rmspe", &statelix::SyntheticControlResult::post_rmspe)
        .def_readonly("rmspe_ratio", &statelix::SyntheticControlResult::rmspe_ratio)
        .def_readonly("pre_treatment_fit", &statelix::SyntheticControlResult::pre_treatment_fit)
        .def_readonly("selected_donors", &statelix::SyntheticControlResult::selected_donors)
        .def_readonly("predictor_balance", &statelix::SyntheticControlResult::predictor_balance)
        .def_readonly("n_donors", &statelix::SyntheticControlResult::n_donors)
        .def_readonly("n_pre", &statelix::SyntheticControlResult::n_pre_periods)
        .def_readonly("n_post", &statelix::SyntheticControlResult::n_post_periods)
        .def_readonly("treatment_period", &statelix::SyntheticControlResult::treatment_period);

    py::class_<statelix::PlaceboResult>(causal, "PlaceboResult")
        .def_readonly("p_value", &statelix::PlaceboResult::p_value)
        .def_readonly("treated_rank", &statelix::PlaceboResult::treated_rank)
        .def_readonly("rmspe_ratios", &statelix::PlaceboResult::rmspe_ratios)
        .def_readonly("placebo_gaps", &statelix::PlaceboResult::placebo_gaps);

    py::class_<statelix::SyntheticControl>(causal, "SyntheticControl")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::SyntheticControl::max_iter)
        .def_readwrite("tol", &statelix::SyntheticControl::tol)
        .def_readwrite("v_penalty", &statelix::SyntheticControl::v_penalty)
        .def_readwrite("normalize", &statelix::SyntheticControl::normalize)
        .def("fit", py::overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXd&, int, int>(
            &statelix::SyntheticControl::fit), 
            py::arg("Y"), py::arg("X"), py::arg("treated_idx"), py::arg("treatment_period"))
        .def("fit", py::overload_cast<const Eigen::MatrixXd&, int, int>(
            &statelix::SyntheticControl::fit),
            py::arg("Y"), py::arg("treated_idx"), py::arg("treatment_period"))
        .def("placebo_test", &statelix::SyntheticControl::placebo_test,
             py::arg("Y"), py::arg("treated_idx"), py::arg("treatment_period"))
        .def("in_time_placebo", &statelix::SyntheticControl::in_time_placebo,
             py::arg("Y"), py::arg("treated_idx"), py::arg("actual_treatment"), py::arg("fake_treatment"));

    py::class_<statelix::GSCResult>(causal, "GSCResult")
        .def_readonly("factors", &statelix::GSCResult::factors)
        .def_readonly("att", &statelix::GSCResult::att)
        .def_readonly("average_att", &statelix::GSCResult::average_att)
        .def_readonly("att_se", &statelix::GSCResult::att_se);

    py::class_<statelix::GeneralizedSyntheticControl>(causal, "GeneralizedSyntheticControl")
        .def(py::init<>())
        .def_readwrite("n_factors", &statelix::GeneralizedSyntheticControl::n_factors)
        .def("fit", &statelix::GeneralizedSyntheticControl::fit, py::arg("Y"), py::arg("D"));

    // --- Dynamic Panel GMM ---
    // (Consolidated above)


    // --- Spatial Econometrics ---

    auto spatial = m.def_submodule("spatial", "Spatial Econometrics");

    py::enum_<statelix::SpatialModel>(spatial, "SpatialModel")
        .value("SAR", statelix::SpatialModel::SAR)
        .value("SEM", statelix::SpatialModel::SEM)
        .value("SDM", statelix::SpatialModel::SDM)
        .value("SAC", statelix::SpatialModel::SAC)
        .export_values();

    py::class_<statelix::SpatialResult>(spatial, "SpatialResult")
        .def_readonly("rho", &statelix::SpatialResult::rho)
        .def_readonly("rho_se", &statelix::SpatialResult::rho_se)
        .def_readonly("lambda", &statelix::SpatialResult::lambda)
        .def_readonly("beta", &statelix::SpatialResult::beta)
        .def_readonly("beta_se", &statelix::SpatialResult::beta_se)
        .def_readonly("log_likelihood", &statelix::SpatialResult::log_likelihood)
        .def_readonly("direct_effects", &statelix::SpatialResult::direct_effects)
        .def_readonly("indirect_effects", &statelix::SpatialResult::indirect_effects)
        .def_readonly("total_effects", &statelix::SpatialResult::total_effects);

    py::class_<statelix::MoranResult>(spatial, "MoranResult")
        .def_readonly("I", &statelix::MoranResult::I)
        .def_readonly("z_stat", &statelix::MoranResult::z_stat)
        .def_readonly("p_value", &statelix::MoranResult::p_value)
        .def_readonly("spatial_autocorrelation", &statelix::MoranResult::spatial_autocorrelation);

    py::class_<statelix::LMSpatialResult>(spatial, "LMSpatialResult")
        .def_readonly("lm_lag", &statelix::LMSpatialResult::lm_lag)
        .def_readonly("lm_lag_pvalue", &statelix::LMSpatialResult::lm_lag_pvalue)
        .def_readonly("lm_error", &statelix::LMSpatialResult::lm_error)
        .def_readonly("lm_error_pvalue", &statelix::LMSpatialResult::lm_error_pvalue)
        .def_readonly("rlm_lag", &statelix::LMSpatialResult::rlm_lag)
        .def_readonly("rlm_error", &statelix::LMSpatialResult::rlm_error)
        .def_readonly("recommendation", &statelix::LMSpatialResult::recommendation);

    py::class_<statelix::SpatialWeights>(spatial, "SpatialWeights")
        .def_static("row_standardize", &statelix::SpatialWeights::row_standardize, py::arg("W"))
        .def_static("knn_weights", &statelix::SpatialWeights::knn_weights, 
                    py::arg("coords"), py::arg("k"))
        .def_static("inverse_distance_weights", &statelix::SpatialWeights::inverse_distance_weights,
                    py::arg("coords"), py::arg("bandwidth")=-1, py::arg("power")=1.0);

    py::class_<statelix::SpatialRegression>(spatial, "SpatialRegression")
        .def(py::init<>())
        .def_readwrite("model", &statelix::SpatialRegression::model)
        .def_readwrite("max_iter", &statelix::SpatialRegression::max_iter)
        .def("fit", &statelix::SpatialRegression::fit, py::arg("y"), py::arg("X"), py::arg("W"))
        .def("moran_test", &statelix::SpatialRegression::moran_test, py::arg("z"), py::arg("W"))
        .def("lm_tests", &statelix::SpatialRegression::lm_tests, py::arg("y"), py::arg("X"), py::arg("W"));

    // --- Quantile Regression ---

    py::class_<statelix::QuantileResult>(m, "QuantileResult")
        .def_readonly("coef", &statelix::QuantileResult::coef)
        .def_readonly("std_errors", &statelix::QuantileResult::std_errors)
        .def_readonly("t_values", &statelix::QuantileResult::t_values)
        .def_readonly("p_values", &statelix::QuantileResult::p_values)
        .def_readonly("conf_lower", &statelix::QuantileResult::conf_lower)
        .def_readonly("conf_upper", &statelix::QuantileResult::conf_upper)
        .def_readonly("tau", &statelix::QuantileResult::tau)
        .def_readonly("pseudo_r_squared", &statelix::QuantileResult::pseudo_r_squared)
        .def_readonly("converged", &statelix::QuantileResult::converged);

    py::class_<statelix::QuantileProcessResult>(m, "QuantileProcessResult")
        .def_readonly("taus", &statelix::QuantileProcessResult::taus)
        .def_readonly("wald_stat", &statelix::QuantileProcessResult::wald_stat)
        .def_readonly("wald_pvalue", &statelix::QuantileProcessResult::wald_pvalue);

    py::class_<statelix::QuantileRegression>(m, "QuantileRegression")
        .def(py::init<>())
        .def_readwrite("max_iter", &statelix::QuantileRegression::max_iter)
        .def_readwrite("bootstrap_se", &statelix::QuantileRegression::bootstrap_se)
        .def_readwrite("bootstrap_reps", &statelix::QuantileRegression::bootstrap_reps)
        .def("fit", &statelix::QuantileRegression::fit, py::arg("y"), py::arg("X"), py::arg("tau"))
        .def("quantile_process", &statelix::QuantileRegression::quantile_process,
             py::arg("y"), py::arg("X"), py::arg("taus"))
        .def("interquantile_range", &statelix::QuantileRegression::interquantile_range,
             py::arg("y"), py::arg("X"), py::arg("tau_low")=0.25, py::arg("tau_high")=0.75);

    // --- Resampling ---
    
    // Resampler Wrapper
    class ResamplerWrapper {
    public:
        statelix::Resampler resampler;
        
        ResamplerWrapper(int seed=42, bool parallel=true, int n_jobs=-1) {
            resampler.seed = seed;
            resampler.parallel = parallel;
            resampler.n_jobs = n_jobs;
        }
        
        // Bootstrap wrapper
        Eigen::MatrixXd bootstrap(
            Eigen::MatrixXd data,
            std::function<Eigen::VectorXd(Eigen::MatrixXd)> func,
            int n_reps
        ) {
            // Disable C++ parallelism when calling Python function to avoid GIL issues
            bool original_parallel = resampler.parallel;
            // Only force serial execution if we suspect the func holds GIL, 
            // which it DOES if it's a Python function wrapped by pybind11.
            resampler.parallel = false; 
            
            auto result = resampler.bootstrap(data, func, n_reps);
            
            resampler.parallel = original_parallel;
            return result;
        }
        
        Eigen::MatrixXd block_bootstrap(
            Eigen::MatrixXd data,
            std::function<Eigen::VectorXd(Eigen::MatrixXd)> func,
            int block_size,
            int n_reps
        ) {
            bool original_parallel = resampler.parallel;
            resampler.parallel = false;
            
            auto result = resampler.block_bootstrap(data, func, block_size, n_reps);
            
            resampler.parallel = original_parallel;
            return result;
        }
        
        Eigen::MatrixXd jackknife(
            Eigen::MatrixXd data,
            std::function<Eigen::VectorXd(Eigen::MatrixXd)> func
        ) {
            bool original_parallel = resampler.parallel;
            resampler.parallel = false;
            
            auto result = resampler.jackknife(data, func);
            
            resampler.parallel = original_parallel;
            return result;
        }
    };
    
    py::class_<ResamplerWrapper>(m, "Resampler")
        .def(py::init<int, bool, int>(), 
             py::arg("seed")=42, py::arg("parallel")=true, py::arg("n_jobs")=-1)
        .def("bootstrap", &ResamplerWrapper::bootstrap,
             py::arg("data"), py::arg("func"), py::arg("n_reps")=1000)
        .def("block_bootstrap", &ResamplerWrapper::block_bootstrap,
             py::arg("data"), py::arg("func"), py::arg("block_size"), py::arg("n_reps")=1000)
        .def("jackknife", &ResamplerWrapper::jackknife,
             py::arg("data"), py::arg("func"));

}
