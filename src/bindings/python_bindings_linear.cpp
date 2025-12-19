
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../linear_model/ols.h" 

namespace py = pybind11;
using namespace statelix;

// Python-facing class wrapper around the functional C++ API
class FitOLS {
public:
    OLSResult result;
    bool is_fitted = false;

    FitOLS() = default;

    FitOLS& fit(
        const Eigen::MatrixXd& X, 
        const Eigen::VectorXd& y,
        bool fit_intercept = true,
        double conf_level = 0.95, 
        std::optional<Eigen::MatrixXd> gram = std::nullopt
    ) {
        const Eigen::MatrixXd* p_gram = nullptr;
        if (gram.has_value()) {
            p_gram = &gram.value();
        }
        result = fit_ols_full(X, y, fit_intercept, conf_level, p_gram);
        is_fitted = true;
        return *this;
    }

    // ... predict ...
    Eigen::VectorXd predict(const Eigen::MatrixXd& X, bool fit_intercept = true) {
        if (!is_fitted) throw std::runtime_error("Model must be fitted before calling predict");
        return predict_ols(result, X, fit_intercept);
    }

    // ... getters ...
    Eigen::VectorXd get_coef() const { 
        if (!is_fitted) return Eigen::VectorXd(0);
        return result.coef; 
    }
    
    double get_intercept() const { 
        if (!is_fitted) return 0.0;
        return result.intercept; 
    }
    
    double get_r_squared() const { return is_fitted ? result.r_squared : 0.0; }
    double get_aic() const { return is_fitted ? result.aic : 0.0; }
    double get_bic() const { return is_fitted ? result.bic : 0.0; }
    double get_log_likelihood() const { return is_fitted ? result.log_likelihood : 0.0; }
};

PYBIND11_MODULE(linear_model, m) {
    m.doc() = "Statelix High-Performance Linear Models";
    
    py::class_<FitOLS>(m, "FitOLS")
        .def(py::init<>())
        .def("fit", &FitOLS::fit, py::return_value_policy::reference, 
             py::arg("X"), py::arg("y"), 
             py::arg("fit_intercept") = true, py::arg("conf_level") = 0.95,
             py::arg("gram") = py::none())
        .def("predict", &FitOLS::predict, 
             py::arg("X"), py::arg("fit_intercept") = true)
        .def_property_readonly("coef_", &FitOLS::get_coef)
        .def_property_readonly("coef", &FitOLS::get_coef) // Alias
        .def_property_readonly("intercept_", &FitOLS::get_intercept)
        .def_property_readonly("intercept", &FitOLS::get_intercept) // Alias
        .def_property_readonly("std_errors", [](const FitOLS& self) { return self.is_fitted ? self.result.std_errors : Eigen::VectorXd(0); })
        .def_property_readonly("t_values", [](const FitOLS& self) { return self.is_fitted ? self.result.t_values : Eigen::VectorXd(0); })
        .def_property_readonly("p_values", [](const FitOLS& self) { return self.is_fitted ? self.result.p_values : Eigen::VectorXd(0); })
        .def_property_readonly("conf_int", [](const FitOLS& self) { return self.is_fitted ? self.result.conf_int : Eigen::MatrixXd(0, 0); })
        .def_property_readonly("vcov", [](const FitOLS& self) { return self.is_fitted ? self.result.vcov : Eigen::MatrixXd(0, 0); })
        .def_property_readonly("r_squared", &FitOLS::get_r_squared)
        .def_property_readonly("rsquared", &FitOLS::get_r_squared) // Alias
        .def_property_readonly("aic", &FitOLS::get_aic)
        .def_property_readonly("bic", &FitOLS::get_bic)
        .def_property_readonly("log_likelihood", &FitOLS::get_log_likelihood)
        ;
}
