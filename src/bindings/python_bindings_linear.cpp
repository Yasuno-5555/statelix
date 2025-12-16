
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../linear_model/ols.h" 

namespace py = pybind11;

PYBIND11_MODULE(linear_model, m) {
    m.doc() = "Statelix Linear Models (Minimal Test)";
    // m.def("hello", []() { return "world"; });
}
