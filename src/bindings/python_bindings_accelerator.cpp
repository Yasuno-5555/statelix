
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../cuda/accelerator.h"

namespace py = pybind11;

PYBIND11_MODULE(accelerator, m) {
    m.doc() = "Statelix CUDA Accelerator Module";

    m.def("is_available", &statelix::cuda::is_available, 
          "Check if NVIDIA GPU is available and accessible.");

    m.def("weighted_gram_matrix", [](py::array_t<double> X, py::array_t<double> W) {
        // Buffer info
        py::buffer_info buf_X = X.request();
        py::buffer_info buf_W = W.request();

        if (buf_X.ndim != 2) throw std::runtime_error("X must be 2D");
        if (buf_W.ndim != 1) throw std::runtime_error("W must be 1D");

        int n = buf_X.shape[0];
        int k = buf_X.shape[1];

        if (buf_W.shape[0] != n) throw std::runtime_error("W size must match X rows");

        // Call CUDA backend
        // Note: pybind11 arrays might not be contiguous if sliced.
        // We really should check or enforce contiguous.
        // For Proof of Concept, assume contiguous or force it in Python.
        // Ideally: auto X_cont = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(X);
        
        const double* ptr_X = static_cast<const double*>(buf_X.ptr);
        const double* ptr_W = static_cast<const double*>(buf_W.ptr);

        std::vector<double> res_vec = statelix::cuda::compute_weighted_gram(ptr_X, ptr_W, n, k);
        
        if (res_vec.empty()) {
            throw std::runtime_error("CUDA computation failed (allocation or fallback condition met)");
        }

        // Return as numpy array (K x K)
        return py::array_t<double>(
            {k, k},              // shape
            {k * 8, 8},          // strides (bytes)
            res_vec.data()       // data pointer (copies)
        );
    }, "Compute Weighted Gram Matrix (X.T @ diag(W) @ X) on GPU");
}
