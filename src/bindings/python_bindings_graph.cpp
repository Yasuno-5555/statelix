#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../graph/louvain.h"
#include "../graph/pagerank.h"

namespace py = pybind11;
using namespace statelix;

PYBIND11_MODULE(graph, m) {
  m.doc() = "Statelix Graph Analysis";

  py::class_<statelix::graph::LouvainResult>(m, "LouvainResult")
      .def_readonly("labels", &statelix::graph::LouvainResult::labels)
      .def_readonly("n_communities",
                    &statelix::graph::LouvainResult::n_communities)
      .def_readonly("modularity", &statelix::graph::LouvainResult::modularity)
      .def_readonly("hierarchy", &statelix::graph::LouvainResult::hierarchy)
      .def_readonly("community_sizes",
                    &statelix::graph::LouvainResult::community_sizes);

  py::class_<statelix::graph::Louvain>(m, "Louvain")
      .def(py::init<>())
      .def_readwrite("resolution", &statelix::graph::Louvain::resolution)
      .def_readwrite("max_iterations",
                     &statelix::graph::Louvain::max_iterations)
      .def_readwrite("randomize_order",
                     &statelix::graph::Louvain::randomize_order)
      .def_readwrite("seed", &statelix::graph::Louvain::seed)
      .def("fit", &statelix::graph::Louvain::fit, py::arg("adjacency"),
           "Detect communities from sparse adjacency matrix. Ensures node IDs "
           "match matrix indices.");

  py::class_<statelix::graph::PageRankResult>(m, "PageRankResult")
      .def_readonly("scores", &statelix::graph::PageRankResult::scores)
      .def_readonly("ranking", &statelix::graph::PageRankResult::ranking)
      .def_readonly("converged", &statelix::graph::PageRankResult::converged);

  py::class_<statelix::graph::PageRank>(m, "PageRank")
      .def(py::init<>())
      .def_readwrite("damping", &statelix::graph::PageRank::damping)
      .def_readwrite("max_iter", &statelix::graph::PageRank::max_iter)
      .def_readwrite("tol", &statelix::graph::PageRank::tol)
      .def("compute", &statelix::graph::PageRank::compute, py::arg("adjacency"))
      .def("personalized", &statelix::graph::PageRank::personalized,
           py::arg("adjacency"), py::arg("seeds"),
           py::arg("restart_prob") = 0.15);
}
