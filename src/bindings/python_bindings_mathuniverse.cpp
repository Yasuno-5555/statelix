#include <MathUniverse/Keirin/persistence.hpp>
#include <MathUniverse/Risan/graph.hpp>
#include <MathUniverse/Ryoshi/causal.hpp>
#include <MathUniverse/Ryoshi/qubit.hpp>
#include <MathUniverse/Shinen/invariants.hpp>
#include <MathUniverse/Shinen/multivector.hpp>
#include <Sokudo/Backends/ZigenCpuBackend.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace MathUniverse;

PYBIND11_MODULE(mathuniverse, m) {
  m.doc() = "MathUniverse Skeleton Bindings for Risan, Keirin, Shinen, Ryoshi, "
            "and Sokudo";

  // --- Risan (Discrete) ---
  auto risan = m.def_submodule("risan", "Discrete mathematics domain");

  py::class_<Risan::Node<std::string>>(risan, "Node")
      .def(py::init<std::string>())
      .def_readwrite("data", &Risan::Node<std::string>::data)
      .def("connect", &Risan::Node<std::string>::connect, py::arg("target"),
           "Connect to another node")
      .def_property_readonly("edges", [](Risan::Node<std::string> &self) {
        std::vector<std::string> target_data;
        for (auto &edge : self.edges) {
          target_data.push_back(edge.target->data);
        }
        return target_data;
      });

  py::class_<Risan::Graph<std::string>::RewireSuggestion>(risan,
                                                          "RewireSuggestion")
      .def_readonly("action",
                    &Risan::Graph<std::string>::RewireSuggestion::action)
      .def_readonly("source",
                    &Risan::Graph<std::string>::RewireSuggestion::source)
      .def_readonly("target",
                    &Risan::Graph<std::string>::RewireSuggestion::target)
      .def_readonly("confidence",
                    &Risan::Graph<std::string>::RewireSuggestion::confidence)
      .def("__repr__",
           [](const Risan::Graph<std::string>::RewireSuggestion &s) {
             return "[" + s.action + "] " + s.source + " -> " + s.target +
                    " (" + std::to_string(s.confidence) + ")";
           });

  py::class_<Risan::Graph<std::string>>(risan, "Graph")
      .def(py::init<>())
      .def("add_node", &Risan::Graph<std::string>::add_node, py::arg("node"))
      .def("suggest_rewiring", &Risan::Graph<std::string>::suggest_rewiring);

  // --- Keirin (Topological) ---
  auto keirin = m.def_submodule("keirin", "Topological data analysis domain");

  py::class_<Keirin::Simplex>(keirin, "Simplex")
      .def(py::init<std::vector<int>, double>())
      .def_readwrite("vertices", &Keirin::Simplex::vertices)
      .def_readwrite("filtration_value", &Keirin::Simplex::filtration_value);

  py::class_<Keirin::Persistence>(keirin, "Persistence")
      .def(py::init<>())
      .def("add_simplex", &Keirin::Persistence::add_simplex,
           py::arg("vertices"), py::arg("filtration_value"))
      .def("compute_homology", &Keirin::Persistence::compute_homology)
      .def("detect_jump", &Keirin::Persistence::detect_jump,
           py::arg("threshold") = 2.0)
      .def_readwrite("complex", &Keirin::Persistence::complex)
      .def_readwrite("structure_score", &Keirin::Persistence::structure_score)
      .def_readwrite("history_scores", &Keirin::Persistence::history_scores);

  // --- Shinen (Geometric) ---
  auto shinen = m.def_submodule("shinen", "Geometric Algebra domain");

  using MV = Shinen::MultiVector<double>;
  py::class_<MV>(shinen, "MultiVector")
      .def(py::init<double, double, double, double, double, double, double,
                    double>(),
           py::arg("s") = 0, py::arg("e1") = 0, py::arg("e2") = 0,
           py::arg("e3") = 0, py::arg("e12") = 0, py::arg("e23") = 0,
           py::arg("e31") = 0, py::arg("e123") = 0)
      .def_readwrite("s", &MV::s)
      .def_readwrite("e1", &MV::e1)
      .def_readwrite("e2", &MV::e2)
      .def_readwrite("e3", &MV::e3)
      .def_readwrite("e12", &MV::e12)
      .def_readwrite("e23", &MV::e23)
      .def_readwrite("e31", &MV::e31)
      .def_readwrite("e123", &MV::e123)
      .def_static("vector", &MV::vector, py::arg("x"), py::arg("y"),
                  py::arg("z"))
      .def_static("bivector", &MV::bivector, py::arg("xy"), py::arg("yz"),
                  py::arg("zx"))
      .def_static("pseudoscalar", &MV::pseudoscalar, py::arg("val"))
      .def_static("scalar", &MV::scalar, py::arg("val"))
      .def_static("rotor", &MV::rotor, py::arg("angle"), py::arg("B_unit"))
      .def("__add__", &MV::operator+)
      .def("__mul__", [](const MV &a, const MV &b) { return a * b; })
      .def("__mul__", [](const MV &a, double b) { return a * b; })
      .def("rotate", &MV::rotate, py::arg("v"))
      .def("magnitude", &MV::magnitude)
      .def("__repr__", [](const MV &m) {
        std::stringstream ss;
        ss << m;
        return ss.str();
      });

  py::class_<Shinen::InvariantScore>(shinen, "InvariantScore")
      .def_readonly("feature_name", &Shinen::InvariantScore::feature_name)
      .def_readonly("score", &Shinen::InvariantScore::score)
      .def("__repr__", [](const Shinen::InvariantScore &s) {
        return s.feature_name + ": " + std::to_string(s.score);
      });

  py::class_<Shinen::InvariantSensors>(shinen, "InvariantSensors")
      .def_static("rank_invariants", &Shinen::InvariantSensors::rank_invariants,
                  py::arg("names"), py::arg("vectors"));

  // --- Ryoshi (Quantum) ---
  auto ryoshi = m.def_submodule("ryoshi", "Quantum computing domain");

  py::class_<Ryoshi::QuantumState>(ryoshi, "QuantumState")
      .def(py::init<int>())
      .def("H", &Ryoshi::QuantumState::H)
      .def("X", &Ryoshi::QuantumState::X)
      .def("CNOT", &Ryoshi::QuantumState::CNOT)
      .def("measure", &Ryoshi::QuantumState::measure)
      .def("print_state", &Ryoshi::QuantumState::print_state);

  py::class_<Ryoshi::CausalSearch>(ryoshi, "CausalSearch")
      .def(py::init<double>(), py::arg("initial_temp") = 100.0)
      .def_readwrite("temperature", &Ryoshi::CausalSearch::temperature)
      .def_readwrite("information_score",
                     &Ryoshi::CausalSearch::information_score)
      .def_readwrite("score_history", &Ryoshi::CausalSearch::score_history)
      .def("step", &Ryoshi::CausalSearch::step, py::arg("current_score"))
      .def("check_early_stopping", &Ryoshi::CausalSearch::check_early_stopping,
           py::arg("tolerance") = 1e-4);

  // --- Sokudo (Stochastic) ---
  auto sokudo = m.def_submodule("sokudo", "Stochastic data domain");

  sokudo.def(
      "generate_normal",
      [](size_t n, int seed) {
        auto result = py::array_t<double>(n);
        double *ptr = static_cast<double *>(result.request().ptr);

        std::mt19937 g(seed);

#pragma omp parallel
        {
          std::mt19937 local_g(seed + omp_get_thread_num());
          std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
          for (size_t i = 0; i < n; ++i) {
            ptr[i] = dist(local_g);
          }
        }
        return result;
      },
      py::arg("n"), py::arg("seed") = 42,
      "Highly parallelized normal sampling");
}
