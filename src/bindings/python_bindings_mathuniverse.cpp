#include <MathUniverse/Keirin/persistence.hpp>
#include <MathUniverse/Risan/graph.hpp>
#include <MathUniverse/Shinen/multivector.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace MathUniverse;

PYBIND11_MODULE(mathuniverse, m) {
  m.doc() = "MathUniverse Skeleton Bindings for Risan, Keirin, and Shinen";

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
      .def_readwrite("complex", &Keirin::Persistence::complex)
      .def_readwrite("structure_score", &Keirin::Persistence::structure_score);

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
      .def_static("scalar", &MV::scalar, py::arg("val"))
      .def_static("rotor", &MV::rotor, py::arg("angle"), py::arg("B_unit"))
      .def("__add__", &MV::operator+)
      .def("__mul__", [](const MV &a, const MV &b) { return a * b; })
      .def("__mul__", [](const MV &a, double b) { return a * b; })
      .def("rotate", &MV::rotate, py::arg("v"))
      .def("__repr__", [](const MV &m) {
        std::stringstream ss;
        ss << m;
        return ss.str();
      });
}
