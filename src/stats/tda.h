#pragma once

#include <MathUniverse/Keirin/Core/homology.hpp>
#include <MathUniverse/Keirin/Core/simplicial_complex.hpp>
#include <MathUniverse/Keirin/TDA/persistence.hpp>
#include <MathUniverse/Keirin/TDA/vietoris_rips.hpp>
#include <vector>

namespace statelix {
namespace stats {
namespace tda {

/**
 * @brief Compute Betti numbers for a given simplicial complex.
 *
 * @param points Point cloud data (N x Dim)
 * @param epsilon Distance threshold for Vietoris-Rips filtration
 * @param dim Dimension to compute Betti number for
 * @return size_t Betti number beta_dim
 */
inline size_t
compute_betti_number(const std::vector<std::vector<double>> &points,
                     double epsilon, int dim) {
  auto filtration = keirin::tda::generate_rips_filtration(points, epsilon);
  keirin::topology::SimplicialComplex K;
  for (const auto &fs : filtration.simplices()) {
    if (fs.value <= epsilon) {
      K.add(fs.simplex);
    }
  }
  return keirin::topology::betti_number(K, dim);
}

/**
 * @brief Compute persistence barcodes für point cloud data.
 *
 * @param points Point cloud data (N x Dim)
 * @param max_epsilon Maximum distance for filtration
 * @return std::vector<keirin::tda::PersistencePair> Birth-Death pairs
 */
inline auto compute_persistence(const std::vector<std::vector<double>> &points,
                                double max_epsilon) {
  auto filtration = keirin::tda::generate_rips_filtration(points, max_epsilon);
  return keirin::tda::compute_persistence(filtration);
}

} // namespace tda
} // namespace stats
} // namespace statelix
