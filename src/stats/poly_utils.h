#pragma once

#include <MathUniverse/Risan/Core/interpolation.hpp>
#include <MathUniverse/Risan/Core/poly_algo.hpp>
#include <MathUniverse/Risan/Core/polynomial.hpp>
#include <utility>
#include <vector>

namespace statelix {
namespace stats {
namespace poly {

/**
 * @brief Performs Lagrange interpolation on a set of points.
 *
 * @param x Vector of x coordinates
 * @param y Vector of y coordinates
 * @return risan::Polynomial<double> Resulting polynomial
 */
inline auto lagrange_interpolate(const std::vector<double> &x,
                                 const std::vector<double> &y) {
  if (x.size() != y.size())
    throw std::invalid_argument("x and y must have same size");
  std::vector<std::pair<double, double>> points;
  points.reserve(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    points.emplace_back(x[i], y[i]);
  }
  return risan::lagrange_interpolate<double>(points);
}

/**
 * @brief Performs Newton interpolation on a set of points.
 *
 * @param x Vector of x coordinates
 * @param y Vector of y coordinates
 * @return risan::Polynomial<double> Resulting polynomial
 */
inline auto newton_interpolate(const std::vector<double> &x,
                               const std::vector<double> &y) {
  if (x.size() != y.size())
    throw std::invalid_argument("x and y must have same size");
  std::vector<std::pair<double, double>> points;
  points.reserve(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    points.emplace_back(x[i], y[i]);
  }
  return risan::newton_interpolate<double>(points);
}

/**
 * @brief Evaluates a polynomial at a given point.
 */
inline double evaluate(const risan::Polynomial<double> &p, double x) {
  return p.evaluate(x);
}

} // namespace poly
} // namespace stats
} // namespace statelix
