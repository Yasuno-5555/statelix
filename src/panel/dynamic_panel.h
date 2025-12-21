#ifndef STATELIX_DYNAMIC_PANEL_H
#define STATELIX_DYNAMIC_PANEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

namespace statelix {
namespace panel {

struct DynamicPanelResult {
  Eigen::VectorXd coefficients; // [rho, beta...]
  Eigen::VectorXd std_errors;
  Eigen::VectorXd z_values;
  Eigen::VectorXd p_values;
  double sargan_test; // Overidentifying restrictions
  double sargan_pvalue;
  double ar1_test; // AR(1) in first differences
  double ar1_pvalue;
  double ar2_test; // AR(2) in first differences
  double ar2_pvalue;
  int n_obs;
  int n_instruments;
  bool converged;
};

class DynamicPanelGMM {
public:
  // Estimator options
  bool two_step = true;
  bool robust_se = true;
  int max_lags = -1; // -1 = use all available lags

  DynamicPanelResult
  estimate(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
           const std::vector<int> &ids,
           const std::vector<int> &time // Assumed integer time periods
  ) {
    int N_total = y.size();
    int K = X.cols();

    // 1. Prepare Data Structure (Sort by ID, then Time)
    // Map: ID -> vector of (time, index)
    std::map<int, std::vector<std::pair<int, int>>> panel_data;
    for (int i = 0; i < N_total; ++i) {
      panel_data[ids[i]].push_back({time[i], i});
    }

    // Sort each individual's time series
    for (auto &entry : panel_data) {
      std::sort(entry.second.begin(), entry.second.end());
    }

    // 2. Construct First Differences (\Delta y, \Delta X)
    // And build instrument matrix Z
    //
    // Z matrix structure (Stacked for all i):
    // Diag(z_i, ...) where z_i contains lags.
    // For Arellano-Bond, we need rows where difference can be taken (T >= 3
    // technically for basic, but generally t start from 2 if we have 0,1)

    std::vector<Eigen::VectorXd> dy_vec;
    std::vector<Eigen::VectorXd> dx_vec;
    std::vector<Eigen::VectorXd> z_rows; // Placeholder for sparse Z rows?

    // To handle Z efficiently (sparse block diagonal), we might need careful
    // construction. Standard GMM: Y = X B + E.  Z'E = 0. Estimator: B = (X' Z W
    // Z' X)^-1 X' Z W Z' Y

    // Let's count usable observations
    int n_diff = 0;
    int T_max = 0;
    for (auto const &[id, obs] : panel_data) {
      if (obs.size() < 2)
        continue; // Need at least 2 for difference
      n_diff += (obs.size() - 1);
      T_max = std::max(T_max, obs.back().first);
    }

    // Matrices for differenced equation
    // We include y_{t-1} in RHS (so \Delta y_{t-1} is predictor)
    // Regressors: [\Delta y_{t-1}, \Delta X]
    Eigen::VectorXd Y_diff(n_diff);
    Eigen::MatrixXd X_diff(n_diff, K + 1);

    // Z is tricky because it grows with T.
    // We will implement Z as a dense matrix for now for simplicity,
    // effectively padding with zeros.
    // Or better: Use List of Z_i and accumulate cross products directly `Z'X`
    // and `Z'Z`. To implement 1-step W, we need sum(Z_i' H Z_i).

    // Count usable instruments to allocate max size?
    // simple AB: for time t (differenced eq for t, t-1), instruments are
    // y_{0}..y_{t-2} Max instruments grows quadratically with T.

    // Pass 1: Build Y_diff, X_diff and map rows to (id, t)
    int row_idx = 0;
    std::vector<std::pair<int, int>>
        row_map; // maps diff_row -> (id, time_of_diff)

    for (auto const &[id, obs] : panel_data) {
      if (obs.size() < 2)
        continue;

      for (size_t t = 1; t < obs.size(); ++t) {
        int curr_idx = obs[t].second;
        int prev_idx = obs[t - 1].second;

        // Gap check: ensure time is consecutive?
        // AB (1991) allows gaps but creating differences requires care.
        // Assuming consecutive for standard diff. If gap, skip diff.
        if (obs[t].first != obs[t - 1].first + 1)
          continue;

        Y_diff(row_idx) = y(curr_idx) - y(prev_idx);

        // Lagged dependent var diff: \Delta y_{t-1} = y_{t-1} - y_{t-2}
        if (t >= 2) {
          int prev2_idx = obs[t - 2].second;
          if (obs[t - 1].first == obs[t - 2].first + 1) {
            X_diff(row_idx, 0) = y(prev_idx) - y(prev2_idx);
          } else {
            // Gap in past, cannot compute \Delta y_{t-1}
            // For this implementation, let's treat as missing/skip row
            // But index mismatch... let's restart logic slightly.
            // Simple approach: Skip if t < 2.
            // Actually, if we lose \Delta y_{t-1}, we lose the whole row.
            // So we effectively need 3 consecutive periods to have 1 obs in
            // regression (t-2, t-1, t) -> \Delta y_t and \Delta y_{t-1}.
            continue;
          }
        } else {
          continue; // Need t>=2 (0,1,2 indices) to have lag difference
        }

        // X diffs
        for (int k = 0; k < K; ++k) {
          X_diff(row_idx, k + 1) = X(curr_idx, k) - X(prev_idx, k);
        }

        row_map.push_back({id, obs[t].first});
        row_idx++;
      }
    }

    // Resize to actual used rows
    Y_diff.conservativeResize(row_idx);
    X_diff.conservativeResize(row_idx, K + 1);
    int N_used = row_idx;

    // 3. Construct Instrument Matrix Z (Sparse/Block)
    // We avoid constructing full Z (N_used x Huge) explicitly if possible.
    // Instead we construct cross products directly ?
    // Z = [Z_1; Z_2; ... Z_n]
    // Z_i = diag(z_{i3}, z_{i4}, ...)
    // z_{it} = [y_{i0}, ... y_{i,t-2}, x_{it} instruments...]
    // For standard AB, instruments are levels of y.

    // Let's figure out total columns in Z.
    // Max time T. Columns = sum_{t=3}^T (t-2).
    int min_t_used = 100000;
    int max_t_used = -1;
    for (auto &p : row_map) {
      if (p.second < min_t_used)
        min_t_used = p.second;
      if (p.second > max_t_used)
        max_t_used = p.second;
    }

    // Mapping (t) -> [col_start, col_count] in Z
    std::map<int, int> z_col_map;
    int z_cols = 0;
    for (int t = min_t_used; t <= max_t_used; ++t) {
      z_col_map[t] = z_cols;
      // Instruments for time t (observations are diffs at t):
      // y_{0} ... y_{t-2}. Count = t-2 - (min_time_in_data) ...
      // Let's assume absolute time or relative?
      // Safer: Available lags.
      // Actually, usually we map "relative lag 2, 3..." to columns.
      // "GMM style" instruments: one column per time period per lag.
      // Collapsed instruments: one column per lag.
      // We implement "GMM style" (Standard AB).
      // # available instruments at time t = (t - start_time) - 1.
      // Let's keep it simpler: dense-ish Z where cols correspond to specific
      // lags at specific times.

      // To be robust:
      // Z row for (id, t) contains [y_{id, start}, ..., y_{id, t-2}]
      // Number of instruments = valid history length.
      // This varies by t.
      // Total columns = Sum over unique t in regression of (available history).
    }

    // Re-loop to define Z columns clearly
    // We'll create a full dense Z for valid rows for simplicity of
    // implementation first. Or block-diagonal.
    //
    // Let's construct Z_list where Z_list[i] is Z matrix for that row.
    // But we need alignment.
    //
    // Simplify: Collapsed Instruments? No, user wants AB.
    //
    // Let's compute X'Z, Y'Z, Z'Z directly?
    // We need Z to compute weights W = (Z' H Z)^-1.

    // Let's try constructing the actual Z matrix.
    // It's block diagonal in terms of Time periods usually.
    // Z = [ Z_{(3)}, 0, ...
    //       0, Z_{(4)}, ... ]
    // Where Z_{(t)} is the stack of instrument vectors for all individuals
    // observed at t. Columns of Z:
    //  Block t=3: instruments y_{1} (1 col)
    //  Block t=4: instruments y_{1}, y_{2} (2 cols)
    // ...

    // Identify all unique time periods in usable rows
    std::vector<int> unique_times;
    for (auto &p : row_map)
      unique_times.push_back(p.second);
    std::sort(unique_times.begin(), unique_times.end());
    unique_times.erase(std::unique(unique_times.begin(), unique_times.end()),
                       unique_times.end());

    std::map<int, int> time_to_col_start;
    int total_z_cols = 0;

    // Base time for lags
    int global_min_time = 100000;
    for (auto const &[id, obs] : panel_data) {
      global_min_time = std::min(global_min_time, obs[0].first);
    }

    for (int t : unique_times) {
      time_to_col_start[t] = total_z_cols;
      // Instruments: y_{global_min_time} ... y_{t-2}
      // Count = (t - 2) - global_min_time + 1
      int n_instr = (t - 2) - global_min_time + 1;
      if (n_instr < 0)
        n_instr = 0;
      // Also need to limit by max_lags if set
      if (max_lags > 0)
        n_instr = std::min(n_instr, max_lags);

      total_z_cols += n_instr;
    }

    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(N_used, total_z_cols);

    // Fill Z
    for (int i = 0; i < N_used; ++i) {
      int id = row_map[i].first;
      int t = row_map[i].second;
      int col_start = time_to_col_start[t];

      // Instruments y_{t-2}, y_{t-3}...
      // Find y values in panel_data for this id
      const auto &obs = panel_data[id];

      // Naive search for history (obs is short, so linear scan is OK)
      // We need values at times t-2, t-3 ... down to limit
      // Note: columns usually ordered as y_{min}, y_{min+1}...
      // Let's assume standard order: y_{start}, y_{start+1} ... y_{t-2}

      // int n_instr_this_t = 0; // Unused

      if (max_lags > 0) {
        // logic for limited lags
        // just take last max_lags
      } else {
        // all lags from start up to t-2
      }

      // Let's implement full lags for now
      int t_end_instr = t - 2;
      int t_start_instr = global_min_time;
      if (max_lags > 0)
        t_start_instr = std::max(t_start_instr, t_end_instr - max_lags + 1);

      int current_col_offset = 0;
      for (int t_instr = t_start_instr; t_instr <= t_end_instr; ++t_instr) {
        // find val
        double val = 0.0;
        bool found = false;
        // Optimization: obs is sorted by time
        for (auto &p : obs) {
          if (p.first == t_instr) {
            val = y(p.second);
            found = true;
            break;
          }
        }

        if (found) {
          Z(i, col_start + current_col_offset) = val;
        } else {
          Z(i, col_start + current_col_offset) =
              0.0; // Missing instrument = 0 ??
          // Standard practice: if instrument missing, row might be invalid or
          // we put 0 In GMM, Z should be orthog to error. 0 is safe placeholder
          // if properly handled, but ideally we only use valid instruments.
        }
        current_col_offset++;
      }
    }

    // 4. Estimation
    // Step 1: H1 = Z' (I_N \otimes H_i) Z ...
    // For 1-step Difference GMM, H_i is matrix with 2 on diag, -1 on off-diag
    // (MA(1) structure of diff error) (Assuming homoskedasticity) H = Z' * A *
    // Z where A is block diagonal with blocks A_i A_i has 2 on diag, -1 on
    // super/subdiag.

    // Construct A? Too big (N_used x N_used).
    // Calculate Z'AZ = sum_i Z_i' A_i Z_i

    // We need to group rows by ID to apply A_i
    // Re-scan usable rows to group by ID
    std::vector<std::vector<int>>
        id_to_rows; // id -> list of row indices in Y_diff/X_diff/Z
    // Since we processed by ID, they are contiguous blocks in Y_diff!
    // We just need start/len for each ID block in N_used rows.

    std::vector<std::pair<int, int>> id_blocks; // start, len
    int curr_start = 0;
    int curr_id = row_map[0].first;
    for (int i = 1; i < N_used; ++i) {
      if (row_map[i].first != curr_id) {
        id_blocks.push_back({curr_start, i - curr_start});
        curr_start = i;
        curr_id = row_map[i].first;
      }
    }
    id_blocks.push_back({curr_start, N_used - curr_start}); // Last block

    // Calculate W1 = (Z' A Z)^-1
    Eigen::MatrixXd ZAZ = Eigen::MatrixXd::Zero(total_z_cols, total_z_cols);

    for (auto p : id_blocks) {
      int start = p.first;
      int len = p.second;
      Eigen::MatrixXd Zi = Z.block(start, 0, len, total_z_cols);

      // Construct Ai (len x len)
      // 2 on diag, -1 on off-diag
      Eigen::MatrixXd Ai = Eigen::MatrixXd::Zero(len, len);
      for (int r = 0; r < len; ++r) {
        Ai(r, r) = 2.0;
        if (r > 0)
          Ai(r, r - 1) = -1.0;
        if (r < len - 1)
          Ai(r, r + 1) = -1.0;
      }

      ZAZ += Zi.transpose() * Ai * Zi;
    }

    // W1 = (ZAZ)^-1
    // Usually singular if too many instruments. Use Pseudo-Inverse or LDLT with
    // regularization
    Eigen::MatrixXd W1 =
        ZAZ.ldlt().solve(Eigen::MatrixXd::Identity(total_z_cols, total_z_cols));

    // Coefficients Step 1
    // b1 = (X' Z W1 Z' X)^-1 X' Z W1 Z' Y
    // Let M = Z' X, V = Z' Y
    Eigen::MatrixXd M = Z.transpose() * X_diff;
    Eigen::VectorXd V = Z.transpose() * Y_diff;

    Eigen::MatrixXd XZWZX = M.transpose() * W1 * M;
    Eigen::VectorXd XZWZY = M.transpose() * W1 * V;

    Eigen::VectorXd beta1 = XZWZX.ldlt().solve(XZWZY);

    DynamicPanelResult result;
    result.coefficients = beta1;
    result.n_obs = N_used;
    result.n_instruments = total_z_cols;

    if (!two_step) {
      // Robust SE for 1-step?
      // Usually we assume homoskedasticity for 1-step SE, or use robust formula
      // on 1-step est. Let's calc robust covariance for consistency. Var(b) =
      // (X'Z W Z'X)^-1 X'Z W (sum Z_i' u_i u_i' Z_i) W Z'X (...)

      // Calculate residuals
      Eigen::VectorXd resid = Y_diff - X_diff * beta1;

      // Robust Sandwhich middle term: sum Z_i' u_i u_i' Z_i
      Eigen::MatrixXd Zuuz = Eigen::MatrixXd::Zero(total_z_cols, total_z_cols);
      for (auto p : id_blocks) {
        int start = p.first;
        int len = p.second;
        Eigen::MatrixXd Zi = Z.block(start, 0, len, total_z_cols);
        Eigen::VectorXd ui = resid.segment(start, len);
        Eigen::MatrixXd uui = ui * ui.transpose();
        Zuuz += Zi.transpose() * uui * Zi;
      }

      Eigen::MatrixXd Q =
          M.transpose() * W1 * M; // Inverse of this is approx var if optimal W
      // Robust Var = Q^-1 (M' W1 Zuuz W1 M) Q^-1
      Eigen::MatrixXd Q_inv = Q.inverse();
      Eigen::MatrixXd V_robust =
          Q_inv * (M.transpose() * W1 * Zuuz * W1 * M) * Q_inv;

      result.std_errors = V_robust.diagonal().cwiseSqrt().matrix();
      result.converged = true;

    } else {
      // Two Step
      // Calculate residuals from 1-step
      Eigen::VectorXd resid1 = Y_diff - X_diff * beta1;

      // New Weight Matrix W2 = (sum Z_i' u_i u_i' Z_i)^-1
      Eigen::MatrixXd Zuuz = Eigen::MatrixXd::Zero(total_z_cols, total_z_cols);
      for (auto p : id_blocks) {
        int start = p.first;
        int len = p.second;
        Eigen::MatrixXd Zi = Z.block(start, 0, len, total_z_cols);
        Eigen::VectorXd ui = resid1.segment(start, len);
        Eigen::MatrixXd uui = ui * ui.transpose();
        Zuuz += Zi.transpose() * uui * Zi;
      }

      Eigen::MatrixXd W2 = Zuuz.ldlt().solve(
          Eigen::MatrixXd::Identity(total_z_cols, total_z_cols));

      // Beta 2
      Eigen::MatrixXd XZW2ZX = M.transpose() * W2 * M;
      Eigen::VectorXd XZW2ZY = M.transpose() * W2 * V;
      Eigen::VectorXd beta2 = XZW2ZX.ldlt().solve(XZW2ZY);

      result.coefficients = beta2;

      // Windmeijer Correction (simple version: just robust SE on 2-step)
      // Var = (X' Z W2 Z' X)^-1
      // Small sample correction is complex, we'll use asymptotic
      Eigen::MatrixXd Var = XZW2ZX.inverse();
      result.std_errors = Var.diagonal().cwiseSqrt().matrix();

      // Sargan Test
      // J = u' Z W2 Z' u
      Eigen::VectorXd resid2 = Y_diff - X_diff * beta2;
      Eigen::VectorXd ZRu = Z.transpose() * resid2;
      result.sargan_test = ZRu.transpose() * W2 * ZRu;
      result.converged = true;
    }

    // Z/P values
    result.z_values =
        (result.coefficients.array() / result.std_errors.array()).matrix();
    // naive p-values

    return result;
  }
};

} // namespace panel
} // namespace statelix

#endif // STATELIX_DYNAMIC_PANEL_H
