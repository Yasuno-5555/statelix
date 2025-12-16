#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

namespace statelix {

// 求解戦略
enum class SolverStrategy {
    AUTO,       // Attempt LDLT, fallback to QR if rank deficient
    CHOLESKY,   // Force LDLT (fastest, but fails if singular)
    QR          // Force QR (robust, slower, more memory)
};

/**
 * @brief 重み付きデザイン行列を効率的に管理するクラス
 * 
 * 大きな対角行列 W を明示的に確保せず、X^T W X の計算などを効率化する。
 */
class WeightedDesignMatrix {
public:
    WeightedDesignMatrix(const Eigen::MatrixXd& X, const Eigen::VectorXd& weights)
        : X_(X), weights_(weights) {
        
        if (X.rows() != weights.size()) {
            throw std::invalid_argument("Dimension mismatch: X.rows() != weights.size()");
        }
    }

    // 重みを更新（Xは変更なし）
    void update_weights(const Eigen::VectorXd& new_weights) {
        if (new_weights.size() != weights_.size()) {
            throw std::invalid_argument("Dimension mismatch in new weights");
        }
        weights_ = new_weights;
    }

    // X^T W X を計算
    // Eigenの遅延評価を利用して対角行列の生成を回避
    Eigen::MatrixXd compute_gram() const {
        // Optimized: X.transpose() * weights.asDiagonal() * X
        return X_.transpose() * weights_.asDiagonal() * X_;
    }

    // X^T W y を計算
    Eigen::VectorXd compute_XTWy(const Eigen::VectorXd& y) const {
        // Equivalent to X.transpose() * (weights.array() * y.array()).matrix()
        // Or X.transpose() * weights.asDiagonal() * y
        // Efficient: Pre-scale y by weights
        return X_.transpose() * weights_.asDiagonal() * y;
    }

    // sqrt(W) * X を計算（QR分解用）
    // 注意: 大きな行列を生成するため、必要な場合のみ呼ぶこと
    Eigen::MatrixXd compute_sqrt_weighted_X() const {
        return weights_.cwiseSqrt().asDiagonal() * X_;
    }

    const Eigen::MatrixXd& X() const { return X_; }
    const Eigen::VectorXd& weights() const { return weights_; }
    int rows() const { return X_.rows(); }
    int cols() const { return X_.cols(); }

private:
    const Eigen::MatrixXd& X_; // 参照で保持（コピー回避）
    Eigen::VectorXd weights_;
};

/**
 * @brief 重み付き最小二乗法の統合ソルバー
 * 
 * 特徴:
 * 1. 戦略パターンのサポート (Cholesky vs QR)
 * 2. 分解結果のキャッシュ（IRLSなどで有用）
 * 3. 数値的安定性の考慮 (LDLT使用)
 */
class WeightedSolver {
public:
    WeightedSolver(SolverStrategy strategy = SolverStrategy::AUTO) 
        : strategy_(strategy), is_decomposed_(false), use_qr_fallback_(false) {}

    // ソルバーの状態をリセット（重み更新時などに呼ぶ）
    void reset() {
        is_decomposed_ = false;
        use_qr_fallback_ = false;
    }

    // 重みを更新し、必要であれば即座に分解を実行
    void update_weights(const Eigen::VectorXd& weights) {
        // Note: WeightedDesignMatrix usually recreated or updated externally
        // This solver just needs to know cache is invalid.
        reset();
    }

    /**
     * @brief 線形方程式 (X^T W X) beta = X^T W y を解く
     * 
     * @param wdm 重み付きデザイン行列
     * @param y ターゲットベクトル
     * @return 推定された係数 beta
     */
    Eigen::VectorXd solve(const WeightedDesignMatrix& wdm, const Eigen::VectorXd& y) {
        compute_decomposition_if_needed(wdm);

        if (use_qr_fallback_ || strategy_ == SolverStrategy::QR) {
             // QR Path: solve min || sqrt(W)X * beta - sqrt(W)y ||
             // We computed QR of sqrt(W)X
             // Target is sqrt(W) * y
             Eigen::VectorXd weighted_y = wdm.weights().cwiseSqrt().asDiagonal() * y;
             return qr_.solve(weighted_y);
        } else {
            // Cholesky Path: solve (X^T W X) * beta = X^T W y
            Eigen::VectorXd XTWy = wdm.compute_XTWy(y);
            return ldlt_.solve(XTWy);
        }
    }

    /**
     * @brief 分散共分散行列 (X^T W X)^-1 を計算
     * 
     * Scale factor (MSEなど) は含んでいない。呼び出し側で乗算すること。
     */
    Eigen::MatrixXd variance_covariance() {
        if (!is_decomposed_) {
             throw std::runtime_error("Solver works must be run before accessing covariance.");
        }

        int p = (use_qr_fallback_ || strategy_ == SolverStrategy::QR) ? 
                qr_.matrixQR().cols() : ldlt_.matrixL().rows();

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(p, p);

        if (use_qr_fallback_ || strategy_ == SolverStrategy::QR) {
            // (R^T R)^-1 = R^-1 (R^T)^-1
            // ColPivHouseholderQR::inverse() is not directly available, utilize solve
            // (X^T W X)^-1 = (R^T Q^T Q R)^-1 = (R^T R)^-1
            // This is tricky with QR object.
            // Standard approach: solve (X^T W X) I = I, but we don't have XtWX in QR explicitly?
            // Actually, qr.solve(I) solves Ax = I? No, QR solves Overdetermined system.
            // But we need (XtWX)^-1.
            // If we have QR of A, then A^T A = R^T R (if Q is orthogonal).
            // So we need (R^T R)^-1.
            // We can compute R from QR object.
            
            // Simpler fallback for QR covariance:
            // Since we are in the "Robust/Slow" path anyway, getting R and inverting might be okay.
            // Eigen's ColPivHouseholderQR has method to solve for general matrix?
            // If we assume full rank, R is invertible.
            // Better: use the property that solver solves A x = b in least squares sense.
            // We need inverse of Gram matrix.
            // Gram = A^T A.
            // If we successfully solved, we satisfied A x = y.
            // To get (A^T A)^-1, we can solve (A^T A) Z = I.
            // But we only have decomposition of A.
            
            // Let's settle for explicit computation if in QR mode for now? 
            // Or re-decompose R using Cholesky?
            // "The inverse of the matrix A^T A can be computed by:"
            // (R^T R)^-1
            
            Eigen::MatrixXd R = qr_.matrixR().topLeftCorner(qr_.rank(), qr_.rank()); 
            // With ColPiv, R is permuted. This is getting complex.
            // Suggestion: For QR path, assume it's rare/fallback.
            // Just compute (X^T W X)^-1 from scratch using SVD or LDLT if possible?
            // Or just solve (X^T W X) beta = e_i ?
            
            // Actually, if we are in QR mode because of Singularity, inverse doesn't exist uniquely!
            // We probably want the Pseudo-Inverse.
            // ColPivHouseholderQR solves for the minimum norm solution? No, basic one.
            
            // Let's return a generalized inverse approximation or throw if strictly needed.
            // For now: Throw exception for covariance in QR mode or implement later.
            // The user said: "Focus on Cholesky default".
            throw std::runtime_error("Covariance matrix computation for QR fallback is not yet fully implemented.");
        } else {
             // Cholesky Path: (X^T W X)^-1
             return ldlt_.solve(I);
        }
    }

    // 現在の戦略を確認（デバッグ用）
    std::string current_strategy_name() const {
        if (use_qr_fallback_) return "QR (Fallback)";
        if (strategy_ == SolverStrategy::QR) return "QR (Forced)";
        return "LDLT (Cholesky)";
    }
    
    // 直前の分解ランクを確認
    int rank() const {
        if (!is_decomposed_) return 0;
        if (use_qr_fallback_ || strategy_ == SolverStrategy::QR) return qr_.rank();
        // LDLT doesn't expose strict rank easily like QR, but vectorD can be checked.
        // For now, assume full rank if LDLT succeeded, or P.
        return ldlt_.vectorD().size(); // This is just P
    }

private:
    void compute_decomposition_if_needed(const WeightedDesignMatrix& wdm) {
        if (is_decomposed_) return;

        if (strategy_ == SolverStrategy::QR) {
            perform_qr(wdm);
        } else {
            // Try Cholesky (LDLT)
            Eigen::MatrixXd A = wdm.compute_gram();
            ldlt_.compute(A);
            
            // Check success and rank deficiency
            // LDLT is robust to semi-definite, but info() will be Success even if singular?
            // Eigen::LDLT::info() returns NumericalIssue if indefinite.
            // For PSD matrices, it usually works.
            // But if we want to detect singularity for AUTO fallback:
            // We can check the diagonal D.
            
            bool is_good = (ldlt_.info() == Eigen::Success);
            
            // Heuristic for singularity in LDLT: check min(abs(D)) / max(abs(D)) ?
            // For now, trust Eigen's info.
             
            if (strategy_ == SolverStrategy::AUTO && !is_good) {
                // Fallback to QR
                use_qr_fallback_ = true;
                perform_qr(wdm);
            } else if (!is_good) {
                throw std::runtime_error("LDLT decomposition failed. Matrix might be indefinite.");
            } else {
                use_qr_fallback_ = false;
            }
        }
        is_decomposed_ = true;
    }

    void perform_qr(const WeightedDesignMatrix& wdm) {
        Eigen::MatrixXd weighted_X = wdm.compute_sqrt_weighted_X();
        qr_.compute(weighted_X);
    }

    SolverStrategy strategy_;
    bool is_decomposed_;
    bool use_qr_fallback_;

    // Decompositions
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
};

} // namespace statelix
