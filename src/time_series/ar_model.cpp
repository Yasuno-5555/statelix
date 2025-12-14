#include "ar_model.h"
#include "../linear_model/ols.h" 
#include <iostream>

namespace statelix {

ARResult fit_ar(const Eigen::VectorXd& series, int p) {
    int n = series.size();
    if (n <= p) {
        throw std::invalid_argument("Time series length must be greater than order p");
    }

    int n_samples = n - p;
    
    // Design matrix: [1, y_{t-1}, ..., y_{t-p}]
    Eigen::MatrixXd X(n_samples, p + 1); // +1 for intercept
    Eigen::VectorXd y(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        // current time t = p + i
        y(i) = series(p + i);
        
        // intercept
        X(i, 0) = 1.0;
        
        // lags
        for (int j = 1; j <= p; ++j) {
            X(i, j) = series(p + i - j);
        }
    }

    // Solve OLS using qr (without adding intercept inside ols, since we added it manually)
    // Wait, fit_ols_qr takes raw X. Does it add intercept?
    // Let's check ols.h/cpp.
    // fit_ols_qr just does qr.solve(y). It treats X as the design matrix.
    // So if X has a column of ones, it fits intercept.
    
    Eigen::VectorXd params = fit_ols_qr(X, y); // This is from ols.h

    // Calculate residuals to estimate sigma2
    Eigen::VectorXd fitted = X * params;
    Eigen::VectorXd resid = y - fitted;
    double rss = resid.squaredNorm();
    double sigma2 = rss / (n_samples - (p + 1));

    return {params, sigma2, p};
}

} // namespace statelix
