#ifndef STATELIX_MATH_UTILS_H
#define STATELIX_MATH_UTILS_H

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace statelix {
namespace stats {

    // Continued fraction for beta function
    inline double beta_cf(double a, double b, double x) {
        double am = 1, bm = 1, az = 1;
        double qab = a + b, qap = a + 1, qam = a - 1;
        double bz = 1.0 - qab * x / qap;
        
        for (int m = 1; m <= 100; ++m) {
            double em = m;
            double d = em * (b - m) * x / ((qam + 2*em) * (a + 2*em));
            double ap = az + d * am, bp = bz + d * bm;
            d = -(a + em) * (qab + em) * x / ((a + 2*em) * (qap + 2*em));
            double app = ap + d * az, bpp = bp + d * bz;
            double aold = az;
            am = ap / bpp; bm = bp / bpp; az = app / bpp; bz = 1.0;
            if (std::abs(az - aold) < 1e-10 * std::abs(az)) break;
        }
        return az;
    }

    // Regularized Incomplete Beta Function Ix(a, b)
    inline double beta_inc(double a, double b, double x) {
        if (x <= 0) return 0.0;
        if (x >= 1) return 1.0;
        
        double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                            a * std::log(x) + b * std::log(1.0 - x));
                            
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return bt * beta_cf(a, b, x) / a;
        } else {
            return 1.0 - bt * beta_cf(b, a, 1.0 - x) / b;
        }
    }

    // F-distribution Cumulative Distribution Function
    inline double f_cdf(double f, int df1, int df2) {
        if (f <= 0) return 0.0;
        double x = (double(df1) * f) / (double(df1) * f + double(df2));
        return beta_inc(df1 / 2.0, df2 / 2.0, x);
    }

    // F-distribution p-value (1 - CDF)
    inline double f_pvalue_approx(double f_stat, int df1, int df2) {
        if (f_stat <= 0.0) return 1.0;
        return 1.0 - f_cdf(f_stat, df1, df2);
    }

    // t-distribution Cumulative Distribution Function approximation
    inline double t_cdf_approx(double t, int df) {
        if (df > 30) {
            // Normal approximation for large df
            return 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)));
        }
        // Simplified t-distribution approximation
        double z = t / std::sqrt(static_cast<double>(df) / (df - 2.0));
        return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    }

} // namespace stats
} // namespace statelix

#endif // STATELIX_MATH_UTILS_H
