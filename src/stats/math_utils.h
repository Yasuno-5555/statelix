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

    
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    // Inverse Error Function
    inline double erfinv(double x) {
        double w = -std::log((1 - x) * (1 + x));
        double p;
        if (w < 5.0) {
            w -= 2.5;
            p = 2.81022636e-08 + w * (3.43273939e-07 + w * (-3.5233877e-06 +
                w * (-4.39150654e-06 + w * (0.00021858087 + w * (-0.00125372503 +
                w * (-0.00417768164 + w * (0.246640727 + w * 0.115956309)))))));
        } else {
            w = std::sqrt(w) - 3.0;
            p = -0.000200214257 + w * (0.000100950558 + w * (0.00134934322 +
                w * (-0.00367342844 + w * (0.00573950773 + w * (-0.0076224613 +
                w * (0.00943887047 + w * (1.00167406 + w * 0.00282095556)))))));
        }
        return p * x;
    }

    // Normal CDF
    inline double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    
    // Normal Quantile
    inline double normal_quantile(double p) {
        return std::sqrt(2.0) * erfinv(2.0 * p - 1.0);
    }
    
    // t-distribution Cumulative Distribution Function (Robust)
    inline double t_cdf(double t, int df) {
        if (df > 100) return normal_cdf(t);
        double x = df / (df + t * t);
        return 0.5 + 0.5 * std::copysign(1.0, t) * (1.0 - beta_inc(df / 2.0, 0.5, x));
    }
    
    // t-distribution Quantile
    inline double t_quantile(double p, int df) {
        if (df > 100) return normal_quantile(p);
        
        double t = (p > 0.5) ? std::sqrt(2.0) * erfinv(2.0 * p - 1.0) : 
                               -std::sqrt(2.0) * erfinv(1.0 - 2.0 * p);
        for (int i = 0; i < 5; ++i) {
            double cdf_val = t_cdf(t, df);
            double pdf = std::tgamma((df + 1.0) / 2.0) / 
                        (std::sqrt(df * M_PI) * std::tgamma(df / 2.0)) *
                        std::pow(1.0 + t * t / df, -(df + 1.0) / 2.0);
            t -= (cdf_val - p) / pdf;
        }
        return t;
    }
    
    // Alias for backward compatibility if needed, but prefer t_cdf
    inline double t_cdf_approx(double t, int df) { return t_cdf(t, df); }

} // namespace stats
} // namespace statelix

#endif // STATELIX_MATH_UTILS_H
