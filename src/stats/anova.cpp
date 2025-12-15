#include "anova.h"
#include <cmath>
#include <numeric>
#include <iostream>

namespace statelix {

// Helper: Approx F-distribution p-value (Copied/Adapted from ols.cpp)
// Helper: Accurate F-distribution p-value using Incomplete Beta Function
namespace {
    // Continued fraction for beta function
    double beta_cf(double a, double b, double x) {
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
    double beta_inc(double a, double b, double x) {
        if (x <= 0) return 0.0;
        if (x >= 1) return 1.0;
        double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                            a * std::log(x) + b * std::log(1.0 - x));
        if (x < (a + 1.0) / (a + b + 2.0)) return bt * beta_cf(a, b, x) / a;
        return 1.0 - bt * beta_cf(b, a, 1.0 - x) / b;
    }

    double f_pvalue_approx(double f_stat, int df1, int df2) {
        if (f_stat <= 0.0) return 1.0;
        // F(d1, d2) cumulative distribution = I_x(d1/2, d2/2) where x = d1*F / (d1*F + d2)
        double x = (double(df1) * f_stat) / (double(df1) * f_stat + double(df2));
        // p-value = 1 - CDF
        return 1.0 - beta_inc(df1 / 2.0, df2 / 2.0, x); 
    }
}

AnoVaResult f_oneway(const std::vector<Eigen::VectorXd>& groups) {
    int k = groups.size();
    if (k < 2) {
        // Not enough groups
        return {}; 
    }

    double total_sum = 0.0;
    int total_n = 0;
    std::vector<double> group_means(k);
    std::vector<int> group_ns(k);

    for (int i = 0; i < k; ++i) {
        group_ns[i] = groups[i].size();
        double sum = groups[i].sum();
        group_means[i] = sum / group_ns[i];
        total_sum += sum;
        total_n += group_ns[i];
    }

    double grand_mean = total_sum / total_n;

    double ss_between = 0.0;
    double ss_within = 0.0;

    for (int i = 0; i < k; ++i) {
        // SS Between = sum(n_i * (mean_i - grand_mean)^2)
        ss_between += group_ns[i] * std::pow(group_means[i] - grand_mean, 2);

        // SS Within = sum((x_ij - mean_i)^2)
        // efficient way: (x - mean).squaredNorm()
        ss_within += (groups[i].array() - group_means[i]).square().sum();
    }

    double ss_total = ss_between + ss_within;

    int df_between = k - 1;
    int df_within = total_n - k;
    int df_total = total_n - 1;

    double ms_between = ss_between / df_between;
    double ms_within = ss_within / df_within;

    double f_stat;
    double p_val;

    if (ms_within < 1e-12) {
        if (ms_between < 1e-12) {
            // Constant data across all groups: F is undefined (0/0), p-value = 1.0 (no difference)
            f_stat = 0.0;
            p_val = 1.0;
        } else {
            // Perfect separation (Within-group variance is 0, but Means differ): F -> Infinity, p-value -> 0.0
            f_stat = std::numeric_limits<double>::infinity();
            p_val = 0.0;
        }
    } else {
        f_stat = ms_between / ms_within;
        p_val = f_pvalue_approx(f_stat, df_between, df_within);
    }
    AnoVaResult res;
    res.f_statistic = f_stat;
    res.p_value = p_val;
    res.ss_between = ss_between;
    res.ss_within = ss_within;
    res.ss_total = ss_total;
    res.df_between = df_between;
    res.df_within = df_within;
    res.df_total = df_total;
    res.ms_between = ms_between;
    res.ms_within = ms_within;

    return res;
}

AnoVaResult f_oneway_flat(const Eigen::VectorXd& data, const Eigen::VectorXi& groups) {
    // Convert flat data to groups
    int k = groups.maxCoeff() + 1; // Assuming 0-indexed groups
    std::vector<Eigen::VectorXd> group_vecs(k);
    
    // First pass count
    std::vector<int> counts(k, 0);
    for(int i=0; i<groups.size(); ++i) {
        if(groups[i] >= 0 && groups[i] < k) counts[groups[i]]++;
    }

    // Allocate
    for(int i=0; i<k; ++i) {
        group_vecs[i].resize(counts[i]);
    }

    // Fill
    std::vector<int> current_idx(k, 0);
    for(int i=0; i<groups.size(); ++i) {
        int g = groups[i];
        if(g >= 0 && g < k) {
            group_vecs[g](current_idx[g]++) = data(i);
        }
    }

    return f_oneway(group_vecs);
}

} // namespace statelix
