/**
 * @file glm_base.h
 * @brief Statelix v1.1 - GLM Foundation: Family & Link Functions
 * 
 * Design principle: Decouple distribution (Family) from link function (Link).
 * Any Family + Link + Penalizer combination becomes possible.
 */
#ifndef STATELIX_GLM_BASE_H
#define STATELIX_GLM_BASE_H

#include <Eigen/Dense>
#include <memory>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace statelix {

// =============================================================================
// Link Functions
// =============================================================================

/**
 * @brief Abstract base class for link functions
 * 
 * Link function g maps mean μ to linear predictor η: g(μ) = η
 * Inverse link maps η back to μ: μ = g^{-1}(η)
 */
class LinkFunction {
public:
    virtual ~LinkFunction() = default;
    
    /**
     * @brief Apply link function: η = g(μ)
     */
    virtual double link(double mu) const = 0;
    
    /**
     * @brief Apply inverse link: μ = g^{-1}(η)
     */
    virtual double inverse(double eta) const = 0;
    
    /**
     * @brief Derivative of inverse link: dμ/dη
     */
    virtual double inverse_derivative(double eta) const = 0;
    
    /**
     * @brief Vectorized versions
     */
    virtual Eigen::VectorXd link(const Eigen::VectorXd& mu) const {
        Eigen::VectorXd result(mu.size());
        for (int i = 0; i < mu.size(); ++i) {
            result(i) = link(mu(i));
        }
        return result;
    }
    
    virtual Eigen::VectorXd inverse(const Eigen::VectorXd& eta) const {
        Eigen::VectorXd result(eta.size());
        for (int i = 0; i < eta.size(); ++i) {
            result(i) = inverse(eta(i));
        }
        return result;
    }
    
    virtual Eigen::VectorXd inverse_derivative(const Eigen::VectorXd& eta) const {
        Eigen::VectorXd result(eta.size());
        for (int i = 0; i < eta.size(); ++i) {
            result(i) = inverse_derivative(eta(i));
        }
        return result;
    }
    
    /**
     * @brief Name for debugging/logging
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Identity link: g(μ) = μ
 */
class IdentityLink : public LinkFunction {
public:
    double link(double mu) const override { return mu; }
    double inverse(double eta) const override { return eta; }
    double inverse_derivative(double) const override { return 1.0; }
    std::string name() const override { return "identity"; }
};

/**
 * @brief Log link: g(μ) = log(μ)
 */
class LogLink : public LinkFunction {
public:
    double link(double mu) const override { 
        return std::log(std::max(mu, 1e-10)); 
    }
    double inverse(double eta) const override { 
        return std::exp(std::min(eta, 20.0));  // Prevent overflow
    }
    double inverse_derivative(double eta) const override { 
        return std::exp(std::min(eta, 20.0)); 
    }
    std::string name() const override { return "log"; }
};

/**
 * @brief Logit link: g(μ) = log(μ/(1-μ))
 */
class LogitLink : public LinkFunction {
public:
    double link(double mu) const override {
        mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
        return std::log(mu / (1.0 - mu));
    }
    double inverse(double eta) const override {
        // Numerically stable sigmoid
        if (eta >= 0) {
            double ez = std::exp(-eta);
            return 1.0 / (1.0 + ez);
        } else {
            double ez = std::exp(eta);
            return ez / (1.0 + ez);
        }
    }
    double inverse_derivative(double eta) const override {
        double p = inverse(eta);
        return p * (1.0 - p);
    }
    std::string name() const override { return "logit"; }
};

/**
 * @brief Probit link: g(μ) = Φ^{-1}(μ) (inverse normal CDF)
 */
class ProbitLink : public LinkFunction {
public:
    double link(double mu) const override {
        mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
        return probit(mu);
    }
    double inverse(double eta) const override {
        return normal_cdf(eta);
    }
    double inverse_derivative(double eta) const override {
        return normal_pdf(eta);
    }
    std::string name() const override { return "probit"; }
    
private:
    // Approximation to inverse normal CDF (Abramowitz & Stegun)
    static double probit(double p) {
        if (p <= 0) return -8.0;
        if (p >= 1) return 8.0;
        
        double t = std::sqrt(-2.0 * std::log(p < 0.5 ? p : 1.0 - p));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        double z = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t);
        return p < 0.5 ? -z : z;
    }
    
    // Standard normal CDF
    static double normal_cdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
    
    // Standard normal PDF
    static double normal_pdf(double x) {
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    }
};

/**
 * @brief Inverse link: g(μ) = 1/μ (canonical for Gamma)
 */
class InverseLink : public LinkFunction {
public:
    double link(double mu) const override {
        return 1.0 / std::max(mu, 1e-10);
    }
    double inverse(double eta) const override {
        return 1.0 / std::max(eta, 1e-10);
    }
    double inverse_derivative(double eta) const override {
        double eta_safe = std::max(eta, 1e-10);
        return -1.0 / (eta_safe * eta_safe);
    }
    std::string name() const override { return "inverse"; }
};

/**
 * @brief Inverse squared link: g(μ) = 1/μ² (canonical for Inverse Gaussian)
 */
class InverseSquaredLink : public LinkFunction {
public:
    double link(double mu) const override {
        return 1.0 / std::max(mu * mu, 1e-20);
    }
    double inverse(double eta) const override {
        return 1.0 / std::sqrt(std::max(eta, 1e-10));
    }
    double inverse_derivative(double eta) const override {
        double eta_safe = std::max(eta, 1e-10);
        return -0.5 / (eta_safe * std::sqrt(eta_safe));
    }
    std::string name() const override { return "inverse_squared"; }
};

// =============================================================================
// Distribution Families
// =============================================================================

/**
 * @brief Abstract base class for exponential family distributions
 */
class Family {
public:
    virtual ~Family() = default;
    
    /**
     * @brief Variance function V(μ): Var(Y) = φ * V(μ)
     */
    virtual double variance(double mu) const = 0;
    
    /**
     * @brief Unit deviance: d(y, μ) = 2 * [log p(y|y) - log p(y|μ)]
     */
    virtual double deviance_unit(double y, double mu) const = 0;
    
    /**
     * @brief Total deviance
     */
    virtual double deviance(const Eigen::VectorXd& y, 
                           const Eigen::VectorXd& mu) const {
        double d = 0.0;
        for (int i = 0; i < y.size(); ++i) {
            d += deviance_unit(y(i), mu(i));
        }
        return d;
    }
    
    /**
     * @brief Check if y values are valid for this family
     */
    virtual bool validate(double y) const = 0;
    
    /**
     * @brief Get the canonical (default) link function for this family
     */
    virtual std::unique_ptr<LinkFunction> canonical_link() const = 0;
    
    /**
     * @brief Family name
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Initialize μ from y (for IRLS starting values)
     */
    virtual double initialize_mu(double y) const { return y; }
    
    /**
     * @brief Vectorized variance
     */
    virtual Eigen::VectorXd variance(const Eigen::VectorXd& mu) const {
        Eigen::VectorXd result(mu.size());
        for (int i = 0; i < mu.size(); ++i) {
            result(i) = variance(mu(i));
        }
        return result;
    }
};

/**
 * @brief Gaussian (Normal) family: Y ~ N(μ, σ²)
 * Variance: V(μ) = 1
 */
class GaussianFamily : public Family {
public:
    double variance(double) const override { return 1.0; }
    
    double deviance_unit(double y, double mu) const override {
        double diff = y - mu;
        return diff * diff;
    }
    
    bool validate(double) const override { return true; }
    
    std::unique_ptr<LinkFunction> canonical_link() const override {
        return std::make_unique<IdentityLink>();
    }
    
    std::string name() const override { return "gaussian"; }
};

/**
 * @brief Binomial family: Y ~ Binomial(n, μ), typically n=1 (Bernoulli)
 * Variance: V(μ) = μ(1-μ)
 */
class BinomialFamily : public Family {
public:
    double variance(double mu) const override {
        mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
        return mu * (1.0 - mu);
    }
    
    double deviance_unit(double y, double mu) const override {
        mu = std::max(1e-10, std::min(1.0 - 1e-10, mu));
        double dev = 0.0;
        if (y > 0) dev += y * std::log(y / mu);
        if (y < 1) dev += (1.0 - y) * std::log((1.0 - y) / (1.0 - mu));
        return 2.0 * dev;
    }
    
    bool validate(double y) const override { return y >= 0 && y <= 1; }
    
    std::unique_ptr<LinkFunction> canonical_link() const override {
        return std::make_unique<LogitLink>();
    }
    
    std::string name() const override { return "binomial"; }
    
    double initialize_mu(double y) const override {
        return (y + 0.5) / 2.0;  // Shrink towards 0.5
    }
};

/**
 * @brief Poisson family: Y ~ Poisson(μ)
 * Variance: V(μ) = μ
 */
class PoissonFamily : public Family {
public:
    double variance(double mu) const override {
        return std::max(mu, 1e-10);
    }
    
    double deviance_unit(double y, double mu) const override {
        mu = std::max(mu, 1e-10);
        if (y == 0) return 2.0 * mu;
        return 2.0 * (y * std::log(y / mu) - (y - mu));
    }
    
    bool validate(double y) const override { return y >= 0; }
    
    std::unique_ptr<LinkFunction> canonical_link() const override {
        return std::make_unique<LogLink>();
    }
    
    std::string name() const override { return "poisson"; }
    
    double initialize_mu(double y) const override {
        return std::max(y, 0.1);  // Avoid log(0)
    }
};

/**
 * @brief Gamma family: Y ~ Gamma(shape, scale)
 * Variance: V(μ) = μ²
 */
class GammaFamily : public Family {
public:
    double variance(double mu) const override {
        mu = std::max(mu, 1e-10);
        return mu * mu;
    }
    
    double deviance_unit(double y, double mu) const override {
        y = std::max(y, 1e-10);
        mu = std::max(mu, 1e-10);
        return 2.0 * ((y - mu) / mu - std::log(y / mu));
    }
    
    bool validate(double y) const override { return y > 0; }
    
    std::unique_ptr<LinkFunction> canonical_link() const override {
        return std::make_unique<InverseLink>();
    }
    
    std::string name() const override { return "gamma"; }
    
    double initialize_mu(double y) const override {
        return std::max(y, 1e-3);
    }
};

/**
 * @brief Negative Binomial family: Y ~ NegBin(r, p) with mean μ
 * Variance: V(μ) = μ + μ²/θ where θ is the dispersion parameter
 */
class NegativeBinomialFamily : public Family {
public:
    double theta = 1.0;  // Dispersion parameter
    
    explicit NegativeBinomialFamily(double th = 1.0) : theta(th) {}
    
    double variance(double mu) const override {
        mu = std::max(mu, 1e-10);
        return mu + mu * mu / theta;
    }
    
    double deviance_unit(double y, double mu) const override {
        y = std::max(y, 0.0);
        mu = std::max(mu, 1e-10);
        double dev = 0.0;
        if (y > 0) {
            dev = y * std::log(y / mu);
        }
        dev -= (y + theta) * std::log((y + theta) / (mu + theta));
        return 2.0 * dev;
    }
    
    bool validate(double y) const override { return y >= 0; }
    
    std::unique_ptr<LinkFunction> canonical_link() const override {
        return std::make_unique<LogLink>();
    }
    
    std::string name() const override { return "negative_binomial"; }
    
    double initialize_mu(double y) const override {
        return std::max(y, 0.1);
    }
};

// =============================================================================
// Factory functions
// =============================================================================

inline std::unique_ptr<Family> make_family(const std::string& name) {
    if (name == "gaussian" || name == "normal") {
        return std::make_unique<GaussianFamily>();
    } else if (name == "binomial" || name == "bernoulli") {
        return std::make_unique<BinomialFamily>();
    } else if (name == "poisson") {
        return std::make_unique<PoissonFamily>();
    } else if (name == "gamma") {
        return std::make_unique<GammaFamily>();
    } else if (name == "negbin" || name == "negative_binomial") {
        return std::make_unique<NegativeBinomialFamily>();
    }
    throw std::invalid_argument("Unknown family: " + name);
}

inline std::unique_ptr<LinkFunction> make_link(const std::string& name) {
    if (name == "identity") {
        return std::make_unique<IdentityLink>();
    } else if (name == "log") {
        return std::make_unique<LogLink>();
    } else if (name == "logit") {
        return std::make_unique<LogitLink>();
    } else if (name == "probit") {
        return std::make_unique<ProbitLink>();
    } else if (name == "inverse") {
        return std::make_unique<InverseLink>();
    } else if (name == "inverse_squared") {
        return std::make_unique<InverseSquaredLink>();
    }
    throw std::invalid_argument("Unknown link: " + name);
}

} // namespace statelix

#endif // STATELIX_GLM_BASE_H
