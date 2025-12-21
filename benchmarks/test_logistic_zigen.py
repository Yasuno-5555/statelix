
import numpy as np
import statelix.linear_model as lm
import statelix

print("Statelix version:", statelix.__version__)

# Create synthetic data for logistic regression
np.random.seed(42)
n_samples = 1000
n_features = 5
X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 0.5, 0.0, 1.0])
intercept = 0.5
logits = X @ true_coef + intercept
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(n_samples) < probs).astype(float)

print("Data created. Running LogisticRegression...")

try:
    model = lm.LogisticRegression()
    model.max_iter = 100
    model.tol = 1e-6
    model.fit_intercept = True

    result = model.fit(X, y)
    print("Optimization finished.")
    print("Converged:", result.converged)
    print("Iterations:", result.iterations)
    print("Coefficients:", result.coef)
    print("Intercept:", result.intercept)
    print("Deviance:", result.deviance)
    print("AIC:", result.aic)
    
    # Check predictions
    pred_probs = model.predict_prob(X, result.coef, result.intercept)
    print("Predicted probs (first 5):", pred_probs[:5])
    
    # Validate against truth
    print("True coef:", true_coef)
    print("True intercept:", intercept)
    
    mse_coef = np.mean((result.coef - true_coef)**2)
    print(f"MSE Coef: {mse_coef:.6f}")
    if mse_coef < 0.1:
        print("✅ Logistic Regression Test Passed")
    else:
        print("❌ Logistic Regression Test Failed (High Error)")

except Exception as e:
    print(f"❌ Error: {e}")
