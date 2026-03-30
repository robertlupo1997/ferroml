# Model Interpretability Guide

How to explain ML model predictions to different audiences using FerroML.

## Global vs Local Explanations

| Type | Question Answered | Methods |
|------|-------------------|---------|
| **Global** | "How does the model work overall?" | Coefficients, feature importances, summary() |
| **Local** | "Why did the model predict X for this input?" | Individual prediction breakdown, uncertainty |

## Model-Specific Interpretation

### Linear Models (OLS, Ridge, Lasso, LogisticRegression)

Linear models are the gold standard for interpretability.

```python
from ferroml.linear import OrdinaryLeastSquares

model = OrdinaryLeastSquares()
model.fit(X_train, y_train)

# Full statistical summary (like statsmodels)
print(model.summary())

# Coefficients with confidence intervals and p-values
coefs = model.coefficients_with_ci(confidence=0.95)
# Returns: [(name, coef, ci_lower, ci_upper, p_value), ...]

# Model-level diagnostics
print(f"R-squared: {model.r_squared():.4f}")
print(f"Adjusted R-squared: {model.adjusted_r_squared():.4f}")
print(f"F-statistic: {model.f_statistic():.2f}")
print(f"AIC: {model.aic():.1f}")
print(f"BIC: {model.bic():.1f}")
```

**What to report:**
- Coefficient sign: positive = increases target, negative = decreases
- Coefficient magnitude: only comparable when features are standardized
- p-value < 0.05: feature is statistically significant
- CI not crossing zero: feature has a reliable effect

### Tree-Based Models (DecisionTree, RandomForest, GBT)

```python
from ferroml.trees import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, seed=42)
model.fit(X_train, y_train)

# Feature importances (mean decrease in impurity)
importances = model.feature_importances()
# Returns array of importance scores, one per feature

# Sort and display
import numpy as np
indices = np.argsort(importances)[::-1]
for i in indices[:10]:
    print(f"  {feature_names[i]:<30} {importances[i]:.4f}")
```

**Caveats:**
- Impurity-based importances are biased toward high-cardinality features
- Use permutation importance for unbiased estimates (see Model-Agnostic section)
- Single trees are fully interpretable; ensembles are not

### Gaussian Process Models

```python
from ferroml.gaussian_process import GaussianProcessRegressor

model = GaussianProcessRegressor()
model.fit(X_train, y_train)

# Predictions with uncertainty
predictions = model.predict_with_uncertainty(X_test, confidence=0.95)
# Returns: (mean, lower_bound, upper_bound)

# High uncertainty = model is unsure = be cautious
```

**What uncertainty tells you:**
- Narrow CI: model is confident (data is similar to training data)
- Wide CI: model is uncertain (extrapolation or sparse region)
- Use uncertainty to flag predictions that need human review

## Model-Agnostic Methods

### Permutation Importance

Measures how much each feature contributes by shuffling it and measuring accuracy drop.

```python
from ferroml.metrics import accuracy_score
import numpy as np

def permutation_importance(model, X, y, n_repeats=10, seed=42):
    """Compute permutation importance for any model."""
    rng = np.random.RandomState(seed)
    baseline = accuracy_score(y, model.predict(X))
    importances = np.zeros(X.shape[1])

    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, col])
            score = accuracy_score(y, model.predict(X_permuted))
            scores.append(baseline - score)
        importances[col] = np.mean(scores)

    return importances
```

**Advantages over built-in importances:**
- Works with any model
- Not biased by feature cardinality
- Accounts for feature interactions (partially)

### Partial Dependence

Shows the marginal effect of a single feature on predictions.

```python
import numpy as np

def partial_dependence(model, X, feature_idx, grid_points=50):
    """Compute partial dependence for a single feature."""
    values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), grid_points)
    pd_values = []

    for val in values:
        X_modified = X.copy()
        X_modified[:, feature_idx] = val
        pd_values.append(model.predict(X_modified).mean())

    return values, np.array(pd_values)
```

## When to Use Which Method

| Audience | Model Type | Method | Complexity |
|----------|-----------|--------|------------|
| Regulator | Any | `summary()`, coefficients_with_ci | Full stats |
| Executive | Any | Top 3 feature importances | Simplified |
| Data scientist | Linear | `summary()`, residual plots | Detailed |
| Data scientist | Tree/Ensemble | Feature importances + permutation | Comparative |
| Data scientist | GP/Bayesian | `predict_with_uncertainty()` | Uncertainty |
| End user | Any | "Feature X had the biggest impact" | Plain English |

## Template Phrases for Non-Technical Audiences

### For Linear Models

> "Each unit increase in **{feature}** is associated with a **{coef:.2f}** {increase/decrease} in **{target}**, holding all other factors constant. We are 95% confident the true effect is between **{ci_lower:.2f}** and **{ci_upper:.2f}**."

> "The most important factors predicting **{target}** are: (1) **{feature_1}**, (2) **{feature_2}**, (3) **{feature_3}**."

### For Tree-Based Models

> "The model considers **{feature}** to be the most important factor, accounting for **{importance*100:.0f}%** of its decision-making."

> "When **{feature}** is above **{threshold}**, the model typically predicts **{outcome}**."

### For Uncertainty

> "The model predicts **{value:.1f}**, but the true value could be anywhere between **{lower:.1f}** and **{upper:.1f}** (95% confidence). {If wide: 'This prediction has high uncertainty -- consider gathering more data before acting on it.'}"

### For General Audiences

> "We tested whether **{feature}** matters, and the answer is **{yes/no}** (p={p_value:.3f}). A p-value below 0.05 means the effect is unlikely to be due to chance."

## Regulatory Requirements

### EU AI Act (2024)

- **Right to explanation**: Users of high-risk AI systems have the right to understand how decisions affecting them were made
- **Documentation**: Must document training data, model design, testing results
- **Use linear models or provide equivalent explanation** for high-risk categories (credit, hiring, healthcare)
- FerroML tools: `summary()`, `coefficients_with_ci()`, `model_card()`

### ECOA / Fair Lending (US)

- **Adverse action notices**: Must explain specific reasons for denial
- **Rank-ordered reasons**: List top factors that led to negative decision
- Use coefficient magnitudes or feature importances to rank factors
- FerroML tools: `coefficients_with_ci()`, `feature_importances()`

### General Best Practices

1. Prefer interpretable models (linear, small trees) for regulated domains
2. Document model selection rationale using `model_card()`
3. Record statistical significance of all features
4. Keep audit trail of experiments (see experiment-tracking-guide)
5. Test for disparate impact across protected classes

## FerroML Interpretability Tools Summary

| Tool | What It Does | Models |
|------|-------------|--------|
| `summary()` | Full statistical report | Linear models |
| `coefficients_with_ci()` | Coefficients + CI + p-values | Linear models |
| `r_squared()` | Variance explained | Linear models |
| `f_statistic()` | Overall model significance | Linear models |
| `aic()` / `bic()` | Model comparison criteria | Linear models |
| `feature_importances()` | Impurity-based importance | Tree models |
| `predict_with_uncertainty()` | Prediction + confidence bounds | GP, Probabilistic |
| `predict_proba()` | Class probabilities | Classifiers |
| `model_card()` | Structured model metadata | All 66 models |
| `recommend()` | Algorithm recommendations | Top-level function |
