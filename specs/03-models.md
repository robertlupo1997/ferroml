# Specification: Machine Learning Models

**Status**: 🔲 NOT IMPLEMENTED

## Overview

ML models with full statistical diagnostics - FerroML's key differentiator.

## Requirements

### Linear Models
- [ ] Linear Regression (OLS) with diagnostics
  - Coefficients with standard errors
  - R², adjusted R²
  - F-statistic
  - Residual diagnostics (normality, homoscedasticity)
  - Prediction intervals
- [ ] Ridge Regression (L2 regularization)
- [ ] Lasso Regression (L1 regularization)
- [ ] Elastic Net (L1 + L2)
- [ ] Logistic Regression
  - Coefficients as odds ratios
  - Wald tests for significance
  - ROC/AUC with CI

### Tree-Based Models
- [ ] Decision Tree (CART)
  - Feature importance
  - Tree visualization data
- [ ] Random Forest
  - OOB error estimation
  - Feature importance with CI
- [ ] Gradient Boosting
  - Learning curves
  - Feature importance

### Support Vector Machines
- [ ] SVC (classification)
- [ ] SVR (regression)

### Model Traits

```rust
pub trait Model: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
    fn feature_importance(&self) -> Option<Array1<f64>>;
    fn search_space(&self) -> SearchSpace;
}

pub trait StatisticalModel: Model {
    fn summary(&self) -> ModelSummary;  // R-style output
    fn diagnostics(&self) -> Diagnostics;
    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo>;
}

pub trait ProbabilisticModel: Model {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval>;
}
```

## Implementation Priority

1. LinearRegression (foundation - tests statistical diagnostics)
2. LogisticRegression (classification baseline)
3. DecisionTree (interpretable)
4. RandomForest (ensemble baseline)
5. GradientBoosting (performance)

## Acceptance Criteria

- [ ] Output matches sklearn within 1e-6 for numerical results
- [ ] All models implement Model trait
- [ ] Linear models include full R-style summary output
- [ ] Prediction intervals implemented for regression models

## Implementation Location

`ferroml-core/src/models/`

## Dependencies

- `nalgebra` for linear algebra
- `statrs` for distributions (t, F, chi-squared)
